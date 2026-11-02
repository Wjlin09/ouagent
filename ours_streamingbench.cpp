// StreamingBench real-time test driver. 
//
// Streams one video on its own timeline, triggering QA pairs at their CSV
// ``time_stamp``. Generation and video ingestion are interleaved, so frames
// that arrive during an active query keep being appended to the persistent KV.
//
// One MP4 == one Agent session: start_session() at the top, end_session() at the
// bottom. A finished query is NOT a session boundary -- the KV/context is never
// cleared between questions, matching the Python ``run_streaming`` loop exactly.
//
// Output layout (benchmark result and execution trace are separate concerns):
//     <out-dir>/benchmark/<video_name>/query_results.csv   (query-level results)
//     <out-dir>/benchmark/<video_name>/video_summary.csv   (video-level aggregate)
//     <out-dir>/trace/<video_name>_trace.csv               (execution trace)

#include "agent/agent.h"
#include "agent/controller.h"
#include "agent/state.h"
#include "runtime/gemma_runtime.h"
#include "video/reader.h"
#include "video/sampler.h"
#include "trace/trace_recorder.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <optional>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#if defined(_WIN32)
#    include <direct.h>
#endif

namespace {

// ---------------------------------------------------------------------------
// Small utilities
// ---------------------------------------------------------------------------

double perf_seconds() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// ``str.strip()`` over ASCII whitespace.
std::string strip(const std::string & s) {
    const char * ws = " \t\r\n\f\v";
    const size_t b = s.find_first_not_of(ws);
    if (b == std::string::npos) {
        return "";
    }
    const size_t e = s.find_last_not_of(ws);
    return s.substr(b, e - b + 1);
}

// Python ``float.__repr__`` equivalent (shortest decimal string that round-trips
// back to the same double, with integral floats keeping a trailing ".0").
std::string py_repr_double(double v) {
    for (int prec = 1; prec <= 17; ++prec) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.*g", prec, v);
        double back = std::strtod(buf, nullptr);
        if (back == v) {
            std::string s = buf;
            if (s.find('.') == std::string::npos &&
                s.find('e') == std::string::npos &&
                s.find('E') == std::string::npos) {
                s += ".0";
            }
            return s;
        }
    }
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    std::string s = buf;
    if (s.find('.') == std::string::npos && s.find('e') == std::string::npos &&
        s.find('E') == std::string::npos) {
        s += ".0";
    }
    return s;
}

// Python ``repr()`` for a string (single/double quote choice + escapes), used
// only by the ``--validate`` schedule dump.
std::string py_repr_string(const std::string & s) {
    const bool has_single = s.find('\'') != std::string::npos;
    const bool has_double = s.find('"') != std::string::npos;
    const char q = (has_single && !has_double) ? '"' : '\'';
    std::string out;
    out += q;
    for (char c : s) {
        if (c == '\\') {
            out += "\\\\";
        } else if (c == q) {
            out += '\\';
            out += c;
        } else if (c == '\n') {
            out += "\\n";
        } else if (c == '\r') {
            out += "\\r";
        } else if (c == '\t') {
            out += "\\t";
        } else {
            out += c;
        }
    }
    out += q;
    return out;
}

std::string py_repr_list(const std::vector<std::string> & v) {
    std::string out = "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) {
            out += ", ";
        }
        out += py_repr_string(v[i]);
    }
    out += "]";
    return out;
}

// ---------------------------------------------------------------------------
// Path helpers (POSIX; no <filesystem> so we avoid the GCC 8 -lstdc++fs flag)
// ---------------------------------------------------------------------------

std::string path_dirname(const std::string & p) {
    const size_t pos = p.find_last_of("/\\");
    return pos == std::string::npos ? std::string() : p.substr(0, pos);
}

std::string path_join(const std::string & a, const std::string & b) {
    if (a.empty()) {
        return b;
    }
    if (a.back() == '/' || a.back() == '\\') {
        return a + b;
    }
    return a + "/" + b;
}

// ``mkdir -p`` equivalent (Python ``Path.mkdir(parents=True, exist_ok=True)``).
void make_dirs(const std::string & path) {
    size_t pos = 0;
    while (pos <= path.size()) {
        const size_t slash = path.find('/', pos);
        const std::string cur = (slash == std::string::npos) ? path : path.substr(0, slash);
        if (!cur.empty()) {
#if defined(_WIN32)
            _mkdir(cur.c_str());  // EEXIST is fine, best-effort
#else
            mkdir(cur.c_str(), 0755);  // EEXIST is fine, best-effort
#endif
        }
        if (slash == std::string::npos) {
            break;
        }
        pos = slash + 1;
    }
}

// ---------------------------------------------------------------------------
// CSV parsing (RFC4180, matching the Python csv module for this dataset)
// ---------------------------------------------------------------------------

std::vector<std::string> split_csv_line(const std::string & line) {
    std::vector<std::string> fields;
    std::string cur;
    bool in_quotes = false;
    for (size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (in_quotes) {
            if (c == '"') {
                if (i + 1 < line.size() && line[i + 1] == '"') {
                    cur += '"';
                    ++i;
                } else {
                    in_quotes = false;
                }
            } else {
                cur += c;
            }
        } else {
            if (c == '"') {
                in_quotes = true;
            } else if (c == ',') {
                fields.push_back(cur);
                cur.clear();
            } else {
                cur += c;
            }
        }
    }
    fields.push_back(cur);
    return fields;
}

struct CsvTable {
    std::vector<std::string> columns;
    std::vector<std::unordered_map<std::string, std::string>> rows;
};

CsvTable load_csv(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("StreamingBench CSV not found: " + path);
    }
    std::string data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    // utf-8-sig: strip a leading BOM if present.
    if (data.size() >= 3 && static_cast<unsigned char>(data[0]) == 0xEF &&
        static_cast<unsigned char>(data[1]) == 0xBB &&
        static_cast<unsigned char>(data[2]) == 0xBF) {
        data.erase(0, 3);
    }

    std::vector<std::string> lines;
    size_t pos = 0;
    while (pos < data.size()) {
        const size_t nl = data.find('\n', pos);
        std::string line = (nl == std::string::npos) ? data.substr(pos)
                                                     : data.substr(pos, nl - pos);
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        lines.push_back(line);
        if (nl == std::string::npos) {
            break;
        }
        pos = nl + 1;
    }
    if (!lines.empty() && lines.back().empty()) {
        lines.pop_back();
    }
    if (lines.empty()) {
        throw std::runtime_error("StreamingBench CSV is empty: " + path);
    }

    CsvTable table;
    table.columns = split_csv_line(lines[0]);
    for (size_t i = 1; i < lines.size(); ++i) {
        const std::vector<std::string> cells = split_csv_line(lines[i]);
        std::unordered_map<std::string, std::string> row;
        for (size_t c = 0; c < table.columns.size() && c < cells.size(); ++c) {
            row[table.columns[c]] = cells[c];
        }
        table.rows.push_back(std::move(row));
    }
    return table;
}

// ---------------------------------------------------------------------------
// StreamingBench QA parsing / mapping (ported 1:1 from the Python script)
// ---------------------------------------------------------------------------

double parse_time_stamp(const std::string & text) {
    const std::string t = strip(text);
    std::vector<std::string> parts;
    std::string cur;
    for (char c : t) {
        if (c == ':') {
            parts.push_back(cur);
            cur.clear();
        } else {
            cur += c;
        }
    }
    parts.push_back(cur);
    if (parts.size() == 2) {
        return std::stoi(parts[0]) * 60 + std::stod(parts[1]);
    }
    if (parts.size() == 3) {
        return std::stoi(parts[0]) * 3600 + std::stoi(parts[1]) * 60 + std::stod(parts[2]);
    }
    throw std::runtime_error("cannot parse time_stamp '" + text + "'");
}

// Escape regex metacharacters (equivalent of Python ``re.escape`` for the keys
// we see here: plain ``sample_NN`` identifiers carry none).
std::string regex_escape(const std::string & s) {
    const std::string meta = "\\.^$|()[]{}*+?";
    std::string out;
    for (char c : s) {
        if (meta.find(c) != std::string::npos) {
            out += '\\';
        }
        out += c;
    }
    return out;
}

// Decode a backslash escape inside a Python string literal. ``e`` is the char
// after the backslash; returns the decoded byte and whether it was a recognized
// single-character escape (so the caller knows how many chars were consumed).
char decode_py_escape(char e) {
    switch (e) {
        case '\\': return '\\';
        case '\'': return '\'';
        case '"': return '"';
        case 'n': return '\n';
        case 't': return '\t';
        case 'r': return '\r';
        case 'b': return '\b';
        case 'f': return '\f';
        case 'a': return '\a';
        case 'v': return '\v';
        default: return e;  // unknown escape: keep the char (Python <3.12)
    }
}

// Parse a Python list literal of strings, e.g. ``['A. ...', 'B. ...']``.
// Returns nullopt on any malformed input (the caller then falls back to ``[text]``,
// exactly like the Python ``ast.literal_eval`` except-branch).
std::optional<std::vector<std::string>> parse_python_string_list(const std::string & text) {
    const size_t n = text.size();
    size_t i = 0;
    auto skip_ws = [&]() {
        while (i < n && std::isspace(static_cast<unsigned char>(text[i]))) {
            ++i;
        }
    };
    skip_ws();
    if (i >= n || text[i] != '[') {
        return std::nullopt;
    }
    ++i;
    std::vector<std::string> out;
    skip_ws();
    if (i < n && text[i] == ']') {
        return out;  // empty list
    }
    while (true) {
        skip_ws();
        if (i >= n || (text[i] != '\'' && text[i] != '"')) {
            return std::nullopt;
        }
        const char q = text[i];
        ++i;
        std::string val;
        bool closed = false;
        while (i < n) {
            const char c = text[i];
            if (c == '\\') {
                if (i + 1 >= n) {
                    return std::nullopt;
                }
                val += decode_py_escape(text[i + 1]);
                i += 2;
                continue;
            }
            if (c == q) {
                ++i;
                closed = true;
                break;
            }
            val += c;
            ++i;
        }
        if (!closed) {
            return std::nullopt;
        }
        out.push_back(val);
        skip_ws();
        if (i < n && text[i] == ',') {
            ++i;
            continue;
        }
        if (i < n && text[i] == ']') {
            ++i;
            break;
        }
        return std::nullopt;
    }
    skip_ws();
    if (i != n) {
        return std::nullopt;
    }
    return out;
}

std::vector<std::string> parse_options(const std::string & raw) {
    const std::string text = strip(raw);
    if (text.empty()) {
        return {};
    }
    std::optional<std::vector<std::string>> parsed = parse_python_string_list(text);
    if (parsed.has_value()) {
        return *parsed;
    }
    return {text};  // ast.literal_eval failed -> return [text]
}

std::vector<std::unordered_map<std::string, std::string>> find_qa_for_video(
    const CsvTable & table, const std::string & video_key) {
    const std::string pattern = "_" + regex_escape(video_key) + "_([0-9]+)$";
    const std::regex re(pattern);

    std::vector<std::pair<int, const std::unordered_map<std::string, std::string> *>> matched;
    for (const auto & row : table.rows) {
        auto it = row.find("question_id");
        if (it == row.end()) {
            continue;
        }
        std::smatch m;
        if (std::regex_search(it->second, m, re)) {
            matched.emplace_back(std::stoi(m[1].str()), &row);
        }
    }
    if (matched.empty()) {
        throw std::runtime_error(
            "no CSV rows match video identifier '" + video_key +
            "' (looked for a '_" + video_key + "_<n>' suffix in question_id)");
    }
    std::sort(matched.begin(), matched.end(),
              [](const auto & a, const auto & b) { return a.first < b.first; });
    std::vector<std::unordered_map<std::string, std::string>> out;
    out.reserve(matched.size());
    for (const auto & p : matched) {
        out.push_back(*p.second);
    }
    return out;
}

struct QaEntry {
    std::string question_id;
    std::string task_type;
    std::string question;
    double time_stamp = 0.0;
    std::string answer;
    std::vector<std::string> options;
    std::string frames_required;
    std::string temporal_clue_type;

    // Transient driver state (the Python "_trigger_ts" / "_arrival_wall" /
    // "_start_wall" keys).
    double trigger_ts = 0.0;
    double arrival_wall = 0.0;
    double start_wall = 0.0;
};

std::vector<QaEntry> build_schedule(
    const std::vector<std::unordered_map<std::string, std::string>> & rows) {
    auto get = [](const std::unordered_map<std::string, std::string> & r,
                  const char * k, const char * def) -> std::string {
        auto it = r.find(k);
        return it != r.end() ? it->second : std::string(def);
    };
    std::vector<QaEntry> out;
    out.reserve(rows.size());
    for (const auto & r : rows) {
        QaEntry q;
        q.question_id = get(r, "question_id", "");
        q.task_type = get(r, "task_type", "");
        q.question = get(r, "question", "");
        q.time_stamp = parse_time_stamp(get(r, "time_stamp", ""));
        q.answer = strip(get(r, "answer", ""));
        q.options = parse_options(get(r, "options", ""));
        q.frames_required = get(r, "frames_required", "");
        q.temporal_clue_type = get(r, "temporal_clue_type", "");
        out.push_back(std::move(q));
    }
    std::stable_sort(out.begin(), out.end(),
                     [](const QaEntry & a, const QaEntry & b) { return a.time_stamp < b.time_stamp; });
    return out;
}

std::string build_prompt(const QaEntry & q) {
    std::string s = q.question + "\n\nOptions:\n";
    for (const std::string & o : q.options) {
        s += o;
        s += "\n";
    }
    s += "\nAnswer with the letter of the correct option (A, B, C, or D).";
    return s;
}

std::optional<std::string> extract_choice(const std::string & text) {
    if (text.empty()) {
        return std::nullopt;
    }
    static const std::regex re(R"(\b([A-D])\b)");
    std::smatch m;
    if (std::regex_search(text, m, re)) {
        return m[1].str();
    }
    return std::nullopt;
}

// ---------------------------------------------------------------------------
// Result records / summary (mirrors utils/benchmark_metrics.py schema)
// ---------------------------------------------------------------------------

struct QueryRecord {
    std::string video_name;
    std::string query_id;      // internal correlation id (from TraceRecorder)
    std::string task_type;
    double scheduled_timestamp = 0.0;
    std::optional<double> actual_trigger_timestamp;
    std::string question;
    std::string reference_answer;
    std::string model_output;
    std::optional<std::string> extracted_choice;
    bool correct = false;
    double queue_wait_time = 0.0;
    double generation_time = 0.0;
    double end_to_end_latency = 0.0;
};

struct VideoSummary {
    std::string video_name;
    double video_duration = 0.0;
    int num_questions = 0;
    int num_correct = 0;
    double accuracy = 0.0;
    double total_wall_time = 0.0;
    double processing_speed_ratio = 0.0;
    double rtf = 0.0;
    double avg_end_to_end_latency = 0.0;
    double avg_generation_time = 0.0;
    double avg_queue_wait_time = 0.0;
};

// Fixed field orders (verbatim from benchmark_metrics.py).
const std::vector<std::string> QUERY_FIELDS = {
    "video_name", "query_id", "task_type", "scheduled_timestamp",
    "actual_trigger_timestamp", "question", "reference_answer", "model_output",
    "extracted_choice", "correct", "queue_wait_time",
    "generation_time", "end_to_end_latency",
};

const std::vector<std::string> VIDEO_SUMMARY_FIELDS = {
    "video_name", "video_duration", "num_questions", "num_correct", "accuracy",
    "total_wall_time", "processing_speed_ratio", "rtf",
    "avg_end_to_end_latency", "avg_generation_time", "avg_queue_wait_time",
};

// A single result cell for CSV output (None already mapped to "").
std::string query_cell(const QueryRecord & q, const std::string & field) {
    if (field == "video_name") return q.video_name;
    if (field == "query_id") return q.query_id;
    if (field == "task_type") return q.task_type;
    if (field == "scheduled_timestamp") return py_repr_double(q.scheduled_timestamp);
    if (field == "actual_trigger_timestamp")
        return q.actual_trigger_timestamp ? py_repr_double(*q.actual_trigger_timestamp) : "";
    if (field == "question") return q.question;
    if (field == "reference_answer") return q.reference_answer;
    if (field == "model_output") return q.model_output;
    if (field == "extracted_choice") return q.extracted_choice ? *q.extracted_choice : "";
    if (field == "correct") return q.correct ? "True" : "False";
    if (field == "queue_wait_time") return py_repr_double(q.queue_wait_time);
    if (field == "generation_time") return py_repr_double(q.generation_time);
    if (field == "end_to_end_latency") return py_repr_double(q.end_to_end_latency);
    return "";
}

std::string summary_cell(const VideoSummary & s, const std::string & field) {
    if (field == "video_name") return s.video_name;
    if (field == "video_duration") return py_repr_double(s.video_duration);
    if (field == "num_questions") return std::to_string(s.num_questions);
    if (field == "num_correct") return std::to_string(s.num_correct);
    if (field == "accuracy") return py_repr_double(s.accuracy);
    if (field == "total_wall_time") return py_repr_double(s.total_wall_time);
    if (field == "processing_speed_ratio") return py_repr_double(s.processing_speed_ratio);
    if (field == "rtf") return py_repr_double(s.rtf);
    if (field == "avg_end_to_end_latency") return py_repr_double(s.avg_end_to_end_latency);
    if (field == "avg_generation_time") return py_repr_double(s.avg_generation_time);
    if (field == "avg_queue_wait_time") return py_repr_double(s.avg_queue_wait_time);
    return "";
}

// ---------------------------------------------------------------------------
// CSV output (utf-8-sig BOM + \r\n + QUOTE_MINIMAL)
// ---------------------------------------------------------------------------

std::string csv_quote(const std::string & s) {
    const bool need = s.find(',') != std::string::npos || s.find('"') != std::string::npos ||
                      s.find('\r') != std::string::npos || s.find('\n') != std::string::npos;
    if (!need) {
        return s;
    }
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') {
            out += "\"\"";
        } else {
            out += c;
        }
    }
    out += "\"";
    return out;
}

void write_csv(const std::string & path, const std::vector<std::string> & fields,
               const std::vector<std::vector<std::string>> & rows) {
    make_dirs(path_dirname(path));
    std::ofstream f(path, std::ios::binary);
    f << "\xEF\xBB\xBF";  // utf-8-sig BOM
    auto emit_row = [&](const std::vector<std::string> & cells) {
        for (size_t i = 0; i < cells.size(); ++i) {
            if (i) {
                f << ",";
            }
            f << csv_quote(cells[i]);
        }
        f << "\r\n";
    };
    emit_row(fields);
    for (const auto & row : rows) {
        emit_row(row);
    }
}

void write_query_csv(const std::string & path, const std::vector<QueryRecord> & queries) {
    std::vector<std::vector<std::string>> rows;
    rows.reserve(queries.size());
    for (const QueryRecord & q : queries) {
        std::vector<std::string> cells;
        cells.reserve(QUERY_FIELDS.size());
        for (const std::string & f : QUERY_FIELDS) {
            cells.push_back(query_cell(q, f));
        }
        rows.push_back(std::move(cells));
    }
    write_csv(path, QUERY_FIELDS, rows);
}

void write_video_summary_csv(const std::string & path, const std::vector<VideoSummary> & summaries) {
    std::vector<std::vector<std::string>> rows;
    rows.reserve(summaries.size());
    for (const VideoSummary & s : summaries) {
        std::vector<std::string> cells;
        cells.reserve(VIDEO_SUMMARY_FIELDS.size());
        for (const std::string & f : VIDEO_SUMMARY_FIELDS) {
            cells.push_back(summary_cell(s, f));
        }
        rows.push_back(std::move(cells));
    }
    write_csv(path, VIDEO_SUMMARY_FIELDS, rows);
}

// ---------------------------------------------------------------------------
// Interleaved streaming driver (the Python ``run_streaming`` control flow)
// ---------------------------------------------------------------------------

struct StreamingResult {
    std::vector<QueryRecord> queries;
    double total_wall_time = 0.0;
};

std::string video_key_from(const std::string & video_path) {
    // Video identifier = the directory containing the mp4 (Python
    // ``video_path.parent.name``), e.g. ``.../video/sample_51/video.mp4`` -> ``sample_51``.
    const std::string parent = path_dirname(video_path);
    const size_t slash = parent.find_last_of("/\\");
    const std::string key = parent.substr(slash + 1);
    if (key.empty() || key == "." || key == ".." || key == "video" || key == "StreamingBench") {
        throw std::runtime_error(
            "cannot derive a video identifier from '" + video_path +
            "'; expected a path like .../video/<sample_id>/video.mp4");
    }
    return key;
}

StreamingResult run_streaming(ours::Agent & agent, ours::FrameSource & reader,
                              ours::FrameSampler & sampler,
                              const std::vector<QaEntry> & schedule,
                              const std::string & video_name, double total_duration,
                              ours::TraceRecorder & trace) {
    const double session_start = perf_seconds();
    agent.start_session();

    std::vector<QaEntry> pending = schedule;
    std::optional<QaEntry> active;
    bool video_done = false;
    double current_ts = 0.0;
    std::vector<QueryRecord> query_records;
    double last_progress = -1.0;

    auto finish_query = [&](const QaEntry & qa, const std::string & output) {
        const double wall_end = ours::now_seconds();
        const std::optional<std::string> choice = extract_choice(output);

        QueryRecord rec;
        rec.video_name = video_name;
        rec.query_id = trace.current_query_id();
        rec.task_type = qa.task_type;
        rec.scheduled_timestamp = qa.time_stamp;
        rec.actual_trigger_timestamp = qa.trigger_ts;
        rec.question = qa.question;
        rec.reference_answer = qa.answer;
        rec.model_output = output;
        rec.extracted_choice = choice;
        rec.correct = qa.answer.empty() ? false : (choice.has_value() && *choice == qa.answer);
        rec.queue_wait_time = qa.start_wall - qa.arrival_wall;
        rec.generation_time = wall_end - qa.start_wall;
        rec.end_to_end_latency = wall_end - qa.arrival_wall;

        query_records.push_back(rec);
        std::printf("[Q%zu/%zu] answer=%s | correct=%s | E2E=%.2fs\n",
                    query_records.size(), schedule.size(),
                    (choice ? choice->c_str() : "-"),
                    rec.correct ? "True" : "False", rec.end_to_end_latency);
    };

    auto skipped_record = [&](const QaEntry & qa) {
        QueryRecord rec;
        rec.video_name = video_name;
        rec.query_id = "";
        rec.task_type = qa.task_type;
        rec.scheduled_timestamp = qa.time_stamp;
        rec.actual_trigger_timestamp = std::nullopt;
        rec.question = qa.question;
        rec.reference_answer = qa.answer;
        rec.model_output = "";
        rec.extracted_choice = std::nullopt;
        rec.correct = false;
        rec.queue_wait_time = 0.0;
        rec.generation_time = 0.0;
        rec.end_to_end_latency = 0.0;
        return rec;
    };

    try {
        while (true) {
            // 1. Advance the video timeline by one sampled frame.
            if (!video_done) {
                ours::VideoFrame f;
                if (sampler.next(reader, f)) {
                    agent.ingest_frame(f.image, f.timestamp);
                    current_ts = f.timestamp;
                    if (f.timestamp - last_progress >= 5.0) {
                        last_progress = f.timestamp;
                        if (total_duration > 0.0) {
                            const double pct = std::min(100.0, f.timestamp / total_duration * 100.0);
                            std::printf("Progress: %.0f%% | video=%.1fs\n", pct, f.timestamp);
                        } else {
                            std::printf("Progress: video=%.1fs\n", f.timestamp);
                        }
                    }
                } else {
                    video_done = true;
                    trace.event("video_eos");
                    agent.flush_chunk();
                }
            }

            // 2. Trigger the next QA the moment the timeline reaches its time_stamp.
            if (!active.has_value() && !pending.empty() &&
                current_ts >= pending.front().time_stamp) {
                QaEntry qa = pending.front();
                pending.erase(pending.begin());
                qa.trigger_ts = current_ts;
                qa.arrival_wall = ours::now_seconds();
                agent.start_query(build_prompt(qa));
                qa.start_wall = ours::now_seconds();
                active = std::move(qa);
            }

            // 3. Tick the Agent: its Controller selects the action (Generate /
            //    AppendVideo / ...), interleaved with further ingestion above.
            const ours::StepOutcome out = agent.step();
            if (active.has_value() && out.query_done) {
                finish_query(*active, out.output);
                active.reset();
            }

            // 4. Termination: video exhausted, no active query, nothing reachable.
            if (video_done && !active.has_value()) {
                if (pending.empty()) {
                    break;
                }
                if (pending.front().time_stamp > current_ts) {
                    break;
                }
            }
        }
    } catch (...) {
        agent.end_session();
        throw;
    }
    agent.end_session();

    // Any QA whose timestamp the video never reached is recorded as skipped.
    for (const QaEntry & qa : pending) {
        query_records.push_back(skipped_record(qa));
    }

    StreamingResult result;
    result.queries = std::move(query_records);
    result.total_wall_time = perf_seconds() - session_start;
    return result;
}

VideoSummary summarize(const std::string & video_name, double total_duration, double total_wall_time,
                       const std::vector<QueryRecord> & queries) {
    VideoSummary s;
    s.video_name = video_name;
    s.video_duration = total_duration;
    s.num_questions = static_cast<int>(queries.size());
    int correct = 0;
    double e2e = 0.0, gen = 0.0, queue = 0.0;
    for (const QueryRecord & q : queries) {
        if (q.correct) {
            ++correct;
        }
        e2e += q.end_to_end_latency;
        gen += q.generation_time;
        queue += q.queue_wait_time;
    }
    s.num_correct = correct;
    s.accuracy = s.num_questions ? static_cast<double>(correct) / s.num_questions : 0.0;
    s.total_wall_time = total_wall_time;
    s.processing_speed_ratio = total_wall_time ? total_duration / total_wall_time : 0.0;
    s.rtf = total_duration ? total_wall_time / total_duration : 0.0;
    s.avg_end_to_end_latency = s.num_questions ? e2e / s.num_questions : 0.0;
    s.avg_generation_time = s.num_questions ? gen / s.num_questions : 0.0;
    s.avg_queue_wait_time = s.num_questions ? queue / s.num_questions : 0.0;
    return s;
}

void print_schedule(const std::string & video_path, const std::string & video_key,
                    const std::vector<QaEntry> & schedule) {
    std::printf("video: %s\n", video_path.c_str());
    std::printf("video identifier: %s\n", video_key.c_str());
    std::printf("questions: %zu\n", schedule.size());
    for (size_t i = 0; i < schedule.size(); ++i) {
        const QaEntry & q = schedule[i];
        std::printf("  [%zu] %s\n", i + 1, q.question_id.c_str());
        std::printf("      time_stamp=%.2fs  answer=%s  clue=%s  frames=%s\n",
                    q.time_stamp, py_repr_string(q.answer).c_str(),
                    q.temporal_clue_type.c_str(), q.frames_required.c_str());
        std::printf("      Q: %s\n", q.question.c_str());
        if (!q.options.empty()) {
            std::printf("      options: %s\n", py_repr_list(q.options).c_str());
        }
    }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct Args {
    std::vector<std::string> videos;
    std::string model;
    std::string mmproj;
    std::string csv = "data/StreamingBench/annotations/Real_Time_Visual_Understanding.csv";
    std::string out_dir = "results/streamingbench";
    uint32_t n_ctx = 65536;
    uint32_t n_batch = 2048;
    uint32_t n_ubatch = 512;
    int32_t n_gpu_layers = -1;
    float temperature = 0.0f;
    float top_p = 0.95f;
    float repeat_penalty = 1.1f;
    uint32_t max_tokens = 1024;
    uint32_t seed = 1234;
    double fps = 1.0;
    int num_frameproc = 4;
    uint32_t resize_max_size = 336;
    std::optional<int> limit;
    bool validate = false;
    bool verbose = false;
};

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --video <path|dir> [<path> ...] --model <gguf> --mmproj <gguf> [options]\n"
        "  --video <path|dir>       input video(s): one file, a directory of samples,\n"
        "                           or several files (required)\n"
        "  --model <gguf>           text model GGUF (required)\n"
        "  --mmproj <gguf>          mmproj GGUF (required)\n"
        "  --csv <path>             StreamingBench QA CSV\n"
        "                           (default data/StreamingBench/annotations/Real_Time_Visual_Understanding.csv)\n"
        "  --out-dir <path>         root output dir (default results/streamingbench)\n"
        "  --fps <float>            sampled frames/second (default 1.0)\n"
        "  --limit <int>            answer only the first N scheduled questions (smoke test)\n"
        "  --validate               parse CSV + print schedule, then exit (no model/video)\n"
        "  --n-ctx <int>            (default 65536)\n"
        "  --n-gpu-layers <int>     -1 = all (default -1)\n"
        "  --temperature <float>    (default 1.0)\n"
        "  --top-p <float>          (default 0.95)\n"
        "  --repeat-penalty <float> (default 1.1)\n"
        "  --max-tokens <int>       per-query cap (default 1024)\n"
        "  --num-frameproc <int>    frames per chunk (default 4)\n"
        "  --resize <int>           long-side cap px, 0 = off (default 336)\n"
        "  --verbose                enable native verbose diagnostics\n",
        argv0);
}

bool parse_uint(const char * s, uint32_t & out) {
    if (s == nullptr || *s == '\0') return false;
    char * end = nullptr;
    const unsigned long v = std::strtoul(s, &end, 10);
    if (end == s || *end != '\0') return false;
    out = static_cast<uint32_t>(v);
    return true;
}

bool parse_int(const char * s, int & out) {
    if (s == nullptr || *s == '\0') return false;
    char * end = nullptr;
    const long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0') return false;
    out = static_cast<int>(v);
    return true;
}

bool parse_float(const char * s, float & out) {
    if (s == nullptr || *s == '\0') return false;
    char * end = nullptr;
    const double v = std::strtod(s, &end);
    if (end == s || *end != '\0') return false;
    out = static_cast<float>(v);
    return true;
}

int parse_args(int argc, char ** argv, Args & a) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        bool ok = true;
        if (arg == "--video") {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for --video\n");
                std::exit(2);
            }
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                a.videos.emplace_back(argv[++i]);
            }
        } else if (arg == "--model") {
            a.model = next("--model");
        } else if (arg == "--mmproj") {
            a.mmproj = next("--mmproj");
        } else if (arg == "--csv") {
            a.csv = next("--csv");
        } else if (arg == "--out-dir") {
            a.out_dir = next("--out-dir");
        } else if (arg == "--fps") {
            float v = 0;
            ok = parse_float(next("--fps"), v);
            a.fps = v;
        } else if (arg == "--limit") {
            int v = 0;
            ok = parse_int(next("--limit"), v);
            a.limit = v;
        } else if (arg == "--validate") {
            a.validate = true;
        } else if (arg == "--n-ctx") {
            ok = parse_uint(next("--n-ctx"), a.n_ctx);
        } else if (arg == "--n-batch") {
            ok = parse_uint(next("--n-batch"), a.n_batch);
        } else if (arg == "--n-ubatch") {
            ok = parse_uint(next("--n-ubatch"), a.n_ubatch);
        } else if (arg == "--n-gpu-layers") {
            uint32_t v = 0;
            ok = parse_uint(next("--n-gpu-layers"), v);
            a.n_gpu_layers = static_cast<int32_t>(v);
        } else if (arg == "--temperature") {
            ok = parse_float(next("--temperature"), a.temperature);
        } else if (arg == "--top-p") {
            ok = parse_float(next("--top-p"), a.top_p);
        } else if (arg == "--repeat-penalty") {
            ok = parse_float(next("--repeat-penalty"), a.repeat_penalty);
        } else if (arg == "--max-tokens") {
            ok = parse_uint(next("--max-tokens"), a.max_tokens);
        } else if (arg == "--num-frameproc") {
            ok = parse_int(next("--num-frameproc"), a.num_frameproc);
        } else if (arg == "--resize") {
            ok = parse_uint(next("--resize"), a.resize_max_size);
        } else if (arg == "--verbose") {
            a.verbose = true;
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            usage(argv[0]);
            return 2;
        }
        if (!ok) {
            std::fprintf(stderr, "invalid value for %s\n", arg.c_str());
            return 2;
        }
    }
    return -1;  // sentinel: continue
}

// -- video path expansion ----------------------------------------------------

bool is_directory(const std::string & path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

// StreamingBench layout: <dir>/<sample_NN>/video.mp4. List the per-sample video
// files directly under ``dir``.
std::vector<std::string> list_sample_videos(const std::string & dir) {
    std::vector<std::string> out;
    DIR * d = opendir(dir.c_str());
    if (d == nullptr) {
        return out;
    }
    while (const struct dirent * e = readdir(d)) {
        const std::string name = e->d_name;
        if (name == "." || name == "..") {
            continue;
        }
        const std::string candidate = path_join(path_join(dir, name), "video.mp4");
        struct stat st;
        if (stat(candidate.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
            out.push_back(candidate);
        }
    }
    closedir(d);
    std::sort(out.begin(), out.end());
    return out;
}

// Turn each --video argument into concrete mp4 paths: directories expand to
// their samples; files are kept as-is.
std::vector<std::string> expand_video_paths(const std::vector<std::string> & paths) {
    std::vector<std::string> out;
    for (const std::string & p : paths) {
        if (is_directory(p)) {
            const std::vector<std::string> found = list_sample_videos(p);
            out.insert(out.end(), found.begin(), found.end());
        } else {
            out.push_back(p);
        }
    }
    return out;
}

} // namespace

int main(int argc, char ** argv) {
    Args a;
    const int early = parse_args(argc, argv, a);
    if (early >= 0) {
        return early;
    }
    // Expand --video args: directories become their per-sample video.mp4 files.
    std::vector<std::string> videos = expand_video_paths(a.videos);
    if (videos.empty() || a.model.empty() || a.mmproj.empty()) {
        usage(argv[0]);
        return 2;
    }

    try {
        const CsvTable csv = load_csv(a.csv);

        // --validate: parse CSV + print the schedule for each video, then exit.
        if (a.validate) {
            for (const std::string & vp : videos) {
                const std::string video_key = video_key_from(vp);
                const std::vector<QaEntry> schedule =
                    build_schedule(find_qa_for_video(csv, video_key));
                print_schedule(vp, video_key, schedule);
            }
            return 0;
        }

        ours::GemmaRuntimeConfig cfg;
        cfg.model_path = a.model;
        cfg.mmproj_path = a.mmproj;
        cfg.n_ctx = a.n_ctx;
        cfg.n_batch = a.n_batch;
        cfg.n_ubatch = a.n_ubatch;
        cfg.n_gpu_layers = a.n_gpu_layers;
        cfg.temperature = a.temperature;
        cfg.top_p = a.top_p;
        cfg.repeat_penalty = a.repeat_penalty;
        cfg.max_tokens = a.max_tokens;
        cfg.seed = a.seed;
        cfg.verbose = a.verbose;

        // The model is loaded once and reused across videos (one Agent session
        // per video; start_session/end_session reset the KV between videos).
        ours::TraceRecorder trace;
        ours::GemmaRuntime backend(cfg, &trace);
        backend.load();

        const std::string benchmark_dir = path_join(a.out_dir, "benchmark");
        const std::string trace_dir = path_join(a.out_dir, "trace");

        for (size_t vi = 0; vi < videos.size(); ++vi) {
            const std::string & vp = videos[vi];
            const std::string video_key = video_key_from(vp);
            std::vector<QaEntry> schedule =
                build_schedule(find_qa_for_video(csv, video_key));
            if (a.limit.has_value()) {
                int lim = *a.limit;
                if (lim < 0) {
                    lim = std::max(0, static_cast<int>(schedule.size()) + lim);
                }
                if (static_cast<size_t>(lim) < schedule.size()) {
                    schedule.resize(static_cast<size_t>(lim));
                }
            }

            ours::VideoReader reader(vp);
            std::string err;
            if (!reader.open(err)) {
                std::fprintf(stderr, "error: %s\n", err.c_str());
                return 1;
            }

            // Uncapped stream (Python passes max_frames=None); one MP4 = one session.
            ours::FrameSampler sampler(a.fps, /*max_frames=*/-1);
            ours::AgentState state;
            ours::Controller controller;
            ours::Agent agent(&backend, state, controller,
                              /*frame_selector=*/nullptr, /*semantic_pruner=*/nullptr,
                              a.num_frameproc, a.resize_max_size, &trace);

            double total_duration = reader.duration();
            if (total_duration <= 0.0) {
                total_duration = 0.0;
                for (const QaEntry & q : schedule) {
                    total_duration = std::max(total_duration, q.time_stamp);
                }
            }

            std::printf("[%zu/%zu] %s.mp4\n", vi + 1, videos.size(), video_key.c_str());

            trace.begin_run();  // fresh trace per video

            StreamingResult res =
                run_streaming(agent, reader, sampler, schedule, video_key, total_duration, trace);

            const VideoSummary summary =
                summarize(video_key, total_duration, res.total_wall_time, res.queries);

            const std::string video_benchmark_dir = path_join(benchmark_dir, video_key);
            write_query_csv(path_join(video_benchmark_dir, "query_results.csv"), res.queries);
            write_video_summary_csv(path_join(video_benchmark_dir, "video_summary.csv"), {summary});
            make_dirs(trace_dir);
            trace.write_csv(path_join(trace_dir, video_key + "_trace.csv"));

            std::printf("[Video Done] %s | accuracy=%d/%d (%.1f%%)\n",
                        video_key.c_str(), summary.num_correct, summary.num_questions,
                        summary.accuracy * 100.0);
            std::printf("[Saved] %s/\n", video_benchmark_dir.c_str());
        }
        return 0;
    } catch (const ours::GemmaRuntimeError & exc) {
        std::fprintf(stderr, "error: %s\n", exc.what());
        return 1;
    } catch (const std::exception & exc) {
        std::fprintf(stderr, "error: %s\n", exc.what());
        return 1;
    }
}
// Centralized execution trace recorder for OURS_Agent.
//
// Agent / StreamingBench / GemmaRuntime only report
// that an event happened or an operation started/ended, and TraceRecorder owns
// the correlation IDs (run -> session -> query -> chunk -> record) plus the CSV
// serialization. Business code never mints its own IDs or writes the CSV itself.
//
// Header-only (like evaluation/latency.h), so no build changes are required.

#ifndef OURS_NATIVE_TRACE_RECORDER_H
#define OURS_NATIVE_TRACE_RECORDER_H

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace ours {

// One structured record (an event or an operation).
struct TraceRecord {
    std::string record_id;
    std::string record_type;       // "event" | "operation"
    std::string run_id;
    std::string session_id;
    std::string query_id;
    std::string chunk_id;
    std::string parent_id;         // reserved for execution hierarchy (empty)
    std::string name;
    double start_epoch = 0.0;      // wall-clock seconds (system_clock)
    double end_epoch = 0.0;        // valid only for operations
    double video_timestamp = -1.0; // seconds on the video timeline; <0 => empty
    std::string action;            // e.g. the Action name for action_selected
};

class TraceSpan;

class TraceRecorder {
public:
    // One run == one StreamingBench video execution.
    void begin_run() {
        records_.clear();
        record_counter_ = 0;
        session_counter_ = 0;
        query_counter_ = 0;
        chunk_counter_ = 0;
        session_id_.clear();
        current_query_id_.clear();
        current_chunk_id_.clear();
    }

    // Record an instant event. A few names advance the correlation context:
    //   "session_start" -> allocates session_id
    //   "query_arrival" -> allocates current_query_id
    //   "chunk_ready"   -> allocates current_chunk_id
    void event(const std::string & name, double video_ts = -1.0,
               const std::string & action = "") {
        if (name == "session_start") {
            session_id_ = "session-" + std::to_string(++session_counter_);
        } else if (name == "query_arrival") {
            current_query_id_ = "query-" + std::to_string(++query_counter_);
        } else if (name == "chunk_ready") {
            current_chunk_id_ = "chunk-" + std::to_string(++chunk_counter_);
        }
        TraceRecord r;
        r.record_id = next_record_id();
        r.record_type = "event";
        r.run_id = run_id_;
        r.session_id = session_id_;
        r.query_id = current_query_id_;
        r.chunk_id = current_chunk_id_;
        r.name = name;
        r.start_epoch = now_epoch();
        r.video_timestamp = video_ts;
        r.action = action;
        records_.push_back(std::move(r));
    }

    // Begin an operation; the returned span finalizes it when it goes out of
    // scope (or on `.end()`).
    TraceSpan begin_operation(const std::string & name, double video_ts = -1.0);

    // Allocate a chunk id for a residual chunk flushed without a chunk_ready
    // event (video end or pre-query partial flush).
    void begin_chunk() {
        current_chunk_id_ = "chunk-" + std::to_string(++chunk_counter_);
    }

    // Correlation accessors (the benchmark joins on these IDs).
    const std::string & run_id() const { return run_id_; }
    const std::string & session_id() const { return session_id_; }
    const std::string & current_query_id() const { return current_query_id_; }
    const std::string & current_chunk_id() const { return current_chunk_id_; }

    // Serialize all records to CSV (utf-8-sig + \r\n, matching benchmark CSVs).
    void write_csv(const std::string & path) const;

private:
    friend class TraceSpan;
    void finish_operation(const std::string & record_id, double end_epoch);
    std::string next_record_id() { return std::to_string(++record_counter_); }
    static double now_epoch() {
        return std::chrono::duration<double>(
                   std::chrono::system_clock::now().time_since_epoch())
            .count();
    }
    static std::string format_wall_time(double epoch);

    std::string run_id_ = "run-1";
    std::string session_id_;
    std::string current_query_id_;
    std::string current_chunk_id_;
    int64_t record_counter_ = 0;
    int64_t session_counter_ = 0;
    int64_t query_counter_ = 0;
    int64_t chunk_counter_ = 0;
    std::vector<TraceRecord> records_;
};

// RAII handle for an in-flight operation.
class TraceSpan {
public:
    TraceSpan() = default;
    TraceSpan(TraceRecorder * r, std::string id)
        : recorder_(r), record_id_(std::move(id)) {}
    TraceSpan(TraceSpan && o) noexcept
        : recorder_(o.recorder_), record_id_(std::move(o.record_id_)) {
        o.recorder_ = nullptr;
    }
    TraceSpan & operator=(TraceSpan && o) noexcept {
        if (this != &o) {
            end();
            recorder_ = o.recorder_;
            record_id_ = std::move(o.record_id_);
            o.recorder_ = nullptr;
        }
        return *this;
    }
    TraceSpan(const TraceSpan &) = delete;
    TraceSpan & operator=(const TraceSpan &) = delete;

    bool is_active() const { return recorder_ != nullptr; }
    void end();
    ~TraceSpan() { end(); }

private:
    TraceRecorder * recorder_ = nullptr;
    std::string record_id_;
};

// -- inline definitions ------------------------------------------------------

inline TraceSpan TraceRecorder::begin_operation(const std::string & name,
                                                double video_ts) {
    std::string id = next_record_id();
    TraceRecord r;
    r.record_id = id;
    r.record_type = "operation";
    r.run_id = run_id_;
    r.session_id = session_id_;
    r.query_id = current_query_id_;
    r.chunk_id = current_chunk_id_;
    r.name = name;
    r.start_epoch = now_epoch();
    r.video_timestamp = video_ts;
    records_.push_back(std::move(r));
    return TraceSpan(this, id);
}

inline void TraceSpan::end() {
    if (recorder_ != nullptr) {
        recorder_->finish_operation(record_id_, TraceRecorder::now_epoch());
        recorder_ = nullptr;
    }
}

inline void TraceRecorder::finish_operation(const std::string & record_id,
                                            double end_epoch) {
    for (TraceRecord & r : records_) {
        if (r.record_id == record_id) {
            r.end_epoch = end_epoch;
            return;
        }
    }
}

inline std::string TraceRecorder::format_wall_time(double epoch) {
    std::time_t secs = static_cast<std::time_t>(std::floor(epoch));
    int ms = static_cast<int>(std::lround((epoch - std::floor(epoch)) * 1000.0));
    if (ms == 1000) {
        secs += 1;
        ms = 0;
    }
    std::tm tmv{};
#if defined(_WIN32)
    localtime_s(&tmv, &secs);
#else
    localtime_r(&secs, &tmv);
#endif
    char buf[40];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tmv);
    char out[48];
    std::snprintf(out, sizeof(out), "%s.%03d", buf, ms);
    return out;
}

inline void TraceRecorder::write_csv(const std::string & path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        return;  // best-effort; the caller guarantees the directory exists
    }
    f << "\xEF\xBB\xBF";  // utf-8-sig BOM

    auto emit_row = [&](const std::vector<std::string> & cells) {
        for (size_t i = 0; i < cells.size(); ++i) {
            if (i) {
                f << ",";
            }
            f << cells[i];
        }
        f << "\r\n";
    };

    emit_row({"record_id", "record_type", "run_id", "session_id", "query_id",
              "chunk_id", "parent_id", "name", "start_wall_time",
              "end_wall_time", "duration_ms", "video_timestamp", "action"});

    char buf[64];
    for (const TraceRecord & r : records_) {
        std::vector<std::string> cells;
        cells.push_back(r.record_id);
        cells.push_back(r.record_type);
        cells.push_back(r.run_id);
        cells.push_back(r.session_id);
        cells.push_back(r.query_id);
        cells.push_back(r.chunk_id);
        cells.push_back(r.parent_id);
        cells.push_back(r.name);
        cells.push_back(format_wall_time(r.start_epoch));
        if (r.record_type == "operation") {
            cells.push_back(format_wall_time(r.end_epoch));
            std::snprintf(buf, sizeof(buf), "%.3f",
                          (r.end_epoch - r.start_epoch) * 1000.0);
            cells.push_back(buf);
        } else {
            cells.push_back("");
            cells.push_back("");
        }
        if (r.video_timestamp >= 0.0) {
            std::snprintf(buf, sizeof(buf), "%.3f", r.video_timestamp);
            cells.push_back(buf);
        } else {
            cells.push_back("");
        }
        cells.push_back(r.action);
        emit_row(cells);
    }
}

} // namespace ours

#endif // OURS_NATIVE_TRACE_RECORDER_H

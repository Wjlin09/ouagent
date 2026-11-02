// C++ implementation of the OURS_Agent native runtime.


#include "gemma_runtime.h"
#include "trace/trace_recorder.h"

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace ours {

namespace {

// ---------------------------------------------------------------------------
// Gemma 4 chat-format constants (verbatim from native_mtmd.py).
// ---------------------------------------------------------------------------

// Matched against the *accumulated* generated text (a marker may be split
// across several sampled tokens). Copied from _EOT_TEXT_MARKERS.
const char * const kEotTextMarkers[] = {
    "<end_of_turn>",
    "<turn|>",          // Gemma 4 uses "<turn|>" instead of "<end_of_turn>"
    "<eos>",
    "<|end_of_turn|>",
    "<|eot_id|>",
    "<|im_end|>",
    "<|end|>",
    "<end_of_utterance>",
};
constexpr size_t kEotTextMarkerCount = sizeof(kEotTextMarkers) / sizeof(kEotTextMarkers[0]);

// The leading <bos> is stripped by the common layer (add_bos=true) and re-added
// by mtmd's add_special, so it is not part of these constants.
//const char * const kGemma4UserOpen    = "<|turn>system\n<turn|>\n<|turn>user\n";
//const char * const kGemma4UserOpen    = "<|turn>user\n";
const char * const kGemma4UserOpen = "<|turn>system\nOutput only the option letter (A, B, C, or D).<turn|>\n<|turn>user\n";
const char * const kGemma4UserClose   = "<turn|>\n";
const char * const kGemma4ModelOpen   = "<|turn>model\n";
const char * const kGemma4ThinkSuffix = "";  // enable_thinking=true: thought channel left open
// Official Gemma 4 end-of-turn marker; appended to close a model turn cut short
// by stop_query() (or the max_tokens cap).
const char * const kGemma4TurnClose   = "<turn|>\n";

// GGML log filter: drop debug/info/cont, keep warn/error on stderr.
void filtered_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void) user_data;
    if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) {
        std::fprintf(stderr, "%s", text);
        std::fflush(stderr);
    }
}

bool file_exists(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    return f.good();
}

// Python str.strip() equivalent (leading/trailing ASCII whitespace).
std::string strip(const std::string & s) {
    const char * ws = " \t\n\r\f\v";
    const size_t b = s.find_first_not_of(ws);
    if (b == std::string::npos) {
        return "";
    }
    const size_t e = s.find_last_not_of(ws);
    return s.substr(b, e - b + 1);
}

void log_info(const char * fmt, ...) {
    std::fprintf(stderr, "[ours-runtime] ");
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
    std::fprintf(stderr, "\n");
}

} // namespace

// ---------------------------------------------------------------------------
// Impl: all llama/mtmd state and the ported inference logic.
// ---------------------------------------------------------------------------

struct GemmaRuntime::Impl {
    GemmaRuntimeConfig cfg;

    llama_model * model = nullptr;
    llama_context * lctx = nullptr;
    const llama_vocab * vocab = nullptr;
    mtmd_context * mtmd_ctx = nullptr;
    llama_sampler * smpl = nullptr;  // sampler *chain* (owns the sub-samplers)
    std::string media_marker;

    // Persistent streaming state (see native_mtmd.py for the semantics).
    int32_t n_past = 0;
    bool session_active = false;
    bool query_active = false;
    bool turn_active = false;
    bool model_turn_open = false;
    std::string query_pieces;   // accumulated generated text (markers stripped)
    uint32_t query_tokens = 0;  // per-query max_tokens counter
    uint32_t n_frames_added = 0;
    uint32_t generation_count = 0;
    bool logits_logged = false;
    int32_t last_context_token_id = -1;

    uint32_t n_batch = 0;  // actual n_batch reported by the context

    TraceRecorder * trace_ = nullptr;  // optional execution trace recorder

    explicit Impl(GemmaRuntimeConfig c, TraceRecorder * trace)
        : cfg(std::move(c)), trace_(trace) {}

    void ensure_loaded();
    void teardown();
    void install_log_filters();
    void clear_session_state();
    void start_session();
    void end_session();

    // inference
    std::string infer(const Frame & frame, const std::string & prompt);
    std::string infer_cli_equivalent(const Frame & frame, const std::string & question);
    void append_video_chunk(const std::vector<Frame> & frames);
    void start_query(const std::string & question);
    void stop_query();
    StepResult step_generation();
    std::string finalize(const std::string & question);
    std::string generate();

    // internals
    int32_t append_to_context(const std::string & text, const std::vector<Frame> & frames,
                              bool add_special, bool logits_last);
    int32_t eval_chunks(const mtmd_input_chunks * chunks, bool logits_last);
    int32_t sample_one_token();
    void close_model_turn_if_open();
    std::string token_to_piece(llama_token token);

    // diagnostics (verbose only)
    void logits_diagnostic();
    void record_last_text_token(const mtmd_input_chunks * chunks);
    void dump_chunks(const mtmd_input_chunks * chunks);
};

// -- lifecycle ---------------------------------------------------------------

void GemmaRuntime::Impl::ensure_loaded() {
    if (model != nullptr) {
        return;
    }

    if (!file_exists(cfg.model_path)) {
        throw GemmaRuntimeError("model file not found: " + cfg.model_path +
                                ". Download the Gemma 4 GGUF on the inference server.");
    }
    if (!file_exists(cfg.mmproj_path)) {
        throw GemmaRuntimeError("mmproj file not found: " + cfg.mmproj_path +
                                ". Download the mmproj GGUF on the inference server.");
    }

    llama_backend_init();
    install_log_filters();

    // -- text model ---------------------------------------------------------
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = cfg.n_gpu_layers;
    model = llama_model_load_from_file(cfg.model_path.c_str(), mparams);
    if (model == nullptr) {
        teardown();
        throw GemmaRuntimeError("failed to load model: " + cfg.model_path);
    }
    vocab = llama_model_get_vocab(model);

    const char * tmpl = llama_model_chat_template(model, nullptr);
    if (tmpl == nullptr) {
        log_info("model has NO embedded chat template -- this looks like a base "
                 "(non-instruction-tuned) model, not an instruct model; the chat "
                 "format below will not be understood");
    } else {
        log_info("model chat template present (len=%zu)", std::strlen(tmpl));
    }

    // -- context ------------------------------------------------------------
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = cfg.n_ctx;
    cparams.n_batch = cfg.n_batch;
    cparams.n_ubatch = cfg.n_ubatch;
    if (cfg.n_threads > 0) {
        cparams.n_threads = cfg.n_threads;
        cparams.n_threads_batch = cfg.n_threads;
    }
    lctx = llama_init_from_model(model, cparams);
    if (lctx == nullptr) {
        teardown();
        throw GemmaRuntimeError("failed to create llama context");
    }
    n_batch = llama_n_batch(lctx);

    // -- multimodal projector (mmproj) --------------------------------------
    mtmd_context_params mctx_params = mtmd_context_params_default();
    mctx_params.use_gpu = cfg.mmproj_use_gpu;
    mctx_params.print_timings = cfg.verbose;
    if (cfg.n_threads > 0) {
        mctx_params.n_threads = cfg.n_threads;
    }
    mtmd_ctx = mtmd_init_from_file(cfg.mmproj_path.c_str(), model, mctx_params);
    if (mtmd_ctx == nullptr) {
        teardown();
        throw GemmaRuntimeError("failed to load mmproj: " + cfg.mmproj_path);
    }

    media_marker = mtmd_default_marker();

    // -- sampler chain (mirrors common/sampling.cpp default order:
    //    penalties -> top_p -> temperature -> dist) --------------------------
    smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    llama_sampler_chain_add(smpl,
        llama_sampler_init_penalties(n_vocab, 64, cfg.repeat_penalty, 0.0f, 0.0f));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(cfg.top_p, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(cfg.temperature));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(cfg.seed));

    log_info("GemmaRuntime loaded (model=%s, mmproj=%s, n_ctx=%u, n_gpu_layers=%d)",
             cfg.model_path.c_str(), cfg.mmproj_path.c_str(), cfg.n_ctx, cfg.n_gpu_layers);
}

void GemmaRuntime::Impl::install_log_filters() {
    mtmd_log_set(filtered_log_callback, nullptr);
    mtmd_helper_log_set(filtered_log_callback, nullptr);
    llama_log_set(filtered_log_callback, nullptr);
}

void GemmaRuntime::Impl::teardown() {
    if (smpl != nullptr) {
        llama_sampler_free(smpl);  // chain owns the sub-samplers
        smpl = nullptr;
    }
    if (mtmd_ctx != nullptr) {
        mtmd_free(mtmd_ctx);
        mtmd_ctx = nullptr;
    }
    if (lctx != nullptr) {
        llama_free(lctx);
        lctx = nullptr;
    }
    if (model != nullptr) {
        llama_model_free(model);
        model = nullptr;
    }
    vocab = nullptr;
}

void GemmaRuntime::Impl::clear_session_state() {
    n_past = 0;
    session_active = false;
    query_active = false;
    turn_active = false;
    model_turn_open = false;
    query_pieces.clear();
    query_tokens = 0;
    n_frames_added = 0;
    generation_count = 0;
    logits_logged = false;
    last_context_token_id = -1;
    if (lctx != nullptr) {
        llama_memory_clear(llama_get_memory(lctx), true);
        if (smpl != nullptr) {
            llama_sampler_reset(smpl);
        }
    }
}

// -- session lifecycle -------------------------------------------------------

void GemmaRuntime::Impl::start_session() {
    clear_session_state();
    session_active = true;
}

void GemmaRuntime::Impl::end_session() {
    clear_session_state();
    session_active = false;
}

// -- streaming video ---------------------------------------------------------

void GemmaRuntime::Impl::append_video_chunk(const std::vector<Frame> & frames) {
    ensure_loaded();
    if (!session_active) {
        throw GemmaRuntimeError("append_video_chunk() requires an active session; "
                                "call start_session() first");
    }
    if (frames.empty()) {
        return;
    }

    TraceSpan span = trace_ ? trace_->begin_operation("chunk_prefill") : TraceSpan{};

    const std::string marker = media_marker;

    if (query_active) {
        // Mid-generation: append media at the current context position and
        // leave the model turn open. logits_last=true so step_generation()
        // continues from the media's final token.
        std::string text = marker;
        for (size_t i = 1; i < frames.size(); ++i) {
            text += "\n" + marker;
        }
        append_to_context(text, frames, /*add_special=*/false, /*logits_last=*/true);
    } else {
        close_model_turn_if_open();
        const bool first = !turn_active;
        const std::string prefix = first ? kGemma4UserOpen : "\n";
        std::string text = prefix + marker;
        for (size_t i = 1; i < frames.size(); ++i) {
            text += "\n" + marker;
        }
        append_to_context(text, frames, /*add_special=*/n_past == 0, /*logits_last=*/false);
        turn_active = true;
    }

    n_frames_added += static_cast<uint32_t>(frames.size());
    if (cfg.verbose) {
        log_info("[session] visual chunk appended: frames=%zu, n_past=%d",
                 frames.size(), n_past);
    }
}

// -- query lifecycle ---------------------------------------------------------

void GemmaRuntime::Impl::start_query(const std::string & question) {
    if (query_active) {
        throw GemmaRuntimeError("start_query() called while a query is already active");
    }
    ensure_loaded();
    if (!session_active) {
        throw GemmaRuntimeError("start_query() requires an active session; "
                                "call start_session() first");
    }

    close_model_turn_if_open();

    const std::string prefix = turn_active ? "" : kGemma4UserOpen;
    const std::string text = prefix + question + kGemma4UserClose +
                             kGemma4ModelOpen + kGemma4ThinkSuffix;
    {
        TraceSpan span =
            trace_ ? trace_->begin_operation("query_prefill") : TraceSpan{};
        append_to_context(text, /*frames=*/{}, /*add_special=*/n_past == 0,
                          /*logits_last=*/true);
    }

    turn_active = false;
    model_turn_open = true;
    query_active = true;
    query_pieces.clear();
    query_tokens = 0;
    generation_count += 1;
    if (cfg.verbose) {
        log_info("[query] started: n_past=%d", n_past);
    }
}

void GemmaRuntime::Impl::stop_query() {
    if (!query_active) {
        return;
    }
    query_active = false;
    if (cfg.verbose) {
        log_info("[query] stopped at n_past=%d (session preserved)", n_past);
    }
}

StepResult GemmaRuntime::Impl::step_generation() {
    if (!query_active) {
        throw GemmaRuntimeError("step_generation() called with no active query");
    }

    if (!logits_logged) {
        if (cfg.verbose) {
            logits_diagnostic();
        }
        logits_logged = true;
    }

    const llama_token token = sample_one_token();
    query_tokens += 1;
    const llama_token eot_token = llama_vocab_eot(vocab);

    if (llama_vocab_is_eog(vocab, token) || token == eot_token) {
        query_active = false;
        model_turn_open = false;
        return StepResult{strip(query_pieces), true};
    }

    query_pieces += token_to_piece(token);
    const std::string text = query_pieces;

    // Defensive fallback: match end-of-turn markers that reach us as literal
    // text (split across several tokens) rather than EOG/control.
    const size_t window_start = text.size() > 64 ? text.size() - 64 : 0;
    for (size_t i = 0; i < kEotTextMarkerCount; ++i) {
        const size_t pos = text.rfind(kEotTextMarkers[i]);
        if (pos != std::string::npos && pos >= window_start) {
            query_active = false;
            model_turn_open = false;
            return StepResult{strip(text.substr(0, pos)), true};
        }
    }

    // Per-query max_tokens cap (reset by start_query, NOT per-session).
    if (query_tokens >= cfg.max_tokens) {
        query_active = false;
        return StepResult{strip(text), true};
    }

    return StepResult{strip(text), false};
}

std::string GemmaRuntime::Impl::finalize(const std::string & question) {
    start_query(question);
    return generate();
}

std::string GemmaRuntime::Impl::generate() {
    if (!query_active) {
        throw GemmaRuntimeError("generate() called with no active query");
    }
    std::string text;
    for (uint32_t i = 0; i < cfg.max_tokens; ++i) {
        StepResult r = step_generation();
        text = r.text;
        if (r.done) {
            break;
        }
    }
    return text;
}

// -- one-shot helpers --------------------------------------------------------

std::string GemmaRuntime::Impl::infer(const Frame & frame, const std::string & prompt) {
    start_session();
    append_video_chunk({frame});
    return finalize(prompt);
}

std::string GemmaRuntime::Impl::infer_cli_equivalent(const Frame & frame,
                                                     const std::string & question) {
    start_session();
    ensure_loaded();

    const std::string text_before = kGemma4UserOpen;
    const std::string text_after = question + kGemma4UserClose + kGemma4ModelOpen +
                                   kGemma4ThinkSuffix;

    mtmd::bitmap_ptr bitmap(mtmd_bitmap_init(frame.width, frame.height, frame.rgb));
    if (!bitmap) {
        throw GemmaRuntimeError("mtmd_bitmap_init failed");
    }

    mtmd::input_chunks_ptr chunks(mtmd_input_chunks_init());
    if (!chunks) {
        throw GemmaRuntimeError("mtmd_input_chunks_init failed");
    }

    mtmd_input_text text0{text_before.c_str(), text_before.size(),
                          /*add_special=*/false, /*parse_special=*/true};
    mtmd_input_text text1{text_after.c_str(), text_after.size(),
                          /*add_special=*/false, /*parse_special=*/true};
    mtmd_input_part part0{&text0, nullptr};
    mtmd_input_part part1{nullptr, bitmap.get()};
    mtmd_input_part part2{&text1, nullptr};
    const mtmd_input_part * parts[3] = {&part0, &part1, &part2};

    const int32_t res = mtmd_tokenize_from_parts(mtmd_ctx, chunks.get(), parts, 3, true);
    if (res != 0) {
        throw GemmaRuntimeError("mtmd_tokenize_from_parts failed: " + std::to_string(res));
    }

    const llama_pos n_pos = mtmd_helper_get_n_pos(chunks.get());
    if (static_cast<int64_t>(n_past) + n_pos > cfg.n_ctx) {
        throw GemmaRuntimeError("image context exceeds model context window: n_past=" +
                                std::to_string(n_past) + " + n_pos=" +
                                std::to_string(n_pos) + " > n_ctx=" +
                                std::to_string(cfg.n_ctx) + ".");
    }

    if (cfg.verbose) {
        record_last_text_token(chunks.get());
        dump_chunks(chunks.get());
    }

    n_past = eval_chunks(chunks.get(), /*logits_last=*/true);
    turn_active = false;
    model_turn_open = true;
    query_active = true;
    query_pieces.clear();
    generation_count += 1;

    const std::string output = generate();
    if (cfg.verbose) {
        log_info("[cli-equivalent] single-image generation finished; answer length=%zu",
                 output.size());
    }
    return output;
}

// -- internals ---------------------------------------------------------------

int32_t GemmaRuntime::Impl::append_to_context(const std::string & text,
                                              const std::vector<Frame> & frames,
                                              bool add_special, bool logits_last) {
    const size_t n_frames = frames.size();

    std::vector<mtmd::bitmap_ptr> bitmaps;
    bitmaps.reserve(n_frames);
    for (const Frame & f : frames) {
        mtmd::bitmap_ptr bmp(mtmd_bitmap_init(f.width, f.height, f.rgb));
        if (!bmp) {
            throw GemmaRuntimeError("mtmd_bitmap_init failed");
        }
        bitmaps.push_back(std::move(bmp));
    }

    mtmd::input_chunks_ptr chunks(mtmd_input_chunks_init());
    if (!chunks) {
        throw GemmaRuntimeError("mtmd_input_chunks_init failed");
    }

    mtmd_input_text input_text{text.c_str(), text.size(),
                               add_special, /*parse_special=*/true};
    std::vector<const mtmd_bitmap *> bitmap_ptrs;
    bitmap_ptrs.reserve(bitmaps.size());
    for (const mtmd::bitmap_ptr & b : bitmaps) {
        bitmap_ptrs.push_back(b.get());
    }

    const int32_t res = mtmd_tokenize(mtmd_ctx, chunks.get(), &input_text,
                                      bitmap_ptrs.empty() ? nullptr : bitmap_ptrs.data(),
                                      bitmap_ptrs.size());
    if (res != 0) {
        throw GemmaRuntimeError("mtmd_tokenize failed: " + std::to_string(res));
    }

    const llama_pos n_pos = mtmd_helper_get_n_pos(chunks.get());

    if (cfg.verbose) {
        record_last_text_token(chunks.get());
        dump_chunks(chunks.get());
    }

    // Deterministic overflow boundary: raise before eval, so n_past and the
    // KV are left exactly as they were (no silent truncation / reset).
    if (static_cast<int64_t>(n_past) + n_pos > cfg.n_ctx) {
        throw GemmaRuntimeError("video context exceeds model context window: n_past=" +
                                std::to_string(n_past) + " + n_pos=" +
                                std::to_string(n_pos) + " > n_ctx=" +
                                std::to_string(cfg.n_ctx) +
                                ". Call reset() to start a new video.");
    }

    n_past = eval_chunks(chunks.get(), logits_last);
    return n_pos;
}

int32_t GemmaRuntime::Impl::eval_chunks(const mtmd_input_chunks * chunks, bool logits_last) {
    llama_pos new_n_past = 0;
    const int32_t res = mtmd_helper_eval_chunks(mtmd_ctx, lctx, chunks, n_past, /*seq_id=*/0,
                                                static_cast<int32_t>(n_batch), logits_last,
                                                &new_n_past);
    if (res != 0) {
        throw GemmaRuntimeError("mtmd_helper_eval_chunks failed: " + std::to_string(res));
    }
    return new_n_past;
}

int32_t GemmaRuntime::Impl::sample_one_token() {
    // The vendored llama_sampler_sample() already *accepts* the sampled token
    // internally (apply + pick + llama_sampler_accept), so we must NOT call
    // llama_sampler_accept() again (a second accept would double-update the
    // repetition-penalty state). See src/llama-sampler.cpp and llama.h.
    const llama_token token = llama_sampler_sample(smpl, lctx, -1);

    llama_token token_val = token;
    llama_pos pos_val = n_past;
    int32_t n_seq_id_val = 1;
    llama_seq_id seq_id_val[1] = {0};
    llama_seq_id * seq_id_ptr[1] = {seq_id_val};
    int8_t logits_val = 1;

    llama_batch batch;
    batch.n_tokens = 1;
    batch.token = &token_val;
    batch.embd = nullptr;
    batch.pos = &pos_val;
    batch.n_seq_id = &n_seq_id_val;
    batch.seq_id = seq_id_ptr;
    batch.logits = &logits_val;

    const int32_t res = llama_decode(lctx, batch);
    if (res != 0) {
        throw GemmaRuntimeError("llama_decode failed: " + std::to_string(res));
    }
    n_past += 1;
    return token;
}

void GemmaRuntime::Impl::close_model_turn_if_open() {
    if (!model_turn_open) {
        return;
    }
    append_to_context(kGemma4TurnClose, /*frames=*/{}, /*add_special=*/false,
                      /*logits_last=*/false);
    model_turn_open = false;
}

std::string GemmaRuntime::Impl::token_to_piece(llama_token token) {
    int32_t size = 64;
    while (true) {
        std::vector<char> buf(static_cast<size_t>(size));
        const int32_t n = llama_token_to_piece(vocab, token, buf.data(), size, 0, false);
        if (n < 0) {
            // Buffer too small: -n is the required length.
            size = -n + 1;
            continue;
        }
        return std::string(buf.data(), static_cast<size_t>(n));
    }
}

// -- diagnostics (verbose only; never affect inference) ----------------------

void GemmaRuntime::Impl::logits_diagnostic() {
    try {
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        float * logits = llama_get_logits_ith(lctx, -1);
        if (logits == nullptr) {
            log_info("logits diagnostic: llama_get_logits_ith(ctx, -1) returned NULL");
            return;
        }
        // Top-20 by logit value (degenerate but useful diagnostic, matches the
        // Python implementation).
        std::vector<std::pair<float, int32_t>> top;
        top.reserve(static_cast<size_t>(n_vocab));
        for (int32_t i = 0; i < n_vocab; ++i) {
            top.emplace_back(logits[i], i);
        }
        std::sort(top.begin(), top.end(),
                  [](const std::pair<float, int32_t> & a, const std::pair<float, int32_t> & b) {
                      return a.first > b.first;
                  });
        const std::string last_piece = last_context_token_id >= 0
                                           ? token_to_piece(last_context_token_id)
                                           : std::string();
        log_info("logits diagnostic: n_past=%d n_vocab=%d last_token_id=%d last_token_piece=%s",
                 n_past, n_vocab, last_context_token_id, last_piece.c_str());
        for (int32_t rank = 1; rank <= 20 && rank - 1 < n_vocab; ++rank) {
            const auto & kv = top[static_cast<size_t>(rank - 1)];
            log_info("  top-20[%02d] token_id=%d piece=%s logit=%.4f", rank, kv.second,
                     token_to_piece(kv.second).c_str(), kv.first);
        }
    } catch (const std::exception & exc) {
        log_info("logits diagnostic skipped: %s", exc.what());
    }
}

void GemmaRuntime::Impl::record_last_text_token(const mtmd_input_chunks * chunks) {
    last_context_token_id = -1;
    try {
        const size_t n_chunks = mtmd_input_chunks_size(chunks);
        for (size_t i = n_chunks; i-- > 0;) {
            const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, i);
            size_t n_tokens = 0;
            const llama_token * tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
            if (tokens != nullptr && n_tokens > 0) {
                last_context_token_id = tokens[n_tokens - 1];
                break;
            }
        }
    } catch (const std::exception & exc) {
        log_info("last-text-token scan skipped: %s", exc.what());
    }
}

void GemmaRuntime::Impl::dump_chunks(const mtmd_input_chunks * chunks) {
    try {
        const size_t n_chunks = mtmd_input_chunks_size(chunks);
        log_info("token dump: n_chunks=%zu (n_past_before=%d)", n_chunks, n_past);
        for (size_t i = 0; i < n_chunks; ++i) {
            const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, i);
            const int ctype = static_cast<int>(mtmd_input_chunk_get_type(chunk));
            if (ctype == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                size_t n_tokens = 0;
                const llama_token * tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
                std::string entries;
                for (size_t j = 0; j < n_tokens; ++j) {
                    if (!entries.empty()) {
                        entries += " ";
                    }
                    entries += std::to_string(tokens[j]) + ":" + token_to_piece(tokens[j]);
                }
                log_info("  chunk[%zu] TEXT n_tokens=%zu %s", i, n_tokens, entries.c_str());
            } else {
                const char * kind = ctype == MTMD_INPUT_CHUNK_TYPE_IMAGE ? "IMAGE"
                                    : ctype == MTMD_INPUT_CHUNK_TYPE_AUDIO ? "AUDIO"
                                                                          : "UNKNOWN";
                log_info("  chunk[%zu] %s n_tokens=%zu", i, kind,
                         mtmd_input_chunk_get_n_tokens(chunk));
            }
        }
    } catch (const std::exception & exc) {
        log_info("token dump skipped: %s", exc.what());
    }
}

// -- public class (thin delegators over Impl) --------------------------------

GemmaRuntime::GemmaRuntime(GemmaRuntimeConfig cfg, TraceRecorder * trace)
    : impl_(new Impl(std::move(cfg), trace)) {}

GemmaRuntime::~GemmaRuntime() {
    close();
}

void GemmaRuntime::load() {
    impl_->ensure_loaded();
    loaded_ = true;
}

void GemmaRuntime::start_session() { impl_->start_session(); }
void GemmaRuntime::end_session() { impl_->end_session(); }
void GemmaRuntime::reset() { impl_->end_session(); }

void GemmaRuntime::append_video_chunk(const std::vector<Frame> & frames) {
    impl_->append_video_chunk(frames);
}

void GemmaRuntime::add_frame(const Frame & frame) {
    impl_->append_video_chunk({frame});
}

void GemmaRuntime::start_query(const std::string & question) {
    impl_->start_query(question);
}

void GemmaRuntime::stop_query() { impl_->stop_query(); }

StepResult GemmaRuntime::step_generation() { return impl_->step_generation(); }

std::string GemmaRuntime::finalize(const std::string & question) {
    return impl_->finalize(question);
}

std::string GemmaRuntime::infer(const Frame & frame, const std::string & prompt) {
    return impl_->infer(frame, prompt);
}

std::string GemmaRuntime::infer_cli_equivalent(const Frame & frame,
                                               const std::string & question) {
    return impl_->infer_cli_equivalent(frame, question);
}

void GemmaRuntime::close() {
    impl_->teardown();
    loaded_ = false;
}

} // namespace ours

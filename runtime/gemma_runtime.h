// OURS_Agent native runtime (C++).
//
// Responsibilities :
//   * Session start/end lifecycle (the ONLY operations that clear the KV cache
//     and reset the sampler);
//   * streaming video: incremental frame/chunk append into a persistent KV
//     cache (one ``mtmd_tokenize`` + one ``mtmd_helper_eval_chunks`` per chunk),
//     allowed both before and DURING an active query;
//   * Query start / per-token generation / stop / restart lifecycle;
//   * ``stop_query()`` never clears the session/KV cache.
//
// This header is self-contained (pimpl) so it can be consumed from a future
// JNI/Android layer without pulling in llama.h/mtmd.h.

#ifndef OURS_NATIVE_GemmaRuntime_H
#define OURS_NATIVE_GemmaRuntime_H

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/chunk_history.h"

namespace ours {

class TraceRecorder;

// A single RGB frame. ``rgb`` points to ``width * height * 3`` bytes in
// ``RGBRGB...`` order. The pointer only needs to remain valid for the duration
// of the call that receives the frame (``mtmd_bitmap_init`` copies the bytes).
struct Frame {
    const uint8_t * rgb;
    uint32_t width;
    uint32_t height;
};

struct GemmaRuntimeConfig {
    std::string model_path;
    std::string mmproj_path;
    uint32_t n_ctx = 8192;
    uint32_t n_batch = 2048;
    uint32_t n_ubatch = 512;
    int32_t n_gpu_layers = -1;   // -1 = offload all layers to GPU
    int32_t n_threads = 0;       // 0 = use the native default (CPU core count)
    bool mmproj_use_gpu = true;
    float temperature = 0.2f;
    float top_p = 0.9f;
    uint32_t max_tokens = 512;
    uint32_t seed = 1234;
    float repeat_penalty = 1.1f;
    bool verbose = false;
};

struct StepResult {
    std::string text;  // full generated text so far (markers stripped)
    bool done = false; // true when generation ended (EOG/EOT or max_tokens cap)
};

// Raised for every native failure (missing library/model, context overflow,
// failed decode, invalid lifecycle call, ...). Replaces the Python
// ``LibmtmdError``.
class GemmaRuntimeError : public std::runtime_error {
public:
    explicit GemmaRuntimeError(const std::string & msg) : std::runtime_error(msg) {}
};

class GemmaRuntime {
public:
    explicit GemmaRuntime(GemmaRuntimeConfig cfg, TraceRecorder * trace = nullptr);
    ~GemmaRuntime();

    GemmaRuntime(const GemmaRuntime &) = delete;
    GemmaRuntime & operator=(const GemmaRuntime &) = delete;

    // Lazy native init (mirrors NativeMTMDBackend._ensure_loaded). Idempotent.
    // Throws GemmaRuntimeError if libmtmd/libllama, the model, or the mmproj
    // cannot be loaded.
    void load();
    bool is_loaded() const { return loaded_; }

    // -- session lifecycle ------------------------------------------------
    // start_session()/end_session() are the ONLY operations that clear the KV
    // cache and reset the sampler. stop_query() never does.
    void start_session();
    void end_session();
    void reset();  // alias for end_session()

    // -- streaming video --------------------------------------------------
    // A chunk is ONE mtmd_tokenize + ONE mtmd_helper_eval_chunks call. Allowed
    // both before and DURING an active query (media injected at the current
    // context position; generation continues from the updated KV).
    void append_video_chunk(const std::vector<Frame> & frames);
    void add_frame(const Frame & frame);

    // -- query lifecycle --------------------------------------------------
    void start_query(const std::string & question);
    void stop_query();  // stops the query, preserves session/KV state
    StepResult step_generation();
    std::string finalize(const std::string & question);

    // -- one-shot helpers (single image) ----------------------------------
    std::string infer(const Frame & frame, const std::string & prompt);
    std::string infer_cli_equivalent(const Frame & frame, const std::string & question);

    void close();

    // KV-cache bytes currently in use for the single sequence (seq 0). Grows as
    // chunks/queries/tokens are appended; resets to 0 on start_session().
    size_t kv_used_bytes() const;

    // Session-level chunk history (one entry per processed video chunk). Empty
    // until a video chunk completes prefill; cleared by start/end_session().
    const std::vector<ChunkHistoryEntry> & chunk_history() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    bool loaded_ = false;
};

} // namespace ours

#endif // OURS_NATIVE_GemmaRuntime_H

// Stable C ABI over GemmaRuntime.
//
// This is the JNI seam: the interface the next phase (Android / JNI / HTP /
// NPU) binds against -- an opaque handle plus plain C structs and scalars, no
// C++ exceptions across the boundary (every call reports failure via a
// non-zero return code and an optional ``err`` buffer).

#ifndef OURS_NATIVE_GemmaRuntime_C_API_H
#define OURS_NATIVE_GemmaRuntime_C_API_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#    if defined(OURS_RUNTIME_BUILD)
#        define OURS_API __declspec(dllexport)
#    else
#        define OURS_API __declspec(dllimport)
#    endif
#else
#    define OURS_API __attribute__((visibility("default")))
#endif

typedef struct ours_gemma_runtime ours_gemma_runtime;

struct ours_frame {
    const uint8_t * rgb;  // width * height * 3 bytes, RGBRGB... order
    uint32_t width;
    uint32_t height;
};

struct ours_gemma_runtime_config {
    const char * model_path;
    const char * mmproj_path;
    uint32_t n_ctx;
    uint32_t n_batch;
    uint32_t n_ubatch;
    int32_t n_gpu_layers;
    int32_t n_threads;        // 0 = native default
    bool mmproj_use_gpu;
    float temperature;
    float top_p;
    uint32_t max_tokens;
    uint32_t seed;
    float repeat_penalty;
    bool verbose;
};

// Create a runtime handle (does NOT load the model; load is lazy, like the
// Python backend). Returns NULL on allocation failure.
OURS_API ours_gemma_runtime * ours_gemma_runtime_create(
    const struct ours_gemma_runtime_config * cfg);

// Destroy the handle and release all native resources.
OURS_API void ours_gemma_runtime_destroy(ours_gemma_runtime * rt);

// Lazy native init. Idempotent. Returns 0 on success, -1 on error (message in
// ``err`` when non-NULL).
OURS_API int32_t ours_gemma_runtime_load(ours_gemma_runtime * rt, char * err, size_t err_cap);

OURS_API int32_t ours_gemma_start_session(ours_gemma_runtime * rt, char * err, size_t err_cap);
OURS_API int32_t ours_gemma_end_session(ours_gemma_runtime * rt, char * err, size_t err_cap);
OURS_API int32_t ours_gemma_reset(ours_gemma_runtime * rt, char * err, size_t err_cap);

OURS_API int32_t ours_gemma_append_video_chunk(
    ours_gemma_runtime * rt, const struct ours_frame * frames, size_t n_frames,
    char * err, size_t err_cap);
OURS_API int32_t ours_gemma_add_frame(
    ours_gemma_runtime * rt, const struct ours_frame * frame, char * err, size_t err_cap);

OURS_API int32_t ours_gemma_start_query(
    ours_gemma_runtime * rt, const char * question, char * err, size_t err_cap);
OURS_API int32_t ours_gemma_stop_query(ours_gemma_runtime * rt, char * err, size_t err_cap);

// Fills ``text`` (up to ``text_cap`` bytes) with the full generated text so
// far; sets ``*done``. Returns 0 on success, -1 on error.
OURS_API int32_t ours_gemma_step_generation(
    ours_gemma_runtime * rt, char * text, size_t text_cap, int32_t * done,
    char * err, size_t err_cap);

OURS_API int32_t ours_gemma_finalize(
    ours_gemma_runtime * rt, const char * question, char * text, size_t text_cap,
    char * err, size_t err_cap);

OURS_API int32_t ours_gemma_infer(
    ours_gemma_runtime * rt, const struct ours_frame * frame, const char * prompt,
    char * text, size_t text_cap, char * err, size_t err_cap);

OURS_API int32_t ours_gemma_infer_cli_equivalent(
    ours_gemma_runtime * rt, const struct ours_frame * frame, const char * question,
    char * text, size_t text_cap, char * err, size_t err_cap);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // OURS_NATIVE_GemmaRuntime_C_API_H

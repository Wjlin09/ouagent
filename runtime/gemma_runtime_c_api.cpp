// C ABI implementation: marshals between the C structs and ours::GemmaRuntime.

#include "gemma_runtime_c_api.h"

#include "gemma_runtime.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace {

int set_err(char * err, size_t err_cap, const std::string & msg) {
    if (err != nullptr && err_cap > 0) {
        std::snprintf(err, err_cap, "%s", msg.c_str());
    }
    return -1;
}

int copy_out(const std::string & text, char * out, size_t out_cap) {
    if (out != nullptr && out_cap > 0) {
        const size_t n = text.size() < out_cap - 1 ? text.size() : out_cap - 1;
        std::memcpy(out, text.data(), n);
        out[n] = '\0';
    }
    return 0;
}

template <typename Fn>
int guard(Fn && fn, char * err, size_t err_cap) {
    try {
        fn();
        return 0;
    } catch (const ours::GemmaRuntimeError & exc) {
        return set_err(err, err_cap, exc.what());
    } catch (const std::exception & exc) {
        return set_err(err, err_cap, std::string("unexpected error: ") + exc.what());
    }
}

} // namespace

struct ours_gemma_runtime {
    ours::GemmaRuntime runtime;
    explicit ours_gemma_runtime(ours::GemmaRuntimeConfig cfg) : runtime(std::move(cfg)) {}
};

extern "C" {

OURS_API ours_gemma_runtime * ours_gemma_runtime_create(
    const struct ours_gemma_runtime_config * cfg) {
    if (cfg == nullptr || cfg->model_path == nullptr || cfg->mmproj_path == nullptr) {
        return nullptr;
    }
    ours::GemmaRuntimeConfig c;
    c.model_path = cfg->model_path;
    c.mmproj_path = cfg->mmproj_path;
    c.n_ctx = cfg->n_ctx;
    c.n_batch = cfg->n_batch;
    c.n_ubatch = cfg->n_ubatch;
    c.n_gpu_layers = cfg->n_gpu_layers;
    c.n_threads = cfg->n_threads;
    c.mmproj_use_gpu = cfg->mmproj_use_gpu;
    c.temperature = cfg->temperature;
    c.top_p = cfg->top_p;
    c.max_tokens = cfg->max_tokens;
    c.seed = cfg->seed;
    c.repeat_penalty = cfg->repeat_penalty;
    c.verbose = cfg->verbose;
    return new ours_gemma_runtime(std::move(c));
}

OURS_API void ours_gemma_runtime_destroy(ours_gemma_runtime * rt) {
    delete rt;
}

OURS_API int32_t ours_gemma_runtime_load(ours_gemma_runtime * rt, char * err, size_t err_cap) {
    if (rt == nullptr) {
        return set_err(err, err_cap, "null runtime");
    }
    return guard([&] { rt->runtime.load(); }, err, err_cap);
}

OURS_API int32_t ours_gemma_start_session(ours_gemma_runtime * rt, char * err, size_t err_cap) {
    return guard([&] { rt->runtime.start_session(); }, err, err_cap);
}

OURS_API int32_t ours_gemma_end_session(ours_gemma_runtime * rt, char * err, size_t err_cap) {
    return guard([&] { rt->runtime.end_session(); }, err, err_cap);
}

OURS_API int32_t ours_gemma_reset(ours_gemma_runtime * rt, char * err, size_t err_cap) {
    return guard([&] { rt->runtime.reset(); }, err, err_cap);
}

OURS_API int32_t ours_gemma_append_video_chunk(
    ours_gemma_runtime * rt, const struct ours_frame * frames, size_t n_frames,
    char * err, size_t err_cap) {
    return guard([&] {
        std::vector<ours::Frame> out;
        out.reserve(n_frames);
        for (size_t i = 0; i < n_frames; ++i) {
            out.push_back(ours::Frame{frames[i].rgb, frames[i].width, frames[i].height});
        }
        rt->runtime.append_video_chunk(out);
    }, err, err_cap);
}

OURS_API int32_t ours_gemma_add_frame(
    ours_gemma_runtime * rt, const struct ours_frame * frame, char * err, size_t err_cap) {
    return guard([&] {
        rt->runtime.add_frame(ours::Frame{frame->rgb, frame->width, frame->height});
    }, err, err_cap);
}

OURS_API int32_t ours_gemma_start_query(
    ours_gemma_runtime * rt, const char * question, char * err, size_t err_cap) {
    return guard([&] { rt->runtime.start_query(question != nullptr ? question : ""); },
                 err, err_cap);
}

OURS_API int32_t ours_gemma_stop_query(ours_gemma_runtime * rt, char * err, size_t err_cap) {
    return guard([&] { rt->runtime.stop_query(); }, err, err_cap);
}

OURS_API int32_t ours_gemma_step_generation(
    ours_gemma_runtime * rt, char * text, size_t text_cap, int32_t * done,
    char * err, size_t err_cap) {
    if (done == nullptr) {
        return set_err(err, err_cap, "null done");
    }
    return guard([&] {
        ours::StepResult r = rt->runtime.step_generation();
        copy_out(r.text, text, text_cap);
        *done = r.done ? 1 : 0;
    }, err, err_cap);
}

OURS_API int32_t ours_gemma_finalize(
    ours_gemma_runtime * rt, const char * question, char * text, size_t text_cap,
    char * err, size_t err_cap) {
    return guard([&] {
        copy_out(rt->runtime.finalize(question != nullptr ? question : ""), text, text_cap);
    }, err, err_cap);
}

OURS_API int32_t ours_gemma_infer(
    ours_gemma_runtime * rt, const struct ours_frame * frame, const char * prompt,
    char * text, size_t text_cap, char * err, size_t err_cap) {
    return guard([&] {
        const ours::Frame f{frame->rgb, frame->width, frame->height};
        copy_out(rt->runtime.infer(f, prompt != nullptr ? prompt : ""), text, text_cap);
    }, err, err_cap);
}

OURS_API int32_t ours_gemma_infer_cli_equivalent(
    ours_gemma_runtime * rt, const struct ours_frame * frame, const char * question,
    char * text, size_t text_cap, char * err, size_t err_cap) {
    return guard([&] {
        const ours::Frame f{frame->rgb, frame->width, frame->height};
        copy_out(rt->runtime.infer_cli_equivalent(f, question != nullptr ? question : ""),
                 text, text_cap);
    }, err, err_cap);
}

} // extern "C"

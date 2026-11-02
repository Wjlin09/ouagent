// Build-level demo / smoke driver for the OURS_Agent native runtime.
//
// Runs a full streaming session against a real Gemma 4 model + mmproj on the
// Linux/GPU inference server:
//
//   start_session -> append_video_chunk (N synthetic frames)
//     -> start_query -> [step_generation interleaved with further appends]
//     -> (stop_query | natural end) -> end_session -> close
//
// It demonstrates the exact lifecycle the C++ runtime must preserve, including
// appending video frames DURING an active query and stop_query() not clearing
// the session/KV cache. Frame decoding is intentionally out of scope here (the
// Python harness decodes via cv2; the Android phase will feed decoded RGB
// frames through the C ABI), so the demo uses synthetic RGB frames.
//
// NOTE: this requires libllama.so / libmtmd.so and the Gemma 4 GGUF files.
#include "runtime/gemma_runtime.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string model;
    std::string mmproj;
    uint32_t n_ctx = 65536;
    uint32_t n_batch = 2048;
    uint32_t n_ubatch = 512;
    int32_t n_gpu_layers = -1;
    float temperature = 1.0f;
    float top_p = 0.95f;
    float repeat_penalty = 1.1f;
    uint32_t max_tokens = 1024;
    uint32_t seed = 1234;
    uint32_t width = 336;
    uint32_t height = 336;
    uint32_t initial_frames = 4;
    uint32_t mid_query_frames = 4;
    std::string prompt = "What is happening in this video?";
    bool verbose = false;
};

void print_usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --model <gguf> --mmproj <gguf> [options]\n"
        "  --model <path>           text model GGUF (required)\n"
        "  --mmproj <path>          multimodal projector GGUF (required)\n"
        "  --n-ctx <int>            context size (default 65536)\n"
        "  --n-gpu-layers <int>     layers offloaded to GPU, -1 = all (default -1)\n"
        "  --temperature <float>    (default 1.0)\n"
        "  --top-p <float>          (default 0.95)\n"
        "  --repeat-penalty <float> (default 1.1)\n"
        "  --max-tokens <int>       per-query cap (default 1024)\n"
        "  --width/--height <int>   synthetic frame size (default 336)\n"
        "  --initial-frames <int>   frames before the query (default 4)\n"
        "  --mid-query-frames <int> frames appended during generation (default 4)\n"
        "  --prompt <text>          question (default 'What is happening in this video?')\n"
        "  --verbose                enable native verbose diagnostics\n",
        argv0);
}

bool parse_uint(const char * s, uint32_t & out) {
    if (s == nullptr || *s == '\0') {
        return false;
    }
    char * end = nullptr;
    const unsigned long v = std::strtoul(s, &end, 10);
    if (end == s || *end != '\0') {
        return false;
    }
    out = static_cast<uint32_t>(v);
    return true;
}

bool parse_float(const char * s, float & out) {
    if (s == nullptr || *s == '\0') {
        return false;
    }
    char * end = nullptr;
    const double v = std::strtod(s, &end);
    if (end == s || *end != '\0') {
        return false;
    }
    out = static_cast<float>(v);
    return true;
}

std::vector<ours::Frame> make_frames(uint32_t n, uint32_t w, uint32_t h) {
    // Synthetic solid-color RGB frames (RGBRGB...); decoded frames in a real
    // deployment would be supplied by the video pipeline.
    std::vector<uint8_t> data(static_cast<size_t>(w) * h * 3, 0x7f);
    std::vector<ours::Frame> frames;
    frames.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        data[0] = static_cast<uint8_t>(i * 16);  // vary the first channel
        frames.push_back(ours::Frame{data.data(), w, h});
    }
    return frames;
}

} // namespace

int main(int argc, char ** argv) {
    Args a;
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
        if (arg == "--model") {
            a.model = next("--model");
        } else if (arg == "--mmproj") {
            a.mmproj = next("--mmproj");
        } else if (arg == "--n-ctx") {
            ok = parse_uint(next("--n-ctx"), a.n_ctx);
        } else if (arg == "--n-batch") {
            ok = parse_uint(next("--n-batch"), a.n_batch);
        } else if (arg == "--n-ubatch") {
            ok = parse_uint(next("--n-ubatch"), a.n_ubatch);
        } else if (arg == "--n-gpu-layers") {
            uint32_t layers = 0;
            ok = parse_uint(next("--n-gpu-layers"), layers);
            a.n_gpu_layers = static_cast<int32_t>(layers);
        } else if (arg == "--temperature") {
            ok = parse_float(next("--temperature"), a.temperature);
        } else if (arg == "--top-p") {
            ok = parse_float(next("--top-p"), a.top_p);
        } else if (arg == "--repeat-penalty") {
            ok = parse_float(next("--repeat-penalty"), a.repeat_penalty);
        } else if (arg == "--max-tokens") {
            ok = parse_uint(next("--max-tokens"), a.max_tokens);
        } else if (arg == "--width") {
            ok = parse_uint(next("--width"), a.width);
        } else if (arg == "--height") {
            ok = parse_uint(next("--height"), a.height);
        } else if (arg == "--initial-frames") {
            ok = parse_uint(next("--initial-frames"), a.initial_frames);
        } else if (arg == "--mid-query-frames") {
            ok = parse_uint(next("--mid-query-frames"), a.mid_query_frames);
        } else if (arg == "--prompt") {
            a.prompt = next("--prompt");
        } else if (arg == "--verbose") {
            a.verbose = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 2;
        }
        if (!ok) {
            std::fprintf(stderr, "invalid value for %s\n", arg.c_str());
            return 2;
        }
    }

    if (a.model.empty() || a.mmproj.empty()) {
        print_usage(argv[0]);
        return 2;
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

    try {
        ours::GemmaRuntime rt(cfg);
        rt.load();

        rt.start_session();
        rt.append_video_chunk(make_frames(a.initial_frames, a.width, a.height));

        rt.start_query(a.prompt);

        // Interleave generation with mid-query video appends: frames keep
        // arriving while the query is generating, and generation continues from
        // the updated KV cache (never reset).
        std::string output;
        bool done = false;
        uint32_t steps = 0;
        while (!done && steps < a.max_tokens) {
            ours::StepResult r = rt.step_generation();
            output = r.text;
            done = r.done;
            ++steps;
            if (!done && steps == a.initial_frames) {
                rt.append_video_chunk(make_frames(a.mid_query_frames, a.width, a.height));
            }
        }

        std::printf("[session] answer = %s\n", output.c_str());
        rt.end_session();
        return 0;
    } catch (const ours::GemmaRuntimeError & exc) {
        std::fprintf(stderr, "[ours-runtime] error: %s\n", exc.what());
        return 1;
    } catch (const std::exception & exc) {
        std::fprintf(stderr, "[ours-runtime] unexpected error: %s\n", exc.what());
        return 1;
    }
}

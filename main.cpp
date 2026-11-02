
#include "agent/agent.h"
#include "agent/controller.h"
#include "agent/state.h"
#include "evaluation/latency.h"
#include "runtime/gemma_runtime.h"
#include "video/reader.h"
#include "video/sampler.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

struct Args {
    std::string video;
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
    double fps = 1.0;
    int max_frames = 16;
    int num_frameproc = 4;
    uint32_t resize_max_size = 336;
    std::string prompt = "What is happening in this video?";
    bool verbose = false;
};

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --video <path> --model <gguf> --mmproj <gguf> [options]\n"
        "  --video <path>           input video (required)\n"
        "  --model <gguf>           text model GGUF (required)\n"
        "  --mmproj <gguf>          mmproj GGUF (required)\n"
        "  --n-ctx <int>            (default 65536)\n"
        "  --n-gpu-layers <int>     -1 = all (default -1)\n"
        "  --temperature <float>    (default 1.0)\n"
        "  --top-p <float>          (default 0.95)\n"
        "  --repeat-penalty <float> (default 1.1)\n"
        "  --max-tokens <int>       per-query cap (default 1024)\n"
        "  --fps <float>            sampled frames/second (default 1.0)\n"
        "  --max-frames <int>       frame cap, -1 = uncapped (default 16)\n"
        "  --num-frameproc <int>    frames per chunk (default 4)\n"
        "  --resize <int>           long-side cap px, 0 = off (default 336)\n"
        "  --prompt <text>          question (default from config)\n"
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
            a.video = next("--video");
        } else if (arg == "--model") {
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
        } else if (arg == "--fps") {
            float v = 0;
            ok = parse_float(next("--fps"), v);
            a.fps = v;
        } else if (arg == "--max-frames") {
            ok = parse_int(next("--max-frames"), a.max_frames);
        } else if (arg == "--num-frameproc") {
            ok = parse_int(next("--num-frameproc"), a.num_frameproc);
        } else if (arg == "--resize") {
            ok = parse_uint(next("--resize"), a.resize_max_size);
        } else if (arg == "--prompt") {
            a.prompt = next("--prompt");
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

} // namespace

int main(int argc, char ** argv) {
    Args a;
    const int early = parse_args(argc, argv, a);
    if (early >= 0) {
        return early;
    }
    if (a.video.empty() || a.model.empty() || a.mmproj.empty()) {
        usage(argv[0]);
        return 2;
    }

    try {
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

        ours::GemmaRuntime backend(cfg);
        backend.load();

        ours::VideoReader reader(a.video);
        std::string err;
        if (!reader.open(err)) {
            std::fprintf(stderr, "error: %s\n", err.c_str());
            return 1;
        }

        ours::FrameSampler sampler(a.fps, a.max_frames);
        ours::AgentState state;
        ours::Controller controller;
        ours::LatencyTracker latency;

        ours::Agent agent(&backend, state, controller,
                          /*frame_selector=*/nullptr,
                          /*semantic_pruner=*/nullptr,
                          a.num_frameproc, a.resize_max_size);

        // Driver loop (formerly Agent::run): VideoReader/FrameSampler feed the
        // Agent through its public API; the Agent's Controller picks the action.
        agent.start_session();

        const double start = ours::now_seconds();

        // Ground the answer in the first visual chunk before opening the query.
        for (int i = 0; i < a.num_frameproc; ++i) {
            ours::VideoFrame f;
            if (!sampler.next(reader, f)) {
                break;
            }
            agent.ingest_frame(f.image, f.timestamp);
        }
        agent.flush_chunk();

        agent.start_query(a.prompt);

        std::string output;
        bool done = false;
        while (!done) {
            const ours::StepOutcome out = agent.step();
            output = out.output;
            done = out.query_done;
            if (done) {
                break;
            }
            // Interleave: pull the next visual chunk while the query is still
            // generating.
            for (int i = 0; i < a.num_frameproc; ++i) {
                ours::VideoFrame f;
                if (!sampler.next(reader, f)) {
                    break;
                }
                agent.ingest_frame(f.image, f.timestamp);
            }
            agent.flush_chunk();
        }

        // Drain any frames that arrived after generation completed.
        ours::VideoFrame f;
        while (sampler.next(reader, f)) {
            agent.ingest_frame(f.image, f.timestamp);
        }
        agent.flush_chunk();

        const double elapsed = ours::now_seconds() - start;
        state.add_message("user", a.prompt);
        state.add_message("assistant", output);
        latency.record(elapsed);

        std::printf("[session] answer = %s\n", output.c_str());
        agent.end_session();

        const ours::LatencySummary s = latency.summary();
        std::printf("Latency summary: count=%d mean=%.3f p50=%.3f p99=%.3f\n",
                    s.count, s.mean, s.p50, s.p99);
        return 0;
    } catch (const ours::GemmaRuntimeError & exc) {
        std::fprintf(stderr, "error: %s\n", exc.what());
        return 1;
    } catch (const std::exception & exc) {
        std::fprintf(stderr, "error: %s\n", exc.what());
        return 1;
    }
}
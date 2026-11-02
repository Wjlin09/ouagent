// Streaming video-understanding agent. 
// Drives a persistent video session with multiple, independently-stoppable
// queries over a continuously-arriving stream of frames. The two lifecycles
// (Session vs Query) mirror ``agent/state.h``:
//
//   * Session lives across many queries and many arriving video chunks.
//   * Query starts/stops independently; stop_query never tears down the session.
//
// The image-normalization role that ``runtime/gemma_backend.py`` played in
// Python (optional aspect-preserving resize) is folded into the Agent's frame
// path here: the ffmpeg reader already yields RGB, and ``resize_max_size`` is
// applied when a chunk is flushed to the runtime backend.

#ifndef OURS_NATIVE_AGENT_AGENT_H
#define OURS_NATIVE_AGENT_AGENT_H

#include "agent/controller.h"
#include "agent/state.h"
#include "algorithms/frame_selector.h"
#include "algorithms/semantic_pruning.h"
#include "runtime/gemma_runtime.h"
#include "trace/trace_recorder.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ours {

// Outcome of one Controller-driven tick (Agent::step). ``query_done`` is set
// when the tick's generation step reached end-of-query; ``output`` then holds
// the final generated text.
struct StepOutcome {
    Action action = Action::Idle;
    std::string output;
    bool query_done = false;
};

class Agent {
public:
    Agent(GemmaRuntime * backend,
          AgentState & state,
          Controller & controller,
          FrameSelector * frame_selector = nullptr,
          SemanticPruner * semantic_pruner = nullptr,
          int num_frameproc = 1,
          uint32_t resize_max_size = 0,
          TraceRecorder * trace = nullptr);

    // -- session lifecycle --------------------------------------------------
    void start_session();
    void end_session();

    // -- visual ingestion (chunked) -----------------------------------------
    void ingest_frame(const Image & frame, double timestamp);
    void flush_chunk();

    // -- query lifecycle ----------------------------------------------------
    void start_query(const std::string & question);
    std::pair<std::string, bool> step_generation();
    void stop_query();

    // -- Controller-driven tick ---------------------------------------------
    // Ask the Controller for the next Action and execute it (AppendVideo /
    // StartQuery / Generate / StopQuery / EndSession). The external driver
    // calls this instead of implementing its own scheduling.
    StepOutcome step();

private:
    std::optional<Image> process_frame(const Image & frame);
    void flush_chunk_impl();

    GemmaRuntime * backend_;
    AgentState & state_;
    Controller & controller_;
    FrameSelector * frame_selector_;
    SemanticPruner * semantic_pruner_;
    int num_frameproc_;
    uint32_t resize_max_size_;
    std::vector<Image> chunk_buffer_;
    // True once a full chunk (num_frameproc_ frames) has been buffered and is
    // waiting for the Controller's AppendVideo action to flush it. Cleared
    // whenever the chunk is flushed.
    bool visual_chunk_ready_ = false;
    TraceRecorder * trace_ = nullptr;
    TraceSpan decode_span_;  // aggregate decode operation for the active query
};

} // namespace ours

#endif // OURS_NATIVE_AGENT_AGENT_H
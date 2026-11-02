// Agent implementation.

#include "agent/agent.h"

namespace ours {

Agent::Agent(GemmaRuntime * backend,
             AgentState & state,
             Controller & controller,
             FrameSelector * frame_selector,
             SemanticPruner * semantic_pruner,
             int num_frameproc,
             uint32_t resize_max_size,
             TraceRecorder * trace)
    : backend_(backend),
      state_(state),
      controller_(controller),
      frame_selector_(frame_selector),
      semantic_pruner_(semantic_pruner),
      num_frameproc_(num_frameproc > 0 ? num_frameproc : 1),
      resize_max_size_(resize_max_size),
      trace_(trace) {}

void Agent::start_session() {
    if (trace_) {
        trace_->event("session_start");
    }
    state_.start_session();
    backend_->start_session();
    chunk_buffer_.clear();
    visual_chunk_ready_ = false;
}

void Agent::end_session() {
    chunk_buffer_.clear();
    visual_chunk_ready_ = false;
    state_.end_session();
    backend_->end_session();
    if (trace_) {
        trace_->event("session_end");
    }
}

// -- visual ingestion --------------------------------------------------------

void Agent::ingest_frame(const Image & frame, double timestamp) {
    std::optional<Image> processed = process_frame(frame);
    if (!processed.has_value()) {
        return;  // dropped by a selector/pruner
    }
    state_.update_frame(*processed, timestamp);
    chunk_buffer_.push_back(std::move(*processed));
    if (static_cast<int>(chunk_buffer_.size()) >= num_frameproc_) {
        // A full chunk has formed: mark it ready so the Controller's next
        // AppendVideo action flushes it (batching stays governed by
        // num_frameproc_, not by an eager flush here).
        visual_chunk_ready_ = true;
        if (trace_) {
            trace_->event("chunk_ready", timestamp);
        }
    }
}

void Agent::flush_chunk() {
    flush_chunk_impl();
}

std::optional<Image> Agent::process_frame(const Image & frame) {
    Image cur = frame;
    // AutoGaze (frame-level input filtering) -- independent of pruning.
    if (frame_selector_ != nullptr) {
        std::optional<Image> out = frame_selector_->select(cur);
        if (!out.has_value()) {
            return std::nullopt;
        }
        cur = std::move(*out);
    }
    // Query-aware semantic pruning -- independent of AutoGaze.
    if (semantic_pruner_ != nullptr) {
        std::optional<Image> out =
            semantic_pruner_->select(cur, state_.current_question, &state_);
        if (!out.has_value()) {
            return std::nullopt;
        }
        cur = std::move(*out);
    }
    return cur;
}

void Agent::flush_chunk_impl() {
    const bool full_chunk = visual_chunk_ready_;
    visual_chunk_ready_ = false;  // any flush consumes the pending chunk
    if (chunk_buffer_.empty()) {
        return;
    }
    // A residual chunk (fewer than num_frameproc_ frames, flushed at video end
    // or pre-query) has no chunk_ready event; allocate a chunk id now so its
    // preprocess/prefill stay traceable.
    if (trace_ && !full_chunk) {
        trace_->begin_chunk();
    }

    // Fold GemmaBackend's image normalization (resize) in here: the runtime
    // backend receives raw RGB ``Frame`` views.
    std::vector<Frame> frames;
    frames.reserve(chunk_buffer_.size());
    {
        TraceSpan span =
            trace_ ? trace_->begin_operation("chunk_preprocess") : TraceSpan{};
        for (Image & img : chunk_buffer_) {
            if (resize_max_size_ > 0) {
                img = resize_image(img, resize_max_size_);
            }
            frames.push_back(Frame{img.rgb.data(), img.width, img.height});
        }
        chunk_buffer_.clear();
    }
    backend_->append_video_chunk(frames);
}

// -- query lifecycle ---------------------------------------------------------

void Agent::start_query(const std::string & question) {
    if (trace_) {
        trace_->event("query_arrival");
    }
    flush_chunk_impl();  // flush pending pre-query frames first
    state_.start_query(question);
    if (trace_) {
        trace_->event("query_start");
    }
    backend_->start_query(question);
}

std::pair<std::string, bool> Agent::step_generation() {
    if (trace_ && !decode_span_.is_active()) {
        decode_span_ = trace_->begin_operation("decode");
    }
    StepResult r = backend_->step_generation();
    state_.query_output = r.text;
    if (r.done) {
        state_.end_query();
        if (trace_) {
            decode_span_.end();
            trace_->event("query_end");
        }
    }
    return {std::move(r.text), r.done};
}

void Agent::stop_query() {
    decode_span_.end();  // close a decode span left open by a forced stop
    state_.stop_query();
    backend_->stop_query();
    state_.end_query();
}

// -- Controller-driven tick --------------------------------------------------

StepOutcome Agent::step() {
    StepOutcome out;
    out.action = controller_.next_action(state_, visual_chunk_ready_ ? 1 : 0);
    if (trace_ && out.action == Action::AppendVideo) {
        trace_->event("action_selected", -1.0, "AppendVideo");
    }
    switch (out.action) {
        case Action::Idle:
            break;
        case Action::StartSession:
            break;  // driven by start_session(); never emitted by next_action()
        case Action::AppendVideo:
            flush_chunk_impl();
            break;
        case Action::StartQuery:
            break;  // queries arrive via start_query(); not emitted here
        case Action::Generate: {
            std::pair<std::string, bool> r = step_generation();
            out.output = std::move(r.first);
            out.query_done = r.second;
            break;
        }
        case Action::StopQuery:
            stop_query();
            break;
        case Action::EndSession:
            //end_session();
            break;
    }
    return out;
}

} // namespace ours
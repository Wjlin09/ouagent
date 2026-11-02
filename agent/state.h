// Mutable state shared across the streaming agent. 
//
// Split into two independent lifecycles so stop_query can never tear down a
// session:
//   * Session (frames, timestamps, history) -- cleared ONLY by end_session.
//   * Query (question, output, stop flag) -- reset by start_query.
//
// STOP QUERY != RESET SESSION.

#ifndef OURS_NATIVE_AGENT_STATE_H
#define OURS_NATIVE_AGENT_STATE_H

#include "common.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ours {

enum class SessionState { Inactive, Active, Ended };
enum class QueryState { Idle, Active, StopRequested, Ended };

struct Message {
    std::string role;
    std::string content;
};

class AgentState {
public:
    // -- SESSION-level ------------------------------------------------------
    SessionState session_state = SessionState::Inactive;
    double session_started_at = 0.0;
    double current_timestamp = 0.0;
    std::vector<Image> frame_buffer;
    std::vector<double> frame_timestamps;
    std::vector<Message> conversation_history;
    std::string last_output;

    // -- QUERY-level --------------------------------------------------------
    QueryState query_state = QueryState::Idle;
    int64_t current_query_id = 0;
    std::string current_question;
    std::string query_output;
    bool stop_requested = false;

    bool session_active() const { return session_state == SessionState::Active; }
    bool query_active() const { return query_state == QueryState::Active; }

    // -- session lifecycle --------------------------------------------------
    void start_session() {
        session_state = SessionState::Active;
        session_started_at = now_seconds();
        current_timestamp = session_started_at;
        frame_buffer.clear();
        frame_timestamps.clear();
        conversation_history.clear();
        last_output.clear();
        reset_query_fields();
        query_state = QueryState::Idle;
    }

    void end_session() {
        session_state = SessionState::Ended;
        session_started_at = 0.0;
        frame_buffer.clear();
        frame_timestamps.clear();
        conversation_history.clear();
        last_output.clear();
        reset_query_fields();
        query_state = QueryState::Idle;
    }

    // -- query lifecycle ----------------------------------------------------
    void start_query(const std::string & question) {
        current_query_id += 1;
        current_question = question;
        query_output.clear();
        stop_requested = false;
        query_state = QueryState::Active;
    }

    void stop_query() {
        stop_requested = true;
        current_question.clear();
        if (query_state == QueryState::Active) {
            query_state = QueryState::StopRequested;
        }
    }

    void end_query() {
        if (!query_output.empty()) {
            last_output = query_output;
        }
        current_question.clear();
        query_state = QueryState::Idle;
        stop_requested = false;
    }

    // -- frames / history ---------------------------------------------------
    void update_frame(const Image & frame, double timestamp) {
        frame_buffer.push_back(frame);
        frame_timestamps.push_back(timestamp);
        current_timestamp = timestamp;
    }

    std::vector<Image> get_recent_frames(int n = -1) const {
        if (n < 0 || frame_buffer.size() <= static_cast<size_t>(n)) {
            return frame_buffer;
        }
        return std::vector<Image>(frame_buffer.end() - n, frame_buffer.end());
    }

    void add_message(const std::string & role, const std::string & content) {
        conversation_history.push_back(Message{role, content});
    }

    void clear_frames() {
        frame_buffer.clear();
        frame_timestamps.clear();
    }

private:
    void reset_query_fields() {
        current_query_id = 0;
        current_question.clear();
        query_output.clear();
        stop_requested = false;
    }
};

} // namespace ours

#endif // OURS_NATIVE_AGENT_STATE_H

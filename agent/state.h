// Mutable state shared across the streaming agent. 
// Split into two independent lifecycles so stop_query can never tear down a
// session:
//   * Session (history) -- cleared ONLY by end_session.
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
        conversation_history.clear();
        last_output.clear();
        reset_query_fields();
        query_state = QueryState::Idle;
    }

    void end_session() {
        session_state = SessionState::Ended;
        session_started_at = 0.0;
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

    // -- history ------------------------------------------------------------
    void add_message(const std::string & role, const std::string & content) {
        conversation_history.push_back(Message{role, content});
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

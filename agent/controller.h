// Scheduler policy.
//
// The loop is a single serial control flow (no threads) because every operation
// on the native llama context/KV must be serialized through one path. A ready
// visual chunk takes priority over a generate step (it may be appended both
// before and DURING an active query).

#ifndef OURS_NATIVE_AGENT_CONTROLLER_H
#define OURS_NATIVE_AGENT_CONTROLLER_H

#include "agent/state.h"

namespace ours {

enum class Action {
    Idle,
    StartSession,
    AppendVideo,
    StartQuery,
    Generate,
    StopQuery,  // NOT EndSession: never resets the session
    EndSession,
};

class Controller {
public:
    Action next_action(const AgentState & state, int pending_visual = 0) const {
        if (state.session_state == SessionState::Ended) {
            return Action::EndSession;
        }
        if (!state.session_active()) {
            return Action::Idle;
        }
        if (state.stop_requested || state.query_state == QueryState::StopRequested) {
            return Action::StopQuery;
        }
        // A ready visual chunk wins, both before and during an active query.
        if (pending_visual > 0) {
            return Action::AppendVideo;
        }
        if (state.query_active()) {
            return Action::Generate;
        }
        return Action::Idle;
    }
};

} // namespace ours

#endif // OURS_NATIVE_AGENT_CONTROLLER_H

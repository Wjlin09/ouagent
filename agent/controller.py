"""Policy that decides the scheduler's next action."""

from __future__ import annotations

from enum import Enum

from .state import AgentState, QueryState, SessionState


class Action(str, Enum):
    """Lifecycle actions for the serial scheduler loop.

    ``START_SESSION`` and ``START_QUERY`` are explicit commands (the Agent's
    ``start_session()`` / ``start_query()``); :meth:`Controller.next_action`
    emits the scheduler-decidable subset: ``APPEND_VIDEO``, ``GENERATE``,
    ``STOP_QUERY``, ``END_SESSION`` (and ``IDLE``).

    ``STOP_QUERY`` is *not* ``END_SESSION``: it never resets the session.
    """

    IDLE = "idle"
    START_SESSION = "start_session"
    APPEND_VIDEO = "append_video"
    START_QUERY = "start_query"
    GENERATE = "generate"
    STOP_QUERY = "stop_query"
    END_SESSION = "end_session"


class Controller:
    """Serial event-loop policy.

    The loop is a single deterministic control flow (no threads) because every
    operation on the native llama context/KV must be serialized through one
    path. ``pending_visual`` reports how many frames are buffered and ready to
    append; a visual chunk may be appended both before and DURING an active
    query (the backend injects the media at the current context position and
    generation continues), so a ready chunk takes priority over a generate step.
    """

    def next_action(self, state: AgentState, pending_visual: int = 0) -> Action:
        if state.session_state is SessionState.ENDED:
            return Action.END_SESSION
        if not state.session_active:
            return Action.IDLE

        if state.stop_requested or state.query_state is QueryState.STOP_REQUESTED:
            return Action.STOP_QUERY

        # A ready visual chunk wins, both before and during an active query.
        if pending_visual > 0:
            return Action.APPEND_VIDEO

        if state.query_active:
            return Action.GENERATE

        return Action.IDLE

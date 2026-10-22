"""Mutable state shared across the streaming video-understanding agent.

State is split into two independent lifecycles so that stopping a query can
never tear down a session:

* **Session lifecycle** (``SessionState``): ``INACTIVE -> ACTIVE -> ENDED``.
  Session-level data (video/frames, timestamps, accumulated visual state,
  conversation history) lives here and is cleared *only* by ``end_session``.
* **Query lifecycle** (``QueryState``): ``IDLE -> ACTIVE -> STOP_REQUESTED /
  ENDED -> IDLE``. Query-level data (question, generated text, stop flag) lives
  here and is reset by ``start_query``.

``STOP QUERY != RESET SESSION``: ``stop_query`` only transitions the *query*
back to idle; it must never clear session-level visual/KV state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class SessionState(str, Enum):
    INACTIVE = "inactive"
    ACTIVE = "active"
    ENDED = "ended"


class QueryState(str, Enum):
    IDLE = "idle"
    ACTIVE = "active"
    STOP_REQUESTED = "stop_requested"
    ENDED = "ended"


@dataclass
class AgentState:
    """State container split into SESSION-level and QUERY-level fields.

    Session-level (cleared only by :meth:`end_session`):
        session_state, session_started_at, current_timestamp, frame_buffer,
        frame_timestamps, conversation_history, last_output.

    Query-level (reset by :meth:`start_query`):
        query_state, current_query_id, current_question, query_output,
        stop_requested.
    """

    # -- SESSION-level -------------------------------------------------------
    session_state: SessionState = SessionState.INACTIVE
    session_started_at: float | None = None
    current_timestamp: float = field(default_factory=time.time)
    frame_buffer: list[np.ndarray] = field(default_factory=list)
    frame_timestamps: list[float] = field(default_factory=list)
    conversation_history: list[dict] = field(default_factory=list)

    # -- QUERY-level ---------------------------------------------------------
    query_state: QueryState = QueryState.IDLE
    current_query_id: int = 0
    current_question: str | None = None
    query_output: str = ""
    stop_requested: bool = False

    # Convenience: most recent completed answer, also kept in
    # ``conversation_history``. Session-level: cleared by start/end_session
    # (alongside the history), never by the query lifecycle.
    last_output: str | None = None

    @property
    def session_active(self) -> bool:
        return self.session_state is SessionState.ACTIVE

    @property
    def query_active(self) -> bool:
        return self.query_state is QueryState.ACTIVE

    # -- session lifecycle ---------------------------------------------------

    def start_session(self) -> None:
        """Enter ACTIVE, clearing any previous session-level data."""
        self.session_state = SessionState.ACTIVE
        self.session_started_at = time.time()
        self.current_timestamp = self.session_started_at
        self.frame_buffer.clear()
        self.frame_timestamps.clear()
        self.conversation_history.clear()
        self.last_output = None
        # A fresh session also starts with no active query.
        self._reset_query_fields()
        self.query_state = QueryState.IDLE

    def end_session(self) -> None:
        """Terminate the whole session and clear session-level state."""
        self.session_state = SessionState.ENDED
        self.session_started_at = None
        self.frame_buffer.clear()
        self.frame_timestamps.clear()
        self.conversation_history.clear()
        self.last_output = None
        self._reset_query_fields()
        self.query_state = QueryState.IDLE

    # -- query lifecycle -----------------------------------------------------

    def start_query(self, question: str) -> None:
        """Begin a new query: bump the id and reset query-level fields.

        Session-level data (frames, timestamps, history) is deliberately left
        untouched.
        """
        self.current_query_id += 1
        self.current_question = question
        self.query_output = ""
        self.stop_requested = False
        self.query_state = QueryState.ACTIVE

    def stop_query(self) -> None:
        """Request the current query to stop (STOP != reset session).

        Clears ``current_question`` so any frames arriving before the next
        query are processed query-agnostically, not with the stopped question.
        """
        self.stop_requested = True
        self.current_question = None
        if self.query_state is QueryState.ACTIVE:
            self.query_state = QueryState.STOP_REQUESTED

    def end_query(self) -> None:
        """Mark the current query finished, returning to IDLE.

        Keeps the generated text in ``query_output`` and ``last_output`` for the
        conversation history; does not clear any session-level state. Clears
        ``current_question`` so the next query starts from a clean question.
        """
        if self.query_output:
            self.last_output = self.query_output
        self.current_question = None
        self.query_state = QueryState.IDLE
        self.stop_requested = False

    def _reset_query_fields(self) -> None:
        self.current_query_id = 0
        self.current_question = None
        self.query_output = ""
        self.stop_requested = False

    # -- frames / history ----------------------------------------------------

    def update_frame(self, frame: np.ndarray, timestamp: float) -> None:
        """Append a freshly read frame and advance the current timestamp."""
        self.frame_buffer.append(frame)
        self.frame_timestamps.append(timestamp)
        self.current_timestamp = timestamp

    def get_recent_frames(self, n: int | None = None) -> list[np.ndarray]:
        """Return the most recent ``n`` frames (all frames if ``n`` is None)."""
        if n is None:
            return list(self.frame_buffer)
        return list(self.frame_buffer[-n:]) if n > 0 else []

    def add_message(self, role: str, content: str) -> None:
        """Record a conversation turn (session-level history)."""
        self.conversation_history.append({"role": role, "content": content})

    def clear_frames(self) -> None:
        """Discard all buffered frames and their timestamps."""
        self.frame_buffer.clear()
        self.frame_timestamps.clear()

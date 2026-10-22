"""Streaming video-understanding agent orchestration.

The agent exposes a persistent interactive session with two decoupled
lifecycles (see :mod:`agent.state`):

* **Session** lives across many queries and many arriving video chunks.
* **Query** starts/stops independently; stopping a query never tears down the
  session.

"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Iterator, Optional

import numpy as np

from .controller import Action, Controller
from .state import AgentState
from algorithms.frame_selector import AutoGazeFrameSelector
from algorithms.semantic_pruning import SemanticPruner
from evaluation.latency import LatencyTracker
from runtime.gemma_backend import GemmaBackend
from video.sampler import FrameSampler

if TYPE_CHECKING:
    from video.reader import VideoReader


class Agent:
    """Drives a persistent video session with multiple, independently-stoppable
    user queries over a continuously-arriving stream of video frames."""

    def __init__(
        self,
        backend: GemmaBackend,
        reader: VideoReader,
        sampler: FrameSampler,
        state: AgentState,
        controller: Controller,
        frame_selector: AutoGazeFrameSelector | None = None,
        semantic_pruner: SemanticPruner | None = None,
        latency_tracker: LatencyTracker | None = None,
        prompt_template: str = "What is happening in this image?",
        num_frameproc: int = 1,
    ) -> None:
        self._backend = backend
        self._reader = reader
        self._sampler = sampler
        self._state = state
        self._controller = controller
        self._frame_selector = frame_selector
        self._semantic_pruner = semantic_pruner
        self.latency_tracker = latency_tracker
        self._prompt_template = prompt_template
        self._num_frameproc = max(1, int(num_frameproc))
        # Frames buffered into the next visual-processing unit (chunk).
        self._chunk_buffer: list[np.ndarray] = []

    # -- session lifecycle ---------------------------------------------------

    def start_session(self) -> None:
        self._state.start_session()
        self._backend.start_session()
        self._chunk_buffer = []

    def end_session(self) -> None:
        self._chunk_buffer = []
        self._state.end_session()
        self._backend.end_session()

    # -- visual ingestion (chunked) ------------------------------------------

    def ingest_frame(self, frame: np.ndarray, timestamp: float) -> None:
        """Process one arriving frame and accumulate it into a visual chunk.

        Pipeline: ``[AutoGaze on/off] -> [semantic pruning on/off] -> buffer``.
        When ``num_frameproc`` frames have accumulated, the chunk is appended to
        the persistent KV in one operation. A chunk is appended both before and
        DURING an active query (the backend injects the media at the current
        context position and generation continues); it is never held back until
        the turn closes.
        """
        processed = self._process_frame(frame)
        if processed is None:
            return
        self._state.update_frame(processed, timestamp)
        self._chunk_buffer.append(processed)
        if len(self._chunk_buffer) >= self._num_frameproc:
            self._flush_chunk()

    def flush_chunk(self) -> None:
        """Append the buffered chunk now (allowed before and during a query)."""
        self._flush_chunk()

    def _process_frame(self, frame: np.ndarray) -> np.ndarray | None:
        # AutoGaze (frame-level input filtering) — independent of pruning.
        if self._frame_selector is not None:
            frame = self._frame_selector.select(frame)
            if frame is None:
                return None
        # Query-aware semantic pruning — independent of AutoGaze. ``query`` is
        # None before the first query (query-agnostic) and the current question
        # afterwards (query-guided).
        if self._semantic_pruner is not None:
            frame = self._semantic_pruner.select(
                frame,
                query=self._state.current_question,
                session_state=self._state,
            )
            if frame is None:
                return None
        return frame

    def _flush_chunk(self) -> None:
        if not self._chunk_buffer:
            return
        chunk = self._chunk_buffer
        self._chunk_buffer = []
        # A visual chunk may be appended both before and DURING an active query:
        # the backend injects the media at the current context position and the
        # in-flight generation continues from the updated KV. No buffering.
        self._backend.append_video_chunk(chunk)

    def _ingest_next_chunk(
        self, frames: Iterator[tuple[int, float, object]]
    ) -> None:
        """Ingest up to one ``num_frameproc`` chunk of frames and flush it.

        Pulls at most ``num_frameproc`` frames from ``frames`` through the
        pipeline and then flushes any partial chunk, so the frames reach the
        backend immediately (mid-query ``step_generation`` must see them).
        """
        for _ in range(self._num_frameproc):
            try:
                _frame_index, timestamp, frame = next(frames)
            except StopIteration:
                break
            self.ingest_frame(frame, timestamp)
        self._flush_chunk()

    # -- query lifecycle -----------------------------------------------------

    def start_query(self, question: str) -> None:
        # Flush any pending pre-query frames into the session first.
        self._flush_chunk()
        self._state.start_query(question)
        self._backend.start_query(question)

    def step_generation(self) -> tuple[str, bool]:
        """Advance the active query by one token; returns ``(text, done)``."""
        text, done = self._backend.step_generation()
        self._state.query_output = text
        if done:
            self._state.end_query()
        return text, done

    def stop_query(self) -> None:
        """Stop the current query (STOP != reset session)."""
        self._state.stop_query()
        self._backend.stop_query()
        self._state.end_query()

    # -- convenience driver --------------------------------------------------

    def iter_frames(self) -> Iterator[tuple[int, float, np.ndarray]]:
        """Yield ``(frame_index, timestamp, frame)`` for the sampled video stream.

        Thin accessor over the configured sampler + reader so an external event
        loop (e.g. an interactive terminal driver) can drive
        :meth:`ingest_frame` itself without reaching into private state.
        """
        return self._sampler.stream(self._reader.read_frames())

    def run(self) -> None:
        """Deterministic single-query demo with interleaved video/generation.
        """
        self.start_session()

        frames = self._sampler.stream(self._reader.read_frames())
        question = self._prompt_template

        # Ground the answer in the first visual chunk before opening the query.
        self._ingest_next_chunk(frames)

        start = time.perf_counter()
        self.start_query(question)

        output, done = "", False
        while not done:
            # The controller decides the next scheduler step (generate / stop /
            # end); STOP_QUERY is never treated as END_SESSION.
            action = self._controller.next_action(self._state)
            if action is Action.STOP_QUERY:
                self.stop_query()
                break
            if action is Action.END_SESSION:
                break
            if action is not Action.GENERATE:
                break  # IDLE: nothing left to generate

            output, done = self.step_generation()
            if done:
                break
            # Interleave: pull the next visual chunk into the persistent KV
            # while the query is still generating.
            self._ingest_next_chunk(frames)

        for _frame_index, timestamp, frame in frames:
            self.ingest_frame(frame, timestamp)
        self._flush_chunk()

        elapsed = time.perf_counter() - start

        self._state.add_message("user", question)
        self._state.add_message("assistant", output)

        if self.latency_tracker is not None:
            self.latency_tracker.record(elapsed)

        print(f"[session] answer = {output}")
        self.end_session()

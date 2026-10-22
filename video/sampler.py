"""Frame sampler that bounds the number of frames fed to the model."""

from __future__ import annotations

from collections.abc import Iterable, Iterator


class FrameSampler:
    """Streaming sampler: forwards at most ``fps`` frames per second, capped at
    ``max_frames`` frames in total.

    ``stream()`` wraps an iterator of ``(frame_index, timestamp, frame)`` tuples
    and yields only the frames that should enter the pipeline, so the video is
    sampled *while* it is being read and is never fully buffered in memory.
    """

    def __init__(self, fps: float = 2.0, max_frames: int | None = 16) -> None:
        self._fps = float(fps)
        # ``max_frames=None`` means uncapped: stream the whole input.
        self._max_frames = None if max_frames is None else int(max_frames)
        # Time (seconds) that must elapse between two selected frames.
        self._interval = 1.0 / self._fps if self._fps > 0.0 else 0.0

    def stream(
        self,
        frames: Iterable[tuple[int, float, object]],
    ) -> Iterator[tuple[int, float, object]]:
        """Yield only the frames selected by the fps / max_frames policy.

        ``frames`` must yield ``(frame_index, timestamp, frame)`` tuples in
        timestamp order.
        """
        sampled = 0
        next_timestamp = 0.0

        for frame_index, timestamp, frame in frames:
            if self._max_frames is not None and sampled >= self._max_frames:
                break
            if self._interval > 0.0 and timestamp < next_timestamp:
                continue

            sampled += 1
            next_timestamp = timestamp + self._interval
            yield frame_index, timestamp, frame

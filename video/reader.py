"""OpenCV-based video frame reading."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoReader:
    """Reads frames from a video file using OpenCV."""

    def __init__(self, video_path: str | Path | None = None) -> None:
        self._video_path = Path(video_path) if video_path is not None else None

    def read_frames(self) -> Generator[tuple[int, float, np.ndarray], None, None]:
        """Yield ``(frame_index, timestamp, frame)`` for every frame.

        If ``video_path`` is None or cannot be opened, logs an error and yields
        nothing (never fabricates frames).
        """
        if self._video_path is None:
            logger.error("video_path is None; no frames will be read")
            return

        if not self._video_path.exists():
            logger.error("video file does not exist: %s", self._video_path)
            return

        capture = cv2.VideoCapture(str(self._video_path))
        if not capture.isOpened():
            logger.error("failed to open video: %s", self._video_path)
            capture.release()
            return

        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                logger.warning("invalid fps (%s); falling back to 30.0", fps)
                fps = 30.0

            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp = frame_index / fps
                yield frame_index, timestamp, frame
                frame_index += 1
        finally:
            capture.release()

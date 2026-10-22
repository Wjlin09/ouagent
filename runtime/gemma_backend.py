"""High-level Gemma 4 multimodal inference backend.

This module is the pure-Python face of the runtime. It converts an image into
raw RGB bytes and delegates all native work to :class:`~runtime.native_mtmd.NativeMTMDBackend`.
Nothing here touches ctypes or C pointers.

The intended call chain is::

    Agent -> GemmaBackend -> NativeMTMDBackend -> libmtmd / libllama -> Gemma 4
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

from .native_mtmd import NativeMTMDBackend

logger = logging.getLogger(__name__)

ImageInput = str | os.PathLike | bytes | np.ndarray


class GemmaBackend:
    """Gemma 4 inference backend: normalizes an image, then calls libmtmd.

    Construction does *not* load the native libraries: ``__init__`` only records
    configuration and builds a lazy :class:`NativeMTMDBackend`. The native
    runtime (and any ``libmtmd``/``libllama`` availability error) is only hit on
    the first :meth:`infer` call.
    """

    def __init__(
        self,
        model_path: str | os.PathLike,
        mmproj_path: str | os.PathLike,
        n_ctx: int = 8192,
        n_batch: int = 2048,
        n_ubatch: int = 512,
        n_gpu_layers: int = -1,
        n_threads: Optional[int] = None,
        mmproj_use_gpu: bool = True,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 512,
        seed: int = 1234,
        repeat_penalty: float = 1.1,
        verbose: bool = False,
        libllama_path: Optional[str] = None,
        libmtmd_path: Optional[str] = None,
        lib_dir: Optional[str] = None,
        resize_max_size: Optional[int] = None,
    ) -> None:
        # ``None`` (or non-positive) means "no resize": the baseline passes
        # frames through at their original resolution, bit-for-bit unchanged.
        self._resize_max_size: Optional[int] = None
        if resize_max_size is not None and int(resize_max_size) > 0:
            self._resize_max_size = int(resize_max_size)
            logger.info(
                "frame resize enabled: long side capped at %d px (aspect preserved)",
                self._resize_max_size,
            )
        self._native = NativeMTMDBackend(
            model_path=os.fspath(model_path),
            mmproj_path=os.fspath(mmproj_path),
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_ubatch=n_ubatch,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            mmproj_use_gpu=mmproj_use_gpu,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
            repeat_penalty=repeat_penalty,
            verbose=verbose,
            libllama_path=libllama_path,
            libmtmd_path=libmtmd_path,
            lib_dir=lib_dir,
        )

    def infer(self, image: ImageInput, prompt: str) -> str:
        """Run inference over a single image and a text prompt.

        Args:
            image: A path to an image file, raw image bytes, or a numpy array
                of shape ``(H, W, 3)`` in BGR order (the OpenCV convention).
            prompt: The text prompt to send alongside the image.

        Returns:
            The model's textual response.
        """
        rgb, width, height = self._to_rgb(image)
        return self._native.infer(rgb, width, height, prompt)

    def infer_cli_equivalent(self, image: ImageInput, prompt: str) -> str:
        """Run the single-image CLI-equivalent debug path (see
        :meth:`NativeMTMDBackend.infer_cli_equivalent`).

        One image + one question -> one ``mtmd_tokenize_from_parts`` -> one
        generation. Exists to isolate the tokenize/eval split from the prompt
        text when diffing Native against llama-mtmd-cli.
        """
        rgb, width, height = self._to_rgb(image)
        return self._native.infer_cli_equivalent(rgb, width, height, prompt)

    def start_session(self) -> None:
        """Begin a fresh session: clear the native KV cache and counters."""
        self._native.start_session()

    def end_session(self) -> None:
        """Terminate the whole session and clear session-level native state."""
        self._native.end_session()

    def append_video_chunk(self, images: list[ImageInput]) -> None:
        """Append a chunk of frames to the persistent context (no generation).

        Each ``image`` is normalized to raw RGB; the whole chunk is appended to
        the KV cache in one native tokenize+eval pass.
        """
        rgb_frames = [self._to_rgb(image) for image in images]
        self._native.append_video_chunk(rgb_frames)

    def add_frame(self, image: ImageInput) -> None:
        """Append one video frame (thin wrapper over :meth:`append_video_chunk`)."""
        self.append_video_chunk([image])

    def start_query(self, question: str) -> None:
        """Open a query: close the user turn and open the model turn (no gen)."""
        self._native.start_query(question)

    def step_generation(self) -> tuple[str, bool]:
        """Advance the active query by one token; returns ``(text, done)``."""
        return self._native.step_generation()

    def stop_query(self) -> None:
        """Stop the current query; session/KV state is preserved."""
        self._native.stop_query()

    def finalize(self, question: str) -> str:
        """Run a single query to completion (convenience wrapper)."""
        return self._native.finalize(question)

    def close(self) -> None:
        """Release native resources (safe to call multiple times)."""
        self._native.close()

    def reset(self) -> None:
        """Start a new conversation/video: clear the native KV cache."""
        self._native.reset()

    # -- image normalization -------------------------------------------------

    @staticmethod
    def _resize_dims(width: int, height: int, max_size: int) -> tuple[int, int]:
        """Target ``(new_w, new_h)`` keeping aspect ratio with long side <= max_size.

        Returns the original size unchanged when the image is already no larger
        than ``max_size`` (never upscales). Pure arithmetic -- no OpenCV -- so it
        is unit-testable without a model or cv2.
        """
        longest = max(width, height)
        if longest <= max_size:
            return width, height
        scale = max_size / float(longest)
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        return new_w, new_h

    def _to_rgb(self, image: ImageInput) -> tuple[bytes, int, int]:
        """Convert ``image`` to ``(rgb_bytes, width, height)``.

        ``rgb_bytes`` is raw ``RGBRGBRGB...`` data of length ``width*height*3``,
        matching what ``mtmd_bitmap_init`` expects.

        When ``resize_max_size`` is set, the RGB frame is downscaled (long side
        capped at ``resize_max_size``, aspect ratio preserved) *before* the bytes
        reach NativeMTMD/mtmd, so a smaller input resolution yields fewer visual
        tokens per frame. ``resize_max_size=None`` keeps the baseline behaviour.
        """
        # Imported lazily so ``import runtime`` / backend construction do not
        # require OpenCV (only actual image conversion does).
        import cv2

        bgr: np.ndarray

        if isinstance(image, (str, os.PathLike)):
            path = Path(image)
            if not path.exists():
                raise FileNotFoundError(f"image file not found: {path}")
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"failed to decode image: {path}")
        elif isinstance(image, bytes):
            bgr = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError("failed to decode image bytes")
        elif isinstance(image, np.ndarray):
            bgr = image
        else:
            raise TypeError(f"unsupported image type: {type(image)!r}")

        if bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError(
                f"expected an image of shape (H, W, 3), got {bgr.shape}"
            )

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if self._resize_max_size is not None:
            height, width = rgb.shape[:2]
            new_w, new_h = self._resize_dims(width, height, self._resize_max_size)
            if (new_w, new_h) != (width, height):
                rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
                logger.debug(
                    "frame resize: %dx%d -> %dx%d", width, height, new_w, new_h
                )

        height, width = rgb.shape[:2]
        return rgb.tobytes(), width, height

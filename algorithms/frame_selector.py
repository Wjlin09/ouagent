"""Frame-level input filtering via AutoGaze (Attend Before Attention).

AutoGaze is a causal, autoregressive "gazing" model that predicts, for each
video frame, which patches carry information and can drop the redundant ones
(see ``third_party/AutoGaze``). Downstream of this project Gemma 4 is called
through llama.cpp's high-level image API, so patches inside Gemma cannot be
skipped; input filtering is therefore applied at the frame level: a frame that
AutoGaze gazes at no patch for is redundant (fully predictable from the frames
already seen) and is dropped before it reaches the model.

AutoGaze and PyTorch are imported lazily so the rest of OURS_Agent has no hard
dependency on them.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# AutoGaze is trained on 16-frame clips; reset its streaming cache every this
# many frames to match the training distribution and keep memory bounded.
_MAX_FRAMES_PER_CLIP = 16


class AutoGazeFrameSelector:
    """Drop redundant frames based on AutoGaze's per-frame gaze density.

    Keeps AutoGaze's causal streaming state across :meth:`select` calls so each
    frame is scored in the context of the frames before it.
    """

    def __init__(
        self,
        model_id: str = "nvidia/AutoGaze",
        gazing_ratio: float = 0.75,
        task_loss_requirement: float = 0.7,
        device: str | None = None,
    ) -> None:
        self._model_id = model_id
        self._gazing_ratio = float(gazing_ratio)
        self._task_loss_requirement = float(task_loss_requirement)
        self._device = device

        self._model = None
        self._processor = None
        self._transform = None
        self._torch = None
        self._past = None  # streaming caches; None == start of a clip
        self._frames_seen = 0

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from autogaze.datasets.video_utils import transform_video_for_pytorch
            from autogaze.models.autogaze import AutoGaze, AutoGazeImageProcessor
        except ImportError as exc:
            raise RuntimeError(
                "AutoGaze input filtering requires PyTorch and the `autogaze` "
                "package. Install it with: pip install -e third_party/AutoGaze"
            ) from exc

        self._torch = torch
        self._transform = transform_video_for_pytorch
        self._processor = AutoGazeImageProcessor.from_pretrained(self._model_id)
        model = AutoGaze.from_pretrained(self._model_id)
        if self._device is not None:
            model = model.to(self._device)
        model.eval()
        self._model = model

    def select(self, frame: np.ndarray) -> np.ndarray | None:
        """Return ``frame`` if it is informative, otherwise ``None`` (drop it)."""
        self._load()

        if self._frames_seen >= _MAX_FRAMES_PER_CLIP:
            self._past = None
            self._frames_seen = 0

        # OpenCV BGR -> RGB, then to the (B, T, C, H, W) layout AutoGaze expects.
        rgb = frame[..., ::-1]
        video = self._transform(rgb[np.newaxis], self._processor)  # (1, C, H, W)
        video = video[None]  # (1, 1, C, H, W)

        past = self._past
        with self._torch.inference_mode():
            outputs = self._model(
                {"video": video},
                gazing_ratio=self._gazing_ratio,
                task_loss_requirement=self._task_loss_requirement,
                generate_only=True,
                use_cache=True,
                past_key_values=past["key_values"] if past else None,
                past_inputs_embeds=past["inputs_embeds"] if past else None,
                past_attention_mask=past["attention_mask"] if past else None,
                past_conv_values=past["conv_values"] if past else None,
            )

        # NB: AutoGaze's input kwarg is ``past_inputs_embeds`` (plural) but the
        # returned key is ``past_input_embeds`` (singular) — verified in source.
        self._past = {
            "key_values": outputs["past_key_values"],
            "inputs_embeds": outputs["past_input_embeds"],
            "attention_mask": outputs["past_attention_mask"],
            "conv_values": outputs["past_conv_values"],
        }
        self._frames_seen += 1

        # A frame with no non-padded gaze is fully redundant: drop it.
        real_gazes = int((~outputs["if_padded_gazing"]).sum().item())
        if real_gazes == 0:
            logger.debug("AutoGaze dropped a redundant frame")
            return None
        return frame

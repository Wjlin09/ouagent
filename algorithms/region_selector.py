"""Region-of-interest selection strategy.

TODO: integrate spatial region selection (saliency / gaze heatmaps) to crop the
most informative image regions before inference, instead of the identity
pass-through currently implemented.
"""

from __future__ import annotations

import numpy as np


class RegionSelector:
    """Selects an image region to attend to. Identity for now."""

    def select(self, frame: np.ndarray) -> np.ndarray:
        # TODO(region selection): return the salient crop instead of the full frame.
        return frame

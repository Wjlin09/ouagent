"""OURS_Agent runtime package.

Exposes the Gemma 4 inference backend chain:

    GemmaBackend (high-level, image normalization)
        -> NativeMTMDBackend (ctypes binding)
            -> libmtmd / libllama -> Gemma 4
"""

from .gemma_backend import GemmaBackend
from .native_mtmd import LibmtmdError, NativeMTMDBackend

__all__ = ["GemmaBackend", "NativeMTMDBackend", "LibmtmdError"]

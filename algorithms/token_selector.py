"""Token pruning strategy (documented boundary — not implemented).

Real token-level semantic pruning requires access to the *internal* visual
token sequence inside the VLM's attention, which the current native interface
does not expose: ``mtmd_helper_eval_chunks`` encodes+decodes an image chunk as
an opaque unit, so Python-side code never sees the per-patch tokens.

Because of that, deleting "tokens" in Python here would **not** be real
attention/token pruning, and modifying the third-party interface is out of
scope. Semantic pruning is therefore abstracted at the highest level the
current architecture supports — query-guided region/patch selection (see
``region_selector.py`` and ``semantic_pruning.py``) — rather than fake token
dropping.

This module exists only to mark that boundary; it is a no-op and must stay one
until the native interface exposes per-token visual selection.
"""

from __future__ import annotations

from typing import Any


class TokenSelector:
    """Identity no-op placeholder. Do NOT treat as real token pruning."""

    def prune(self, tokens: list[Any]) -> list[Any]:
        return tokens

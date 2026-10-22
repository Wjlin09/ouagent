"""Query-aware semantic pruning interface (integration point).

This is the seam where a *real* query-guided visual selector will plug in. It
is deliberately independent from AutoGaze (:mod:`algorithms.frame_selector`):
enabling semantic pruning must never implicitly enable AutoGaze, and vice
versa.

Status: **no concrete implementation ships yet.** The identity pass-through is
*not* a semantic algorithm, so it is never presented as one. When this is
enabled but no scorer has been provided, construction fails loudly rather than
silently pretending to prune.
"""

from __future__ import annotations

from typing import Any


class SemanticPruner:
    """Query-aware semantic pruning interface.

    Subclasses must implement :meth:`select`. ``query`` is ``None`` before the
    first ``start_query`` (query-agnostic filtering) and the current question
    afterwards (query-guided filtering). ``session_state`` carries accumulated
    session context a real scorer may need (e.g. seen frames/timestamps).
    """

    def select(
        self,
        frame: Any,
        query: str | None = None,
        session_state: Any = None,
    ) -> Any:
        """Return the frame (possibly transformed) or ``None`` to drop it.

        Raises ``NotImplementedError``: there is no concrete scorer yet, and a
        pass-through must not masquerade as real semantic pruning.
        """
        raise NotImplementedError(
            "no concrete query-aware semantic pruner is implemented; "
            "enable algorithms.semantic_pruning only after providing a real "
            "scorer (an identity pass-through is not a semantic algorithm)"
        )

"""langchain-halu: a thin LangChain integration for Komplex AI — a hallucination detector for LLM output.

This package adapts the ``halu`` SDK (Komplex AI's hallucination detector) to
LangChain, exposing three complementary patterns:

* :func:`halu_annotate` -- a Runnable that enriches an answer with a verdict
  (non-blocking "annotate").
* :func:`halu_guard` -- a Runnable that passes clean answers through and raises
  on flagged ones (blocking "guard").
* :class:`HaluCallbackHandler` -- a callback that passively monitors LLM
  completions and logs flagged answers (observability "monitor").
* :func:`generate_clean` -- regenerate an answer until it passes the detector
  (the "retry" pattern).

The ``halu`` error and result types are re-exported for convenience so callers
do not need a separate ``import halu`` just to catch a flagged answer.

Set ``HALU_API_KEY`` in the environment (or pass ``api_key`` through ``halu``)
before use. See https://detector.komplexai.io/guide for details and limits.
"""

from __future__ import annotations

from halu import (
    DetectResult,
    HaluError,
    HaluHallucinationFlagged,
)

from .callbacks import HaluCallbackHandler
from .retry import generate_clean
from .runnables import halu_annotate, halu_guard

__version__ = "0.1.0"

__all__ = [
    "halu_annotate",
    "halu_guard",
    "HaluCallbackHandler",
    "generate_clean",
    # Re-exported from halu for convenience.
    "DetectResult",
    "HaluError",
    "HaluHallucinationFlagged",
    "__version__",
]

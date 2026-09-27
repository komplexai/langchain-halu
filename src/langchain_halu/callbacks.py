"""LangChain callback handler for passive hallucination monitoring.

``HaluCallbackHandler`` observes LLM completions as they finish and runs the
Komplex AI detector on each one, logging a warning (and stashing the last
result) whenever an answer is flagged. It never blocks or alters generation —
this is the "monitor" pattern, meant for observability, dashboards, and alerts.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

import halu

from ._common import require_langchain_core

logger = logging.getLogger("langchain_halu")

require_langchain_core()
from langchain_core.callbacks import BaseCallbackHandler  # noqa: E402


class HaluCallbackHandler(BaseCallbackHandler):
    """Run hallucination detection on every LLM completion, passively.

    On ``on_llm_end`` the handler extracts each generated text, runs
    ``halu.detect`` on it, and — when the result is flagged — emits a warning
    log record and stores the result. Generation is never interrupted; use
    :func:`langchain_halu.halu_guard` if you need to block flagged answers.

    Parameters
    ----------
    threshold:
        Probability at or above which a warning is logged. The detector's own
        ``.flag`` is authoritative for the flagged/monitor decision; this
        threshold is used as an additional gate so callers can tighten
        sensitivity without changing the SDK default. Defaults to 0.5.
    prompt:
        Optional static prompt/context passed to the detector for every call.
        Most callers leave this ``None`` and rely on response-only detection,
        since the callback does not receive the original prompt text.
    log_level:
        Logging level used for flagged answers. Defaults to ``logging.WARNING``.

    Attributes
    ----------
    last_result:
        The most recent :class:`halu.DetectResult`, or ``None`` if no
        completion has been processed yet.
    flagged_results:
        A list of all flagged :class:`halu.DetectResult` objects seen so far.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        prompt: Optional[str] = None,
        log_level: int = logging.WARNING,
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.prompt = prompt
        self.log_level = log_level
        self.last_result: Optional["halu.DetectResult"] = None
        self.flagged_results: List["halu.DetectResult"] = []

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Detect on each generated text when an LLM run completes.

        Parameters
        ----------
        response:
            A LangChain ``LLMResult``-like object exposing ``.generations``
            (a list of lists of generation objects, each with ``.text``).
        """
        for text in self._iter_texts(response):
            if not text:
                continue
            try:
                result = halu.detect(text, prompt=self.prompt)
            except halu.HaluError as exc:
                # Monitoring must never break the host application; log and move on.
                logger.warning("langchain-halu: detection failed: %s", exc)
                continue

            self.last_result = result
            if result.flag or result.p_hallucination >= self.threshold:
                self.flagged_results.append(result)
                logger.log(
                    self.log_level,
                    "langchain-halu: possible hallucination flagged "
                    "(p=%.3f, top_regime=%s, request_id=%s)",
                    result.p_hallucination,
                    result.top_regime,
                    result.request_id,
                )

    @staticmethod
    def _iter_texts(response: Any) -> List[str]:
        """Best-effort extraction of generated texts from an ``LLMResult``."""
        texts: List[str] = []
        generations = getattr(response, "generations", None)
        if not generations:
            return texts
        for batch in generations:
            for generation in batch:
                text = getattr(generation, "text", None)
                if isinstance(text, str) and text:
                    texts.append(text)
                    continue
                # Chat generations may carry the text on a nested message.
                message = getattr(generation, "message", None)
                content = getattr(message, "content", None)
                if isinstance(content, str) and content:
                    texts.append(content)
        return texts

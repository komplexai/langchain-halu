"""LangChain ``Runnable`` factories for the Komplex AI hallucination detector.

These are thin adapters over ``halu.detect``. They accept whatever an upstream
LLM/chain step emits (a ``str`` or an ``AIMessage``-like object) and plug into a
LangChain Expression Language (LCEL) pipeline via the ``|`` operator.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import halu

from ._common import require_langchain_core, to_text


def halu_annotate(prompt: Optional[str] = None) -> "Any":
    """Build a ``Runnable`` that annotates an LLM answer with a detection verdict.

    The returned runnable takes an LLM output (a ``str`` or an object with a
    ``.content`` attribute, such as ``AIMessage``) and returns a dict::

        {
            "answer": <the original answer text>,
            "p_hallucination": <float 0-1>,
            "flag": <bool>,
            "top_regime": <str>,
        }

    This is the non-blocking "annotate" pattern: the answer always passes
    through, enriched with the detector's verdict so downstream code (or a
    human) can decide what to do.

    Parameters
    ----------
    prompt:
        Optional prompt/context string passed to ``halu.detect`` to improve
        detection. If ``None``, response-only detection is used.

    Returns
    -------
    Runnable[Any, Dict[str, Any]]
        A LangChain runnable suitable for use in an LCEL pipeline.

    Examples
    --------
    >>> from langchain_core.output_parsers import StrOutputParser
    >>> chain = llm | StrOutputParser() | halu_annotate()  # doctest: +SKIP
    """
    require_langchain_core()
    from langchain_core.runnables import RunnableLambda

    def _annotate(output: Any) -> Dict[str, Any]:
        answer = to_text(output)
        result = halu.detect(answer, prompt=prompt)
        return {
            "answer": answer,
            "p_hallucination": result.p_hallucination,
            "flag": result.flag,
            "top_regime": result.top_regime,
        }

    return RunnableLambda(_annotate)


def halu_guard(threshold: float = 0.5, prompt: Optional[str] = None) -> "Any":
    """Build a ``Runnable`` that gates an LLM answer on the detector's verdict.

    The returned runnable takes an LLM output (a ``str`` or an object with a
    ``.content`` attribute) and either:

    * returns the answer text unchanged when it is not flagged, or
    * raises :class:`halu.HaluHallucinationFlagged` when the estimated
      probability of hallucination meets or exceeds ``threshold``.

    This is the blocking "guard" pattern: a flagged answer should not reach the
    end user. Wrap downstream calls in ``try/except halu.HaluHallucinationFlagged``
    to handle the rejection (e.g. show a fallback message or retry).

    Parameters
    ----------
    threshold:
        Probability at or above which the answer is rejected. Defaults to 0.5.
    prompt:
        Optional prompt/context string passed to the detector.

    Returns
    -------
    Runnable[Any, str]
        A LangChain runnable that returns the answer text or raises.

    Examples
    --------
    >>> chain = llm | StrOutputParser() | halu_guard(threshold=0.5)  # doctest: +SKIP
    """
    require_langchain_core()
    from langchain_core.runnables import RunnableLambda

    def _guard(output: Any) -> str:
        answer = to_text(output)
        # detect_or_raise applies the threshold and raises
        # HaluHallucinationFlagged (with .detection_result) when flagged;
        # it returns the DetectResult otherwise.
        halu.detect_or_raise(answer, threshold=threshold, prompt=prompt)
        return answer

    return RunnableLambda(_guard)

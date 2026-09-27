"""Retry helper: regenerate an answer until the detector considers it clean.

This wraps ``halu.regenerate_until_clean`` so it can drive a LangChain chain or
any plain callable. It is the "retry" pattern: ask the model, check the answer,
and if it is flagged, ask again with feedback about the failing regime, up to a
bounded number of attempts.
"""

from __future__ import annotations

from typing import Any, Callable, Union

import halu

from ._common import to_text


def generate_clean(
    chain_or_callable: Union[Callable[[str], Any], Any],
    prompt: str,
    max_retries: int = 2,
    threshold: float = 0.5,
) -> Any:
    """Generate an answer and retry until it passes the detector.

    Parameters
    ----------
    chain_or_callable:
        Either a LangChain ``Runnable`` (anything exposing ``.invoke``) or a
        plain callable. It is called with a single string argument (the prompt,
        possibly augmented with regeneration feedback) and is expected to
        return an answer as a ``str`` or an ``AIMessage``-like object.
    prompt:
        The prompt/question to send to the model. Also passed to the detector
        as context to improve detection quality.
    max_retries:
        Maximum number of regeneration attempts after the first try.
        Defaults to 2.
    threshold:
        Acceptance threshold; an answer with ``p_hallucination`` below this is
        accepted. Defaults to 0.5.

    Returns
    -------
    Any
        Whatever ``halu.regenerate_until_clean`` returns — typically the
        accepted answer text (see the ``halu`` documentation for the exact
        shape and behavior when no clean answer is found within the retry
        budget).

    Notes
    -----
    The underlying ``ask`` callback normalizes chain output to text via
    :func:`langchain_halu._common.to_text`, so both string-returning and
    message-returning chains work.

    Examples
    --------
    >>> chain = prompt_template | llm  # doctest: +SKIP
    >>> answer = generate_clean(chain, "Who discovered penicillin?")  # doctest: +SKIP
    """
    if hasattr(chain_or_callable, "invoke"):
        invoke = chain_or_callable.invoke
    elif callable(chain_or_callable):
        invoke = chain_or_callable
    else:  # pragma: no cover - defensive
        raise TypeError(
            "chain_or_callable must be a LangChain Runnable (with .invoke) "
            "or a callable."
        )

    def ask(augmented_prompt: str) -> str:
        return to_text(invoke(augmented_prompt))

    return halu.regenerate_until_clean(
        ask,
        prompt=prompt,
        max_retries=max_retries,
        acceptance_threshold=threshold,
        feedback_detail="regime",
    )

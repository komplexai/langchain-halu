"""Shared helpers for the langchain-halu integration.

This module isolates the small amount of glue logic used across the
Runnable factories, the callback handler, and the retry helper: importing
``langchain_core`` with a friendly error, and normalizing the many shapes a
LangChain LLM step can hand us into a plain string.
"""

from __future__ import annotations

from typing import Any


def require_langchain_core() -> None:
    """Raise a clear, actionable error if ``langchain-core`` is not installed.

    ``langchain-core`` is a declared dependency, but users occasionally end up
    with a broken environment (e.g. after a partial install). Surfacing a
    targeted message beats a bare ``ModuleNotFoundError`` from deep in an
    import chain.
    """
    try:
        import langchain_core  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "langchain-halu requires 'langchain-core' (>=0.3). "
            "Install it with `pip install langchain-core` "
            "or `pip install langchain-halu`."
        ) from exc


def to_text(output: Any) -> str:
    """Normalize a LangChain LLM/chain output into a plain string.

    Handles the common shapes seen at the tail of a LangChain pipeline:

    * a plain ``str`` (e.g. from ``StrOutputParser``);
    * a message-like object exposing ``.content`` (e.g. ``AIMessage``),
      whose content may itself be a string or a list of content blocks;
    * anything else, which is coerced with ``str()`` as a last resort.

    Parameters
    ----------
    output:
        The value produced by an upstream LLM or chain step.

    Returns
    -------
    str
        The best-effort text representation of ``output``.
    """
    if isinstance(output, str):
        return output

    content = getattr(output, "content", None)
    if content is not None:
        return _content_to_text(content)

    return str(output)


def _content_to_text(content: Any) -> str:
    """Flatten a message ``.content`` payload into a string.

    Modern LangChain messages may carry ``content`` as a list of content
    blocks (dicts with a ``"text"`` key, or already-plain strings) instead of
    a single string. We concatenate the textual parts and ignore non-text
    blocks (e.g. images/tool calls), which are not meaningful to a
    natural-language hallucination detector.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, (list, tuple)):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)

    return str(content)

"""Light smoke tests for langchain-halu.

These verify that the package imports and that its runnables, callback handler,
and retry helper construct correctly without any network access. The single
live end-to-end test is skipped by default and only runs when
``HALU_LIVE_TEST=1`` is set in the environment (and a real ``HALU_API_KEY`` is
available).

Run with: ``pytest``
"""

import logging
import os

import pytest

# The package depends on halu and langchain-core; if either is missing from the
# test environment, skip the whole module rather than error out.
pytest.importorskip("halu")
pytest.importorskip("langchain_core")

import langchain_halu  # noqa: E402
from langchain_halu._common import to_text  # noqa: E402


class _FakeMessage:
    """Minimal AIMessage-like object exposing ``.content``."""

    def __init__(self, content):
        self.content = content


def test_public_exports():
    for name in (
        "halu_annotate",
        "halu_guard",
        "HaluCallbackHandler",
        "generate_clean",
        "DetectResult",
        "HaluError",
        "HaluHallucinationFlagged",
    ):
        assert hasattr(langchain_halu, name), name
    assert langchain_halu.__version__ == "0.1.0"


def test_to_text_str():
    assert to_text("hello") == "hello"


def test_to_text_message_content_str():
    assert to_text(_FakeMessage("hi there")) == "hi there"


def test_to_text_message_content_blocks():
    msg = _FakeMessage([{"type": "text", "text": "a"}, "b", {"image": "x"}])
    assert to_text(msg) == "ab"


def test_to_text_fallback():
    assert to_text(123) == "123"


def test_halu_annotate_constructs():
    runnable = langchain_halu.halu_annotate()
    assert hasattr(runnable, "invoke")


def test_halu_guard_constructs():
    runnable = langchain_halu.halu_guard(threshold=0.7)
    assert hasattr(runnable, "invoke")


def test_callback_handler_constructs():
    handler = langchain_halu.HaluCallbackHandler(threshold=0.6)
    assert handler.last_result is None
    assert handler.flagged_results == []
    assert handler.threshold == 0.6


def test_callback_handler_is_base_callback_handler():
    from langchain_core.callbacks import BaseCallbackHandler

    assert isinstance(langchain_halu.HaluCallbackHandler(), BaseCallbackHandler)


def test_callback_iter_texts_empty_response():
    class _EmptyResult:
        generations = []

    handler = langchain_halu.HaluCallbackHandler()
    # Should not raise and should not call the network on empty generations.
    handler.on_llm_end(_EmptyResult())
    assert handler.last_result is None


def test_generate_clean_type_error_on_bad_input():
    with pytest.raises(TypeError):
        langchain_halu.generate_clean(object(), "prompt")


@pytest.mark.skipif(
    os.environ.get("HALU_LIVE_TEST") != "1",
    reason="live network test; set HALU_LIVE_TEST=1 and HALU_API_KEY to run",
)
def test_annotate_live():
    runnable = langchain_halu.halu_annotate()
    out = runnable.invoke("The Eiffel Tower is located in Berlin, Germany.")
    assert set(out) == {"answer", "p_hallucination", "flag", "top_regime"}
    assert 0.0 <= out["p_hallucination"] <= 1.0
    logging.getLogger(__name__).info("live annotate result: %s", out)

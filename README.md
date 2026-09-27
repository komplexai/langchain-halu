# langchain-halu

A thin [LangChain](https://python.langchain.com/) integration for
[**Komplex AI**](https://detector.komplexai.io) — a hallucination detector for
LLM output. It adapts the [`halu`](https://pypi.org/project/halu/) Python SDK to
LangChain so you can annotate, guard, monitor, or retry generations from inside
an LCEL pipeline.

## Install

```bash
pip install langchain-halu
```

This pulls in `halu` and `langchain-core` (not the full `langchain`).

Set your API key in the environment:

```bash
export HALU_API_KEY="sk-..."
```

Get a key at <https://detector.komplexai.io>.

## Three patterns

### 1. Annotate (non-blocking)

Enrich an answer with a verdict; the answer always passes through.

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI  # any LangChain chat model works
from langchain_halu import halu_annotate

llm = ChatOpenAI(model="gpt-4o-mini")
chain = llm | StrOutputParser() | halu_annotate()

result = chain.invoke("Who painted the Mona Lisa?")
# {
#   "answer": "Leonardo da Vinci painted the Mona Lisa.",
#   "p_hallucination": 0.02,
#   "flag": False,
#   "top_regime": "NORMAL",
# }
```

### 2. Guard (blocking)

Reject flagged answers so they never reach the user.

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_halu import halu_guard, HaluHallucinationFlagged

chain = llm | StrOutputParser() | halu_guard(threshold=0.5)

try:
    answer = chain.invoke("Summarize the plot of a book that does not exist.")
except HaluHallucinationFlagged as exc:
    result = exc.detection_result
    print(f"Blocked (p={result.p_hallucination:.2f}, regime={result.top_regime})")
    answer = "Sorry, I couldn't produce a reliable answer."
```

### 3. Monitor (passive callback)

Observe every completion and log flagged answers, without changing behavior.

```python
from langchain_halu import HaluCallbackHandler

monitor = HaluCallbackHandler(threshold=0.5)
llm.invoke("What year did the Great Fire of London happen?", config={"callbacks": [monitor]})

if monitor.last_result and monitor.last_result.flag:
    print("Flagged:", monitor.last_result.top_regime)
# monitor.flagged_results holds every flagged DetectResult seen so far.
```

### Bonus: Retry until clean

Regenerate with regime feedback until the answer passes the detector.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_halu import generate_clean

prompt = ChatPromptTemplate.from_template("{input}")
chain = prompt | llm | StrOutputParser()

answer = generate_clean(
    chain,
    prompt="Who discovered penicillin, and in what year?",
    max_retries=2,
    threshold=0.5,
)
```

`generate_clean` accepts any LangChain `Runnable` (anything with `.invoke`) or a
plain callable that takes a prompt string and returns text or an
`AIMessage`-like object.

## API

| Name | Kind | Purpose |
|------|------|---------|
| `halu_annotate(prompt=None)` | `Runnable` factory | Return `{"answer", "p_hallucination", "flag", "top_regime"}` |
| `halu_guard(threshold=0.5, prompt=None)` | `Runnable` factory | Pass the answer through or raise `HaluHallucinationFlagged` |
| `HaluCallbackHandler(threshold=0.5, prompt=None, log_level=WARNING)` | Callback | Detect on `on_llm_end`, log + store flagged results |
| `generate_clean(chain_or_callable, prompt, max_retries=2, threshold=0.5)` | Function | Retry generation until clean |

`DetectResult`, `HaluError`, and `HaluHallucinationFlagged` are re-exported from
`halu` for convenience.

## Configuration

Detection reads `HALU_API_KEY` from the environment by default and talks to
`https://api.komplexai.io`. All the standard `halu` options (`api_key`,
`base_url`, `timeout`) are honored through `halu`'s own configuration; see the
[`halu` docs](https://pypi.org/project/halu/).

## Scope & limits

The detector is optimized for a specific slice, and results outside it are not
reliable:

- **English, natural-language** responses only.
- Responses up to **2048 characters**.
- The **first call after an idle period can cold-start for ~10–30 s**; subsequent
  calls are fast. Build a warm-up call into latency-sensitive apps.
- Regimes reported: `NORMAL`, `FABRICATED`, `NEAR_FALSE`, `CF_AUTH` (fake or
  misattributed citation), `FALSE_REFUSAL`, `Other`.

Full guidance and best practices: <https://detector.komplexai.io/guide>.

## License

Apache-2.0. See [LICENSE](LICENSE).

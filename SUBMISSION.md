# DRAFT — LangChain integration listing for langchain-halu

> **Status: DRAFT for internal review. Do NOT submit yet.** Nothing here has been
> filed with LangChain. Review the wording, verify every `halu` API detail
> against the shipped SDK, and confirm the public URLs before opening any PR.

This file contains (A) a proposed provider/integration docs page for the
LangChain documentation site, and (B) a checklist and notes for actually getting
listed.

---

## A. Proposed docs page

Suggested path in the `langchain-ai/langchain` docs tree:
`docs/docs/integrations/providers/komplexai.mdx` (provider page), optionally with
a companion how-to under `docs/docs/integrations/tools/` or
`docs/docs/integrations/callbacks/`.

````markdown
# Komplex AI

> [Komplex AI](https://detector.komplexai.io) provides a hallucination detector
> for LLM output. The `langchain-halu` package is a thin LangChain integration
> around the [`halu`](https://pypi.org/project/halu/) SDK.

## Installation and setup

```bash
pip install langchain-halu
export HALU_API_KEY="..."   # get a key at https://detector.komplexai.io
```

`langchain-halu` depends only on `halu` and `langchain-core`.

## Runnables

Drop these into any LCEL pipeline with `|`.

```python
from langchain_halu import halu_annotate, halu_guard

# Annotate: never blocks; enriches the answer with a verdict.
annotated = (llm | StrOutputParser() | halu_annotate()).invoke("...")
# {"answer": ..., "p_hallucination": ..., "flag": ..., "top_regime": ...}

# Guard: raises halu.HaluHallucinationFlagged on a flagged answer.
guarded = (llm | StrOutputParser() | halu_guard(threshold=0.5)).invoke("...")
```

## Callbacks

```python
from langchain_halu import HaluCallbackHandler

monitor = HaluCallbackHandler()
llm.invoke("...", config={"callbacks": [monitor]})
monitor.last_result       # most recent halu.DetectResult
monitor.flagged_results   # all flagged results this run
```

## Retry pattern

```python
from langchain_halu import generate_clean

answer = generate_clean(chain, prompt="...", max_retries=2, threshold=0.5)
```

## Scope and limits

English natural-language responses, up to 2048 characters. The first call after
idle may cold-start (~10–30 s). See <https://detector.komplexai.io/guide>.
````

---

## B. Listing checklist & notes

### Pre-submission (do these first)

- [ ] Publish `langchain-halu` to PyPI and confirm `pip install langchain-halu`
      works in a clean venv (the docs PR links to the PyPI package).
- [ ] Confirm the public URLs resolve: `https://detector.komplexai.io`,
      `https://detector.komplexai.io/guide`, `https://github.com/komplexai`.
- [ ] Re-verify every `halu` symbol used in the examples matches the shipped SDK
      (`detect`, `detect_or_raise`, `regenerate_until_clean`, `DetectResult`
      fields, `HaluHallucinationFlagged.detection_result`).
- [ ] Decide on the canonical source repo URL (a dedicated
      `github.com/komplexai/langchain-halu` repo is expected by reviewers) and
      put it in `pyproject.toml` `[project.urls] Source` before publishing.
- [ ] Run the smoke tests (`pytest`) and, once, the live test
      (`HALU_LIVE_TEST=1 pytest -k live`) against production.

### Option 1 — Docs-only listing (lowest friction, recommended first)

LangChain maintains an integrations catalog in the main docs repo. A provider
page plus a short how-to is usually enough to be listed.

- [ ] Fork `github.com/langchain-ai/langchain`.
- [ ] Add `docs/docs/integrations/providers/komplexai.mdx` (page A above).
- [ ] Optionally add a callbacks/tools how-to page and link it from the provider
      page.
- [ ] Follow the repo's docs contribution guide (`docs/README.md`); build docs
      locally to confirm the page renders and links pass the link checker.
- [ ] Open a PR titled e.g. `docs: add Komplex AI (langchain-halu) integration`,
      describing the package and linking to PyPI + source repo.
- [ ] Note in the PR that it is a third-party/community integration maintained by
      Komplex AI.

### Option 2 — `langchain-community` contribution (optional, later)

Only if maintainers prefer the code live in-tree rather than as an external
package. Higher review bar (tests, lint, typing, no heavy deps).

- [ ] Discuss with maintainers first (issue or discussion) — many integrations
      are now preferred as standalone `langchain-*` packages rather than added to
      `langchain-community`.
- [ ] If accepted, port the runnables/callback under
      `libs/community/langchain_community/…` following their module conventions,
      keeping `halu` an optional import.

### Option 3 — Partner package (optional, much later)

Formal `langchain-<partner>` packages are co-maintained with LangChain and imply
an ongoing support commitment. Pursue only if the integration gets real traction
and LangChain invites it. Requires a standalone repo, CI, and a maintenance SLA.

### Notes

- Keep the dependency surface minimal: `langchain-core` only, never the full
  `langchain`. Reviewers check for this.
- The three-pattern framing (annotate / guard / monitor) mirrors how LangChain
  documents guardrail-style integrations; keep it.
- Be honest about scope/limits in the docs page — reviewers dislike over-broad
  accuracy claims, and the English/2048-char/cold-start constraints are real.

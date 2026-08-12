# Case Study: httpx Codebase & Docs Q&A Assistant

## Business Problem

**Client scenario:** a mid-size engineering org maintaining an internal
Python library (stand-in here: httpx, used as a realistic open-source
proxy for "our own library"). New engineers and cross-team consumers
constantly ask the same questions — "how do I stream a response," "what
does `Client.request` actually do," "how is auth implemented" — and the
answers live scattered across docstrings, source code, and documentation
that's often stale relative to the code. Senior engineers lose real hours
per week answering these in Slack instead of doing their own work, and
answers vary by who answered.

**Who has this problem:** any engineering org with an internal library,
SDK, or platform that's non-trivial enough that "read the source" isn't a
realistic onboarding answer, but small enough that a dedicated
documentation team isn't justified.

## Solution

A RAG assistant that indexes both a library's **documentation and its
actual source code** together, and answers questions with citations back
to the exact function or doc section it used — so an engineer can verify
the answer instead of trusting it blindly.

## Architecture

```
docs/*.md + code/*.py (real corpus: httpx's own source + docs)
        |
        v
Structure-aware chunking (headings/paragraphs for docs,
AST function/class boundaries for code)          -> 470 chunks
        |
        v
sentence-transformers embeddings -> Chroma vector store
        |
        v
Retrieval (top-k) -> prompt construction -> Groq (Llama 3.3 70B)
        |
        v
FastAPI: auth, rate limiting, structured logs, Prometheus metrics
        |
        v
Web UI: landing page, playground (live SSE progress), eval dashboard
```

## Key Architectural Decisions (ADR Log)

### ADR-001: Structure-aware chunking over fixed-size chunking
**Decision:** split markdown on headings then paragraphs, with a
hard-length fallback; split Python source on AST function/class
boundaries.
**Why:** fixed-size (character-count) chunking slices a sentence or a
function in half at an arbitrary point, with no relationship to where one
idea ends and another begins. A chunk boundary needs to line up with a
complete, meaningful unit for retrieval to return something coherent.
**Trade-off:** more code than `text.split()`, but retrieval quality
depends entirely on this — bad chunking can't be fixed downstream by a
better prompt.

### ADR-002: `all-MiniLM-L6-v2` local embeddings over a larger hosted model
**Decision:** local, free, 384-dim sentence-transformers model.
**Why:** standard lightweight baseline for small-corpus retrieval (470
chunks here); no per-call cost, no API key, fast enough on CPU.
**Trade-off:** would revisit for a much larger or more nuanced corpus
where retrieval quality testing showed it missing relevant chunks —
that's a measurable trigger to reconsider, not a guess.

### ADR-003: Groq (hosted) over Ollama (local) for generation
**Decision:** moved off a local Ollama model onto Groq's hosted API
partway through the build.
**Context:** local inference is free and private but CPU-bound — minutes
per answer on a laptop with no GPU. That's a real usability blocker for a
tool meant to be used repeatedly during a workday.
**Why Groq specifically:** LPU-based inference, unusually fast even
among hosted providers — verified directly (~1s response vs. 30-45s
local).
**Trade-off:** data leaves the machine, subject to a third party's
pricing/rate limits. For an internal-tool prototype answering questions
about open-source library internals, that's an acceptable trade for
usability; would reconsider for a corpus containing actual proprietary
code.

### ADR-004: AST-based code chunking, not text-splitting code like prose
**Decision:** parse Python source with the stdlib `ast` module; one
chunk per top-level function or per class method, using
`ast.get_source_segment` for the exact original text.
**Why:** splitting code by character count or blank lines produces
syntactically broken fragments — a function cut off mid-`if`-statement is
useless context for an LLM. A chunk boundary must be a complete,
syntactically valid unit.
**Trade-off:** doesn't chunk nested/inner functions separately, and
module-level constants outside any function/class aren't indexed —
acceptable gap for a first version, documented rather than hidden.

### ADR-005: Real codebase + docs corpus over generic framework docs
**Decision:** pivoted the corpus from FastAPI's public documentation to
httpx's actual source code and documentation together.
**Context:** the original version answered questions from prose docs
only — a legitimate RAG demo, but a shallow one that doesn't require
solving the harder problem (retrieving over two structurally different
content types). It also read as generic rather than business-relevant.
**Trade-off:** cost a full re-ingestion (chunker rewrite, re-embedding,
new eval questions) partway through the build — worth it for what the
project now actually demonstrates.

### ADR-006: Live progress via Server-Sent Events, not a blank wait
**Decision:** long-running actions (eval runs, ~10 sequential LLM calls)
stream per-step progress to the UI via SSE instead of blocking silently.
**Why:** a user watching a spinner for over a minute with zero feedback
can't tell "working" from "broken." SSE required no new infrastructure
(no websockets, no polling) — FastAPI's `StreamingResponse` over the
existing HTTP connection was sufficient.

## AI Engineering Notes

- **Grounding via prompt instruction + citations, not fine-tuning.** The
  prompt explicitly tells the model to answer only from retrieved context
  and to admit when it doesn't know — verified directly: a query about
  `Client.request` internals correctly returned "I don't know" when the
  exact right chunk wasn't retrieved, rather than hallucinating a
  plausible-sounding wrong answer.
- **Retrieval quality matters more than prompt engineering.** If the
  wrong chunks come back, no amount of prompt tuning fixes the answer —
  this is why chunking strategy got as much design attention as the
  generation step.
- **Where AI was NOT the answer:** the eval harness (Project 2) uses a
  hand-written LLM judge for quality scoring rather than a framework,
  precisely because the judging logic needed to be fully understood, not
  imported.

## Engineering Challenges Solved

This project inherited a partially-built scaffold from another AI coding
tool (OpenCode) with several real, non-obvious bugs found and fixed by
actually running the code, not just reading it:

- **Starlette `TemplateResponse` signature mismatch** — the installed
  Starlette version required `(request, name, context)`; the scaffold
  used the older deprecated `(name, context)` form, which silently passed
  the whole context dict as the template *name*, crashing with
  `unhashable type: dict` on every page.
- **Jinja2 macro syntax error** — `**attrs` in a macro's parameter list
  isn't valid Jinja2 (Python-style kwargs capture doesn't exist there;
  Jinja2 provides an implicit `kwargs` variable instead) — broke every
  page that imported the shared component library, only surfacing at
  render time.
- **`chromadb.errors.InvalidCollectionException` no longer exists** in
  the installed Chroma version (renamed to `NotFoundError`) — another
  dependency-version drift bug, same pattern as the Starlette one.
- **Duplicate route registration** — `/api/query` and `/health` were each
  registered twice (once generically in a shared app factory, once with
  real logic in the app) — Starlette silently uses whichever was
  registered first, so the real logic was being shadowed.
- **Missing `load_dotenv()`** in a standalone process (later, in the MCP
  server for Project 4) caused a required env var to be silently missing
  only in that process, not others.
- **Groq 429 rate limits crashed requests** with an unhandled 500 instead
  of retrying — fixed with exponential backoff, a real production
  resilience gap that would affect actual usage, not just a demo path.

Each was found by actually running the app end-to-end and reading the
real traceback, not by inspecting code statically — a running app text at
"looks correct" is a hypothesis, not a verification.

## Results

- 470 indexed chunks (125 docs, 345 code) across 19 source files
- 18/18 tests passing
- Verified retrieval + generation correctness on multiple real queries,
  including a deliberate "does it admit uncertainty" check
- Sub-second query latency on Groq vs. 30-45s on local Ollama (measured,
  not estimated)

## Known Limitations

- Retrieval isn't perfect — demonstrated directly with the `Client.request`
  example; no reranking or hybrid (keyword + vector) search yet
- No conversation memory — every question answered independently
- Root-level walkthrough scripts (`01_chunk.py` etc.) reflect the
  original FastAPI-docs version and were not migrated to the new corpus —
  kept as the "how RAG actually works" reference, not a second
  maintained implementation
- README's Quick Start section still describes the original Ollama +
  docker-compose setup from before the Groq migration — flagged here as
  a known documentation gap, not yet corrected in the README itself

## Future Roadmap (Phase 2)

- Hybrid search (BM25 + vector) and cross-encoder reranking
- Conversation memory for multi-turn follow-up questions
- RAGAS or similar integration alongside the hand-rolled Project 2 judge,
  for comparison
- Multi-tenancy — one deployment indexing multiple internal repos

---

## Interview Readiness

**30-second explanation:** "It's a RAG assistant that answers questions
about a Python library using its actual source code and docs together,
not just prose documentation — every answer cites exactly which function
or doc section it came from."

**2-minute explanation:** covers the business problem (engineers
re-answering the same onboarding questions), the two-content-type
retrieval challenge (docs vs. code needing different chunking), and the
grounding/citation mechanism that makes answers verifiable.

**5-minute technical explanation:** walk the architecture diagram —
ingestion (chunking strategies per content type) → embedding → Chroma →
retrieval → Groq generation with a grounding-instruction prompt → FastAPI
with auth/observability → SSE-streamed UI. Emphasize why chunking
strategy dominates retrieval quality.

**Deep technical explanation (per ADR):** be ready to defend each ADR
above on its own — especially ADR-003 (Groq vs local) and ADR-004
(AST chunking) as they involve real measured trade-offs, not defaults.

**AI explanation:** grounding is enforced by prompt instruction plus a
verifiable citation trail, not model fine-tuning; the system was directly
observed refusing to answer rather than hallucinating when retrieval
under-delivered — that's the actual reliability property being
demonstrated, not just "it uses an LLM."

**Engineering trade-offs at 10x scale:** local sentence-transformers
embedding would need to move to a batched/GPU-backed service; Chroma's
single-process embedded mode would need to become a proper Chroma server
or a managed vector DB; rate limiting would need per-tenant quotas rather
than a single API key.

**Failure scenario:** Groq API down or rate-limited → the retry-with-
backoff logic handles transient 429s, but a sustained outage has no
fallback model configured yet — a real gap, worth naming rather than
hiding, and a natural Phase 2 item (model routing/fallback).

**Business explanation:** a company would pay for this because the
alternative — engineers manually answering the same questions repeatedly
in Slack — has a real, calculable cost in senior engineer time, and a
tool that gives verifiable (cited) answers reduces the trust barrier that
makes AI chatbots risky for technical questions specifically.

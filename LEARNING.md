# httpx Codebase & Docs Q&A Assistant — Learning Notes

Notes on how this project works and why it's built this way. Written to be
able to defend every design choice in an interview, not just describe what
the code does.

## Project history (why there are two implementations in this repo)

This project was built in two stages, and both are still in the repo on
purpose:

1. **`01_chunk.py` / `02_embed.py` / `rag.py` / `03_generate.py` / `app.py`**
   at the repo root — a minimal, single-file-per-step walkthrough built
   first to understand the core RAG mechanics (chunk → embed → retrieve →
   generate) with nothing else in the way. It indexed FastAPI's own
   documentation pages and used a local Ollama model.
2. **`src/`** — a production-structured FastAPI app (config, auth, rate
   limiting, structured logging, Prometheus metrics, an eval harness, a
   web UI) that supersedes the root scripts. It now indexes **httpx's
   actual source code and documentation together**, and generates via a
   hosted API (Groq) instead of a local model.

`src/` is the canonical, currently-running implementation. The root
scripts are kept as the "how RAG actually works, stripped to the studs"
reference — genuinely useful for explaining the core idea without the
production scaffolding around it, but they were not migrated to the httpx
corpus or Groq; they still reflect the original FastAPI-docs version.

## The problem

An LLM's knowledge is frozen at training time and general-purpose. Ask it
a specific question about a library's actual behavior and it will often
answer from vague memory of the library's public docs — which misses
implementation details, or is just wrong about a library's less-documented
corners. Retrieval-Augmented Generation (RAG) fixes this by finding the
relevant source text first (documentation *and*, here, actual source code),
then having the LLM answer using that text as grounding, instead of
answering from memory.

## Why a codebase, not just docs

The original version of this project indexed only prose documentation.
That's a reasonable RAG demo, but it's a shallow one — the interesting,
defensible engineering problem is retrieving over **two different kinds of
text with different structure**: markdown prose (chunk on headings/
paragraphs) and Python source code (chunk on function/class boundaries, or
you get syntactically broken fragments). This version indexes both the
[httpx](https://github.com/encode/httpx) library's documentation *and* a
core subset of its actual source (`_client.py`, `_api.py`, `_models.py`,
`_auth.py`, `_exceptions.py`, `_config.py`, `_urls.py`, `_content.py`), so
the assistant can answer both "how do I use this" (docs) and "what does
this actually do" (code) questions, citing whichever it actually used.

## Pipeline overview

```
httpx docs (data/raw/docs/*.md)   httpx source (data/raw/code/*.py)
            |                                |
            v                                v
   markdown chunker                   AST-based code chunker
   (headings -> paragraphs)           (function/class boundaries)
            \                                /
             \                              /
              v                            v
        [1] chunking (src/ingestion/chunker.py)  -> data/chunks.jsonl
                              |
                              v
        [2] embedding (src/ingestion/embedder.py) -> data/chroma/
                              |
                              v
        [3] retrieval + generation (src/rag/pipeline.py)
                              |
                              v
        [4] FastAPI app (src/main.py, src/api/routes.py)
```

Steps 1 and 2 run once, offline, to build the knowledge base (470 chunks:
125 from docs, 345 from code). Steps 3 and 4 run per question, at request
time.

## Step 1 — Chunking (`src/ingestion/chunker.py`)

**Docs side (`chunk_document`)** — the same structure-aware approach as the
original version: split on markdown headings first, then on paragraph
breaks if a section is still too long, with a hard word-boundary fallback
for a paragraph that has no blank lines to split on at all. Tiny leftover
fragments get merged into the previous chunk so a lone heading never ends
up as its own orphan chunk.

**Code side (`chunk_code_file`) — the actual new piece:**

**Why not reuse the same character/paragraph splitter for code:**
splitting Python source by character count or blank lines will cut a
function in half at an arbitrary point — you'd get a chunk ending
mid-`if`-statement with no closing brace context, which is useless for an
LLM trying to explain what the function does. A chunk boundary needs to
line up with a *complete, syntactically valid unit*.

**What it does instead:** parses each file with Python's built-in `ast`
module (no dependency needed — it's stdlib) and walks the module's
top-level statements:
- a top-level function becomes one chunk
- a class with methods becomes **one chunk per method**, headed
  `module > ClassName.method_name` (e.g. `_client > Client.request`)
- a class with no methods (a plain data class) becomes one chunk for the
  whole class

Each chunk's exact source text is pulled with `ast.get_source_segment`,
which hands back the precise original source slice for that AST node —
not a re-serialized/reformatted version, the real code as written,
docstring and all. A handful of very long methods (e.g.
`Client.request`) still route through the same paragraph/length fallback
used for oversized doc sections, so nothing blows the embedding context.

**Result:** 470 chunks total — 125 from 11 doc pages, 345 from 8 source
files, avg ~490 chars.

## Step 2 — Embeddings + vector store (`src/ingestion/embedder.py`)

Unchanged in approach from the original version: `all-MiniLM-L6-v2` via
`sentence-transformers` (local, free, 384-dim, fast on CPU — the standard
lightweight baseline for small-scale retrieval work), stored in a Chroma
collection (`httpx_kb`) that's dropped and recreated on every run so
re-running never leaves stale vectors behind.

One thing that matters more here than in the docs-only version: the same
embedding model has to make sense of *both* prose and code in the same
vector space, since a query like "how do I stream a response" needs to be
able to match both a doc paragraph about streaming *and* the
`Client.stream` method's source. `all-MiniLM-L6-v2` isn't code-specialized,
but it holds up well enough in practice — verified directly by the sanity
check below, not just assumed.

**Sanity check, now genuinely informative:** the embedder runs five test
queries after indexing and logs the top 3 chunks for each. For "how do I
stream a response," the top hits were the actual `_api.py::stream`
function and both `Client.stream`/`AsyncClient.stream` methods — code
chunks, correctly outranking the docs for a question about runtime
behavior. That's the concrete evidence the hybrid docs+code retrieval is
doing its job, not just a hope.

## Step 3 — Retrieval + generation (`src/rag/pipeline.py`)

`RagPipeline` loads the embedding model and opens the Chroma connection
once, in `initialize()` (called from the FastAPI `lifespan` handler at
startup) — not per request.

**Retrieval:** embed the question with the same embedding model used on
the chunks, query Chroma for the top-k (default 4) closest chunks by
cosine distance, converted to a similarity score.

**Prompt construction:** retrieved chunks (prose or code, indistinguishable
in format — each is just `[heading]\ntext`) get joined into a fixed prompt
that tells the model to answer using *only* the given context, covering
both documentation and source excerpts, and to say it doesn't know rather
than guess. That instruction is what keeps the model grounded — without
it, an LLM will often answer from its own training memory of the library
instead of the actual excerpts it was given.

**Generation:** a plain HTTP POST to Groq's hosted API
(`api.groq.com/openai/v1/chat/completions`), the same OpenAI-compatible
chat-completions shape most hosted LLM providers use — a `messages` list
in, `choices[0].message.content` out. Model: `llama-3.3-70b-versatile`.

**Local vs. hosted inference, a real tradeoff worth being able to discuss:**
this project ran on a local Ollama model first (`qwen3:8b`, then
`phi3:mini` for speed). Local inference is free and private but CPU-bound —
minutes per answer on a laptop with no GPU. A hosted inference API like
Groq returns in about a second, because the model runs on dedicated
inference hardware (Groq specifically uses custom LPU chips, unusually
fast even among hosted providers), at the cost of sending data to a third
party and being subject to their rate limits/pricing.

## Step 4 — FastAPI app (`src/main.py`, `src/api/routes.py`)

**Endpoints:**
- `GET /health` — checks the pipeline, Chroma, and Groq are all actually
  reachable, not just that the process is alive
- `POST /api/query` — JSON in/out, requires an API key (`X-API-Key` header)
- `POST /api/query/ui` — the same pipeline call, but returns an HTML
  fragment for the HTMX-driven playground UI, and does not require auth
  (it's server-rendered, same-origin only)
- `POST /api/eval/run` — runs the evaluation dataset (see below) against
  the live pipeline
- `/playground`, `/eval` — the web UI

**Security/observability layer**, genuinely new territory beyond the
original walkthrough version:
- API-key auth (`src/shared/auth.py`) — off automatically in dev if
  `API_KEY` isn't set, enforced once it is
- Rate limiting per route via `slowapi`
- Structured JSON/console logging via `structlog`, with a request ID
  threaded through every log line for a given request
- Prometheus metrics (`/metrics`) tracking request counts, latencies, and
  per-operation (retrieve/generate/embed) timing

**Evaluation harness** (`src/eval/runner.py`): runs a fixed set of httpx
questions through the live pipeline and scores each answer on four
heuristic metrics — context precision/recall (keyword overlap between
retrieved chunks and expected topics), faithfulness (does the answer's
vocabulary actually appear in the retrieved context, or did the model
wander off), and answer relevancy (does the answer address the question's
own terms). These are lexical heuristics, not the LLM-as-judge scoring
Project 2 will add — a legitimate first pass, with a known, honestly-stated
ceiling.

## How to run it

```bash
# set GROQ_API_KEY (and optionally API_KEY) in .env — see .env.example
# get a free Groq key at console.groq.com/keys

# one-time ingestion (already done in this repo, listed for reference)
python -m src.ingestion.chunker    # data/raw/{docs,code} -> data/chunks.jsonl
python -m src.ingestion.embedder   # chunks -> data/chroma/ (embeds + sanity-checks retrieval)

# start the API
uvicorn src.main:app --port 8000

# then either:
curl -X POST http://localhost:8000/api/query -H "Content-Type: application/json" \
  -H "X-API-Key: <your API_KEY from .env>" \
  -d '{"question": "What is the difference between httpx.Client and httpx.AsyncClient?"}'
# or open http://localhost:8000/playground for the UI
# or http://localhost:8000/docs for the interactive API docs
```

## Things worth being able to explain in an interview

- Why code needs a different chunking strategy than prose, and why AST
  parsing (chunk boundary = complete function/method) beats character or
  line-count splitting for source code specifically.
- Why `ast.get_source_segment` returns the real original source, not a
  reformatted reconstruction — and why that distinction matters (you want
  the LLM reading the code as actually written).
- Why the embedding model used for the query has to be the same one used
  for the chunks, and why one general-purpose embedding model still has to
  work across two different content types (prose and code) here.
- Why retrieval quality matters more than prompt engineering in a RAG
  system — garbage in, garbage out. Concrete example from this project:
  asking about `Client.request` internals returned `build_request` and the
  top-level `request()` function instead of the actual `Client.request`
  method (which exists in the corpus, just didn't rank in the top 4) — the
  model correctly said "I don't know" rather than hallucinate, which is the
  grounding constraint working exactly as designed even when retrieval
  itself is imperfect.
- The local-vs-hosted-inference tradeoff (cost/privacy vs. latency) that
  led to moving off Ollama onto Groq's hosted API.
- What the auth/rate-limiting/observability layer actually does and why a
  "production-ready" claim needs more than a working happy path — API key
  enforcement, structured logs with request IDs, and Prometheus metrics
  are the concrete things that back that claim up.
- The heuristic (non-LLM-judge) evaluation metrics used here, and why
  that's an honest first pass rather than the final word on quality —
  which is exactly the gap Project 2 exists to close.

## Known limitations (honest, not hidden)

- Retrieval isn't perfect — demonstrated directly above with the
  `Client.request` example. No reranking or hybrid (keyword + vector)
  search yet.
- Evaluation metrics are lexical heuristics (keyword/word-overlap based),
  not LLM-as-judge scoring — a legitimate first pass, not the final word.
- No conversation memory — every question is answered independently, no
  multi-turn context.
- The code chunker only handles top-level functions and class methods; it
  doesn't chunk nested/inner functions separately, and module-level
  constants or imports outside any function/class aren't indexed at all.
- The root-level scripts (`01_chunk.py` etc.) were not migrated to the
  httpx corpus or Groq — they're kept as the original walkthrough, not as
  a second maintained implementation.

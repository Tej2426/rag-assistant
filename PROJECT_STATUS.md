# RAG Document Q&A Assistant — Project Status

## Context
This is Project 1 of a 5-project AI/ML portfolio, built for an Associate AI/ML Engineer job search (positioning: applied GenAI / AI Engineer track, not classical ML). All work is being done hands-on as a learning process — explain concepts clearly, don't just generate code silently, and check understanding before moving to the next step.

## Full 5-project roadmap
1. **RAG Document Q&A Assistant** (this project) — Python, LangChain, Chroma (vector DB), Ollama/DeepSeek, FastAPI
2. **LLM Evaluation & Observability Harness** — built on top of Project 1, using Ragas/DeepEval
3. **Multi-Agent System** — CrewAI or LangGraph, framed around a real business use case, uses Project 1 as a callable tool
4. **Agentic Tool-Use via MCP** — Model Context Protocol, extends the Project 3 agent
5. **End-to-End Classical ML Model with Deployment** — upgrades an existing Customer Churn analysis (currently just a Power BI dashboard) into a real trained + evaluated + Dockerized model

Deliberately excluded: a Microsoft Fabric/Lakehouse/Power BI + agent project — too similar to a private client project that isn't being showcased publicly.

Per project, once built: produce resume bullet points + interview Q&A as a deliverable.

## Project 1 status: chunking done, embeddings next

### Done
- Downloaded a real corpus: 8 pages of FastAPI's official documentation (`data/raw/*.md`)
- Built a structure-aware chunker (`01_chunk.py`): splits on markdown headings first, falls back to paragraph-level splitting for long sections, merges tiny orphan fragments
- Ran it: **131 chunks** from 8 source pages, avg ~631 chars/chunk
- Known limitation surfaced and intentionally left undocumented-but-flagged: a few chunks (up to ~2,500 chars) came out oversized because the chunker splits on blank lines (`\n\n`), and some FastAPI doc pages have dense HTML/Jinja blocks with no blank lines to split on. Worth a hard-length fallback eventually, but treated as a known edge case rather than silently patched.

### Files in this project
- `01_chunk.py` — the chunking script
- `data/raw/*.md` — 8 FastAPI doc pages (source corpus)
- `data/chunks.jsonl` — 131 chunks, one JSON object per line: `{id, source, heading, text, n_chars}`

### Next step: embeddings
1. Pick an embedding model (a local/free sentence-transformers model is a reasonable start — no API cost)
2. Generate a vector embedding for each of the 131 chunks
3. Store the embeddings in Chroma (local vector DB, zero infra to start)
4. Sanity-check retrieval with a couple of manual test queries before moving to generation

## Working style notes for whoever picks this up
- The user is early-career, positioning for Associate AI/ML Engineer roles, and has explicitly said a lot of past work was "vibe-coded" (AI-generated, run without full understanding). The goal of this project is to close that gap — don't just hand over working code, make sure each step is explained and understood before moving on.
- The user wants to be able to defend every line of this project in an interview. Prioritize clarity and defensibility over speed or feature-completeness.
- No fabricated claims: skills/tools only get added to the resume once they're genuinely understood, not just used.

"""
Step 3: Generation (the "G" in RAG) - CLI entry point.

WHY WE DON'T JUST ASK THE LLM DIRECTLY:
An LLM's knowledge is frozen at training time and general-purpose - it
doesn't know the specific contents of our FastAPI doc corpus, and it will
happily hallucinate a plausible-sounding but wrong answer if asked cold.
RAG fixes this by retrieving the most relevant chunks first (Step 2) and
stuffing them into the prompt as grounding context, then asking the LLM to
answer USING that context. This is why retrieval quality matters more than
prompt cleverness - if the wrong chunks come back, no prompt fixes it.

WHY A LOCAL MODEL VIA OLLAMA:
Zero API cost, no data leaving the machine, and it's the same shape of
integration (HTTP call, JSON in/out) as a hosted API would be - the concepts
transfer directly to OpenAI/Anthropic-style APIs later.

The actual pipeline (retrieve -> prompt -> generate) lives in rag.py so the
FastAPI app (app.py) can reuse it without duplicating this logic.
"""

import sys

from rag import RagPipeline


def main():
    pipeline = RagPipeline()

    question = " ".join(sys.argv[1:]) or "How do I add validation to a query parameter?"
    print(f"Question: {question}\n")

    reply, chunks = pipeline.answer(question)

    print("Retrieved chunks:")
    for doc, meta in chunks:
        print(f"  - [{meta['source']}] {meta['heading']}")

    print(f"\nAnswer:\n{reply}")


if __name__ == "__main__":
    main()

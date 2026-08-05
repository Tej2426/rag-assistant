"""
Step 1: Chunking

WHY WE CHUNK AT ALL:
Embedding models and LLM context windows both have limits. A whole doc page
(sometimes 500+ lines) is too big and too unfocused to embed as one vector —
the embedding would blur together a dozen unrelated topics, so similarity
search against it would be vague and unreliable. We split each doc into
smaller, semantically coherent pieces so that:
  1. Each chunk's embedding represents ONE idea, not twenty
  2. Retrieval can return just the relevant paragraph, not the whole page
  3. We don't blow the LLM's context window when we stuff retrieved chunks
     into a prompt

CHUNKING STRATEGY CHOICES (this is a real design decision, not a detail):
- Fixed-size character chunking: simplest, but can slice a code block or a
  sentence in half at an arbitrary character count. Fast, but sloppy.
- Semantic / structure-aware chunking: split on natural boundaries (markdown
  headings, paragraphs) first, and only fall back to size-based splitting
  if a section is still too big. Slightly more work, much better retrieval
  quality — because a chunk boundary lines up with a meaning boundary.

We're doing structure-aware chunking here: split on markdown headings first,
then on paragraphs within a section if it's still too long. This is closer
to what you'd defend in an interview than "I split every 500 characters."
"""

import json
import re
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/chunks.jsonl")

MAX_CHUNK_CHARS = 1200      # target chunk size
MIN_CHUNK_CHARS = 200       # don't keep tiny orphan chunks; merge them forward


def split_by_headings(text: str):
    """Split a markdown doc into (heading_path, section_text) blocks."""
    lines = text.split("\n")
    sections = []
    current_heading_stack = []
    current_lines = []

    def flush():
        if current_lines:
            heading = " > ".join(current_heading_stack) if current_heading_stack else "(intro)"
            sections.append((heading, "\n".join(current_lines).strip()))

    heading_re = re.compile(r"^(#{1,6})\s+(.*)")
    for line in lines:
        m = heading_re.match(line)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            # trim the heading stack to this level, then push the new heading
            current_heading_stack = current_heading_stack[: level - 1] + [title]
            current_lines = []
        else:
            current_lines.append(line)
    flush()
    return [s for s in sections if s[1]]


def split_by_length(text: str):
    """Hard fallback for a single paragraph with no blank lines to split on
    (e.g. dense HTML/Jinja blocks). Slices on whitespace boundaries so we
    don't cut a word in half."""
    words = text.split(" ")
    chunks, current = [], ""
    for w in words:
        if len(current) + len(w) + 1 <= MAX_CHUNK_CHARS:
            current = f"{current} {w}" if current else w
        else:
            if current:
                chunks.append(current)
            current = w
    if current:
        chunks.append(current)
    return chunks


def split_long_section(heading: str, text: str):
    """If a section under a heading is still too long, split on paragraph
    breaks and greedily pack paragraphs up to MAX_CHUNK_CHARS. A paragraph
    that's still oversized on its own (no blank lines to split on) gets a
    hard length-based fallback split."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for p in paragraphs:
        if len(p) > MAX_CHUNK_CHARS:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(split_by_length(p))
        elif len(current) + len(p) + 2 <= MAX_CHUNK_CHARS:
            current = f"{current}\n\n{p}" if current else p
        else:
            if current:
                chunks.append(current)
            current = p
    if current:
        chunks.append(current)
    return chunks


def merge_tiny_chunks(chunks):
    """Fold very small chunks into the previous one so we don't end up with
    orphan fragments like a lone heading and one sentence."""
    merged = []
    for c in chunks:
        if merged and len(c) < MIN_CHUNK_CHARS:
            merged[-1] = merged[-1] + "\n\n" + c
        else:
            merged.append(c)
    return merged


def chunk_document(path: Path):
    text = path.read_text(encoding="utf-8")
    sections = split_by_headings(text)
    doc_chunks = []
    for heading, section_text in sections:
        for piece in split_long_section(heading, section_text):
            doc_chunks.append({"heading": heading, "text": piece})

    texts_only = merge_tiny_chunks([c["text"] for c in doc_chunks])
    # re-attach headings after merge (approximate — good enough for v1)
    result = []
    for i, t in enumerate(texts_only):
        heading = doc_chunks[min(i, len(doc_chunks) - 1)]["heading"]
        result.append({"heading": heading, "text": t})
    return result


def main():
    all_chunks = []
    for path in sorted(RAW_DIR.glob("*.md")):
        doc_chunks = chunk_document(path)
        for i, c in enumerate(doc_chunks):
            all_chunks.append({
                "id": f"{path.stem}::{i}",
                "source": path.stem,
                "heading": c["heading"],
                "text": c["text"],
                "n_chars": len(c["text"]),
            })
        print(f"{path.name:45s} -> {len(doc_chunks):3d} chunks")

    OUT_PATH.write_text(
        "\n".join(json.dumps(c) for c in all_chunks), encoding="utf-8"
    )
    sizes = [c["n_chars"] for c in all_chunks]
    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Avg size: {sum(sizes)//len(sizes)} chars | Min: {min(sizes)} | Max: {max(sizes)}")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

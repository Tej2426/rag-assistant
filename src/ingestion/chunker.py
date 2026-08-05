"""
Document ingestion - Chunking.
Structure-aware chunking for markdown docs, and AST-based chunking for
Python source - splitting code on function/class boundaries rather than
character count, so a chunk is always one complete, syntactically valid
unit (a whole method, not half of one).
"""

import ast
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

from src.shared import get_logger

logger = get_logger(__name__)

DOCS_DIR = Path("data/raw/docs")
CODE_DIR = Path("data/raw/code")
OUT_PATH = Path("data/chunks.jsonl")

MAX_CHUNK_CHARS = 1200
MIN_CHUNK_CHARS = 200


def clean_html(text: str) -> str:
    """Remove HTML tags and clean up."""
    soup = BeautifulSoup(text, "html.parser")
    for script in soup(["script", "style", "meta", "link"]):
        script.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def remove_frontmatter(text: str) -> str:
    """Remove YAML frontmatter from markdown."""
    if text.startswith('---'):
        parts = text.split('---', 2)
        if len(parts) >= 3:
            return parts[2].strip()
    return text


def normalize_text(text: str) -> str:
    """Normalize text for chunking."""
    text = clean_html(text)
    text = remove_frontmatter(text)
    text = text.replace('\u00a0', ' ')
    text = text.replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2014', '--').replace('\u2013', '-')
    return text


def split_by_headings(text: str) -> list[dict[str, str]]:
    """Split markdown by headings, preserving heading hierarchy."""
    lines = text.split("\n")
    sections = []
    current_heading_stack = []
    current_lines = []
    
    def flush():
        if current_lines:
            heading = " > ".join(current_heading_stack) if current_heading_stack else "(intro)"
            content = "\n".join(current_lines).strip()
            if content:
                sections.append({"heading": heading, "text": content})
    
    heading_re = re.compile(r"^(#{1,6})\s+(.*)")
    
    for line in lines:
        m = heading_re.match(line)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            current_heading_stack = current_heading_stack[:level - 1] + [title]
            current_lines = []
        else:
            current_lines.append(line)
    
    flush()
    return sections


def split_long_section(heading: str, text: str) -> list[str]:
    """Split a long section on paragraphs, with hard fallback."""
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


def split_by_length(text: str) -> list[str]:
    """Hard fallback: split on word boundaries."""
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


def merge_tiny_chunks(chunks: list[dict[str, str]]) -> list[dict[str, str]]:
    """Merge tiny chunks into previous chunk."""
    merged = []
    for c in chunks:
        if merged and len(c["text"]) < MIN_CHUNK_CHARS:
            merged[-1]["text"] = merged[-1]["text"] + "\n\n" + c["text"]
        else:
            merged.append(c)
    return merged


def _chunk_from_node(node: ast.AST, heading: str, source: str) -> dict[str, str] | None:
    text = ast.get_source_segment(source, node)
    if not text:
        return None
    return {"heading": heading, "text": text}


def chunk_code_file(path: Path) -> list[dict[str, str]]:
    """Chunk a Python source file on function/class boundaries via ast.

    Each top-level function and each method of a class becomes its own
    chunk, tagged with a heading like 'ClassName.method_name'. Unlike
    character-count chunking, a boundary here always lines up with a
    complete, syntactically valid unit - a whole method, never half of
    one - which matters for retrieval quality on code.
    """
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    module = path.stem

    raw_chunks = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunk = _chunk_from_node(node, f"{module} > {node.name}", source)
            if chunk:
                raw_chunks.append(chunk)
        elif isinstance(node, ast.ClassDef):
            methods = [
                n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            if not methods:
                chunk = _chunk_from_node(node, f"{module} > {node.name}", source)
                if chunk:
                    raw_chunks.append(chunk)
                continue
            for method in methods:
                heading = f"{module} > {node.name}.{method.name}"
                chunk = _chunk_from_node(method, heading, source)
                if chunk:
                    raw_chunks.append(chunk)

    # A handful of methods (e.g. Client.request) are still long enough to
    # need the same length fallback used for oversized doc paragraphs.
    result = []
    for c in raw_chunks:
        for piece in split_long_section(c["heading"], c["text"]):
            result.append({"heading": c["heading"], "text": piece})
    return result


def chunk_document(path: Path) -> list[dict[str, str]]:
    """Chunk a single markdown document."""
    raw_text = path.read_text(encoding="utf-8")
    text = normalize_text(raw_text)
    
    sections = split_by_headings(text)
    doc_chunks = []
    
    for section in sections:
        for piece in split_long_section(section["heading"], section["text"]):
            doc_chunks.append({"heading": section["heading"], "text": piece})
    
    merged = merge_tiny_chunks(doc_chunks)
    
    result = []
    for i, chunk in enumerate(merged):
        heading = doc_chunks[min(i, len(doc_chunks) - 1)]["heading"]
        result.append({"heading": heading, "text": chunk["text"]})
    
    return result


def main():
    """Main chunking entry point."""
    all_chunks = []

    for path in sorted(DOCS_DIR.glob("*.md")):
        logger.info("chunking_document", file=path.name)
        doc_chunks = chunk_document(path)

        for i, c in enumerate(doc_chunks):
            all_chunks.append({
                "id": f"docs::{path.stem}::{i}",
                "source": f"docs/{path.stem}",
                "heading": c["heading"],
                "text": c["text"],
                "n_chars": len(c["text"]),
            })

        logger.info("document_chunked", file=path.name, chunks=len(doc_chunks))

    for path in sorted(CODE_DIR.glob("*.py")):
        logger.info("chunking_code", file=path.name)
        code_chunks = chunk_code_file(path)

        for i, c in enumerate(code_chunks):
            all_chunks.append({
                "id": f"code::{path.stem}::{i}",
                "source": f"code/{path.stem}",
                "heading": c["heading"],
                "text": c["text"],
                "n_chars": len(c["text"]),
            })

        logger.info("code_chunked", file=path.name, chunks=len(code_chunks))

    OUT_PATH.write_text(
        "\n".join(json.dumps(c) for c in all_chunks), encoding="utf-8"
    )
    
    sizes = [c["n_chars"] for c in all_chunks]
    logger.info(
        "chunking_complete",
        total_chunks=len(all_chunks),
        avg_size=sum(sizes) // len(sizes),
        min_size=min(sizes),
        max_size=max(sizes),
        output=str(OUT_PATH),
    )


if __name__ == "__main__":
    main()
from __future__ import annotations

from typing import Any, Dict, List, Optional
import re


def clean_text(text: str) -> str:
    """
    Clean text before chunking so the chunks are more consistent.

    This function:
    - normalizes line endings
    - removes trailing spaces
    - collapses excessive blank lines
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def estimate_token_count(text: str) -> int:
    """
    Rough token estimate.

    This is only a helper for debugging or future tuning.
    A simple rule of thumb is about 4 characters per token in English.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def split_markdown_sections(text: str) -> List[Dict[str, Optional[str]]]:
    """
    Split Markdown text into sections based on headings.

    Returns a list like:
    [
        {"section": "Overview", "text": "..."},
        {"section": "Eligibility", "text": "..."}
    ]

    Notes:
    - Supports headings like: # Title, ## Overview, ### Something
    - Content before the first heading is kept as one section with section=None
    """
    lines = text.splitlines()

    sections: List[Dict[str, Optional[str]]] = []
    current_section: Optional[str] = None
    current_lines: List[str] = []

    heading_pattern = re.compile(r"^(#{1,6})\s+(.*)$")

    for line in lines:
        match = heading_pattern.match(line.strip())

        if match:
            # Save the previous section before starting a new one
            if current_lines:
                section_text = "\n".join(current_lines).strip()
                if section_text:
                    sections.append(
                        {
                            "section": current_section,
                            "text": section_text,
                        }
                    )
                current_lines = []

            current_section = match.group(2).strip()
            current_lines.append(line)
        else:
            current_lines.append(line)

    # Save the final section
    if current_lines:
        section_text = "\n".join(current_lines).strip()
        if section_text:
            sections.append(
                {
                    "section": current_section,
                    "text": section_text,
                }
            )

    return sections


def split_text_with_overlap(
    text: str,
    chunk_size: int = 700,
    chunk_overlap: int = 100,
) -> List[str]:
    """
    Fallback chunking by characters with overlap.

    Example:
    - chunk_size=700
    - chunk_overlap=100

    This means:
    - chunk 1 = chars 0..699
    - chunk 2 = chars 600..1299
    and so on

    Why overlap matters:
    It preserves context between neighboring chunks.
    """
    text = clean_text(text)

    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: List[str] = []
    start = 0
    step = chunk_size - chunk_overlap

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start += step

    return chunks


def chunk_section_text(
    section_text: str,
    section_name: Optional[str],
    chunk_size: int = 700,
    chunk_overlap: int = 100,
) -> List[Dict[str, Optional[str]]]:
    """
    Chunk one section of text.

    If the section is already short enough, return it as one chunk.
    Otherwise, split it using overlap-based character chunking.
    """
    section_text = clean_text(section_text)

    if not section_text:
        return []

    if len(section_text) <= chunk_size:
        return [{"section": section_name, "text": section_text}]

    raw_chunks = split_text_with_overlap(
        text=section_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return [{"section": section_name, "text": chunk} for chunk in raw_chunks]


def build_chunk_record(
    doc: Dict[str, Any],
    chunk_text: str,
    chunk_index: int,
    section: Optional[str],
) -> Dict[str, Any]:
    """
    Build one chunk record while preserving metadata from the source document.

    Expected source doc format:
    {
        "text": "...",
        "title": "...",
        "source": "pto_policy.md",
        "doc_id": "pto_policy",
        "page": None,
        "section": None
    }
    """
    doc_id = doc["doc_id"]
    chunk_id = f"{doc_id}_chunk_{chunk_index}"

    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "text": chunk_text,
        "title": doc.get("title"),
        "source": doc.get("source"),
        "page": doc.get("page"),
        "section": section,
    }


def chunk_document(
    doc: Dict[str, Any],
    chunk_size: int = 700,
    chunk_overlap: int = 100,
) -> List[Dict[str, Any]]:
    """
    Chunk a single normalized document.

    Strategy:
    1. If the document looks like Markdown, split by headings first.
    2. Then chunk each section if needed.
    3. If not Markdown, fall back to character chunking.

    Returns a list of chunk dictionaries with metadata.
    """
    text = clean_text(doc.get("text", ""))
    source = str(doc.get("source", "")).lower()

    if not text:
        return []

    chunk_records: List[Dict[str, Any]] = []
    chunk_index = 1

    # Treat .md files as heading-aware Markdown documents
    if source.endswith(".md"):
        sections = split_markdown_sections(text)

        for section_data in sections:
            section_name = section_data["section"]
            section_text = section_data["text"]

            section_chunks = chunk_section_text(
                section_text=section_text,
                section_name=section_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            for item in section_chunks:
                chunk_records.append(
                    build_chunk_record(
                        doc=doc,
                        chunk_text=item["text"],
                        chunk_index=chunk_index,
                        section=item["section"],
                    )
                )
                chunk_index += 1

    else:
        # Fallback chunking for HTML, TXT, PDF text, etc.
        raw_chunks = split_text_with_overlap(
            text=text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for raw_chunk in raw_chunks:
            chunk_records.append(
                build_chunk_record(
                    doc=doc,
                    chunk_text=raw_chunk,
                    chunk_index=chunk_index,
                    section=doc.get("section"),
                )
            )
            chunk_index += 1

    return chunk_records


def chunk_documents(
    docs: List[Dict[str, Any]],
    chunk_size: int = 700,
    chunk_overlap: int = 100,
) -> List[Dict[str, Any]]:
    """
    Chunk a list of documents.

    Returns one flat list of chunk records.
    """
    all_chunks: List[Dict[str, Any]] = []

    for doc in docs:
        doc_chunks = chunk_document(
            doc=doc,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        all_chunks.extend(doc_chunks)

    return all_chunks


if __name__ == "__main__":
    # Small self-test so you can run this file directly if needed.
    sample_doc = {
        "text": """# PTO Policy

## Overview
Employees receive paid leave for vacation, illness, and personal matters.

## Eligibility
All full-time employees are eligible. Part-time staff receive prorated benefits.

## Carryover
Unused leave may be carried over up to 5 days into the next year.
""",
        "title": "PTO Policy",
        "source": "pto_policy.md",
        "doc_id": "pto_policy",
        "page": None,
        "section": None,
    }

    chunks = chunk_document(sample_doc, chunk_size=120, chunk_overlap=20)

    print(f"Created {len(chunks)} chunks\n")
    for chunk in chunks:
        print(chunk)
        print("-" * 80)
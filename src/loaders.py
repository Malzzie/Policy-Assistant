from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import re

# HTML parsing
from bs4 import BeautifulSoup

# PDF parsing
from pypdf import PdfReader


def make_doc(
    text: str,
    title: str,
    source: str,
    doc_id: str,
    page: Optional[int] = None,
    section: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create one normalized document dictionary.

    Output format:
    {
        "text": "...",
        "title": "...",
        "source": "pto_policy.md",
        "doc_id": "pto_policy",
        "page": None,
        "section": None
    }
    """
    return {
        "text": clean_text(text),
        "title": title.strip(),
        "source": source,
        "doc_id": doc_id,
        "page": page,
        "section": section,
    }


def clean_text(text: str) -> str:
    """
    Clean text so all loaders return consistent output.

    What this does:
    - replaces multiple spaces/tabs with one space
    - reduces repeated blank lines
    - trims leading/trailing whitespace
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def file_stem_to_title(path: Path) -> str:
    """
    Convert a filename like 'pto_policy.md' into 'Pto Policy'.

    You can improve this later with a custom mapping for nicer titles.
    """
    return path.stem.replace("_", " ").replace("-", " ").title()


def extract_markdown_title(text: str, fallback: str) -> str:
    """
    Try to extract the first Markdown H1 heading as the title.
    Example: '# Paid Time Off (PTO) Policy'
    """
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return fallback


def extract_html_title(html: str, fallback: str) -> str:
    """
    Try to extract a title from HTML:
    1. <title>
    2. first <h1>
    3. fallback
    """
    soup = BeautifulSoup(html, "html.parser")

    if soup.title and soup.title.string:
        return soup.title.string.strip()

    h1 = soup.find("h1")
    if h1:
        return h1.get_text(" ", strip=True)

    return fallback


def load_markdown_file(path: str | Path) -> Dict[str, Any]:
    """
    Load a Markdown file into the normalized document format.
    """
    path = Path(path)
    raw_text = path.read_text(encoding="utf-8")
    fallback_title = file_stem_to_title(path)
    title = extract_markdown_title(raw_text, fallback_title)

    return make_doc(
        text=raw_text,
        title=title,
        source=path.name,
        doc_id=path.stem,
        page=None,
        section=None,
    )


def load_text_file(path: str | Path) -> Dict[str, Any]:
    """
    Load a plain text file into the normalized document format.
    """
    path = Path(path)
    raw_text = path.read_text(encoding="utf-8")
    title = file_stem_to_title(path)

    return make_doc(
        text=raw_text,
        title=title,
        source=path.name,
        doc_id=path.stem,
        page=None,
        section=None,
    )


def load_html_file(path: str | Path) -> Dict[str, Any]:
    """
    Load an HTML file and extract the visible text.
    """
    path = Path(path)
    raw_html = path.read_text(encoding="utf-8")
    fallback_title = file_stem_to_title(path)
    title = extract_html_title(raw_html, fallback_title)

    soup = BeautifulSoup(raw_html, "html.parser")
    text = soup.get_text(separator="\n", strip=True)

    return make_doc(
        text=text,
        title=title,
        source=path.name,
        doc_id=path.stem,
        page=None,
        section=None,
    )


def load_pdf_file(path: str | Path) -> List[Dict[str, Any]]:
    """
    Load a PDF file and return one document per page.

    Returning one doc per page is useful later for retrieval,
    because PDFs usually work better when page numbers are preserved.
    """
    path = Path(path)
    reader = PdfReader(str(path))

    docs: List[Dict[str, Any]] = []

    # Try to get title from PDF metadata first
    pdf_title = None
    if reader.metadata and reader.metadata.title:
        pdf_title = str(reader.metadata.title).strip()

    fallback_title = file_stem_to_title(path)
    title = pdf_title if pdf_title else fallback_title

    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        docs.append(
            make_doc(
                text=text,
                title=title,
                source=path.name,
                doc_id=path.stem,
                page=i,
                section=None,
            )
        )

    return docs


def load_documents(directory: str | Path) -> List[Dict[str, Any]]:
    """
    Load all supported documents from a directory.

    Supported file types:
    - .md
    - .txt
    - .html / .htm
    - .pdf

    Returns:
        A list of normalized document dictionaries.
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    all_docs: List[Dict[str, Any]] = []

    for path in sorted(directory.iterdir()):
        if path.is_dir():
            continue

        suffix = path.suffix.lower()

        if suffix == ".md":
            all_docs.append(load_markdown_file(path))

        elif suffix == ".txt":
            all_docs.append(load_text_file(path))

        elif suffix in {".html", ".htm"}:
            all_docs.append(load_html_file(path))

        elif suffix == ".pdf":
            pdf_docs = load_pdf_file(path)
            all_docs.extend(pdf_docs)

        else:
            print(f"Skipping unsupported file: {path.name}")

    return all_docs


if __name__ == "__main__":
    # Example test run
    docs = load_documents("policies")

    print(f"Loaded {len(docs)} documents/pages.\n")

    # Print the first few docs so you can inspect the structure
    for doc in docs[:5]:
        print(doc)
        print("-" * 80)
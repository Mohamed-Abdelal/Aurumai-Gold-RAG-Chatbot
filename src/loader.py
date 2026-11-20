"""Clean .txt / .md knowledge base loader with 4-line metadata header parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class Document:
    """Represents a single cleaned knowledge-base article loaded from data/."""

    doc_id: str
    filename: str
    title: str
    category: str
    date: str
    source: str
    text: str


_HEADER_RE = re.compile(r"^(Title|Category|Date|Source)\s*:\s*(.+)$", re.IGNORECASE)


def parse_header_and_body(raw_text: str, fallback_title: str) -> Tuple[Dict[str, str], str]:
    """
    Parse the standard 4-line header (`Title:`, `Category:`, `Date:`, `Source:`)
    at the top of each .txt/.md article and return (metadata_dict, clean_body).
    """
    normalized = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")

    meta: Dict[str, str] = {
        "title": fallback_title,
        "category": "General Gold Investing",
        "date": "",
        "source": "AurumAI Knowledge Base",
    }

    body_start_idx = 0
    for idx, line in enumerate(lines[:10]):
        stripped = line.strip()
        if not stripped:
            if body_start_idx > 0:
                body_start_idx = idx + 1
                break
            continue
        match = _HEADER_RE.match(stripped)
        if match:
            key = match.group(1).lower()
            meta[key] = match.group(2).strip()
            body_start_idx = idx + 1
        else:
            break

    body = "\n".join(lines[body_start_idx:]).strip()
    body = re.sub(r"\n{3,}", "\n\n", body)
    return meta, body


def load_text_document(file_path: Path) -> Document:
    """Load and parse a single .txt or .md knowledge-base file."""
    raw_text = file_path.read_text(encoding="utf-8")
    fallback_title = file_path.stem.replace("_", " ").strip().title()
    meta, body = parse_header_and_body(raw_text, fallback_title=fallback_title)

    return Document(
        doc_id=file_path.stem,
        filename=file_path.name,
        title=meta["title"],
        category=meta["category"],
        date=meta["date"],
        source=meta["source"],
        text=body,
    )


def load_corpus_documents(data_dir: Path) -> List[Document]:
    """
    Load all .txt and .md files in data_dir sorted alphabetically.
    No PDF parser is required unless PDFs are explicitly introduced.
    """
    if not data_dir.exists():
        return []

    supported_files = sorted(
        p for p in data_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".txt", ".md"}
    )

    documents: List[Document] = []
    for path in supported_files:
        doc = load_text_document(path)
        if doc.text:
            documents.append(doc)

    return documents

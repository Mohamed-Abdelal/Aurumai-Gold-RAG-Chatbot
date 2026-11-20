"""Word-based document chunking with sliding overlap and title-anchored embedding text."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Dict, List

from src.config import CHUNK_OVERLAP_WORDS, CHUNK_SIZE_WORDS, MIN_CHUNK_WORDS
from src.loader import Document


@dataclass
class TextChunk:
    """A single indexed text chunk with full document provenance."""

    chunk_id: str
    doc_id: str
    filename: str
    title: str
    category: str
    date: str
    source: str
    chunk_index: int
    text: str
    word_count: int
    char_count: int

    def embedding_text(self) -> str:
        """
        Prepend `Title: <doc_title>\n\n` before encoding with all-MiniLM-L6-v2
        so mid-article paragraphs stay anchored to their topic during cosine search.
        """
        return f"Title: {self.title}\n\n{self.text}"

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "TextChunk":
        return cls(**data)


def split_text_by_words(
    text: str,
    chunk_size: int = CHUNK_SIZE_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS,
    min_chunk_words: int = MIN_CHUNK_WORDS,
) -> List[str]:
    """
    Split text into overlapping word-bounded chunks (`chunk_size` words, `overlap` words).

    Uses a clean sliding window over whitespace-delimited words while collapsing
    excess blank lines so every chunk stays focused and semantically coherent.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and strictly smaller than chunk_size.")

    cleaned = re.sub(r"[ \t]+", " ", text.strip())
    words = cleaned.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [" ".join(words)] if len(words) >= min_chunk_words else [" ".join(words)]

    step = chunk_size - overlap
    chunks: List[str] = []
    start = 0

    while start < len(words):
        window = words[start : start + chunk_size]
        if len(window) < min_chunk_words and chunks:
            # Merge a tiny trailing tail into the previous chunk rather than dropping facts
            chunks[-1] = f"{chunks[-1]} {' '.join(window[overlap:])}".strip()
            break
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
        start += step

    return chunks


def chunk_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS,
    min_chunk_words: int = MIN_CHUNK_WORDS,
) -> List[TextChunk]:
    """Convert loaded Document objects into word-chunked TextChunk objects."""
    all_chunks: List[TextChunk] = []

    for doc in documents:
        raw_chunks = split_text_by_words(
            doc.text,
            chunk_size=chunk_size,
            overlap=overlap,
            min_chunk_words=min_chunk_words,
        )
        for idx, chunk_text in enumerate(raw_chunks):
            chunk_id = f"{doc.doc_id}_c{idx}"
            all_chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    filename=doc.filename,
                    title=doc.title,
                    category=doc.category,
                    date=doc.date,
                    source=doc.source,
                    chunk_index=idx,
                    text=chunk_text,
                    word_count=len(chunk_text.split()),
                    char_count=len(chunk_text),
                )
            )

    return all_chunks

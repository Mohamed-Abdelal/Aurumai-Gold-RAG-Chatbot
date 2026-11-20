"""Unit tests for AurumAI core RAG components (header parsing, word chunking, title prefix, pure NumPy)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.chunker import TextChunk, chunk_documents, split_text_by_words
from src.loader import Document, parse_header_and_body
from src.retriever import NumPyVectorIndex, compute_cosine_similarity


def _make_chunk(idx: int, filename: str = "01_gold.txt", category: str = "Beginner Investment Vehicles") -> TextChunk:
    return TextChunk(
        chunk_id=f"{Path(filename).stem}_c{idx}",
        doc_id=Path(filename).stem,
        filename=filename,
        title=f"Gold Guide {idx}",
        category=category,
        date="2026-11-15",
        source="AurumAI Desk",
        chunk_index=idx,
        text=f"Sample gold investment chunk number {idx} for testing.",
        word_count=9,
        char_count=52,
    )


def test_cosine_similarity_exact_math() -> None:
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    docs = np.array(
        [
            [2.0, 0.0, 0.0],   # Identical direction -> +1.0
            [0.0, 5.0, 0.0],   # Orthogonal          ->  0.0
            [-3.0, 0.0, 0.0],  # Opposite direction  -> -1.0
            [1.0, 1.0, 0.0],   # 45 degrees          -> ~0.7071
        ],
        dtype=np.float32,
    )
    scores = compute_cosine_similarity(query, docs)
    assert scores.shape == (4,)
    assert np.isclose(scores[0], 1.0, atol=1e-5)
    assert np.isclose(scores[1], 0.0, atol=1e-5)
    assert np.isclose(scores[2], -1.0, atol=1e-5)
    assert np.isclose(scores[3], np.sqrt(2) / 2, atol=1e-5)


def test_cosine_similarity_handles_zero_vector_safely() -> None:
    query = np.array([0.0, 0.0], dtype=np.float32)
    docs = np.array([[1.0, 2.0], [0.0, 0.0]], dtype=np.float32)
    scores = compute_cosine_similarity(query, docs)
    assert np.all(np.isfinite(scores))
    assert np.allclose(scores, 0.0)


def test_cosine_similarity_dimension_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="dimension mismatch"):
        compute_cosine_similarity(np.ones(3), np.ones((2, 4)))


def test_numpy_vector_index_top_k_and_filters(tmp_path: Path) -> None:
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],   # chunk 0 (doc A, cat 1) -> score 1.0
            [0.8, 0.6, 0.0],   # chunk 1 (doc B, cat 2) -> score 0.8
            [0.1, 0.99, 0.0],  # chunk 2 (doc A, cat 1) -> score ~0.1
            [0.6, 0.8, 0.0],   # chunk 3 (doc B, cat 2) -> score 0.6
        ],
        dtype=np.float32,
    )
    chunks = [
        _make_chunk(0, "doc_a.txt", "Egypt Gold Buying Guide"),
        _make_chunk(1, "doc_b.txt", "Portfolio Allocation & Strategies"),
        _make_chunk(2, "doc_a.txt", "Egypt Gold Buying Guide"),
        _make_chunk(3, "doc_b.txt", "Portfolio Allocation & Strategies"),
    ]
    index = NumPyVectorIndex(embeddings=embeddings, chunks=chunks)
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    top2 = index.search(query, top_k=2, min_similarity=0.2)
    assert [r.chunk.chunk_index for r in top2] == [0, 1]
    assert [r.rank for r in top2] == [1, 2]
    assert top2[0].score > top2[1].score

    filtered_doc = index.search(query, top_k=4, min_similarity=0.0, doc_filter="doc_b.txt")
    assert all(r.chunk.filename == "doc_b.txt" for r in filtered_doc)
    assert [r.chunk.chunk_index for r in filtered_doc] == [1, 3]

    filtered_cat = index.search(
        query, top_k=4, min_similarity=0.0, category_filter="Egypt Gold Buying Guide"
    )
    assert all(r.chunk.category == "Egypt Gold Buying Guide" for r in filtered_cat)

    # Save / Load roundtrip
    emb_file = tmp_path / "embeddings.npz"
    chk_file = tmp_path / "chunks.json"
    index.save(emb_file, chk_file)
    reloaded = NumPyVectorIndex.load(emb_file, chk_file)
    assert reloaded.num_chunks == 4
    assert np.allclose(reloaded.embeddings, embeddings)


def test_parse_four_line_header_and_body() -> None:
    raw = (
        "Title: Egypt 24K Bullion Bars vs. 21K Gold Pounds\n"
        "Category: Egypt Gold Buying Guide\n"
        "Date: 2026-11-15\n"
        "Source: AurumAI Egypt Market Desk\n\n"
        "A standard 21K gold pound weighs 8 grams.\n"
    )
    meta, body = parse_header_and_body(raw, fallback_title="Fallback")
    assert meta["title"] == "Egypt 24K Bullion Bars vs. 21K Gold Pounds"
    assert meta["category"] == "Egypt Gold Buying Guide"
    assert meta["date"] == "2026-11-15"
    assert meta["source"] == "AurumAI Egypt Market Desk"
    assert body == "A standard 21K gold pound weighs 8 grams."


def test_word_chunking_220_40_and_title_prefix() -> None:
    words = [f"word{i}" for i in range(460)]
    doc = Document(
        doc_id="16_egypt",
        filename="16_egypt.txt",
        title="Egypt Gold Guide",
        category="Egypt Gold Buying Guide",
        date="2026-11-15",
        source="AurumAI",
        text=" ".join(words),
    )
    chunks = chunk_documents([doc], chunk_size=220, overlap=40, min_chunk_words=25)
    assert len(chunks) == 3
    assert chunks[0].word_count == 220
    # Verify exact 40-word sliding overlap between chunk 0 and chunk 1
    assert chunks[0].text.split()[-40:] == chunks[1].text.split()[:40]
    # Verify Title prefix applied before embedding
    assert chunks[1].embedding_text().startswith("Title: Egypt Gold Guide\n\n")

"""
Build the pure-NumPy vector index for AurumAI from .txt/.md articles in data/:
    data/*.txt -> Header Parsing -> 220-word / 40-overlap Chunking
    -> `Title: <doc_title>\n\n` Prefixed Embeddings (all-MiniLM-L6-v2)
    -> index/embeddings.npz + index/chunks.json

Usage:
    python scripts/build_index.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.chunker import chunk_documents
from src.config import (
    CHUNK_OVERLAP_WORDS,
    CHUNK_SIZE_WORDS,
    CHUNKS_PATH,
    DATA_DIR,
    EMBEDDING_MODEL_NAME,
    EMBEDDINGS_PATH,
    MIN_CHUNK_WORDS,
)
from src.embeddings import EmbeddingModel
from src.loader import load_corpus_documents
from src.retriever import NumPyVectorIndex


def build_index() -> None:
    t0 = time.perf_counter()
    print(f"1. Loading .txt/.md articles from: {DATA_DIR}")
    documents = load_corpus_documents(DATA_DIR)
    if not documents:
        raise RuntimeError(f"No .txt or .md documents found in {DATA_DIR}.")
    categories = sorted({d.category for d in documents})
    print(f"   Loaded {len(documents)} articles across {len(categories)} categories.")

    print(
        f"2. Chunking articles (chunk_size={CHUNK_SIZE_WORDS} words, "
        f"overlap={CHUNK_OVERLAP_WORDS} words, min_words={MIN_CHUNK_WORDS})..."
    )
    chunks = chunk_documents(
        documents,
        chunk_size=CHUNK_SIZE_WORDS,
        overlap=CHUNK_OVERLAP_WORDS,
        min_chunk_words=MIN_CHUNK_WORDS,
    )
    print(f"   Generated {len(chunks)} overlapping word chunks.")

    print(f"3. Generating title-anchored embeddings with {EMBEDDING_MODEL_NAME}...")
    embedder = EmbeddingModel(model_name=EMBEDDING_MODEL_NAME)
    # Critical accuracy fix: prepend `Title: <doc_title>\n\n` to every chunk before encoding
    # so mid-article paragraphs stay tied to their topic during NumPy cosine similarity search.
    texts_to_embed = [c.embedding_text() for c in chunks]
    embeddings = embedder.embed_texts(
        texts_to_embed,
        batch_size=32,
        show_progress_bar=True,
    )
    print(f"   Embeddings matrix shape: {embeddings.shape} ({embeddings.dtype})")

    print(f"4. Saving pure-NumPy index to {EMBEDDINGS_PATH} and {CHUNKS_PATH}...")
    index = NumPyVectorIndex(embeddings=embeddings, chunks=chunks)
    index.save(EMBEDDINGS_PATH, CHUNKS_PATH)

    emb_kb = EMBEDDINGS_PATH.stat().st_size / 1024.0
    chk_kb = CHUNKS_PATH.stat().st_size / 1024.0
    elapsed = time.perf_counter() - t0
    print(
        f"\nAurumAI index built successfully in {elapsed:.1f}s!\n"
        f"  Articles indexed : {len(documents)}\n"
        f"  Total chunks     : {len(chunks)}\n"
        f"  Embedding dim    : {embeddings.shape[1]}\n"
        f"  Index size       : {emb_kb:.1f} KB (.npz) + {chk_kb:.1f} KB (chunks.json)"
    )


if __name__ == "__main__":
    build_index()

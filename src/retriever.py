"""Pure-NumPy Cosine Similarity and Top-K Retrieval Engine."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.chunker import TextChunk


@dataclass
class RetrievedChunk:
    """A retrieved document chunk paired with its NumPy cosine similarity score."""

    chunk: TextChunk
    score: float
    rank: int


def compute_cosine_similarity(
    query_embedding: np.ndarray,
    doc_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity between a query vector q (D,) and N document
    vectors D_mat (N, D) using pure NumPy:

        cosine_similarity(q, d_i) = (q · d_i) / (||q||_2 * ||d_i||_2)
    """
    q = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
    docs = np.asarray(doc_embeddings, dtype=np.float32)

    if docs.ndim != 2 or docs.shape[0] == 0:
        return np.array([], dtype=np.float32)
    if docs.shape[1] != q.shape[0]:
        raise ValueError(
            f"Embedding dimension mismatch: query has {q.shape[0]}, docs have {docs.shape[1]}."
        )

    dot_products = np.dot(docs, q)
    query_norm = float(np.linalg.norm(q))
    doc_norms = np.linalg.norm(docs, axis=1)
    denominators = np.maximum(doc_norms * query_norm, 1e-10)

    similarities = dot_products / denominators
    return np.clip(similarities, -1.0, 1.0).astype(np.float32)


class NumPyVectorIndex:
    """Simple vector store backed strictly by a NumPy embedding matrix and JSON metadata."""

    def __init__(self, embeddings: np.ndarray, chunks: List[TextChunk]):
        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Embeddings rows ({embeddings.shape[0]}) != chunks count ({len(chunks)})."
            )
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.chunks = chunks

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)

    @property
    def embedding_dim(self) -> int:
        return int(self.embeddings.shape[1]) if self.embeddings.ndim == 2 else 0

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 4,
        min_similarity: float = 0.0,
    ) -> List[RetrievedChunk]:
        """Retrieve the Top-K most similar chunks using pure-NumPy cosine similarity."""
        if self.num_chunks == 0 or top_k <= 0:
            return []

        scores = compute_cosine_similarity(query_embedding, self.embeddings)
        sorted_indices = np.argsort(scores)[::-1]

        results: List[RetrievedChunk] = []
        for idx in sorted_indices:
            score = float(scores[idx])
            if not np.isfinite(score) or score < min_similarity:
                break
            results.append(
                RetrievedChunk(
                    chunk=self.chunks[int(idx)],
                    score=score,
                    rank=len(results) + 1,
                )
            )
            if len(results) >= top_k:
                break

        return results

    def save(self, embeddings_path: Path, chunks_path: Path) -> None:
        """Save the NumPy matrix (.npz) and chunk metadata (.json) to disk."""
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        chunks_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(embeddings_path, embeddings=self.embeddings)
        with chunks_path.open("w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in self.chunks], f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, embeddings_path: Path, chunks_path: Path) -> "NumPyVectorIndex":
        """Load a pre-computed NumPy index from disk."""
        if not embeddings_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"Index files not found at {embeddings_path} and {chunks_path}. "
                "Run `python scripts/build_index.py` first."
            )

        data = np.load(embeddings_path)
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)

        with chunks_path.open("r", encoding="utf-8") as f:
            raw_chunks = json.load(f)
        chunks = [TextChunk.from_dict(item) for item in raw_chunks]

        return cls(embeddings=embeddings, chunks=chunks)

    def get_unique_documents(self) -> List[Dict]:
        """Return a summary list of all indexed articles."""
        seen: Dict[str, Dict] = {}
        for chunk in self.chunks:
            if chunk.filename not in seen:
                seen[chunk.filename] = {
                    "filename": chunk.filename,
                    "title": chunk.title,
                    "category": chunk.category,
                    "chunk_count": 0,
                }
            seen[chunk.filename]["chunk_count"] += 1
        return sorted(seen.values(), key=lambda x: x["filename"])

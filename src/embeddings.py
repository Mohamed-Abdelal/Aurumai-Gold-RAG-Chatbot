"""Dense text embedding generation using SentenceTransformers."""

from __future__ import annotations

import os
from typing import List, Optional

# Ensure standard HTTPS downloads on Windows machines where optional hf_xet DLL is blocked
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL_NAME


class EmbeddingModel:
    """Thin wrapper around SentenceTransformer producing float32 NumPy vectors."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, device: Optional[str] = None):
        self.model_name = model_name
        try:
            # Prefer locally cached weights for instant startup and offline reliability
            self._model = SentenceTransformer(model_name, device=device, local_files_only=True)
        except Exception:
            self._model = SentenceTransformer(model_name, device=device)

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Encode a list of document chunks into an (N, D) float32 NumPy array."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=False,  # Keep raw vectors so NumPy cosine formula is explicit
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Encode a single user query into a 1D (D,) float32 NumPy vector."""
        vector = self._model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return np.asarray(vector, dtype=np.float32).reshape(-1)

"""Turn text into vectors with the required model (all-MiniLM-L6-v2).

The model only reads about the first 256 tokens of its input. That is why we
split each page into shorter views in chunk.py, so important text is not cut off.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from utils import EMBEDDING_MODEL_NAME

_model = None  # lazily-loaded SentenceTransformer singleton


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        # No explicit device: sentence-transformers auto-selects the GPU (CUDA)
        # when available and falls back to CPU otherwise.
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(
    texts: Sequence[str],
    *,
    batch_size: int = 256,
    show_progress_bar: bool = False,
) -> np.ndarray:
    """Return L2-normalized embeddings, shape (n, 384), float32."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    model = get_model()
    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def embed_queries(queries: List[str], *, batch_size: int = 64) -> np.ndarray:
    return embed_texts(queries, batch_size=batch_size)

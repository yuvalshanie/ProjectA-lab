"""Embedding utilities.

All embeddings (corpus and queries) use a single model,
``sentence-transformers/all-MiniLM-L6-v2``, as required by the assignment.
Vectors are L2-normalized so that inner product equals cosine similarity.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from utils import EMBEDDING_MODEL_NAME, MAX_SEQ_LENGTH

EMBED_DIM = 384  # all-MiniLM-L6-v2 output dimension

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Load (once) and return the shared sentence-transformer model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        # Pin the context window to the empirically chosen value (see utils).
        _model.max_seq_length = MAX_SEQ_LENGTH
    return _model


def embed_texts(texts: Sequence[str], *, batch_size: int = 128) -> np.ndarray:
    """Return L2-normalized embeddings of ``texts``, shape (n, EMBED_DIM)."""
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)
    model = get_model()
    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def embed_queries(queries: List[str], *, batch_size: int = 128) -> np.ndarray:
    """Embed query strings with the same model/normalization as the corpus."""
    return embed_texts(queries, batch_size=batch_size)

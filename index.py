"""Offline index build and load.

The build step (run on our machine via ``scripts/build_index.py``) embeds the
full corpus and writes the artifacts that ``run()`` loads at grading time:

    artifacts/index_vectors.npy   float32 (num_units, 384), L2-normalized
    artifacts/index_meta.json     page_id (and chunk_id) for each vector row

The grader never rebuilds the index; it only calls ``run()``, which loads these
files. See README for formats.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from chunk import Chunk, chunk_corpus
from embed import EMBED_DIM, embed_texts
from utils import (
    ARTIFACTS_DIR,
    EMBEDDING_MODEL_NAME,
    MAX_SEQ_LENGTH,
    ensure_artifacts_dir,
    iter_entries,
)

INDEX_VECTORS_NAME = "index_vectors.npy"
INDEX_META_NAME = "index_meta.json"


def build_index(
    *,
    entries_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, List[int]]:
    """Embed the full corpus and persist artifacts.

    Returns ``(vectors, page_ids)`` where ``vectors[i]`` corresponds to
    ``page_ids[i]``.
    """
    out_dir = artifacts_dir or ensure_artifacts_dir()
    records = list(iter_entries(entries_dir))
    chunks: List[Chunk] = chunk_corpus(records)
    texts = [c.text for c in chunks]
    vectors = embed_texts(texts)
    page_ids = [c.page_id for c in chunks]

    np.save(out_dir / INDEX_VECTORS_NAME, vectors.astype(np.float32))
    meta = {
        "page_ids": page_ids,
        "chunk_ids": [c.chunk_id for c in chunks],
        "model": EMBEDDING_MODEL_NAME,
        "max_seq_length": MAX_SEQ_LENGTH,
        "embed_dim": EMBED_DIM,
        "num_vectors": len(page_ids),
    }
    (out_dir / INDEX_META_NAME).write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return vectors, page_ids


def load_index(
    artifacts_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load precomputed vectors and the per-row page_id map from ``artifacts/``.

    Returns ``(vectors, page_ids)`` with ``vectors`` float32 (n, dim) and
    ``page_ids`` an int64 array of length n.
    """
    root = artifacts_dir or ARTIFACTS_DIR
    vectors = np.load(root / INDEX_VECTORS_NAME).astype(np.float32)
    meta = json.loads((root / INDEX_META_NAME).read_text(encoding="utf-8"))
    page_ids = np.asarray([int(x) for x in meta["page_ids"]], dtype=np.int64)
    return vectors, page_ids

"""Query-time retrieval (the timed stage at grading).

Hybrid of two signals, each scored over the whole corpus and min-max normalized
per query, then linearly fused:

    score = ALPHA_DENSE * dense_cosine + (1 - ALPHA_DENSE) * bm25

* dense  — cosine similarity between the query and page embeddings
           (all-MiniLM-L6-v2, L2-normalized -> inner product). Exact search by a
           dense matmul; at 27k x 384 this is well under a second, so an ANN
           index (FAISS) is unnecessary here and we keep exact recall.
* bm25   — lexical overlap from the prebuilt inverted index (see ``lexical.py``),
           which captures the exact years / populations / scores / names that
           dense embeddings blur.

There is one vector per page, so each corpus row maps directly to a page_id and
no chunk aggregation is needed.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from embed import embed_queries
from index import load_index
from lexical import LexicalIndex
from utils import ALPHA_DENSE, K_EVAL


def _minmax_rows(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize each row to [0, 1] (constant rows -> 0)."""
    lo = scores.min(axis=1, keepdims=True)
    hi = scores.max(axis=1, keepdims=True)
    return (scores - lo) / (hi - lo + 1e-9)


class Retriever:
    """Holds the dense vectors, page_id map, and the BM25 index (built once)."""

    def __init__(self, artifacts_dir: Optional[Path] = None) -> None:
        vectors, page_ids = load_index(artifacts_dir)
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.page_ids = page_ids
        self.lexical = LexicalIndex(artifacts_dir)
        if self.lexical.num_docs != len(page_ids):
            raise ValueError(
                "Dense/lexical index mismatch: "
                f"{len(page_ids)} pages vs {self.lexical.num_docs} lexical docs"
            )

    def search(self, queries: List[str], top_k: int = K_EVAL) -> List[List[int]]:
        if not queries:
            return []
        if self.vectors.shape[0] == 0:
            return [[] for _ in queries]

        query_vectors = embed_queries(queries)
        dense = query_vectors @ self.vectors.T          # (Q, n_pages)
        bm25 = self.lexical.scores(queries)             # (Q, n_pages)
        fused = ALPHA_DENSE * _minmax_rows(dense) + (1.0 - ALPHA_DENSE) * _minmax_rows(bm25)

        results: List[List[int]] = []
        for row in fused:
            top = np.argpartition(-row, min(top_k, row.size - 1))[:top_k]
            top = top[np.argsort(-row[top])]
            results.append([int(self.page_ids[i]) for i in top])
        return results


_retriever: Retriever | None = None


def search_batch(
    queries: List[str],
    *,
    top_k: int = K_EVAL,
    artifacts_dir: Optional[Path] = None,
) -> List[List[int]]:
    """Return ranked page_id lists (best first) for each query."""
    global _retriever
    if _retriever is None or artifacts_dir is not None:
        _retriever = Retriever(artifacts_dir)
    return _retriever.search(queries, top_k=top_k)

"""Query-time retrieval (the timed stage at grading).

Loads the prebuilt index from ``artifacts/``, embeds the query batch with the
same model, and runs exact inner-product (cosine, since vectors are normalized)
search with FAISS. Scores are aggregated to the page level by taking each
page's best unit, then the top-``K_EVAL`` page_ids are returned per query.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np

from embed import embed_queries
from index import load_index
from utils import K_EVAL

# How many units to pull from FAISS before aggregating to pages. With one unit
# per page this only needs to exceed K_EVAL; the margin keeps results exact if a
# multi-unit index is ever used.
_CANDIDATE_UNITS = 256


class Retriever:
    """Holds the FAISS index and the unit->page_id map (built once)."""

    def __init__(self, artifacts_dir: Optional[Path] = None) -> None:
        vectors, page_ids = load_index(artifacts_dir)
        self.page_ids = page_ids
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        if vectors.shape[0]:
            self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    def search(self, queries: List[str], top_k: int = K_EVAL) -> List[List[int]]:
        if not queries:
            return []
        query_vectors = embed_queries(queries)
        n_units = self.index.ntotal
        if n_units == 0:
            return [[] for _ in queries]

        n_probe = min(n_units, max(_CANDIDATE_UNITS, top_k * 4))
        scores, idx = self.index.search(
            np.ascontiguousarray(query_vectors, dtype=np.float32), n_probe
        )

        results: List[List[int]] = []
        for row_idx, row_scores in zip(idx, scores):
            best_per_page: dict[int, float] = {}
            for unit, score in zip(row_idx, row_scores):
                if unit < 0:  # FAISS pads with -1 when fewer than n_probe hits
                    continue
                pid = int(self.page_ids[unit])
                if score > best_per_page.get(pid, -1e30):
                    best_per_page[pid] = float(score)
            ranked = sorted(best_per_page, key=best_per_page.get, reverse=True)
            results.append(ranked[:top_k])
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

"""Query-time retrieval. This is the timed part the autograder runs.

For each query we combine two scores:
1. Dense score: embed the query with MiniLM, search the FAISS index of page
views, and give each page the sum of its 3 best view matches.
2. Lexical score: keyword matching from the inverted index (see lexical.py).

We put both scores on a 0-1 scale (divide by the per query maximum), add them,
and give a small extra bonus to pages that score well in both. The best 10
pages are returned.

We also tried a cross encoder reranker on top. It did not improve the score here
and was slower, so the final pipeline uses only these two signals.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from embed import embed_queries
from index import load_index
from lexical import LexicalIndex
from utils import K_EVAL

DENSE_DEPTH = 2000          # how many view vectors to pull from FAISS per query
# A page's dense score is the sum of its 3 best view matches. We use 3 because a
# page usually matches well on its title/lead/full views; adding more matches
# only brings in noise from body chunks. The exact weights are not sensitive.
HIT_WEIGHTS = (1.0, 0.5, 0.3)
LEXICAL_WEIGHT = 0.75       # how much the keyword score counts in the sum
CONSENSUS_BONUS = 0.7       # extra reward for pages that score well in both

_CACHE: Optional[Tuple["object", Dict, LexicalIndex]] = None


def _get_index(artifacts_dir: Optional[Path]):
    global _CACHE
    if _CACHE is None:
        _CACHE = load_index(artifacts_dir)
    return _CACHE


def _dense_page_scores(
    sims: np.ndarray, idxs: np.ndarray, page_ids: List[int]
) -> Dict[int, float]:
    #Turn view matches into one score per page (sum of a page's best hits).
    hits: Dict[int, List[float]] = {}
    for sim, vi in zip(sims.tolist(), idxs.tolist()):
        if vi < 0:
            continue
        hits.setdefault(page_ids[vi], []).append(float(sim))
    scores: Dict[int, float] = {}
    for pid, values in hits.items():
        values.sort(reverse=True)
        scores[pid] = sum(w * values[i] for i, w in enumerate(HIT_WEIGHTS) if i < len(values))
    return scores


def _bymax(scores: Dict[int, float]) -> Dict[int, float]:
    #Scale scores to 0-1 by dividing by the highest score for this query.
    if not scores:
        return {}
    hi = max(scores.values()) or 1.0
    return {pid: s / hi for pid, s in scores.items()}


def search_batch(
    queries: List[str],
    *,
    top_k: int = K_EVAL,
    artifacts_dir: Optional[Path] = None,
) -> List[List[int]]:
    index, meta, lexical = _get_index(artifacts_dir)
    page_ids = meta["page_ids"]

    qvecs = embed_queries(queries)
    if qvecs.size == 0:
        return [[] for _ in queries]
    qvecs = np.ascontiguousarray(qvecs.astype(np.float32, copy=False))
    depth = min(DENSE_DEPTH, index.ntotal)
    sims, idxs = index.search(qvecs, depth)

    results: List[List[int]] = []
    for q, row_sim, row_idx in zip(queries, sims, idxs):
        dense = _bymax(_dense_page_scores(row_sim, row_idx, page_ids))
        sparse = _bymax(lexical.score(q))
        fused: Dict[int, float] = {}
        for pid in set(dense) | set(sparse):
            d, s = dense.get(pid, 0.0), sparse.get(pid, 0.0)
            # add the two scores, plus a bonus when both are high (d*s is only
            # large if neither score is near zero).
            fused[pid] = d + LEXICAL_WEIGHT * s + CONSENSUS_BONUS * (d * s) ** 0.5
        results.append(sorted(fused, key=fused.get, reverse=True)[:top_k])
    return results

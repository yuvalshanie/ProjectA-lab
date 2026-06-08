"""Lexical (BM25) retrieval stage — pure numpy + standard library.

Dense MiniLM similarity blurs exact tokens (years, populations, scores, proper
names); BM25 matches them directly, which is what rescues the hard multi-fact
queries. We precompute BM25 *postings with the term weights already baked in*
(the weight of a (term, doc) pair does not depend on the query), prune
near-stopword terms, and cap each term to its strongest postings. The result is
a compact inverted index (~22 MB) that ships in ``artifacts/`` and is scored at
query time with a sparse scatter-add.

Artifacts written/read:
    artifacts/bm25_postings.npz   indptr (int64), docs (int32), weights (float16)
    artifacts/bm25_vocab.txt      one term per line (row == column id)
    artifacts/bm25_meta.json      {num_docs, k1, b, df_max_ratio, postings_cap}
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils import (
    ARTIFACTS_DIR,
    BM25_B,
    BM25_DF_MAX_RATIO,
    BM25_K1,
    BM25_POSTINGS_CAP,
    ensure_artifacts_dir,
)

POSTINGS_NAME = "bm25_postings.npz"
VOCAB_NAME = "bm25_vocab.txt"
META_NAME = "bm25_meta.json"

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokenization (shared by build and query time)."""
    return _TOKEN.findall(text.lower())


def build_lexical(
    texts: Sequence[str],
    *,
    artifacts_dir: Optional[Path] = None,
) -> None:
    """Build and persist the pruned/capped BM25 inverted index for ``texts``.

    ``texts[i]`` must correspond to the same document row ``i`` as the dense
    index (both are built from ``utils.iter_entries`` order).
    """
    out_dir = artifacts_dir or ensure_artifacts_dir()
    n_docs = len(texts)
    k1, b = BM25_K1, BM25_B
    df_max = int(BM25_DF_MAX_RATIO * n_docs)

    doc_tokens = [tokenize(t) for t in texts]
    doc_len = np.array([len(t) for t in doc_tokens], dtype=np.float32)
    avgdl = float(doc_len.mean()) if n_docs else 0.0

    term_freq: List[Dict[str, int]] = []
    doc_freq: Dict[str, int] = defaultdict(int)
    for tokens in doc_tokens:
        tf: Dict[str, int] = defaultdict(int)
        for w in tokens:
            tf[w] += 1
        term_freq.append(tf)
        for w in tf:
            doc_freq[w] += 1

    # Precompute weighted postings per surviving term.
    postings: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for doc_id, tf in enumerate(term_freq):
        length_norm = k1 * (1.0 - b + b * doc_len[doc_id] / avgdl) if avgdl else k1
        for w, c in tf.items():
            n = doc_freq[w]
            if n > df_max:  # near-stopword: ~zero idf, huge posting list -> drop
                continue
            idf = math.log(1.0 + (n_docs - n + 0.5) / (n + 0.5))
            postings[w].append((doc_id, idf * (c * (k1 + 1.0)) / (c + length_norm)))

    vocab: List[str] = []
    col_docs: List[np.ndarray] = []
    col_weights: List[np.ndarray] = []
    for w, plist in postings.items():
        if len(plist) > BM25_POSTINGS_CAP:
            plist = sorted(plist, key=lambda x: -x[1])[:BM25_POSTINGS_CAP]
        vocab.append(w)
        col_docs.append(np.fromiter((p[0] for p in plist), dtype=np.int32, count=len(plist)))
        col_weights.append(np.fromiter((p[1] for p in plist), dtype=np.float32, count=len(plist)))

    indptr = np.zeros(len(vocab) + 1, dtype=np.int64)
    for i, d in enumerate(col_docs):
        indptr[i + 1] = indptr[i] + d.size
    docs = np.concatenate(col_docs) if col_docs else np.zeros(0, dtype=np.int32)
    weights = (
        np.concatenate(col_weights).astype(np.float16)
        if col_weights
        else np.zeros(0, dtype=np.float16)
    )

    np.savez_compressed(out_dir / POSTINGS_NAME, indptr=indptr, docs=docs, weights=weights)
    (out_dir / VOCAB_NAME).write_text("\n".join(vocab), encoding="utf-8")
    (out_dir / META_NAME).write_text(
        json.dumps(
            {
                "num_docs": n_docs,
                "k1": k1,
                "b": b,
                "df_max_ratio": BM25_DF_MAX_RATIO,
                "postings_cap": BM25_POSTINGS_CAP,
                "vocab_size": len(vocab),
                "nnz": int(docs.size),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


class LexicalIndex:
    """Loads the BM25 postings and scores query batches by sparse scatter-add."""

    def __init__(self, artifacts_dir: Optional[Path] = None) -> None:
        root = artifacts_dir or ARTIFACTS_DIR
        npz = np.load(root / POSTINGS_NAME)
        self.indptr = npz["indptr"]
        self.docs = npz["docs"]
        self.weights = npz["weights"].astype(np.float32)
        vocab = (root / VOCAB_NAME).read_text(encoding="utf-8").split("\n")
        # An empty file splits to [""]; guard against that edge case.
        if vocab == [""]:
            vocab = []
        self.term_id: Dict[str, int] = {w: i for i, w in enumerate(vocab)}
        self.num_docs = int(json.loads((root / META_NAME).read_text(encoding="utf-8"))["num_docs"])

    def scores(self, queries: Sequence[str]) -> np.ndarray:
        """Return BM25 scores, shape (len(queries), num_docs)."""
        out = np.zeros((len(queries), self.num_docs), dtype=np.float32)
        for i, q in enumerate(queries):
            row = out[i]
            for w in set(tokenize(q)):
                j = self.term_id.get(w)
                if j is None:
                    continue
                a, b = self.indptr[j], self.indptr[j + 1]
                np.add.at(row, self.docs[a:b], self.weights[a:b])
        return out

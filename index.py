"""Build the index files (offline) and load them (at query time).

build_index() reads the corpus and writes three files to artifacts:
dense_vectors.npy: the embedding of every page view (float16)
index_meta.json: the page id for each vector, plus model name and size
lexical.json.gz: the keyword (inverted) index

load_index() reads these back. It must not re-embed or rebuild anything, because
it runs inside the timed run() call.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import faiss
import numpy as np

from chunk import Chunk, chunk_corpus
from embed import embed_texts
from lexical import LexicalIndex, build_lexical, save_lexical
from utils import ARTIFACTS_DIR, EMBEDDING_MODEL_NAME, ensure_artifacts_dir, iter_entries

DENSE_NAME = "dense_vectors.npy"   #float16 vectors, rebuilt into FAISS at load
META_NAME = "index_meta.json"
LEXICAL_NAME = "lexical.json.gz"


def _build_dense(records: List[Dict[str, Any]], out_dir: Path) -> None:
    #Embed every page view and save the vectors plus the page-id list.
    chunks: List[Chunk] = chunk_corpus(records)
    print(f"[build] {len(records)} pages -> {len(chunks)} views; embedding...")
    vectors = embed_texts([c.text for c in chunks], show_progress_bar=True)
    # float16 halves the file size so it stays under GitHub's 100 MB limit; the
    # rounding does not affect ranking. We rebuild a float32 FAISS index at load.
    np.save(out_dir / DENSE_NAME, vectors.astype(np.float16))
    meta = {
        "page_ids": [c.page_id for c in chunks],
        "model": EMBEDDING_MODEL_NAME,
        "num_vectors": len(chunks),
        "dim": int(vectors.shape[1]),
    }
    (out_dir / META_NAME).write_text(json.dumps(meta), encoding="utf-8")


def _build_sparse(records: List[Dict[str, Any]], out_dir: Path) -> None:
    print("[build] building lexical index...")
    save_lexical(build_lexical(records), out_dir / LEXICAL_NAME)


def build_index(
    *,
    entries_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
) -> None:
    #Build all artifacts that run() needs. Offline only, not timed at grading.
    out_dir = artifacts_dir or ensure_artifacts_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    records = list(iter_entries(entries_dir))
    _build_dense(records, out_dir)
    _build_sparse(records, out_dir)
    print(f"[build] done. artifacts in {out_dir}")


def load_index(
    artifacts_dir: Optional[Path] = None,
) -> Tuple["faiss.Index", Dict[str, Any], LexicalIndex]:
    #Load the dense vectors (as a FAISS index), the metadata, and the lexical index.
    root = artifacts_dir or ARTIFACTS_DIR
    vectors = np.load(root / DENSE_NAME).astype(np.float32)
    index = faiss.IndexFlatIP(int(vectors.shape[1]))
    index.add(vectors)
    meta = json.loads((root / META_NAME).read_text(encoding="utf-8"))
    meta["page_ids"] = [int(x) for x in meta["page_ids"]]
    lexical = LexicalIndex(root / LEXICAL_NAME)
    return index, meta, lexical

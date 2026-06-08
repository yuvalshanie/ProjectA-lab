# Section B — Wikipedia Retrieval Pipeline

End-to-end retrieval over a corpus of ~27k Wikipedia-style pages. Given a batch
of natural-language queries, `run(queries)` returns, for each query, a ranked
list of `page_id`s (most relevant first). Only the top-10 per query are scored
(mean NDCG@10, binary relevance).

Retrieval is a **hybrid of dense (MiniLM) and lexical (BM25)** scoring. All
embeddings use **`sentence-transformers/all-MiniLM-L6-v2`** (assignment
requirement); BM25 is implemented from scratch with `numpy` + the standard
library. The only third-party dependencies are `numpy`, `sentence-transformers`,
and `faiss-cpu`.

**Public self-test result: mean NDCG@10 = 0.277** (dense-only baseline: 0.224).

---

## Pipeline

The pipeline is split into small, single-responsibility modules:

| Stage    | File          | What it does |
|----------|---------------|--------------|
| Chunk    | `chunk.py`    | Turns each page into retrieval unit(s). We use **one unit per page** = `title + "\n\n" + content` (see *Design decisions*). |
| Embed    | `embed.py`    | Encodes text with all-MiniLM-L6-v2, L2-normalized, at `max_seq_length=256` (chosen empirically — see below). |
| Index    | `index.py`    | Offline: embeds the whole corpus, builds the BM25 index, and writes `artifacts/`. Also loads the dense part back. |
| Lexical  | `lexical.py`  | Offline: builds a pruned/capped BM25 inverted index (numpy + stdlib). Query-time: scores a query batch against it. |
| Retrieve | `retrieve.py` | Timed: embeds the queries, scores them with **dense cosine + BM25**, min-max normalizes each signal per query, fuses them (`0.6·dense + 0.4·bm25`), and returns the top-10 page_ids. |
| Entry    | `main.py`     | `run(queries)` → `retrieve.search_batch(queries)`. |

`utils.py` holds shared paths/constants; `eval.py` (read-only) computes NDCG@10.

> **Why not FAISS here?** Exact dense search is a single `(50 × 27074 × 384)`
> matmul — well under a second — so an approximate index would only cost recall.
> `faiss-cpu` stays in `requirements.txt` (it is in the sanctioned dependency
> set) but is unnecessary at this corpus size.

### Why this design

The corpus mixes long real-Wikipedia "distractor" articles (median ~7.6k chars)
with **short synthetic answer pages** (the relevant pages have a median length
of ~1k chars; 79% are under 2k chars). The queries are abstract "riddle-style"
descriptions that rarely name the target entity.

**Step 1 — the dense representation.** Because the answer pages are short, a
single page vector represents them faithfully. Among MiniLM-only options the
native window (256) was best; everything fancier added noise:

| Dense strategy (all MiniLM-only)                    | NDCG@10 |
|-----------------------------------------------------|:-------:|
| **Whole-page vector, `max_seq_length=256`**         | **0.224** |
| Whole-page vector, `max_seq_length=384` / `512`     | 0.213 / 0.217 |
| 256 / 512 score ensembles                           |  ≤0.220 |
| Title-only vector                                   |  0.008  |
| Whole + title ensemble                              |  ≤0.13  |
| Pseudo-relevance feedback (query expansion)         |  ≤0.22  |
| Query decomposition (clause sub-queries)            |  ≤0.21  |
| Document-neighbor / cluster expansion               |  ≤0.20  |

A diagnostic shows why dense plateaus: **71/100** relevant pages land in the
top-100 but only **26/100** in the top-10 — recall is fine, ranking *precision*
is the bottleneck. **Single-answer** queries score ~0.34, while **multi-fact
"what links X, Y and Z"** queries (half the set) score only ~0.10 and dominate
the gap.

**Step 2 — add lexical (BM25), the real win.** Those queries carry exact tokens
(`1,456,779 residents`, `September 1958`, `seven-game`, proper names) that dense
embeddings blur but term matching nails. Fusing the two signals — each min-max
normalized per query, then `0.6·dense + 0.4·bm25` — gives the final result:

| Final retriever                          | NDCG@10 | single | multi |
|------------------------------------------|:-------:|:------:|:-----:|
| Dense only                               |  0.224  | 0.345  | 0.104 |
| BM25 only                                |  0.224  | 0.274  | 0.174 |
| **Hybrid dense + BM25 (final)**          | **0.277** | 0.350 | 0.183 |

The BM25 parameters are textbook defaults (`k1=1.5`, `b=0.75`) and the fusion
weight is flat across 0.5–0.65, so the gain is not an overfit to the public set.
Capping each term to its 200 strongest postings both shrinks the index and
removes noisy low-weight matches (it slightly *raised* quality). The whole index
stays small enough to commit **without Git LFS**.

---

## Artifacts (required — committed to the repo)

The grader does **not** rebuild the index; it loads these files from disk:

| Path                            | Format | Contents |
|---------------------------------|--------|----------|
| `artifacts/index_vectors.npy`   | `float32` `(num_pages, 384)` | L2-normalized page embeddings (~41 MB). |
| `artifacts/index_meta.json`     | JSON   | `page_ids` (row → page_id), `chunk_ids`, `model`, `max_seq_length`, `embed_dim`, `num_vectors`. |
| `artifacts/bm25_postings.npz`   | npz    | BM25 inverted index: `indptr`, `docs` (int32), `weights` (float16). ~22 MB. |
| `artifacts/bm25_vocab.txt`      | text   | One term per line; line number = column id used by `indptr`. ~4 MB. |
| `artifacts/bm25_meta.json`      | JSON   | `num_docs`, `k1`, `b`, pruning params, `vocab_size`, `nnz`. |

Row `i` of `index_vectors.npy` and BM25 document id `i` both correspond to
`page_ids[i]` (both are built from the same corpus iteration order). Every file
is well under GitHub's 100 MB per-file limit, so **no Git LFS is required** and a
fresh clone runs out of the box.

> The raw corpus (`data/Wikipedia Entries/`, ~430 MB) is **git-ignored**: it is
> only needed to *rebuild* the index, never at grading time. The 50 labelled
> public queries (`data/public_queries.json`) are committed so `eval_public.py`
> runs on a fresh clone.

---

## Setup & usage

```bash
pip install -r requirements.txt
```

**Evaluate (uses the committed index — no rebuild needed):**

```bash
python scripts/eval_public.py
# -> public_queries=50 / mean_ndcg@10=0.2765 / query_phase_time≈2s
```

**Rebuild the index from the corpus (offline, our machine only):**
Place the corpus under `data/Wikipedia Entries/` (one JSON per page), then:

```bash
python scripts/build_index.py     # writes artifacts/
```

---

## Repository layout

```
main.py            run(queries) entry point (called by the grader)
chunk.py           page -> retrieval unit(s)
embed.py           all-MiniLM-L6-v2 encoder (shared by corpus & queries)
lexical.py         BM25 inverted index: offline build + query-time scoring
index.py           offline build (dense + BM25) + load of artifacts/
retrieve.py        hybrid dense+BM25 scoring and fusion -> top-10 page_ids
utils.py           paths, constants, corpus iteration
eval.py            NDCG@10 (read-only)
scripts/
  build_index.py   offline index build (read-only)
  eval_public.py   public self-test (read-only)
artifacts/         committed prebuilt index, loaded at grading time:
  index_vectors.npy / index_meta.json     dense MiniLM page embeddings
  bm25_postings.npz / bm25_vocab.txt / bm25_meta.json   BM25 lexical index
data/
  public_queries.json   50 labelled public queries
  Wikipedia Entries/    full corpus (git-ignored; rebuild input only)
requirements.txt
```

---

## Video

Presentation video: **<ADD VIDEO LINK HERE>**

## Team

- ID 315335315
- ID 212134845

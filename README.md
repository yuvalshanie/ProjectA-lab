# Section B — Wikipedia Retrieval Pipeline

End-to-end retrieval over a corpus of ~27k Wikipedia-style pages. Given a batch
of natural-language queries, `run(queries)` returns, for each query, a ranked
list of `page_id`s (most relevant first). Only the top-10 per query are scored
(mean NDCG@10, binary relevance).

All embeddings use **`sentence-transformers/all-MiniLM-L6-v2`** (assignment
requirement). The only third-party dependencies are `numpy`,
`sentence-transformers`, and `faiss-cpu`.

---

## Pipeline

The pipeline is split into small, single-responsibility modules:

| Stage    | File          | What it does |
|----------|---------------|--------------|
| Chunk    | `chunk.py`    | Turns each page into retrieval unit(s). We use **one unit per page** = `title + "\n\n" + content` (see *Design decisions*). |
| Embed    | `embed.py`    | Encodes text with all-MiniLM-L6-v2, L2-normalized, at `max_seq_length=256` (chosen empirically — see below). |
| Index    | `index.py`    | Offline: embeds the whole corpus and writes `artifacts/`. Also loads them back. |
| Retrieve | `retrieve.py` | Timed: embeds the queries and runs exact inner-product (cosine) search with a **FAISS `IndexFlatIP`**, aggregating to the page level (best unit per page) and returning the top-10 page_ids. |
| Entry    | `main.py`     | `run(queries)` → `retrieve.search_batch(queries)`. |

`utils.py` holds shared paths/constants; `eval.py` (read-only) computes NDCG@10.

### Why this design

The corpus mixes long real-Wikipedia "distractor" articles (median ~7.6k chars)
with **short synthetic answer pages** (the relevant pages have a median length
of ~1k chars; 79% are under 2k chars). Because the answer pages are short, a
single extended-context vector represents them faithfully. We measured several
alternatives on the 50 public queries (mean NDCG@10) and kept the simplest thing
that worked best:

| Strategy (all MiniLM-only)                          | NDCG@10 |
|-----------------------------------------------------|:-------:|
| **Whole-page vector, `max_seq_length=256` (final)** | **0.224** |
| Whole-page vector, `max_seq_length=384`             |  0.213  |
| Whole-page vector, `max_seq_length=512`             |  0.217  |
| 256 / 512 score ensembles                           |  ≤0.220 |
| Title-only vector                                   |  0.008  |
| Whole + title ensemble                              |  ≤0.13  |
| Pseudo-relevance feedback (query expansion)         |  ≤0.22  |
| Query decomposition (clause sub-queries)            |  ≤0.21  |
| Document-neighbor / cluster expansion               |  ≤0.20  |

A diagnostic explains why: under the whole-page index, **71/100** relevant pages
land in the top-100 but only **26/100** in the top-10 — recall is fine, ranking
precision is the bottleneck. Splitting by query type, **single-answer** queries
score ~0.34 while **multi-fact "what links X, Y and Z"** queries (half the set)
score only ~0.10 and dominate the gap.

Take-aways that shaped the final design:
- The queries are abstract "riddle-style" descriptions that rarely name the
  entity, so **title-only** and **query-expansion** tricks add noise.
- The multi-fact queries are the hard part; generic expansion and
  cluster-neighbor reranking helped them slightly but hurt the (more numerous)
  single-answer queries more, so every variant lost overall.
- A **larger context window** helps the multi-fact queries a little but dilutes
  the short single-answer pages; native **256** was the best single-vector
  representation, and it keeps the index small enough to commit **without Git
  LFS**.

---

## Artifacts (required — committed to the repo)

The grader does **not** rebuild the index; it loads these files from disk:

| Path                          | Format | Contents |
|-------------------------------|--------|----------|
| `artifacts/index_vectors.npy` | `float32` `(num_pages, 384)` | L2-normalized page embeddings. |
| `artifacts/index_meta.json`   | JSON   | `page_ids` (row → page_id), `chunk_ids`, `model`, `max_seq_length`, `embed_dim`, `num_vectors`. |

Row `i` of `index_vectors.npy` corresponds to `page_ids[i]` in the metadata.
Total size is well under GitHub's 100 MB per-file limit, so **no Git LFS is
required** and a fresh clone runs out of the box.

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
# -> public_queries=50 / mean_ndcg@10=0.2241 / query_phase_time≈2s
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
index.py           offline build + load of artifacts/
retrieve.py        FAISS inner-product search + page-level aggregation
utils.py           paths, constants, corpus iteration
eval.py            NDCG@10 (read-only)
scripts/
  build_index.py   offline index build (read-only)
  eval_public.py   public self-test (read-only)
artifacts/         committed prebuilt index (loaded at grading time)
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

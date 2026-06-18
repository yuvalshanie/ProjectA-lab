# Section B - Retrieval pipeline

This project answers queries over about 27,000 Wikipedia-style pages. The
function "run(queries)" returns, for each query, a list of page ids ordered from
most to least relevant. Only the top 10 per query are scored (mean NDCG@10).

## How to run

```bash
pip install -r requirements.txt     # see the GPU notes inside requirements.txt
python scripts/build_index.py       # build the index (offline, a few minutes)
python scripts/eval_public.py       # score on the public queries (no rebuild)
```

'build_index.py' writes the files in 'artifacts/'. After that, 'eval_public.py'
only loads those files, so a fresh clone runs without rebuilding. One call to
'run()' on 50 queries finishes in about 14 seconds (the limit is 60).

## How it works

We combine two kinds of search and merge their scores for each query.

1. **Chunk** ('chunk.py'). The embedding model only reads the first ~256 tokens
   of its input, so we split every page into several short "views": the title,
   the lead (title + first paragraph), the full page (for short pages), and a
   few overlapping windows of the body. Each view is embedded on its own.

2. **Embed** ('embed.py'). Every view is turned into a vector with the required
   model 'all-MiniLM-L6-v2'. The vectors are L2-normalized.

3. **Index** ('index.py'). All view vectors go into a FAISS index (dense search).
   In parallel we build a keyword index ('lexical.py'): a tf-idf inverted index
   with word pairs/triples, a boost for numbers, and decade tokens (a page that
   mentions 1825 also gets a "1820s" token).

4. **Retrieve** ('retrieve.py'). For each query we get a dense score (sum of a
   page's 3 best view matches) and a keyword score. We scale each to 0–1, add
   them (keyword weight 0.75), and add a small bonus for pages that score well
   in both. The top 10 pages are returned.

## Files

| File | What it does |
|---|---|
| 'main.py' | `run(queries)` (called by the autograder) and the offline build entry. |
| 'chunk.py' | Split a page into views. |
| 'embed.py' | Load the model and embed text. |
| 'lexical.py' | Build and score the keyword (inverted) index. |
| 'index.py' | Build the artifacts; load them at query time. |
| 'retrieve.py' | Search, score, and merge the two signals. |
| 'utils.py' | Paths and small helpers (read corpus, read queries). |

## Artifacts (in `artifacts/`, loaded at query time, never rebuilt during grading)

| File | Contents |
|---|---|
| 'dense_vectors.npy' | All view embeddings, saved as float16 (under 100 MB, so no Git LFS). A FAISS index is rebuilt from these at load. |
| 'index_meta.json' | The page id for each vector, the model name, and sizes. |
| 'lexical.json.gz' | The keyword index (page lengths, idf, and postings). |

'run()' uses only these files. It does not read the corpus at query time.

## Results (public queries, mean NDCG@10)

| Setup | NDCG@10 |
|---|---|
| Dense only | 0.22 |
| Keyword only | 0.41 |
| Both combined (final) | 0.49 |

Single-answer questions score about 0.63. The hardest ones are the broad
"what links A, B, and C" questions with many correct pages (about 0.20): their
pages are retrievable but hard to rank into the top 10.

### Things we tried and dropped
- A cross-encoder reranker. It helped a weaker setup but lowered our score here
  (0.49 to 0.43) and was slower, so we did not use it.
- BM25 length normalization. The tf-idf formula with a mild length penalty was
  clearly better here (0.41 vs 0.32 keyword-only).


## Submit
Public GitHub repo with this code and the committed 'artifacts/'.
Video link: https://github.com/yuvalshanie/ProjectA-lab/blob/main/presentation.mp4


"""Keyword search: a small inverted index scored with tf-idf.

The dense embeddings are good at meaning but weak on exact words. Many queries
need an exact match on a year, a number, or a multi-word name, so we add a
keyword search. It uses only the standard library and numpy.

On top of plain tf-idf we add three things:
-phrase features: we also index word pairs and triples, so "memorial arena"
 counts as one feature instead of two separate words.
-digit boost: pure numbers (years, populations) count more, because they
 identify a page strongly.
-decade indexing: a page that mentions a year like 1825 also gets a "1820s"
 token, so a query about "the 1820s" can still match it.

The index is built once and saved as a gzipped JSON file, then loaded at query
time.
"""
from __future__ import annotations

import gzip
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

LENGTH_NORM = 0.002         # how much to penalize long pages (small = mild)
PHRASE_BOOST = 3.0          # extra weight per extra word in a phrase feature
DIGIT_BOOST = 5.0           # extra weight for tokens that are pure numbers
MIN_IDF = 1.4               # ignore very common words at query time
MAX_DF = 3000               # ignore words that appear in too many pages
TITLE_REPEAT = 2            # count the title this many times (gives it weight)
MAX_DOC_WORDS = 1200        # only index the first part of very long pages

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "and", "for", "that", "with", "from", "this", "what", "when", "where",
    "which", "who", "how", "did", "was", "were", "are", "about", "also", "into",
    "after", "before", "during", "over", "under", "they", "their", "its", "his",
    "her", "can", "had", "has", "have", "been", "of", "to", "in", "on", "at",
    "by", "as", "an",
}


def tokenize(text: str) -> List[str]:
    """Lowercase the text and split into words, dropping stopwords and 1-letter tokens."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 2 and t not in _STOPWORDS]


def _ngram_features(tokens: List[str]) -> List[str]:
    feats = list(tokens)
    feats += [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
    feats += [f"{tokens[i]}_{tokens[i+1]}_{tokens[i+2]}" for i in range(len(tokens) - 2)]
    return feats


def _decade_tokens(tokens: List[str]) -> List[str]:
    """For each 4-digit year token (e.g. 1825) emit its decade token (1820s)."""
    out: List[str] = []
    for t in tokens:
        if len(t) == 4 and t.isdigit() and 1000 <= int(t) <= 2099:
            out.append(t[:3] + "0s")
    return out


def query_features(query: str) -> List[str]:
    """Query features: single words plus word pairs and triples (no duplicates).

    Decades are handled on the document side at build time, so the query needs
    no special handling here."""
    return list(dict.fromkeys(_ngram_features(tokenize(query))))


def build_lexical(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build the inverted index from all pages.

    Returns a dict with: the page id and length of every document, and for every
    feature its idf and the list of (document, count) pairs (its postings)."""
    docs: List[Dict[str, Any]] = []
    postings: Dict[str, Dict[int, int]] = {}
    for i, rec in enumerate(records):
        title = str(rec.get("title", ""))
        body = " ".join(str(rec.get("content", "")).split()[:MAX_DOC_WORDS])
        text = (" " + title) * TITLE_REPEAT + " " + body
        tokens = tokenize(text)
        counts: Dict[str, int] = {}
        for feat in _ngram_features(tokens) + _decade_tokens(tokens):
            counts[feat] = counts.get(feat, 0) + 1
        docs.append({"page_id": int(rec["page_id"]), "length": max(1, sum(counts.values()))})
        for feat, c in counts.items():
            postings.setdefault(feat, {})[i] = c

    n_docs = len(docs)
    terms: Dict[str, Any] = {}
    for feat, posting in postings.items():
        df = len(posting)
        if (df <= 1 and not feat.isdigit()) or df > MAX_DF:
            continue
        idf = math.log((n_docs + 1) / (df + 0.5))
        terms[feat] = [round(idf, 6), list(posting.items())]
    avgdl = sum(d["length"] for d in docs) / max(1, n_docs)
    return {"docs": docs, "terms": terms, "avgdl": round(avgdl, 3)}


def save_lexical(index: Dict[str, Any], path: Path) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(index, f, separators=(",", ":"))


class LexicalIndex:
    """The inverted index loaded at query time, with a method to score a query."""

    def __init__(self, path: Path):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        self.docs = data["docs"]
        self.terms = data["terms"]
        self.avgdl = float(data.get("avgdl", 1.0))

    def score(self, query: str) -> Dict[int, float]:
        """Return a keyword score for every page that shares a feature with the query."""
        scores: Dict[int, float] = {}
        for feat in query_features(query):
            item = self.terms.get(feat)
            if item is None:
                continue
            idf, postings = item
            idf = float(idf)
            if idf < MIN_IDF:
                continue
            parts = feat.count("_") + 1
            boost = DIGIT_BOOST if feat.isdigit() else 1.0 + PHRASE_BOOST * (parts - 1)
            for doc_idx, count in postings:
                doc = self.docs[int(doc_idx)]
                # tf-idf score: rarer words (idf) and more mentions (count) score
                # higher, divided by a mild penalty for long pages.
                weight = idf * (1.0 + math.log1p(float(count))) / (1.0 + LENGTH_NORM * float(doc["length"]))
                pid = int(doc["page_id"])
                scores[pid] = scores.get(pid, 0.0) + boost * weight
        return scores

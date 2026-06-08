"""Chunking stage: turn each corpus page into retrieval unit(s).

Design note
-----------
We evaluated fixed-size and overlapping word-window chunking with per-page
max-pooling, but a single unit per page (the full title + content) scored best
on the public queries: the relevant answer pages in this corpus are short, so
one extended-context vector (see ``MAX_SEQ_LENGTH`` in ``utils``) represents a
page faithfully, while multi-chunk indexes mostly added near-duplicate
distractor vectors. We therefore keep one unit per page. The ``Chunk``
abstraction and the page-level aggregation in ``retrieve`` are kept general, so
a multi-chunk strategy can be reintroduced by changing ``chunk_entry`` alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from utils import entry_text


@dataclass
class Chunk:
    """One retrieval unit. ``chunk_id`` distinguishes units within a page."""

    page_id: int
    chunk_id: int
    text: str


def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    """Split one corpus entry into retrieval units (one whole-page unit)."""
    page_id = int(record["page_id"])
    return [Chunk(page_id=page_id, chunk_id=0, text=entry_text(record))]


def chunk_corpus(records: List[Dict[str, Any]]) -> List[Chunk]:
    """Flatten the whole corpus into a list of retrieval units."""
    chunks: List[Chunk] = []
    for record in records:
        chunks.extend(chunk_entry(record))
    return chunks

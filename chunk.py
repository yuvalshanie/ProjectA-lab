"""Split each page into several short pieces ("views") to embed.

The embedding model only reads about the first 256 tokens of its input, so one
vector per page would miss anything written later. Instead we make several
vectors per page:
- title:  just the title
- lead:   title + first paragraph
- full:   the whole page (only for short pages)
- chunks: overlapping windows over the body

This way a query can match a page through any of these pieces. This is the
chunking idea from Lecture 3.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from utils import entry_text

CHUNK_WORDS = 180          # window size (~ MiniLM token budget with the title)
CHUNK_OVERLAP = 45         # keep facts that straddle a window boundary
MAX_BODY_WORDS = 800       # cap work on very long pages
MAX_CHUNKS = 4             # at most this many body windows per page
SHORT_PAGE_WORDS = 1000    # pages this short also get a single "full" view


@dataclass
class Chunk:
    page_id: int
    chunk_id: int
    text: str


def _first_paragraph(content: str) -> str:
    for para in content.replace("\r\n", "\n").split("\n\n"):
        para = para.strip()
        if para:
            return para
    return content.strip()


def _word_windows(words: List[str]) -> List[str]:
    """Split the body into overlapping word windows (at most MAX_CHUNKS of them)."""
    windows: List[str] = []
    step = max(1, CHUNK_WORDS - CHUNK_OVERLAP)
    for start in range(0, min(len(words), MAX_BODY_WORDS), step):
        part = words[start : start + CHUNK_WORDS]
        if len(part) < 30:
            break
        windows.append(" ".join(part))
        if len(windows) >= MAX_CHUNKS or start + CHUNK_WORDS >= len(words):
            break
    return windows


def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    """Produce the multi-view retrieval units for one page."""
    page_id = int(record["page_id"])
    title = str(record.get("title", "")).strip()
    content = str(record.get("content", "")).strip()
    words = content.split()

    chunks: List[Chunk] = []

    def add(text: str) -> None:
        if text.strip():
            chunks.append(Chunk(page_id, len(chunks), text.strip()))

    if title:
        add(title)                                  # title view
    lead = _first_paragraph(content)
    if lead:
        add(f"{title}\n\n{lead}" if title else lead)  # lead view

    if len(words) <= SHORT_PAGE_WORDS:
        add(entry_text(record))                       # full-page view
        for w in _word_windows(words):
            add(f"{title}\n\n{w}" if title else w)     # body-window views
    else:
        # Long page: index its opening window (title-prefixed).
        head = " ".join(words[:CHUNK_WORDS])
        add(f"{title}\n\n{head}" if title else head)

    if not chunks:  # empty-page safety net
        add(entry_text(record) or str(page_id))
    return chunks


def chunk_corpus(records: List[Dict[str, Any]]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for record in records:
        chunks.extend(chunk_entry(record))
    return chunks

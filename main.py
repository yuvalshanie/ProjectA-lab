"""Entry point. The autograder imports this file and calls run(queries).

All loading happens inside run() (the index is cached after the first call), so
everything stays within the time limit.
"""
from __future__ import annotations

from typing import List

from index import build_index
from retrieve import search_batch


def run(queries: List[str]) -> List[List[int]]:
    """Return a ranked list of page ids for each query, best first.

    Only the first 10 ids per query are scored."""
    return search_batch(queries)


def build_offline_index() -> None:
    """Build the artifacts/ files. Run once locally; not timed at grading."""
    build_index()


if __name__ == "__main__":
    build_offline_index()
    print("Index built under artifacts/. Run: python scripts/eval_public.py")

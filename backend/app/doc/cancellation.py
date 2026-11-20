"""In-process cancellation flags shared by API requests and pipeline workers."""

from __future__ import annotations

_cancelled: set[str] = set()


def request(run_id: str) -> None:
    _cancelled.add(run_id)


def requested(run_id: str) -> bool:
    return run_id in _cancelled


def clear(run_id: str) -> None:
    _cancelled.discard(run_id)

"""Thread-pool helpers for running independent model calls concurrently.

Model calls are network-bound and the OpenAI client is thread-safe, so a plain
thread pool is enough. Callbacks (``log_fn``) often touch Streamlit, which only
works on the script thread, so both helpers deliver results/events on the
caller's thread.

``MAX_CONCURRENCY`` bounds in-flight HTTP calls process-wide via
:func:`call_slot`, so nested pools (seeds × calibration samples) never exceed
it. ``MAX_CONCURRENCY=1`` restores fully sequential behaviour.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from contextlib import AbstractContextManager
from typing import TypeVar

from frontier_lens.config import MAX_CONCURRENCY

T = TypeVar("T")
R = TypeVar("R")

_call_slots = threading.BoundedSemaphore(MAX_CONCURRENCY)


def call_slot() -> AbstractContextManager:
    """Hold one of the process-wide model-call slots for the duration of a request."""
    return _call_slots


def _workers(n_items: int, max_workers: int | None) -> int:
    return max(1, min(max_workers or MAX_CONCURRENCY, n_items))


def map_unordered(
    fn: Callable[[T], R],
    items: Iterable[T],
    max_workers: int | None = None,
) -> Iterator[tuple[int, R]]:
    """Run *fn* over *items* concurrently; yield ``(index, result)`` as each finishes.

    Results are yielded on the caller's thread. An exception from *fn* is raised
    at the yield point; closing the iterator early cancels work not yet started.
    """
    items = list(items)
    if not items:
        return
    workers = _workers(len(items), max_workers)
    if workers == 1:
        for i, item in enumerate(items):
            yield i, fn(item)
        return
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {pool.submit(fn, item): i for i, item in enumerate(items)}
        for fut in as_completed(futures):
            yield futures[fut], fut.result()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def map_with_events(
    fn: Callable[[T, Callable[..., None]], R],
    items: Iterable[T],
    log_fn: Callable[..., None] | None = None,
    max_workers: int | None = None,
) -> list[R]:
    """Run ``fn(item, emit)`` concurrently and return results in input order.

    Workers call ``emit(event, **fields)``; events are queued and forwarded to
    *log_fn* on the caller's thread so Streamlit callbacks stay safe.
    """
    items = list(items)
    if not items:
        return []

    def _noop(event: str, **fields) -> None:
        pass

    if _workers(len(items), max_workers) == 1:
        return [fn(item, log_fn or _noop) for item in items]

    events: queue.Queue = queue.Queue()

    def emit(event: str, **fields) -> None:
        events.put((event, fields))

    def drain() -> None:
        while True:
            try:
                event, fields = events.get_nowait()
            except queue.Empty:
                return
            if log_fn:
                log_fn(event, **fields)

    results: list = [None] * len(items)
    pool = ThreadPoolExecutor(max_workers=_workers(len(items), max_workers))
    try:
        futures = {pool.submit(fn, item, emit): i for i, item in enumerate(items)}
        pending = set(futures)
        while pending:
            done, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
            drain()
            for fut in done:
                results[futures[fut]] = fut.result()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        drain()
    return results

import threading
import time

import pytest

from frontier_lens.parallel import map_unordered, map_with_events


def test_map_unordered_runs_concurrently_and_indexes_results():
    def slow_square(x: int) -> int:
        time.sleep(0.3)
        return x * x

    t0 = time.perf_counter()
    out = dict(map_unordered(slow_square, range(4), max_workers=4))
    assert out == {0: 0, 1: 1, 2: 4, 3: 9}
    assert time.perf_counter() - t0 < 0.9  # sequential would take 1.2s


def test_map_unordered_propagates_errors():
    def boom(x: int) -> int:
        if x == 2:
            raise ValueError("bad")
        return x

    with pytest.raises(ValueError):
        list(map_unordered(boom, range(4), max_workers=2))


def test_map_with_events_forwards_on_caller_thread_in_input_order():
    caller = threading.get_ident()
    seen: list[tuple[str, int, bool]] = []

    def log_fn(event: str, **fields):
        seen.append((event, fields["i"], threading.get_ident() == caller))

    def work(i: int, emit) -> int:
        time.sleep(0.05 * (3 - i))
        emit("done", i=i)
        return i * 10

    results = map_with_events(work, range(3), log_fn, max_workers=3)
    assert results == [0, 10, 20]
    assert sorted(i for _, i, _ in seen) == [0, 1, 2]
    assert all(on_caller for _, _, on_caller in seen)


def test_map_with_events_sequential_when_one_worker():
    order: list[int] = []
    results = map_with_events(lambda i, emit: order.append(i) or i, range(3), max_workers=1)
    assert results == [0, 1, 2]
    assert order == [0, 1, 2]

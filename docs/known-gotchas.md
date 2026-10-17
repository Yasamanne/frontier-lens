# Known gotchas

These hand-written Python items used to sit in the seed bank as hidden-verifier pitfalls. They are **not** used for training or eval. The seed bank is the procedural generators under `src/frontier_lens/templates/`.

They are short, named language traps (mutable defaults, C3, aliases, live-list iteration). That is the opposite of what the generators do: difficulty comes from step length, the stem shows the program or the problem, and the answer is a large-space value.

## Removed items

1. Class `D(B, C)` with `super()` along the C3 order. The stem named C3 and asked for `D().f()`.
2. A shared mutable default list plus `try`/`else`/`finally`. The stem described the default and the clauses.
3. `for x in xs` appending to the live list when `x` is odd.
4. `defaultdict(list)` where a lookup creates a missing key.
5. Slice assignment `a[1:5] = a[3:7]` through an alias.
6. `deque.rotate` and `extendleft`.
7. `Counter.subtract` allowing negative counts.
8. A small `itertools.product` enumeration.
9. Collatz step counts from 1 through 40.
10. A deque elimination loop.
11. A default argument aliased to a list, plus a chained comparison.
12. Python floor division and remainder with mixed signs.

The verifier snippets that used to accompany those stems were deleted from `python_state.py` with the lists. Do not paste them back into `specs()`.

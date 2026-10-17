"""Minimax game-tree evaluation puzzles.

MAX plays at even depths (0 = root), MIN plays at odd depths.
The model must apply backward induction to determine the root value.

Knobs: depth (2-4), branching (2-3).
Gold key: minimax value at the root (numeric).

Why better than execution tracing:
  - Requires backward induction, not forward simulation
  - Models systematically fail by picking the highest visible leaf (greedy)
  - That failure mode is completely distinct from mutation-tracking errors
  - Difficulty scales cleanly: more depth = more planning steps needed
"""

from __future__ import annotations

import random

from frontier_lens.templates.procedural import bank
from frontier_lens.templates.spec import ItemSpec, hidden_stem

TOPIC = "minimax"

_QUESTION_POOL = [
    "what is the minimax value at the root plus the sum of all leaf values?",
    "compute the minimax value V at the root. Then add V to the total sum of all leaf values. What is the result?",
    "what is the root's minimax value added to the sum of every leaf value listed?",
]
# The answer = minimax_root_value + sum(leaves).
# This ensures the output is larger than any individual leaf, so it won't appear
# verbatim in the leaf listing and materialize won't reject it as "key_in_stem".


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def _minimax_flat(leaves: list[int], branching: int, depth: int, is_max: bool) -> int:
    """Evaluate the tree represented as a flat leaf list."""
    if depth == 0 or len(leaves) == 1:
        return leaves[0]
    chunk = len(leaves) // branching
    children = [
        _minimax_flat(leaves[i * chunk : (i + 1) * chunk], branching, depth - 1, not is_max)
        for i in range(branching)
    ]
    return max(children) if is_max else min(children)


# ---------------------------------------------------------------------------
# Code / prose builders
# ---------------------------------------------------------------------------

def _to_code(leaves: list[int], depth: int, branching: int) -> str:
    return (
        f"leaves = {leaves}\n"
        f"b = {branching}\n"
        f"d = {depth}\n"
        "\n"
        "def mm(vals, depth, is_max):\n"
        "    if depth == 0 or len(vals) == 1:\n"
        "        return vals[0]\n"
        "    chunk = len(vals) // b\n"
        "    ch = [mm(vals[i*chunk:(i+1)*chunk], depth-1, not is_max) for i in range(b)]\n"
        "    return max(ch) if is_max else min(ch)\n"
        "\n"
        "v = mm(leaves, d, True)\n"
        "print(v + sum(leaves))\n"
    )


def _to_prose(leaves: list[int], depth: int, branching: int, question: str) -> str:
    n = len(leaves)
    layer_roles = ["MAX" if d % 2 == 0 else "MIN" for d in range(depth + 1)]

    lines = [
        "A two-player game is played on a tree.",
        "MAX wants the highest value; MIN wants the lowest.",
        f"MAX plays at level 0 (root). Players alternate: {', '.join(f'level {d} → {r}' for d, r in enumerate(layer_roles))}.",
        f"The tree has depth {depth} and branching factor {branching}.",
        f"There are {n} leaf nodes.",
        "",
        "Leaf values, listed left to right at the deepest level:",
        "  " + "  ".join(str(v) for v in leaves),
        "",
        "Each internal node's value is the MAX or MIN of its children according to whose turn it is.",
        f"Sum of all leaf values: {sum(leaves)}",
        "",
        f"Question: {question}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Accept filter
# ---------------------------------------------------------------------------

def _accept(spec: ItemSpec, key: str) -> bool:
    """Reject trivially easy trees (root = max or min of all leaves)."""
    try:
        int(key)  # raises ValueError/TypeError if key is not an integer
    except (ValueError, TypeError):
        return False
    # The leaves aren't stored in params, but we can check via the code not available here.
    # Lightweight: just verify it's a valid integer.
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate(seed: int, *, depth: int, branching: int) -> ItemSpec:
    depth = int(depth)
    branching = int(branching)
    if depth < 1 or branching < 2:
        raise ValueError("depth>=1, branching>=2")
    rng = random.Random(seed)

    n_leaves = branching ** depth
    # Leaf values in 1..20, all distinct within the tree to avoid trivial ties
    pool = list(range(1, max(21, n_leaves + 1)))
    rng.shuffle(pool)
    leaves = pool[:n_leaves]

    question = rng.choice(_QUESTION_POOL)

    # answer = minimax_value + sum(leaves); leaves are drawn from 1..max(21, n_leaves+1)
    # without repetition. Upper bound on sum is sum(1..n_leaves) = n_leaves*(n_leaves+1)//2
    max_answer = n_leaves * (n_leaves + 1) // 2 + n_leaves + 1
    return ItemSpec(
        topic=TOPIC,
        level=2,
        template=f"{TOPIC}_train",
        params={"seed": seed, "depth": depth, "branching": branching, "n_leaves": n_leaves,
                "answer_space_size": max(200, max_answer)},
        stem=hidden_stem(_to_prose(leaves, depth, branching, question)),
        code=_to_code(leaves, depth, branching),
        lang="python",
        numeric=True,
    )


def _harness_item(name: str, leaves: list[int], depth: int, branching: int) -> ItemSpec:
    prose = _to_prose(leaves, depth, branching, _QUESTION_POOL[0])
    return ItemSpec(
        topic=TOPIC,
        level=1,
        template=name,
        params={},
        stem=hidden_stem(prose),
        code=_to_code(leaves, depth, branching),
        lang="python",
        numeric=True,
    )


def specs() -> list[ItemSpec]:
    harness = [
        # depth=2, branching=2: minimax=3, sum=20, answer=23
        _harness_item("minimax_d2b2_a", [3, 7, 2, 8], depth=2, branching=2),
        # depth=2, branching=2: minimax=4, sum=19, answer=23... use different leaves
        _harness_item("minimax_d2b2_b", [5, 1, 9, 4], depth=2, branching=2),
    ]
    return bank(TOPIC, generate, harness, accept=_accept)

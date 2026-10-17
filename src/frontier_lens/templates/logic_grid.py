"""Positional logic-grid (ordering) puzzles.

N named items are placed in a row at positions 0 … N-1.
The model reads prose clues and deduces the position of one target item.

Knobs: n_items (3-6), n_clue_types (1-4 — higher means more clue variety).
Gold key: 0-indexed position of the queried item (numeric).

Why better than constraint_counting:
  - Requires deductive propagation, not independent per-character checks
  - Constraints interact: knowing A's position restricts B's and C's
  - Systematic failure mode: models apply the first matching clue and stop
  - Very different cognitive skill from counting or execution tracing
"""

from __future__ import annotations

import random
from itertools import permutations

from frontier_lens.templates.procedural import bank
from frontier_lens.templates.spec import ItemSpec, hidden_stem

TOPIC = "logic_grid"

_NAME_SETS = [
    ["Alice", "Bob", "Carol", "Dan", "Eve", "Frank"],
    ["Apple", "Banana", "Cherry", "Date", "Elderberry", "Fig"],
    ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"],
    ["Train", "Plane", "Car", "Ship", "Bus", "Bike"],
    ["Ruby", "Slate", "Amber", "Jade", "Onyx", "Pearl"],
]

# Clue kinds, ordered from simplest to most complex
_CLUE_KINDS = ["exact", "before", "adjacent", "distance", "not_adjacent"]

_CHECKSUM_QUESTION_POOL = [
    "what four-digit checksum does this arrangement produce?",
    "what is the four-digit output of the checksum procedure above?",
    "after determining all positions, what four-digit number does the checksum produce?",
]

_CHECKSUM_DESCRIPTION = """\
After deducing all positions, compute a checksum as follows:
  - Start with checksum = 0.
  - Process items in the order they are listed (top to bottom above).
  - For each item: checksum = (checksum * 31 + item's_position) mod 10000.
Report the checksum as a four-digit zero-padded number (e.g. "0042").\
"""


# ---------------------------------------------------------------------------
# Clue satisfaction
# ---------------------------------------------------------------------------

def _satisfies(perm: tuple[int, ...], clue: dict) -> bool:
    """perm[i] = position of item i."""
    k = clue["kind"]
    i = clue["i"]
    j = clue.get("j", 0)
    if k == "exact":
        return perm[i] == clue["pos"]
    if k == "before":
        return perm[i] < perm[j]
    if k == "adjacent":          # i immediately left of j
        return perm[j] == perm[i] + 1
    if k == "distance":          # i exactly d to the left of j
        return perm[j] - perm[i] == clue["d"]
    if k == "not_adjacent":
        return abs(perm[i] - perm[j]) > 1
    return True


def _satisfies_all(perm: tuple[int, ...], clues: list[dict]) -> bool:
    return all(_satisfies(perm, c) for c in clues)


def _count_solutions(solution: tuple[int, ...], clues: list[dict], n: int) -> int:
    count = 0
    for perm in permutations(range(n)):
        if _satisfies_all(perm, clues):
            count += 1
            if count > 1:
                return count
    return count


# ---------------------------------------------------------------------------
# Clue generation
# ---------------------------------------------------------------------------

def _candidate_clues(rng: random.Random, perm: list[int], n: int, n_clue_types: int) -> list[dict]:
    """Return all clues of the allowed types that are true for perm."""
    kinds = _CLUE_KINDS[:n_clue_types + 1]
    candidates: list[dict] = []
    for i in range(n):
        for k in kinds:
            if k == "exact":
                candidates.append({"kind": "exact", "i": i, "pos": perm[i]})
            elif k == "before":
                for j in range(n):
                    if i != j and perm[i] < perm[j]:
                        candidates.append({"kind": "before", "i": i, "j": j})
            elif k == "adjacent":
                for j in range(n):
                    if i != j and perm[j] == perm[i] + 1:
                        candidates.append({"kind": "adjacent", "i": i, "j": j})
            elif k == "distance":
                for j in range(n):
                    d = perm[j] - perm[i]
                    if 2 <= d <= n - 2:
                        candidates.append({"kind": "distance", "i": i, "j": j, "d": d})
            elif k == "not_adjacent":
                for j in range(i + 1, n):
                    if abs(perm[i] - perm[j]) > 1:
                        candidates.append({"kind": "not_adjacent", "i": i, "j": j})
    rng.shuffle(candidates)
    return candidates


def _generate_clues(rng: random.Random, perm: list[int], n: int, n_clue_types: int) -> list[dict]:
    """Greedy: add clues until the solution is unique."""
    t_perm = tuple(perm)
    clues: list[dict] = []

    # Always start with one exact clue to anchor the search
    anchor = rng.randrange(n)
    clues.append({"kind": "exact", "i": anchor, "pos": perm[anchor]})

    if _count_solutions(t_perm, clues, n) == 1:
        return clues

    candidates = _candidate_clues(rng, perm, n, n_clue_types)
    for c in candidates:
        if c in clues:
            continue
        clues.append(c)
        if _count_solutions(t_perm, clues, n) == 1:
            break

    # Fallback: add exact clues for all remaining items
    if _count_solutions(t_perm, clues, n) != 1:
        for i in range(n):
            c = {"kind": "exact", "i": i, "pos": perm[i]}
            if c not in clues:
                clues.append(c)

    rng.shuffle(clues)
    return clues


# ---------------------------------------------------------------------------
# Code / prose builders
# ---------------------------------------------------------------------------

def _clue_to_sentence(names: list[str], clue: dict) -> str:
    k = clue["kind"]
    a = names[clue["i"]]
    if k == "exact":
        return f"{a} is at position {clue['pos']}."
    b = names[clue["j"]]
    if k == "before":
        return f"{a} is somewhere to the left of {b}."
    if k == "adjacent":
        return f"{a} is immediately to the left of {b} (no gap between them)."
    if k == "distance":
        return f"{a} is exactly {clue['d']} position(s) to the left of {b}."
    if k == "not_adjacent":
        return f"{a} and {b} are not next to each other."
    return ""


def _to_code(n: int, clues: list[dict], names: list[str]) -> str:
    """Solver + rolling checksum over all positions in listing order."""
    lines = [
        "from itertools import permutations",
        "",
        f"n = {n}",
        "",
        "def check(perm):",
    ]
    for c in clues:
        k = c["kind"]
        i = c["i"]
        j = c.get("j", 0)
        if k == "exact":
            lines.append(f"    if perm[{i}] != {c['pos']}: return False")
        elif k == "before":
            lines.append(f"    if not (perm[{i}] < perm[{j}]): return False")
        elif k == "adjacent":
            lines.append(f"    if not (perm[{j}] == perm[{i}] + 1): return False")
        elif k == "distance":
            lines.append(f"    if not (perm[{j}] - perm[{i}] == {c['d']}): return False")
        elif k == "not_adjacent":
            lines.append(f"    if not (abs(perm[{i}] - perm[{j}]) > 1): return False")
    lines += [
        "    return True",
        "",
        "for perm in permutations(range(n)):",
        "    if check(perm):",
        "        acc = 0",
        "        for pos in perm:",
        "            acc = (acc * 31 + pos) % 10000",
        '        print("%04d" % acc)',
        "        break",
    ]
    return "\n".join(lines) + "\n"


def _to_prose(names: list[str], clues: list[dict], question: str) -> str:
    n = len(names)
    name_list = ", ".join(names)
    lines = [
        f"{n} items — {name_list} — are arranged in a row.",
        f"Positions are numbered 0 (leftmost) to {n - 1} (rightmost).",
        "Each item occupies exactly one unique position.",
        "",
        "Clues:",
    ]
    lines += [f"- {_clue_to_sentence(names, c)}" for c in clues]
    lines += ["", _CHECKSUM_DESCRIPTION, "", f"Question: {question}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate(seed: int, *, n_items: int, n_clue_types: int) -> ItemSpec:
    n_items = int(n_items)
    n_clue_types = int(n_clue_types)
    rng = random.Random(seed)

    names = list(rng.choice(_NAME_SETS)[:n_items])

    # Random permutation: perm[i] = position of item i
    perm = list(range(n_items))
    rng.shuffle(perm)

    clues = _generate_clues(rng, perm, n_items, n_clue_types)
    question = rng.choice(_CHECKSUM_QUESTION_POOL)

    return ItemSpec(
        topic=TOPIC,
        level=2,
        template=f"{TOPIC}_train",
        params={"seed": seed, "n_items": n_items, "n_clue_types": n_clue_types,
                "answer_space_size": 10000},
        stem=hidden_stem(_to_prose(names, clues, question)),
        code=_to_code(n_items, clues, names),
        lang="python",
        numeric=True,
    )


def _accept(spec: ItemSpec, key: str) -> bool:
    """Accept any 4-digit zero-padded checksum."""
    return len(key.strip()) == 4 and key.strip().isdigit()


def _harness_item(name: str, names: list[str], clues: list[dict]) -> ItemSpec:
    prose = _to_prose(names, clues, _CHECKSUM_QUESTION_POOL[0])
    return ItemSpec(
        topic=TOPIC,
        level=1,
        template=name,
        params={},
        stem=hidden_stem(prose),
        code=_to_code(len(names), clues, names),
        lang="python",
        numeric=True,
    )


def specs() -> list[ItemSpec]:
    # Harness A: Alice=2, Bob=0, Carol=1
    # checksum: acc = (0*31+2)%10000=2, (2*31+0)%10000=62, (62*31+1)%10000=1923
    harness_a_names = ["Alice", "Bob", "Carol"]
    harness_a_clues = [
        {"kind": "exact", "i": 1, "pos": 0},  # Bob at 0
        {"kind": "exact", "i": 2, "pos": 1},  # Carol at 1
        {"kind": "exact", "i": 0, "pos": 2},  # Alice at 2
    ]

    # Harness B: Train=0, Plane=2, Car=1
    harness_b_names = ["Train", "Plane", "Car"]
    harness_b_clues = [
        {"kind": "exact", "i": 0, "pos": 0},   # Train at 0
        {"kind": "adjacent", "i": 2, "j": 1},  # Car immediately left of Plane
        {"kind": "exact", "i": 2, "pos": 1},   # Car at 1
    ]

    harness = [
        _harness_item("logic_3a", harness_a_names, harness_a_clues),
        _harness_item("logic_3b", harness_b_names, harness_b_clues),
    ]
    return bank(TOPIC, generate, harness, accept=_accept)

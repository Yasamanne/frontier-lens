"""Procedural counting problems. The stem is prose; the sandbox brute-forces the key.

Knobs: length (8..12), alphabet size (3..4), number of constraints (4..6).
Constraints are meant to interact, not sit as near-no-ops.
"""

from __future__ import annotations

import random

from frontier_lens.templates.procedural import bank
from frontier_lens.templates.spec import ItemSpec

TOPIC = "constraint_counting"
_LETTERS = "abcde"


_KINDS = (
    "adjacent",
    "count",
    "forbidden",
    "palindrome",
    "runs",
    "parity",
    "position",
    "distinct",
    "required",
    "order",
    "modular",
)


def _pick(rng: random.Random, alpha: str, length: int, n_constraints: int) -> list[dict]:
    chosen = rng.sample(_KINDS, n_constraints)
    has_adjacent = "adjacent" in chosen
    specs: list[dict] = []
    for kind in chosen:
        if kind == "adjacent":
            specs.append({"kind": "adjacent"})
        elif kind == "count":
            ch = alpha[rng.randrange(len(alpha))]
            lo = rng.randrange(1, max(2, length // 3 + 1))
            hi = rng.randrange(lo, min(length - 1, lo + 2) + 1)
            specs.append({"kind": "count", "ch": ch, "lo": lo, "hi": hi})
        elif kind == "forbidden":
            a = alpha[rng.randrange(len(alpha))]
            b = alpha[rng.randrange(len(alpha))]
            if has_adjacent and a == b and len(alpha) > 1:
                b = alpha[(alpha.index(a) + 1) % len(alpha)]
            specs.append({"kind": "forbidden", "sub": a + b})
        elif kind == "palindrome":
            mode = rng.choice(("same", "different")) if length < 10 else rng.choice(("ends", "different"))
            specs.append({"kind": "palindrome", "mode": mode})
        elif kind == "runs":
            lo = rng.randrange(2, max(3, length // 2))
            hi = rng.randrange(lo, min(length - 1, lo + 3) + 1)
            specs.append({"kind": "runs", "lo": lo, "hi": hi})
        elif kind == "parity":
            ch = alpha[rng.randrange(len(alpha))]
            even = bool(rng.randrange(2))
            specs.append({"kind": "parity", "ch": ch, "even": even})
        elif kind == "position":
            idx = rng.randrange(length)
            ch = alpha[rng.randrange(len(alpha))]
            eq = bool(rng.randrange(2))
            specs.append({"kind": "position", "i": idx, "ch": ch, "eq": eq})
        elif kind == "distinct":
            lo = 2 if len(alpha) > 2 else 1
            hi = rng.randrange(lo, len(alpha) + 1)
            specs.append({"kind": "distinct", "lo": lo, "hi": hi})
        elif kind == "required":
            a = alpha[rng.randrange(len(alpha))]
            b = alpha[rng.randrange(len(alpha))]
            specs.append({"kind": "required", "sub": a + b})
        elif kind == "order":
            pair = rng.sample(list(alpha), 2) if len(alpha) >= 2 else [alpha[0], alpha[0]]
            specs.append({"kind": "order", "a": pair[0], "b": pair[1]})
        else:
            pair = rng.sample(list(alpha), 2) if len(alpha) >= 2 else [alpha[0], alpha[0]]
            specs.append({"kind": "modular", "a": pair[0], "b": pair[1], "mod": 3})
    return specs


def _sentence(spec: dict) -> str:
    kind = spec["kind"]
    if kind == "adjacent":
        return "No two neighboring characters are equal."
    if kind == "count":
        return (
            f"The character {spec['ch']} occurs at least {spec['lo']} times "
            f"and at most {spec['hi']} times."
        )
    if kind == "forbidden":
        return f"The substring {spec['sub']} does not occur."
    if kind == "palindrome":
        mode = spec["mode"]
        if mode == "same":
            return "The string is the same when the characters are reversed."
        if mode == "ends":
            return "The first character equals the last character."
        return "The string is different when the characters are reversed."
    if kind == "runs":
        return (
            f"The number of runs is at least {spec['lo']} and at most {spec['hi']}, "
            "where a run is a maximal contiguous block of one character."
        )
    if kind == "parity":
        parity = "even" if spec["even"] else "odd"
        return f"The number of {spec['ch']} characters is {parity}."
    if kind == "position":
        place = spec["i"] + 1
        if spec["eq"]:
            return f"The character in position {place} (counting from 1) is {spec['ch']}."
        return f"The character in position {place} (counting from 1) is not {spec['ch']}."
    if kind == "distinct":
        return (
            f"The number of distinct characters is at least {spec['lo']} "
            f"and at most {spec['hi']}."
        )
    if kind == "required":
        return f"The substring {spec['sub']} occurs at least once."
    if kind == "order":
        return (
            f"Both {spec['a']} and {spec['b']} occur, and the first {spec['a']} "
            f"appears before the first {spec['b']}."
        )
    return (
        f"The number of {spec['a']} characters minus the number of {spec['b']} "
        "characters is a multiple of 3, including 0 and negative multiples."
    )


def _check(spec: dict, length: int) -> str:
    kind = spec["kind"]
    if kind == "adjacent":
        return f"all(s[i] != s[i + 1] for i in range({length - 1}))"
    if kind == "count":
        return f"{spec['lo']} <= s.count({spec['ch']!r}) <= {spec['hi']}"
    if kind == "forbidden":
        return f"{spec['sub']!r} not in s"
    if kind == "palindrome":
        mode = spec["mode"]
        if mode == "same":
            return "s == s[::-1]"
        if mode == "ends":
            return "s[0] == s[-1]"
        return "s != s[::-1]"
    if kind == "runs":
        return f"{spec['lo']} <= runs(s) <= {spec['hi']}"
    if kind == "parity":
        want = 0 if spec["even"] else 1
        return f"s.count({spec['ch']!r}) % 2 == {want}"
    if kind == "position":
        op = "==" if spec["eq"] else "!="
        return f"s[{spec['i']}] {op} {spec['ch']!r}"
    if kind == "distinct":
        return f"{spec['lo']} <= len(set(s)) <= {spec['hi']}"
    if kind == "required":
        return f"{spec['sub']!r} in s"
    if kind == "order":
        a, b = spec["a"], spec["b"]
        return f"{a!r} in s and {b!r} in s and s.find({a!r}) < s.find({b!r})"
    return f"(s.count({spec['a']!r}) - s.count({spec['b']!r})) % {spec['mod']} == 0"


def _prose(alpha: str, length: int, specs: list[dict]) -> str:
    letters = ", ".join(alpha)
    lines = [
        (
            f"Count the strings of length {length} whose characters are chosen from "
            f"{{{letters}}} and that satisfy every condition below."
        )
    ]
    lines.extend(f"- {_sentence(spec)}" for spec in specs)
    lines.append("")
    lines.append("Question: how many such strings are there?")
    return "\n".join(lines)


def _code(alpha: str, length: int, specs: list[dict]) -> str:
    checks = "\n".join(f"    if not ({_check(spec, length)}):\n        return False" for spec in specs)
    needs_runs = any(spec["kind"] == "runs" for spec in specs)
    runs_fn = ""
    if needs_runs:
        runs_fn = (
            "def runs(s):\n"
            "    total = 1\n"
            "    for i in range(1, len(s)):\n"
            "        if s[i] != s[i - 1]:\n"
            "            total += 1\n"
            "    return total\n\n"
        )
    return (
        "from itertools import product\n"
        f"alpha = {alpha!r}\n\n"
        f"{runs_fn}"
        "def ok(s):\n"
        f"{checks}\n"
        "    return True\n\n"
        "n = 0\n"
        f"for cand in product(alpha, repeat={length}):\n"
        "    if ok(''.join(cand)):\n"
        "        n += 1\n"
        "print(n)\n"
    )


def generate(seed: int, *, length: int, alphabet_size: int, n_constraints: int) -> ItemSpec:
    length = int(length)
    alphabet_size = int(alphabet_size)
    n_constraints = int(n_constraints)
    if not 1 <= alphabet_size <= len(_LETTERS):
        raise ValueError("alphabet_size out of range")
    alpha = _LETTERS[:alphabet_size]
    specs = _pick(random.Random(seed), alpha, length, n_constraints)
    universe = alphabet_size**length
    return ItemSpec(
        topic=TOPIC,
        level=2,
        template=f"{TOPIC}_train",
        params={
            "seed": seed,
            "length": length,
            "alphabet_size": alphabet_size,
            "n_constraints": n_constraints,
            "answer_space_size": universe + 1,
        },
        stem=_prose(alpha, length, specs),
        code=_code(alpha, length, specs),
        lang="python",
        numeric=True,
    )


def _accept(spec: ItemSpec, key: str) -> bool:
    universe = spec.params["alphabet_size"] ** spec.params["length"]
    try:
        n = int(key)
    except ValueError:
        return False
    return 0 < n < universe


def _harness(name: str, alpha: str, length: int, prose: str) -> ItemSpec:
    code = (
        "from itertools import product\n"
        f"alpha = {alpha!r}\n"
        "n = 0\n"
        f"for _cand in product(alpha, repeat={length}):\n"
        "    n += 1\n"
        "print(n)\n"
    )
    return ItemSpec(
        topic=TOPIC,
        level=1,
        template=name,
        params={"alphabet_size": len(alpha), "length": length},
        stem=prose + "\n\nQuestion: how many such strings are there?",
        code=code,
        lang="python",
        numeric=True,
    )


def specs() -> list[ItemSpec]:
    harness = [
        _harness(
            "all_len2_ab",
            "ab",
            2,
            "Count the strings of length 2 whose characters are chosen from {a, b}.",
        ),
        _harness(
            "all_len2_abc",
            "abc",
            2,
            "Count the strings of length 2 whose characters are chosen from {a, b, c}.",
        ),
    ]
    return bank(TOPIC, generate, harness, accept=_accept)

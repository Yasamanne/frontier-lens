"""Finite-state machine simulation puzzles.

The model sees the FSM definition in prose and an input string.
It must trace the machine character-by-character without running code.

Knobs: n_states (3-7), n_symbols (2-3), string_length (4-14).
Gold key: index of the final state (numeric, 0-indexed).

Why better than execution tracing:
  - Discrete state-transition reasoning, not arithmetic/mutation tracking
  - Model cannot shortcut by recognising a checksum pattern
  - Failure mode is state-loss after N steps, cleanly distinguishable
"""

from __future__ import annotations

import random

from frontier_lens.templates.procedural import bank
from frontier_lens.templates.spec import ItemSpec, hidden_stem

TOPIC = "fsm_queries"
_ALPHABET = "abcde"

_QUESTION_POOL = [
    "what four-digit number does the simulation produce?",
    "what is the four-digit output of the simulation after processing the entire string?",
    "after reading all input characters, what four-digit value does the machine output?",
]

# The verifier accumulates a rolling hash over each visited state index so the
# 4-digit output is unlikely to appear verbatim in the prose description.
# Model must trace every transition to reproduce the accumulator.


# ---------------------------------------------------------------------------
# FSM generation
# ---------------------------------------------------------------------------

def _generate_transitions(
    rng: random.Random,
    n_states: int,
    symbols: list[str],
) -> dict[tuple[str, str], str]:
    """Random total transition function; every state reachable from S0."""
    states = [f"S{i}" for i in range(n_states)]

    # Start with a fully random assignment
    trans: dict[tuple[str, str], str] = {}
    for s in states:
        for sym in symbols:
            trans[(s, sym)] = rng.choice(states)

    # Repair: BFS from S0, redirect one transition to reach each unvisited state
    reachable: set[str] = {"S0"}
    frontier: list[str] = ["S0"]
    while frontier:
        s = frontier.pop()
        for sym in symbols:
            t = trans[(s, sym)]
            if t not in reachable:
                reachable.add(t)
                frontier.append(t)

    unreachable = [s for s in states if s not in reachable]
    for dst in unreachable:
        src = rng.choice(list(reachable))
        sym = rng.choice(symbols)
        trans[(src, sym)] = dst
        reachable.add(dst)

    return trans


def _simulate(trans: dict[tuple[str, str], str], input_str: str) -> str:
    state = "S0"
    for ch in input_str:
        state = trans[(state, ch)]
    return state


# ---------------------------------------------------------------------------
# Code / prose builders
# ---------------------------------------------------------------------------

def _to_code(trans: dict[tuple[str, str], str], input_str: str) -> str:
    """Verifier: rolling hash over visited state indices → 4-digit output."""
    lines = ["trans = {"]
    for (s, sym), t in sorted(trans.items()):
        lines.append(f"    ({s!r}, {sym!r}): {t!r},")
    lines += [
        "}",
        "state = 'S0'",
        "acc = 0",
        f"for ch in {input_str!r}:",
        "    state = trans[(state, ch)]",
        "    acc = (acc * 31 + int(state[1:])) % 10000",
        'print("%04d" % acc)',
    ]
    return "\n".join(lines) + "\n"


def _to_prose(
    n_states: int,
    symbols: list[str],
    trans: dict[tuple[str, str], str],
    input_str: str,
    question: str,
) -> str:
    states = [f"S{i}" for i in range(n_states)]
    sym_str = ", ".join(f'"{s}"' for s in symbols)
    lines = [
        f"A finite state machine has {n_states} states: {', '.join(states)}.",
        f"The machine starts in state S0.",
        f"It reads the input string one character at a time over alphabet {{{sym_str}}}.",
        "Each (state, character) pair maps to exactly one next state.",
        "",
        "Transition table:",
    ]
    # Column header
    header = "  State  |  " + "  |  ".join(f'"{sym}"' for sym in symbols)
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for s in states:
        row = "  |  ".join(trans[(s, sym)] for sym in symbols)
        lines.append(f"  {s}    |  {row}")
    lines += [
        "",
        f'Input string: "{input_str}"',
        "",
        "Simulation procedure:",
        "  Start: state = S0, accumulator = 0",
        "  For each character in the input string (left to right):",
        "    1. Look up the next state from the transition table.",
        "    2. Update accumulator = (accumulator * 31 + state_index) mod 10000",
        '  Output the accumulator as a zero-padded four-digit number (e.g. "0042").',
        "",
        f"Question: {question}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate(seed: int, *, n_states: int, n_symbols: int, string_length: int) -> ItemSpec:
    n_states = int(n_states)
    n_symbols = int(n_symbols)
    string_length = int(string_length)
    rng = random.Random(seed)
    symbols = list(_ALPHABET[:n_symbols])

    trans = _generate_transitions(rng, n_states, symbols)
    input_str = "".join(rng.choice(symbols) for _ in range(string_length))
    question = rng.choice(_QUESTION_POOL)

    return ItemSpec(
        topic=TOPIC,
        level=2,
        template=f"{TOPIC}_train",
        params={
            "seed": seed,
            "n_states": n_states,
            "n_symbols": n_symbols,
            "string_length": string_length,
            "answer_space_size": 10000,   # 4-digit hash 0000-9999
        },
        stem=hidden_stem(_to_prose(n_states, symbols, trans, input_str, question)),
        code=_to_code(trans, input_str),
        lang="python",
        numeric=True,
    )


def _accept(spec: ItemSpec, key: str) -> bool:
    """Accept any 4-digit zero-padded hash value (0000-9999)."""
    return len(key.strip()) == 4 and key.strip().isdigit()


def _harness_item(
    name: str,
    trans_triples: list[tuple[str, str, str]],
    input_str: str,
    symbols: list[str],
) -> ItemSpec:
    n_states = len({s for s, _, _ in trans_triples})
    trans = {(s, sym): t for s, sym, t in trans_triples}
    prose = _to_prose(n_states, symbols, trans, input_str, _QUESTION_POOL[0])
    return ItemSpec(
        topic=TOPIC,
        level=1,
        template=name,
        params={},
        stem=hidden_stem(prose),
        code=_to_code(trans, input_str),
        lang="python",
        numeric=True,
    )


def specs() -> list[ItemSpec]:
    # Two simple hand-crafted harness items (2 states, 2 symbols)
    harness = [
        _harness_item(
            "fsm_2s2sym_a",
            [("S0", "a", "S1"), ("S0", "b", "S0"), ("S1", "a", "S0"), ("S1", "b", "S1")],
            "abba",
            ["a", "b"],
        ),
        _harness_item(
            "fsm_2s2sym_b",
            [("S0", "a", "S0"), ("S0", "b", "S1"), ("S1", "a", "S1"), ("S1", "b", "S0")],
            "aabb",
            ["a", "b"],
        ),
    ]
    return bank(TOPIC, generate, harness, accept=_accept)

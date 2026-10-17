"""Procedural shell pipelines over a seeded CSV.

Knobs: rows (80..320) and stages (4..8). The command prints one derived
number (a column sum, a count-weighted sum, or one group's sum).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from frontier_lens.templates.procedural import bank
from frontier_lens.templates.spec import ItemSpec, shell_stem

TOPIC = "shell_pipelines"
_CATS = (
    "amber",
    "birch",
    "cedar",
    "dune",
    "elm",
    "flint",
    "grove",
    "hazel",
    "ivy",
    "jade",
    "kelp",
    "larch",
)


@dataclass
class Frame:
    rows: list[list[str]]
    id_field: int = 2
    num_field: int | None = 4
    cat_field: int | None = 3


def _table(rng: random.Random, rows: int) -> list[list[str]]:
    table: list[list[str]] = []
    for i in range(rows):
        if i > 0 and rng.random() < 0.18:
            table.append(list(table[rng.randrange(i)]))
            continue
        bucket = rng.randrange(3)
        if bucket == 0:
            num = rng.randrange(0, 10)
        elif bucket == 1:
            num = rng.randrange(10, 100)
        else:
            num = rng.randrange(100, 400)
        table.append(
            [
                f"{i:04d}",
                str(rng.randrange(1000, 10000)),
                _CATS[rng.randrange(len(_CATS))],
                str(num),
                f"{rng.randrange(5000):04d}",
            ]
        )
    return table


def _csv(rows: list[list[str]]) -> str:
    return "\n".join(",".join(row) for row in rows) + "\n"


def _sort(frame: Frame, rng: random.Random, *, source: str = "") -> tuple[Frame, str]:
    candidates: list[tuple[int, bool]] = [(0, True), (0, False)]
    candidates.append((frame.id_field - 1, True))
    candidates.append((frame.id_field - 1, False))
    if frame.num_field:
        candidates.append((frame.num_field - 1, True))
        candidates.append((frame.num_field - 1, False))
    if frame.cat_field:
        candidates.append((frame.cat_field - 1, False))
    field, numeric = candidates[rng.randrange(len(candidates))]
    reverse = bool(rng.randrange(2))

    def key(row: list[str]):
        raw = row[field]
        primary = int(raw) if numeric else raw
        return (primary, row[0])

    ordered = sorted(frame.rows, key=key)
    if reverse:
        ordered = list(reversed(ordered))
    flag = "n" if numeric else ""
    rev = "r" if reverse else ""
    cmd = f"sort -t, -k{field + 1},{field + 1}{flag}{rev} -k1,1{rev}"
    if source:
        cmd = f"{cmd} {source}"
    return Frame(ordered, frame.id_field, frame.num_field, frame.cat_field), cmd


def _filter(frame: Frame, rng: random.Random) -> tuple[Frame, str] | None:
    if not frame.num_field or len(frame.rows) < 4:
        return None
    idx = frame.num_field - 1
    vals = sorted(int(row[idx]) for row in frame.rows)
    if vals[0] == vals[-1]:
        return None
    threshold = vals[len(vals) // 3]
    kept = [row for row in frame.rows if int(row[idx]) >= threshold]
    if len(kept) < 2 or len(kept) == len(frame.rows):
        threshold = vals[len(vals) // 2]
        kept = [row for row in frame.rows if int(row[idx]) >= threshold]
    if len(kept) < 2 or len(kept) == len(frame.rows):
        return None
    prog = f"${frame.num_field}+0 >= {threshold} {{print $0}}"
    cmd = "awk -F, '" + prog + "'"
    return Frame(kept, frame.id_field, frame.num_field, frame.cat_field), cmd


def _head(frame: Frame, rng: random.Random) -> tuple[Frame, str] | None:
    if len(frame.rows) < 4:
        return None
    lo = max(2, len(frame.rows) // 5)
    hi = max(lo + 1, (4 * len(frame.rows)) // 5)
    n = rng.randint(lo, hi)
    n = min(n, len(frame.rows) - 1)
    nxt = Frame(frame.rows[:n], frame.id_field, frame.num_field, frame.cat_field)
    return nxt, f"head -n {n}"


def _tail(frame: Frame, rng: random.Random) -> tuple[Frame, str] | None:
    if len(frame.rows) < 4:
        return None
    lo = max(2, len(frame.rows) // 5)
    hi = max(lo + 1, (4 * len(frame.rows)) // 5)
    n = rng.randint(lo, hi)
    n = min(n, len(frame.rows) - 1)
    nxt = Frame(frame.rows[-n:], frame.id_field, frame.num_field, frame.cat_field)
    return nxt, f"tail -n {n}"


def _uniq(frame: Frame) -> tuple[Frame, str] | None:
    out: list[list[str]] = []
    for row in frame.rows:
        if not out or out[-1] != row:
            out.append(row)
    if len(out) < 2 or len(out) == len(frame.rows):
        return None
    return Frame(out, frame.id_field, frame.num_field, frame.cat_field), "uniq"


def _grep(frame: Frame, rng: random.Random) -> tuple[Frame, str] | None:
    if not frame.cat_field:
        return None
    idx = frame.cat_field - 1
    counts: dict[str, int] = {}
    for row in frame.rows:
        counts[row[idx]] = counts.get(row[idx], 0) + 1
    options = [cat for cat, n in counts.items() if 2 <= n < len(frame.rows)]
    if not options:
        return None
    cat = options[rng.randrange(len(options))]
    kept = [row for row in frame.rows if row[idx] == cat]
    return Frame(kept, frame.id_field, frame.num_field, frame.cat_field), f"grep -F -- ',{cat},'"


def _cut(frame: Frame) -> tuple[Frame, str] | None:
    if frame.num_field is None or frame.id_field != 2:
        return None
    num_idx = frame.num_field - 1
    if num_idx == 1:
        return None
    kept = [[row[0], row[1], row[num_idx]] for row in frame.rows]
    fields = f"1,2,{frame.num_field}"
    return Frame(kept, 2, 3, None), f"cut -d, -f{fields}"


def _fallback(frame: Frame) -> tuple[Frame, str]:
    ordered = sorted(frame.rows, key=lambda row: row[0])
    return Frame(ordered, frame.id_field, frame.num_field, frame.cat_field), "sort -t, -k1,1"


def _middle(frame: Frame, rng: random.Random, *, allow_cut: bool) -> tuple[Frame, str]:
    ops = ["filter", "head", "tail", "uniq", "grep", "sort"]
    if allow_cut:
        ops.append("cut")
    rng.shuffle(ops)
    for op in ops:
        if op == "filter":
            got = _filter(frame, rng)
        elif op == "head":
            got = _head(frame, rng)
        elif op == "tail":
            got = _tail(frame, rng)
        elif op == "uniq":
            got = _uniq(frame)
        elif op == "grep":
            got = _grep(frame, rng)
        elif op == "cut":
            got = _cut(frame)
        else:
            got = _sort(frame, rng)
        if got is not None:
            return got
    return _fallback(frame)


def _sum_cmd(id_field: int) -> str:
    return "awk -F, '" + f"{{s+=${id_field}}} END {{print s}}" + "'"


def _family_sum(rng: random.Random, stages: int, table: list[list[str]]) -> str:
    frame = Frame([list(row) for row in table])
    frame, cmd = _sort(frame, rng, source="data.csv")
    parts = [cmd]
    while len(parts) < stages - 1:
        frame, cmd = _middle(frame, rng, allow_cut=True)
        parts.append(cmd)
    parts.append(_sum_cmd(frame.id_field))
    return " | ".join(parts)


def _family_uniq_c(rng: random.Random, stages: int, table: list[list[str]]) -> str:
    frame = Frame([list(row) for row in table])
    frame, cmd = _sort(frame, rng, source="data.csv")
    parts = [cmd]
    for _ in range(stages - 3):
        frame, cmd = _middle(frame, rng, allow_cut=False)
        parts.append(cmd)
    parts.append("uniq -c")
    field = frame.id_field
    prog = f'{{split($2,a,","); s+=$1*a[{field}]}} END {{print s}}'
    parts.append("awk '" + prog + "'")
    return " | ".join(parts)


def _family_group(rng: random.Random, stages: int, table: list[list[str]]) -> str:
    frame = Frame([list(row) for row in table])
    frame, cmd = _sort(frame, rng, source="data.csv")
    parts = [cmd]
    for _ in range(stages - 5):
        frame, cmd = _middle(frame, rng, allow_cut=False)
        parts.append(cmd)
    cat_field = frame.cat_field
    id_field = frame.id_field
    prog = f"{{s[${cat_field}]+=${id_field}}} END {{for (k in s) print k \",\" s[k]}}"
    parts.append("awk -F, '" + prog + "'")
    parts.append("sort -t, -k1,1")
    parts.append("tail -n 1")
    parts.append("cut -d, -f2")
    return " | ".join(parts)


def _command(rng: random.Random, stages: int, table: list[list[str]]) -> str:
    family = rng.randrange(3)
    if family == 0 and stages >= 5:
        cmd = _family_group(rng, stages, table)
    elif family == 1:
        cmd = _family_uniq_c(rng, stages, table)
    else:
        cmd = _family_sum(rng, stages, table)
    if cmd.count("|") + 1 != stages:
        raise RuntimeError(f"pipeline has {cmd.count('|') + 1} stages, wanted {stages}")
    return cmd


def generate(seed: int, *, rows: int, stages: int) -> ItemSpec:
    rows = int(rows)
    stages = int(stages)
    if not 3 <= stages <= 8:
        raise ValueError("stages must be 3..8")
    if rows < 4:
        raise ValueError("rows must be at least 4")
    rng = random.Random(seed)
    table = _table(rng, rows)
    command = _command(rng, stages, table)
    files = {"data.csv": _csv(table)}
    return ItemSpec(
        topic=TOPIC,
        level=2,
        template=f"{TOPIC}_train",
        params={
            "seed": seed,
            "rows": rows,
            "stages": stages,
            "answer_space_size": 9000,
        },
        stem=shell_stem(files, command, "what number does this print?"),
        code=command,
        lang="shell",
        input_files=files,
        numeric=True,
    )


def _harness(name: str, files: dict[str, str], command: str) -> ItemSpec:
    return ItemSpec(
        topic=TOPIC,
        level=1,
        template=name,
        params={},
        stem=shell_stem(files, command, "what number does this print?"),
        code=command,
        lang="shell",
        input_files=files,
        numeric=True,
    )


def specs() -> list[ItemSpec]:
    harness = [
        _harness(
            "uniq_sort_count",
            {"colors.txt": "red\nblue\nred\ngreen\n"},
            "sort -u colors.txt | wc -l",
        ),
        _harness(
            "line_count",
            {"notes.txt": "a\nb\nc\n"},
            "cat notes.txt | wc -l",
        ),
    ]
    return bank(TOPIC, generate, harness)

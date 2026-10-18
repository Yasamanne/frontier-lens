"""Label uploaded questions that have no topic and/or difficulty, using the judge model.

Only missing fields are filled; anything the file supplies is kept. The model reads
questions in batches and returns a topic and a band for each:

  topic   short snake_case label; it must reuse an existing topic when one fits
  band    ``standard`` (routine, one or two steps) or ``hard`` (multi-step or competition-level)

Band maps to level (standard 2, hard 3) like the ``difficulty`` upload column. Rows the
model cannot label keep the upload defaults (``custom``, level 2). Labelled rows record
``label_source`` in their params so the origin of a label stays visible.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass

from frontier_lens.parallel import map_unordered

BATCH_SIZE = 20
MAX_STEM_CHARS = 600
BAND_TO_LEVEL = {"standard": 2, "hard": 3}
_VAGUE_TOPICS = {"", "unknown", "other", "others", "general", "misc", "miscellaneous", "none"}

_PROMPT = """\
You label exam questions so a dashboard can group them by subject and difficulty.

For each numbered question return a topic and a band.

Topic rules:
- a BROAD subject area in snake_case, at most 3 words, so that a whole dataset ends up with
  roughly 10 to 20 topics (examples: algebra, geometry, calculus, probability, statistics,
  number_theory, physics, biology, law, ethics, public_health)
- prefer the broad field over a specific sub-field: "calculus", not "vector_calculus" or
  "double_integrals"; "geometry", not "circle_geometry"
- REUSE one of the existing topics below whenever it fits; invent a new topic only when none does
- never use vague words such as "other", "general" or "misc"

Band rules:
- "standard": routine, solvable in one or two steps
- "hard": multi-step reasoning, competition level, or easy to get wrong

Existing topics: {topics}

Questions:
{questions}

Reply with a JSON array only, one object per question, for example:
[{{"id": 1, "topic": "algebra", "band": "standard"}}]
"""


@dataclass
class LabelReport:
    rows_needing_labels: int = 0
    rows_labeled: int = 0
    batches: int = 0
    failed_batches: int = 0
    new_topics: tuple[str, ...] = ()


def needs_label(row: dict) -> bool:
    return bool(row.get("topic_missing") or row.get("level_missing"))


def n_calls(rows: list[dict], batch_size: int = BATCH_SIZE) -> int:
    n = sum(1 for r in rows if needs_label(r))
    return -(-n // batch_size)


def normalize_topic(raw: object) -> str | None:
    """snake_case, at most 40 chars; None if empty or vague."""
    topic = re.sub(r"[^a-z0-9]+", "_", str(raw or "").lower()).strip("_")[:40].strip("_")
    if topic in _VAGUE_TOPICS or not topic[:1].isalpha():
        return None
    return topic


def parse_labels(text: str, n: int) -> dict[int, tuple[str | None, str | None]]:
    """Map 1-based question number to (topic, band); invalid values become None."""
    cleaned = re.sub(r"```(?:json)?", "", text or "")
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start < 0 or end <= start:
        raise ValueError("no JSON array in the labelling response")
    items = json.loads(cleaned[start : end + 1])
    out: dict[int, tuple[str | None, str | None]] = {}
    for item in items:
        try:
            idx = int(item["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if not 1 <= idx <= n:
            continue
        band = str(item.get("band") or "").strip().lower()
        out[idx] = (normalize_topic(item.get("topic")), band if band in BAND_TO_LEVEL else None)
    return out


def _prompt(batch: list[dict], topics: list[str]) -> str:
    lines = []
    for i, row in enumerate(batch, 1):
        stem = " ".join(str(row["stem"]).split())[:MAX_STEM_CHARS]
        lines.append(f"{i}. {stem}  [answer: {str(row['key'])[:60]}]")
    return _PROMPT.format(topics=", ".join(topics) or "(none yet)", questions="\n".join(lines))


def _apply(batch: list[dict], labels: dict[int, tuple[str | None, str | None]]) -> int:
    """Fill missing fields from *labels*; return how many rows received at least one."""
    done = 0
    for i, row in enumerate(batch, 1):
        topic, band = labels.get(i, (None, None))
        sources = row.setdefault("labels", {})
        got = False
        if row.get("topic_missing") and topic:
            row["topic"] = topic
            sources["topic"] = "llm"
            got = True
        if row.get("level_missing") and band:
            row["level"] = BAND_TO_LEVEL[band]
            sources["band"] = "llm"
            got = True
        done += got
    return done


def default_complete() -> Callable[[str], str]:
    """Prompt -> text using the configured judge model."""
    from frontier_lens.providers.openai_compatible import OpenAICompatibleProvider
    from frontier_lens.registry import judge_spec

    spec = judge_spec()
    if spec is None:
        raise RuntimeError("no judge model configured in models.yaml")
    provider = OpenAICompatibleProvider(spec.base_url or "", spec.api_key_env)

    def complete(prompt: str) -> str:
        return provider.complete(
            prompt, spec.model, thinking=bool(spec.thinking), skip_correction=True
        ).text

    return complete


def label_rows(
    rows: list[dict],
    *,
    known_topics: list[str],
    complete: Callable[[str], str] | None = None,
    batch_size: int = BATCH_SIZE,
    progress: Callable[[int, int], None] | None = None,
) -> LabelReport:
    """Fill missing topic/level on *rows* in place and report what happened.

    The first batch runs alone so the topics it creates join the list shown to the
    remaining batches (which then run concurrently); that keeps topic names consistent.
    A batch that errors or returns nothing usable leaves its rows on the defaults.
    """
    todo = [r for r in rows if needs_label(r)]
    report = LabelReport(rows_needing_labels=len(todo))
    if not todo:
        return report
    complete = complete or default_complete()
    batches = [todo[i : i + batch_size] for i in range(0, len(todo), batch_size)]
    report.batches = len(batches)
    topics = sorted({t for t in known_topics if t})
    initial = set(topics)
    done = 0

    def run(batch: list[dict], shown: list[str]) -> int | None:
        try:
            return _apply(batch, parse_labels(complete(_prompt(batch, shown)), len(batch)))
        except Exception:
            return None

    def note(result: int | None) -> None:
        nonlocal done
        done += 1
        if result is None:
            report.failed_batches += 1
        else:
            report.rows_labeled += result
        if progress:
            progress(done, len(batches))

    note(run(batches[0], topics))
    seen = {r["topic"] for r in batches[0] if r.get("labels", {}).get("topic")}
    shown = sorted(set(topics) | seen)
    rest = batches[1:]
    for _, result in map_unordered(lambda b: run(b, shown), rest):
        note(result)

    used = {r["topic"] for r in todo if r.get("labels", {}).get("topic")}
    report.new_topics = tuple(sorted(used - initial))
    return report

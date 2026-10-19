from __future__ import annotations

from sqlalchemy import text

from frontier_lens.db import get_engine
from frontier_lens.failure_analysis import failure_modes, failure_summary_text
from frontier_lens.feedback import context_for as sme_feedback_for
from frontier_lens.heatmap import weakest_cell
from frontier_lens.providers.openai_compatible import OpenAICompatibleProvider
from frontier_lens.registry import sync_models
from frontier_lens.split import EVAL_TEMPLATES
from frontier_lens.templates import all_specs

from .candidates import _store_candidate
from .prompts import ONE_SHOT_EVAL, ONE_SHOT_TRAIN
from .utils import generator_kwargs


def _eval_cells() -> list[tuple[str, str]]:
    seen: list[tuple[str, str]] = []
    for spec in all_specs():
        if spec.template in EVAL_TEMPLATES:
            cell = (spec.topic, spec.band)
            if cell not in seen:
                seen.append(cell)
    return seen or [("python_state", "hard")]


def _failures(run_id: str, model_id: str, topic: str, band: str, limit: int = 6) -> str:
    """Return concatenated stems of questions the model mostly fails."""
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT q.stem, q.id, AVG(a.is_correct) as acc
                FROM attempts a JOIN questions q ON q.id=a.question_id
                WHERE a.run_id=:r AND a.model_id=:m AND q.topic=:t AND q.band=:b AND q.split='train'
                GROUP BY q.id
                HAVING acc < 0.5
                LIMIT :n
                """
            ),
            {"r": run_id, "m": model_id, "t": topic, "b": band, "n": limit},
        ).mappings().all()
    if not rows:
        return "(no listed failures)"
    return "\n---\n".join(r["stem"][:800] for r in rows)


def _failure_analysis_text(run_id: str, model_id: str, topic: str, band: str) -> str:
    """Return a human-readable failure mode analysis for the prompt."""
    analysis = failure_modes(run_id, model_id, topic, band, split="train")
    return failure_summary_text(analysis)


def propose_one_shot(
    run_id: str,
    model_id: str,
    n: int = 10,
    *,
    rl_split: str = "train",
) -> list[str]:
    if rl_split not in {"train", "eval"}:
        raise ValueError(f"bad split {rl_split}")
    specs = {m.id: m for m in sync_models()}
    gen = next(m for m in specs.values() if m.role == "generator")
    provider = OpenAICompatibleProvider(gen.base_url or "", gen.api_key_env)
    ids = []
    if rl_split == "eval":
        cells = _eval_cells()
        for i in range(n):
            topic, band = cells[i % len(cells)]
            prompt = ONE_SHOT_EVAL.format(topic=topic, band=band)
            completion = provider.complete(
                prompt, gen.model, skip_correction=True, **generator_kwargs(gen)
            )
            cid = _store_candidate(
                completion.text,
                topic,
                band,
                gen.id,
                "one_shot",
                [],
                rl_split="eval",
                calibrate=False,
            )
            ids.append(cid)
        return ids

    cell = weakest_cell(run_id, model_id)
    if not cell:
        raise RuntimeError("no heatmap cell")
    fails = _failures(run_id, model_id, cell.topic, cell.band)
    fa_text = _failure_analysis_text(run_id, model_id, cell.topic, cell.band)
    sme_feedback = sme_feedback_for(cell.topic)
    for _ in range(n):
        prompt = ONE_SHOT_TRAIN.format(
            topic=cell.topic, band=cell.band, failures=fails, failure_analysis=fa_text,
            sme_feedback=sme_feedback,
        )
        completion = provider.complete(
            prompt, gen.model, skip_correction=True, **generator_kwargs(gen)
        )
        cid = _store_candidate(
            completion.text,
            cell.topic,
            cell.band,
            gen.id,
            "one_shot",
            [model_id],
            run_id=run_id,
            rl_split="train",
            calibrate=True,
        )
        ids.append(cid)
    return ids

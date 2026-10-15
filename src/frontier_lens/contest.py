from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import text

from frontier_lens.config import POLICY_K, REFERENCE_K
from frontier_lens.db import get_engine
from frontier_lens.ids import item_set_hash
from frontier_lens.parse import parse_final
from frontier_lens.prompt import contest_prompt
from frontier_lens.providers.fixture import FixtureProvider
from frontier_lens.providers.openai_compatible import OpenAICompatibleProvider
from frontier_lens.registry import ModelSpec, sync_models
from frontier_lens.reward import grade


def samples_for(spec: ModelSpec, k_override: int | None = None) -> int:
    if k_override is not None:
        return k_override
    if spec.role == "policy":
        return POLICY_K
    return REFERENCE_K


def _questions(conn, levels: list[int]):
    rows = conn.execute(
        text(
            "SELECT id, stem, key, numeric FROM questions WHERE source IN ('seed','generated') AND level IN :lv"
        ).bindparams(lv=tuple(levels)),
    ).mappings().all()
    return list(rows)


def run_contest(
    model_ids: list[str] | None,
    k: int | None = None,
    levels: list[int] | None = None,
    replay: bool = False,
    write_fixtures: bool = False,
) -> str:
    levels = levels or [1, 2, 3]
    specs = {m.id: m for m in sync_models()}
    engine = get_engine()
    with engine.begin() as conn:
        qrows = conn.execute(
            text("SELECT id, stem, key, numeric, level FROM questions WHERE source IN ('seed','generated')")
        ).mappings().all()
        qrows = [q for q in qrows if q["level"] in levels]
        if not qrows:
            raise RuntimeError("no questions; run python -m frontier_lens.seed")
        ids = [
            m
            for m in (model_ids or list(specs))
            if specs.get(m) and specs[m].role in {"policy", "reference"}
        ]
        if not ids:
            raise RuntimeError("no evaluated models selected")
        run_id = uuid.uuid4().hex[:12]
        conn.execute(
            text(
                """
                INSERT INTO eval_runs (id, created_at, item_set_hash, status, k, notes)
                VALUES (:id, :ts, :h, 'running', :k, :notes)
                """
            ),
            {
                "id": run_id,
                "ts": datetime.now(timezone.utc).isoformat(),
                "h": item_set_hash([q["id"] for q in qrows]),
                "k": max(POLICY_K, REFERENCE_K) if k is None else k,
                "notes": "replay" if replay else "live",
            },
        )

    fixture = FixtureProvider() if replay else None
    recorded: list[dict] = []
    try:
        for mid in ids:
            spec: ModelSpec = specs[mid]
            provider = None if replay else OpenAICompatibleProvider(spec.base_url or "", spec.api_key_env)
            n_samples = samples_for(spec, k)
            for q in qrows:
                for sample in range(1, n_samples + 1):
                    with engine.begin() as conn:
                        exists = conn.execute(
                            text(
                                "SELECT 1 FROM attempts WHERE run_id=:r AND model_id=:m AND question_id=:q AND sample_index=:s"
                            ),
                            {"r": run_id, "m": mid, "q": q["id"], "s": sample},
                        ).first()
                    if exists:
                        continue
                    if replay:
                        completion = fixture.complete_lookup(mid, q["id"], sample)
                    else:
                        completion = provider.complete(contest_prompt(q["stem"]), spec.model)
                    numeric = bool(q["numeric"])
                    parsed = parse_final(completion.text, numeric=numeric)
                    reward, parsed_ans, unparsable = grade(
                        completion.text, q["key"], numeric=numeric
                    )
                    correct = reward == 1
                    if parsed_ans is not None:
                        parsed.answer = parsed_ans
                    parsed.unparsable = unparsable
                    with engine.begin() as conn:
                        conn.execute(
                            text(
                                """
                                INSERT INTO attempts (
                                    run_id, model_id, question_id, sample_index, raw_output, parsed_answer,
                                    is_correct, unparsable, latency_ms, tokens_in, tokens_out
                                ) VALUES (
                                    :r, :m, :q, :s, :raw, :ans, :ok, :unp, :lat, :ti, :to
                                )
                                """
                            ),
                            {
                                "r": run_id,
                                "m": mid,
                                "q": q["id"],
                                "s": sample,
                                "raw": completion.text,
                                "ans": parsed.answer,
                                "ok": int(correct),
                                "unp": int(parsed.unparsable),
                                "lat": completion.latency_ms,
                                "ti": completion.tokens_in,
                                "to": completion.tokens_out,
                            },
                        )
                    recorded.append(
                        {
                            "model_id": mid,
                            "question_id": q["id"],
                            "sample_index": sample,
                            "raw_output": completion.text,
                            "parsed_answer": parsed.answer,
                            "is_correct": correct,
                            "latency_ms": completion.latency_ms,
                        }
                    )
                    print(f"{mid} {q['id']} k={sample} correct={correct} unparsable={parsed.unparsable}")
        with engine.begin() as conn:
            conn.execute(text("UPDATE eval_runs SET status='complete' WHERE id=:id"), {"id": run_id})
        if write_fixtures and not replay:
            from frontier_lens.config import FIXTURES_PATH

            FIXTURES_PATH.parent.mkdir(exist_ok=True)
            header = {
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "run_id": run_id,
                "k": k,
                "models": [
                    {
                        "id": specs[m].id,
                        "provider": specs[m].provider,
                        "model_id": specs[m].model,
                        "quant": specs[m].quant,
                        "thinking": specs[m].thinking,
                    }
                    for m in ids
                ],
                "rows": recorded,
            }
            FIXTURES_PATH.write_text(json.dumps(header, indent=2))
            print(f"wrote {FIXTURES_PATH}")
    except Exception:
        with engine.begin() as conn:
            conn.execute(text("UPDATE eval_runs SET status='failed' WHERE id=:id"), {"id": run_id})
        raise
    return run_id


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--live", action="store_true")
    p.add_argument("--replay", action="store_true")
    p.add_argument("--k", type=int, default=None, help="override samples; default policy=8, reference=3")
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--levels", nargs="*", type=int, default=[1, 2, 3])
    p.add_argument("--write-fixtures", action="store_true")
    args = p.parse_args()
    if not args.live and not args.replay:
        raise SystemExit("pass --live or --replay")
    run_id = run_contest(
        args.models,
        k=args.k,
        levels=args.levels,
        replay=args.replay,
        write_fixtures=args.write_fixtures,
    )
    print("run", run_id)


if __name__ == "__main__":
    main()

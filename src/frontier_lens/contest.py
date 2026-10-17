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
from frontier_lens.providers.base import Completion
from frontier_lens.providers.fixture import FixtureProvider
from frontier_lens.providers.openai_compatible import OpenAICompatibleProvider
from frontier_lens.registry import ModelSpec, sync_models
from frontier_lens.reward import grade
from frontier_lens.runlog import write_event


def select_questions(
    levels: list[int] | None = None,
    limit: int | None = None,
    question_ids: list[str] | None = None,
) -> list[dict]:
    levels = levels or [1, 2, 3]
    engine = get_engine()
    with engine.connect() as conn:
        qrows = conn.execute(
            text(
                """
                SELECT id, stem, key, numeric, level, topic, band, split, template,
                       params_json, source
                FROM questions WHERE source IN ('seed','generated','uploaded')
                """
            )
        ).mappings().all()
    qrows = [dict(q) for q in qrows if q["level"] in levels]
    if question_ids:
        wanted = set(question_ids)
        qrows = [q for q in qrows if q["id"] in wanted]
    qrows = sorted(qrows, key=lambda q: (q["level"], q["id"]))
    if limit is not None:
        qrows = qrows[:limit]
    return qrows


def replay_gap(
    model_ids: list[str] | None,
    levels: list[int] | None = None,
    *,
    k: int | None = None,
    limit: int | None = None,
    question_ids: list[str] | None = None,
) -> str | None:
    """Why replay cannot score this selection, or None when every sample is recorded."""
    specs = {m.id: m for m in sync_models()}
    ids = [
        mid
        for mid in (model_ids or list(specs))
        if specs.get(mid) and specs[mid].role in {"policy", "reference"}
    ]
    questions = select_questions(levels, limit, question_ids)
    fixture = FixtureProvider()
    for mid in ids:
        n_samples = samples_for(specs[mid], k)
        for question in questions:
            for sample in range(1, n_samples + 1):
                if (mid, question["id"], sample) not in fixture.rows:
                    return (
                        f"Replay has no recorded answer for {mid} {question['id']} "
                        f"sample {sample}. Fixtures belong to an older seed bank. "
                        "Choose live to call the model, or record a new file with "
                        "`uv run python -m frontier_lens.contest --live --write-fixtures`."
                    )
    return None


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
    limit: int | None = None,
    question_ids: list[str] | None = None,
    notes: str | None = None,
    reuse_prev_runs: bool = False,
    judge_model_id: str | None = None,
) -> str:
    levels = levels or [1, 2, 3]
    specs = {m.id: m for m in sync_models()}
    qrows = select_questions(levels, limit, question_ids)
    engine = get_engine()
    with engine.begin() as conn:
        if not qrows:
            raise RuntimeError("no questions; run python -m frontier_lens.seed")
        ids = [
            m
            for m in (model_ids or list(specs))
            if specs.get(m) and specs[m].role in {"policy", "reference"}
        ]
        if not ids:
            raise RuntimeError("no evaluated models selected")
        if replay:
            gap = replay_gap(ids, levels, k=k, limit=limit, question_ids=question_ids)
            if gap:
                raise RuntimeError(gap)
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
                "notes": notes or ("replay" if replay else "live"),
            },
        )

    fixture = FixtureProvider() if replay else None

    # Build judge_fn if a judge model is configured
    judge_fn = None
    if judge_model_id:
        _judge_spec = specs.get(judge_model_id)
        if _judge_spec:
            _judge_provider = OpenAICompatibleProvider(
                _judge_spec.base_url or "", _judge_spec.api_key_env
            )
            def judge_fn(prompt: str) -> str:
                return _judge_provider.complete(
                    prompt, _judge_spec.model, skip_correction=True
                ).text

    recorded: list[dict] = []
    write_event(
        run_id,
        "run_start",
        mode="replay" if replay else "live",
        status="running",
        notes=notes or ("replay" if replay else "live"),
        judge_model=judge_model_id,
        models=[{"id": specs[m].id, "model": specs[m].model, "role": specs[m].role} for m in ids],
        k=max(POLICY_K, REFERENCE_K) if k is None else k,
        levels=levels,
        limit=limit,
        n_items=len(qrows),
    )
    _log_status(run_id, "running", models=",".join(ids), items=len(qrows))
    try:
        for mid in ids:
            spec: ModelSpec = specs[mid]
            provider = None if replay else OpenAICompatibleProvider(spec.base_url or "", spec.api_key_env)
            n_samples = samples_for(spec, k)
            provider_misses = 0
            for q in qrows:
                for sample in range(1, n_samples + 1):
                    with engine.begin() as conn:
                        if reuse_prev_runs:
                            # Skip this sample if ANY completed run has it for this model
                            exists = conn.execute(
                                text(
                                    """
                                    SELECT 1 FROM attempts a
                                    JOIN eval_runs r2 ON r2.id = a.run_id
                                    WHERE a.model_id = :m
                                      AND a.question_id = :q
                                      AND a.sample_index = :s
                                      AND (a.run_id = :r OR r2.status = 'complete')
                                    LIMIT 1
                                    """
                                ),
                                {"r": run_id, "m": mid, "q": q["id"], "s": sample},
                            ).first()
                        else:
                            exists = conn.execute(
                                text(
                                    "SELECT 1 FROM attempts WHERE run_id=:r AND model_id=:m AND question_id=:q AND sample_index=:s"
                                ),
                                {"r": run_id, "m": mid, "q": q["id"], "s": sample},
                            ).first()
                    if exists:
                        continue
                    prompt = contest_prompt(q["stem"])
                    provider_note = None
                    if replay:
                        completion = fixture.complete_lookup(mid, q["id"], sample)
                    else:
                        try:
                            completion = provider.complete(
                                prompt, spec.model,
                                thinking=bool(spec.thinking),
                                json_mode=bool(getattr(spec, "json_mode", False)),
                            )
                            provider_misses = 0
                        except Exception as exc:
                            provider_misses += 1
                            provider_note = f"{type(exc).__name__}: {exc}"
                            completion = Completion(
                                text=f"[provider_error] {provider_note}",
                                latency_ms=0,
                                tokens_in=None,
                                tokens_out=None,
                            )
                    numeric = bool(q["numeric"])
                    # grade() is the canonical scorer — its parsed_ans and
                    # unparsable flag are the source of truth.
                    # parse_final() is only used as a fallback display value
                    # when grade() could not extract an answer.
                    reward, parsed_ans, unparsable = grade(
                        completion.text, q["key"], numeric=numeric,
                        judge_fn=judge_fn,
                    )
                    correct = reward == 1
                    parsed = parse_final(completion.text, numeric=numeric)
                    # Always prefer grade's parsed answer to avoid divergence
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
                    write_event(
                        run_id,
                        "contest_sample",
                        model_id=mid,
                        provider_model=spec.model,
                        question_id=q["id"],
                        sample_index=sample,
                        source=q["source"],
                        template=q["template"],
                        level=q["level"],
                        split=q["split"],
                        topic=q["topic"],
                        band=q["band"],
                        params=json.loads(q["params_json"] or "{}"),
                        gold_key=q["key"],
                        prompt=prompt,
                        raw_output=completion.text,
                        parsed_answer=parsed.answer,
                        reward=reward,
                        is_correct=correct,
                        unparsable=parsed.unparsable,
                        latency_ms=completion.latency_ms,
                        tokens_in=completion.tokens_in,
                        tokens_out=completion.tokens_out,
                    )
                    print(
                        f"{mid} {q['id']} k={sample} correct={correct} unparsable={parsed.unparsable}"
                        + (f" provider_error={provider_note}" if provider_note else ""),
                        flush=True,
                    )
                    if provider_note and provider_misses >= 3:
                        raise RuntimeError(
                            f"provider failed {provider_misses} samples in a row; stopping. "
                            f"Last error: {provider_note}"
                        )
        with engine.begin() as conn:
            conn.execute(text("UPDATE eval_runs SET status='complete' WHERE id=:id"), {"id": run_id})
        write_event(run_id, "run_end", status="complete")
        _log_status(run_id, "complete")
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
            print(f"wrote {FIXTURES_PATH}", flush=True)
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        with engine.begin() as conn:
            conn.execute(text("UPDATE eval_runs SET status='failed' WHERE id=:id"), {"id": run_id})
        write_event(run_id, "run_end", status="failed", error=err)
        _log_status(run_id, "failed", error=err)
        raise
    return run_id


def run_verifier(
    run_id: str,
    verifier_model_id: str,
    failed_question_ids: list[str],
    k: int,
) -> None:
    """Run the verifier model on questions the primary model completely failed.

    Attempts are stored in the normal ``attempts`` table with the verifier's
    model_id, so they participate in all existing queries transparently.
    """
    specs = {m.id: m for m in sync_models()}
    spec = specs.get(verifier_model_id)
    if spec is None:
        raise RuntimeError(f"unknown model {verifier_model_id}")

    engine = get_engine()
    id_placeholders = ",".join(f":fq{i}" for i in range(len(failed_question_ids)))
    id_params = {f"fq{i}": qid for i, qid in enumerate(failed_question_ids)}
    with engine.connect() as conn:
        q_rows = conn.execute(
            text(
                "SELECT id, stem, key, numeric, level, topic, band, split, template, "
                f"params_json, source FROM questions WHERE id IN ({id_placeholders})"
            ),
            id_params,
        ).mappings().all()

    seen: set[str] = set()
    qrows: list[dict] = []
    wanted = set(failed_question_ids)
    for q in q_rows:
        if q["id"] not in seen and q["id"] in wanted:
            seen.add(q["id"])
            qrows.append(dict(q))

    provider = OpenAICompatibleProvider(spec.base_url or "", spec.api_key_env)
    for q in qrows:
        for sample in range(1, k + 1):
            with engine.begin() as conn:
                exists = conn.execute(
                    text(
                        "SELECT 1 FROM attempts WHERE run_id=:r AND model_id=:m "
                        "AND question_id=:q AND sample_index=:s"
                    ),
                    {"r": run_id, "m": verifier_model_id, "q": q["id"], "s": sample},
                ).first()
                if exists:
                    continue
            try:
                completion = provider.complete(
                    contest_prompt(q["stem"]), spec.model, thinking=bool(spec.thinking)
                )
            except Exception as exc:
                completion = Completion(
                    text=f"[verifier_error] {exc}", latency_ms=0,
                    tokens_in=None, tokens_out=None,
                )
            numeric = bool(q["numeric"])
            parsed = parse_final(completion.text, numeric=numeric)
            reward, parsed_ans, unparsable = grade(completion.text, q["key"], numeric=numeric)
            if parsed_ans is not None:
                parsed.answer = parsed_ans
            parsed.unparsable = unparsable
            with engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        INSERT OR IGNORE INTO attempts (
                            run_id, model_id, question_id, sample_index, raw_output,
                            parsed_answer, is_correct, unparsable, latency_ms,
                            tokens_in, tokens_out
                        ) VALUES (
                            :r, :m, :q, :s, :raw, :ans, :ok, :unp, :lat, :ti, :to
                        )
                        """
                    ),
                    {
                        "r": run_id, "m": verifier_model_id, "q": q["id"], "s": sample,
                        "raw": completion.text, "ans": parsed.answer,
                        "ok": int(reward), "unp": int(parsed.unparsable),
                        "lat": completion.latency_ms,
                        "ti": completion.tokens_in, "to": completion.tokens_out,
                    },
                )
            print(
                f"verifier {verifier_model_id} {q['id']} k={sample} "
                f"correct={bool(reward)} unparsable={parsed.unparsable}",
                flush=True,
            )


def generate_results_csv(
    run_id: str,
    primary_model_ids: list[str],
    verifier_model_id: str | None = None,
) -> str:
    """Return a wide CSV string with one row per question.

    Columns: question metadata, then for each model:
      {model}_response_1..k, {model}_final_answer_1..k, {model}_correct_1..k,
      {model}_pass_rate.
    Final column: needs_sme_review (True if all attempts across all models failed).
    """
    import csv
    import io as _io

    engine = get_engine()
    all_model_ids = primary_model_ids + ([verifier_model_id] if verifier_model_id else [])

    # Fetch all attempts for this run grouped by model+question
    with engine.connect() as conn:
        attempts = conn.execute(
            text(
                """
                SELECT a.model_id, a.question_id, a.sample_index,
                       a.raw_output, a.parsed_answer, a.is_correct,
                       q.stem, q.key, q.topic, q.band, q.numeric
                FROM attempts a
                JOIN questions q ON q.id = a.question_id
                WHERE a.run_id = :r
                ORDER BY a.question_id, a.model_id, a.sample_index
                """
            ),
            {"r": run_id},
        ).mappings().all()

    # Index: {question_id: {model_id: {sample_index: row}}}
    by_q: dict[str, dict] = {}
    q_meta: dict[str, dict] = {}
    for a in attempts:
        qid = a["question_id"]
        mid = a["model_id"]
        if qid not in by_q:
            by_q[qid] = {}
            q_meta[qid] = {
                "stem": a["stem"], "key": a["key"],
                "topic": a["topic"], "band": a["band"], "numeric": a["numeric"],
            }
        by_q[qid].setdefault(mid, {})[a["sample_index"]] = dict(a)

    if not by_q:
        return "question_id,error\n,no attempts found for this run\n"

    # Fetch run metadata for the timestamp columns
    with engine.connect() as conn:
        run_meta = conn.execute(
            text("SELECT id, created_at, notes FROM eval_runs WHERE id=:r"),
            {"r": run_id},
        ).mappings().first()
    run_created_at = run_meta["created_at"] if run_meta else ""
    run_notes = run_meta["notes"] if run_meta else ""

    # Determine max k per model
    k_per_model: dict[str, int] = {}
    for qid, models in by_q.items():
        for mid, samples in models.items():
            k_per_model[mid] = max(k_per_model.get(mid, 0), max(samples.keys()))

    # Build header
    meta_cols = ["run_id", "run_created_at", "batch", "question_id", "stem", "topic", "band", "key", "numeric"]
    model_cols: list[str] = []
    for mid in all_model_ids:
        k = k_per_model.get(mid, 0)
        safe = mid.replace("-", "_").replace(".", "_")
        for i in range(1, k + 1):
            model_cols.append(f"{safe}_response_{i}")
        for i in range(1, k + 1):
            model_cols.append(f"{safe}_final_answer_{i}")
        for i in range(1, k + 1):
            model_cols.append(f"{safe}_correct_{i}")
        model_cols.append(f"{safe}_pass_rate")
    all_cols = meta_cols + model_cols + ["needs_sme_review"]

    out = _io.StringIO()
    writer = csv.DictWriter(out, fieldnames=all_cols, extrasaction="ignore")
    writer.writeheader()

    for qid, models in sorted(by_q.items()):
        meta = q_meta[qid]
        def _oneline(s: str) -> str:
            return (s or "").replace("\r\n", " ").replace("\n", " ").replace("\r", " ").strip()

        row: dict = {
            "run_id": run_id,
            "run_created_at": run_created_at,
            "batch": run_notes,
            "question_id": qid,
            "stem": _oneline(meta["stem"])[:300],
            "topic": meta["topic"],
            "band": meta["band"],
            "key": f'="{_oneline(meta["key"])}"',
            "numeric": meta["numeric"],
        }
        all_correct_counts: list[int] = []
        for mid in all_model_ids:
            k = k_per_model.get(mid, 0)
            safe = mid.replace("-", "_").replace(".", "_")
            samples = models.get(mid, {})
            n_correct = 0
            for i in range(1, k + 1):
                s = samples.get(i, {})
                raw = (s.get("raw_output") or "").replace("\r\n", " ").replace("\n", " ").replace("\r", " ").strip()
                # Truncate long reasoning chains — final_answer column has the extracted answer
                if len(raw) > 400:
                    raw = raw[:200] + " … " + raw[-150:]
                row[f"{safe}_response_{i}"] = raw
                # Prefix with ="..." so Excel/Numbers treats the value as plain text
                # and doesn't auto-convert fractions like 4/5 to dates (5-Apr).
                ans = s.get("parsed_answer") or ""
                row[f"{safe}_final_answer_{i}"] = f'="{ans}"' if ans else ""
                correct = bool(s.get("is_correct", False))
                row[f"{safe}_correct_{i}"] = correct
                if correct:
                    n_correct += 1
            row[f"{safe}_pass_rate"] = round(n_correct / k, 3) if k else 0
            all_correct_counts.append(n_correct)
        row["needs_sme_review"] = all(c == 0 for c in all_correct_counts)
        writer.writerow(row)

    return out.getvalue()


def _log_status(run_id: str, status: str, **fields) -> None:
    parts = [f"run {run_id} status={status}"]
    for key, value in fields.items():
        text_value = " ".join(str(value).split())
        if len(text_value) > 300:
            text_value = text_value[:300] + "..."
        parts.append(f"{key}={text_value}")
    print(" ".join(parts), flush=True)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--live", action="store_true")
    p.add_argument("--replay", action="store_true")
    p.add_argument("--k", type=int, default=None, help="override samples; default policy=8, reference=3")
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--levels", nargs="*", type=int, default=[1, 2, 3])
    p.add_argument("--write-fixtures", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="score only the first N items (smoke test)")
    p.add_argument("--questions", nargs="*", default=None, help="score only these question ids")
    p.add_argument("--notes", default=None, help="batch label stored in eval_runs.notes")
    p.add_argument("--reuse-prev-runs", action="store_true",
                   help="skip (model, question, sample) triples already present in any completed run")
    p.add_argument("--judge-model", default=None,
                   help="model id to use as LLM judge when exact match fails")
    args = p.parse_args()
    if not args.live and not args.replay:
        raise SystemExit("pass --live or --replay")
    if args.replay:
        gap = replay_gap(
            args.models,
            args.levels,
            k=args.k,
            limit=args.limit,
            question_ids=args.questions,
        )
        if gap:
            raise SystemExit(gap)
    run_id = run_contest(
        args.models,
        k=args.k,
        levels=args.levels,
        replay=args.replay,
        write_fixtures=args.write_fixtures,
        limit=args.limit,
        question_ids=args.questions,
        notes=args.notes,
        reuse_prev_runs=args.reuse_prev_runs,
        judge_model_id=args.judge_model,
    )
    print(f"run {run_id}", flush=True)


if __name__ == "__main__":
    main()

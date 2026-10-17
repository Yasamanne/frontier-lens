from __future__ import annotations

import json

from sqlalchemy import text

from frontier_lens.calibrate import print_knob_report, require_live_if_configured
from frontier_lens.db import get_engine
from frontier_lens.ids import item_id
from frontier_lens.sandbox import execute
from frontier_lens.split import leakage, rl_split
from frontier_lens.templates import all_specs


def assign_splits(specs) -> list[tuple]:
    out = [(spec, rl_split(spec.template, spec.level)) for spec in specs]
    leaked = leakage([(spec.template, split) for spec, split in out])
    if leaked:
        raise RuntimeError(f"template leakage across train/eval: {leaked}")
    return out


def seed(*, force: bool = False) -> int:
    require_live_if_configured()
    engine = get_engine()
    with engine.connect() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM questions WHERE source='seed'")).scalar_one()
    if n and not force:
        print(f"seed already present ({n} rows). pass --force to rebuild.")
        return n
    print_knob_report()

    # Run every key in the sandbox before touching the table, so the write
    # transaction stays short (other writers, e.g. job heartbeats, are not
    # blocked) and a failing spec leaves the existing bank untouched.
    rows = []
    for spec, split in assign_splits(all_specs()):
        result = execute(spec.lang, spec.code, spec.input_files, twice=True)
        if not result.ok:
            raise RuntimeError(
                f"seed failed {spec.topic} L{spec.level} {spec.template}: {result.reason} {result.stderr}"
            )
        rows.append(
            {
                "id": item_id(spec.template, spec.params, spec.code, spec.input_files),
                "topic": spec.topic,
                "level": spec.level,
                "band": spec.band,
                "split": split,
                "template": spec.template,
                "params": json.dumps(spec.params, sort_keys=True),
                "stem": spec.stem,
                "code": spec.code,
                "lang": spec.lang,
                "files": json.dumps(spec.input_files, sort_keys=True),
                "key": result.stdout,
                "numeric": int(spec.numeric),
            }
        )
    with engine.begin() as conn:
        if force:
            conn.execute(text("DELETE FROM questions WHERE source='seed'"))
        for row in rows:
            conn.execute(
                text(
                    """
                    INSERT OR REPLACE INTO questions (
                        id, topic, level, band, split, template, params_json, stem, code, lang,
                        input_files_json, key, numeric, source
                    ) VALUES (
                        :id, :topic, :level, :band, :split, :template, :params, :stem, :code, :lang,
                        :files, :key, :numeric, 'seed'
                    )
                    """
                ),
                row,
            )
    inserted = len(rows)
    print(f"seeded {inserted} items")
    return inserted


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    seed(force=args.force)


if __name__ == "__main__":
    main()

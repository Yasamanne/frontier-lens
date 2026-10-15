from __future__ import annotations

import json

from sqlalchemy import text

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
    engine = get_engine()
    with engine.begin() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM questions WHERE source='seed'")).scalar_one()
        if n and not force:
            print(f"seed already present ({n} rows). pass --force to rebuild.")
            return n
        if force:
            conn.execute(text("DELETE FROM questions WHERE source='seed'"))

    labeled = assign_splits(all_specs())
    inserted = 0
    with engine.begin() as conn:
        for spec, split in labeled:
            result = execute(spec.lang, spec.code, spec.input_files, twice=True)
            if not result.ok:
                raise RuntimeError(
                    f"seed failed {spec.topic} L{spec.level} {spec.template}: {result.reason} {result.stderr}"
                )
            qid = item_id(spec.template, spec.params, spec.code, spec.input_files)
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
                {
                    "id": qid,
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
                },
            )
            inserted += 1
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

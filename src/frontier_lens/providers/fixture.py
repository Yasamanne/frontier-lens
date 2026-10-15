from __future__ import annotations

import json
from pathlib import Path

from frontier_lens.config import FIXTURES_PATH
from frontier_lens.providers.base import Completion


class FixtureProvider:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or FIXTURES_PATH
        self.rows: dict[tuple[str, str, int], dict] = {}
        if self.path.exists():
            data = json.loads(self.path.read_text())
            for row in data.get("rows", []):
                key = (row["model_id"], row["question_id"], int(row["sample_index"]))
                self.rows[key] = row

    def complete_lookup(self, model_id: str, question_id: str, sample_index: int) -> Completion:
        row = self.rows.get((model_id, question_id, sample_index))
        if not row:
            raise RuntimeError(
                f"no fixture for {model_id} {question_id} sample {sample_index}"
            )
        return Completion(row.get("raw_output") or "", int(row.get("latency_ms") or 0), None, None)

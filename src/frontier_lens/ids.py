from __future__ import annotations

import hashlib
import json


def item_id(template: str, params: dict, code: str, input_files: dict[str, str]) -> str:
    payload = json.dumps(
        {"template": template, "params": params, "code": code, "files": input_files},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def item_set_hash(question_ids: list[str]) -> str:
    joined = ",".join(sorted(question_ids))
    return hashlib.sha256(joined.encode()).hexdigest()[:16]

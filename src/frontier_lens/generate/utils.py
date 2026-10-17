from __future__ import annotations

import json
import re as _re


def _extract_json(text: str) -> dict:
    """Robustly extract the first JSON object from a model response.

    Tries (in order):
    1. Markdown fenced block  ``` json { … } ```
    2. Outermost ``{`` … ``}`` in the raw text
    3. Each line that starts with ``{``

    Raises ``ValueError`` if nothing parses.
    """
    # 1. markdown code fence (```json or plain ```)
    m = _re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, _re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 2. outermost { … }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    # 3. scan line-by-line for a self-contained JSON object
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    raise ValueError(f"no valid JSON object found in response ({len(text)} chars)")


def _label_or_infer(value: str | None, field: str) -> str:
    """Return the value if set, else a prompt placeholder telling the model to infer it."""
    if value and value.strip():
        return value.strip()
    return f"(unknown — infer {field} from the seed question)"


def _extract_stem_from_raw(raw: str) -> str:
    """Extract just the stem from a raw generator response for use in fix prompts."""
    try:
        return _extract_json(raw).get("stem", raw[:400])
    except Exception:
        return raw[:400]

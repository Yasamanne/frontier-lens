from __future__ import annotations

import re
from dataclasses import dataclass

FINAL_RE = re.compile(r"FINAL:\s*(.+)\s*$", re.MULTILINE)


@dataclass
class Parsed:
    answer: str | None
    unparsable: bool


def normalize(text: str, numeric: bool) -> str:
    text = " ".join(text.strip().split())
    if numeric:
        try:
            value = float(text.replace(",", ""))
            if value.is_integer():
                return str(int(value))
            return format(value, "g")
        except ValueError:
            pass
    return text


def parse_final(raw: str, numeric: bool = True) -> Parsed:
    if not raw:
        return Parsed(None, True)
    matches = FINAL_RE.findall(raw)
    if not matches:
        return Parsed(None, True)
    return Parsed(normalize(matches[-1], numeric), False)

from __future__ import annotations

import re
from dataclasses import dataclass

# Primary: exact match on any FINAL: line (case-insensitive to catch typos)
FINAL_RE = re.compile(r"FINAL:\s*(.+)\s*$", re.MULTILINE)

# Fallback patterns tried in order when no FINAL: line is found.
# Each captures the candidate answer in group 1.
_FALLBACKS = [
    # "final =" or "final=" (space before colon omitted)
    re.compile(r"final\s*=\s*(.+)\s*$", re.MULTILINE | re.IGNORECASE),
    # "Answer: 42" / "The answer: 42" / "Result: 42" on a line
    re.compile(r"^(?:the\s+)?(?:answer|result|output)\s*:\s*(.+)\s*$", re.MULTILINE | re.IGNORECASE),
    # "The answer is 42" / "result is 42" anywhere
    re.compile(r"(?:answer|result|output)\s+is\s+([+-]?\d[\d,\.]*)", re.IGNORECASE),
    # "= 42" at end of line (last numeric after equals sign)
    re.compile(r"=\s*([+-]?\d[\d,\.]+)\s*$", re.MULTILINE),
    # A bare number on its own line (last such line wins)
    re.compile(r"^([+-]?\d[\d,\.]+)\s*$", re.MULTILINE),
]


@dataclass
class Parsed:
    answer: str | None
    unparsable: bool


def normalize(text: str, numeric: bool) -> str:
    """Normalise an answer string for comparison.

    Only strips presentation-level LaTeX wrappers that carry no mathematical
    meaning — outer ``$...$`` delimiters and ``\\boxed{...}`` — so that
    ``$B$`` == ``B`` and ``\\boxed{42}`` == ``42``.

    Complex expressions like ``256 + \\frac{176}{3}\\pi`` are left intact
    so their meaning is preserved.
    """
    text = " ".join(text.strip().split())

    # 1. Unwrap \boxed{...} → keep inner content
    text = re.sub(r"\\boxed\{([^}]*)\}", r"\1", text)

    # 2. Strip outer $ delimiters: $B$ → B, $42$ → 42
    #    (only strip $ that wrap the whole value, not mid-expression ones)
    text = re.sub(r"^\$(.+)\$$", r"\1", text.strip())

    # 3. Remove any remaining bare $ signs (e.g. $B$ after step 2 already removed them)
    text = re.sub(r"\$", "", text)

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


def _fallback_answer(raw: str, numeric: bool) -> str | None:
    """Try secondary patterns when no FINAL: line is present."""
    # Non-numeric questions need an explicit FINAL: — don't guess
    if not numeric:
        return None
    for pat in _FALLBACKS:
        matches = pat.findall(raw)
        if matches:
            candidate = matches[-1].strip()
            # Only accept if it looks like a number after normalisation
            norm = normalize(candidate, numeric=True)
            try:
                float(norm)
                return norm
            except (ValueError, TypeError):
                continue
    return None


def parse_final(raw: str, numeric: bool = True) -> Parsed:
    if not raw:
        return Parsed(None, True)

    # Primary: look for FINAL: (case-sensitive as required, but we also catch
    # the case-insensitive variant via FINAL_RE already being case-sensitive —
    # fallback handles the rest)
    matches = FINAL_RE.findall(raw)
    if matches:
        return Parsed(normalize(matches[-1], numeric), False)

    # Case-insensitive fallback for "final:" typos
    ci_matches = re.findall(r"final:\s*(.+)\s*$", raw, re.MULTILINE | re.IGNORECASE)
    if ci_matches:
        return Parsed(normalize(ci_matches[-1], numeric), False)

    # Secondary fallback patterns (numeric only)
    fb = _fallback_answer(raw, numeric)
    if fb is not None:
        return Parsed(fb, False)

    return Parsed(None, True)

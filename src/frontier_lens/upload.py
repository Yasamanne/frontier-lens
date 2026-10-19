"""Validate and import user-supplied seed datasets.

Supported formats: CSV, JSONL.
Required columns: stem, key.
Optional columns (with defaults): topic, difficulty (or level), numeric, split,
                                   solution_steps.

``difficulty`` is accepted as a friendly alias for ``level``:
  easy → 1  |  medium → 2  |  hard → 3  (or raw integers 1/2/3)
If both ``difficulty`` and ``level`` are present, ``difficulty`` wins.

``solution_steps`` is stored in params_json and shown in the SME queue.
It is never passed to the policy model.

Inserted with source='uploaded' so they survive seed --force rebuilds
and are included in all contest runs alongside seed/generated items.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import unicodedata
from dataclasses import dataclass, field

from sqlalchemy import text

from frontier_lens.db import get_engine

# ── Schema ────────────────────────────────────────────────────────────────

REQUIRED_COLUMNS = {"stem", "key"}

OPTIONAL_DEFAULTS = {
    "topic": "custom",
    "level": 2,           # L2 standard
    "numeric": True,
    "split": "train",
    "solution_steps": "",
}

VALID_LEVELS = {1, 2, 3}
VALID_SPLITS = {"train", "eval", "harness"}
BAND_MAP = {1: "harness", 2: "standard", 3: "hard"}

# Friendly difficulty strings → internal level integer
DIFFICULTY_MAP = {
    "easy": 1, "simple": 1, "1": 1,
    "medium": 2, "standard": 2, "normal": 2, "2": 2,
    "hard": 3, "difficult": 3, "3": 3,
}

MAX_ROWS = 10_000
MAX_FILE_BYTES = 10 * 1024 * 1024   # 10 MB
MIN_ROWS_RECOMMENDED = 5


# ── Validation result ─────────────────────────────────────────────────────

@dataclass
class FuzzyPair:
    """Two rows that appear identical after normalisation."""
    row_a: dict   # first occurrence (will be kept)
    row_b: dict   # second occurrence (would be skipped if deduped)
    same_key: bool


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)   # cleaned rows ready to insert
    n_duplicates_skipped: int = 0
    n_already_in_db: int = 0
    fuzzy_pairs: list[FuzzyPair] = field(default_factory=list)  # near-duplicate pairs


# ── Parsers ───────────────────────────────────────────────────────────────

def _parse_csv(data: bytes) -> list[dict]:
    import csv
    text = data.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    return [dict(row) for row in reader]


def _parse_jsonl(data: bytes) -> list[dict]:
    rows = []
    for i, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSONL line {i}: {exc}") from exc
    return rows


# ── Column coercion ───────────────────────────────────────────────────────

def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    return s not in {"0", "false", "no", "f", "n", ""}


def _infer_numeric(key: str) -> bool:
    """Return True only if the key is a plain number.

    Uses the same light LaTeX stripping as ``parse.normalize()``:
    only unwraps ``\\boxed{...}`` and strips outer ``$...$``.
    Keys that still contain LaTeX commands (``\\sqrt``, ``\\frac``, ``\\pi``)
    or braces after that are symbolic expressions and return False.

    Examples:
      ``$42$``       → True   (digit in math mode)
      ``\\boxed{7}`` → True   (boxed digit)
      ``4\\sqrt{3}`` → False  (symbolic, not a plain number)
      ``[1, \\infty)``→ False (interval)
      ``Yes``        → False  (text)
    """
    # 1. Unwrap \boxed{...} → keep inner content
    clean = re.sub(r"\\boxed\{([^}]*)\}", r"\1", key)
    # 2. Strip outer $...$ delimiters
    clean = re.sub(r"^\$(.+)\$$", r"\1", clean.strip())
    # 3. Remove bare $ signs
    clean = re.sub(r"\$", "", clean).strip()
    # If the result still contains LaTeX commands or braces → symbolic
    if "\\" in clean or "{" in clean or "}" in clean:
        return False
    if not clean:
        return False
    try:
        float(clean)
        return True
    except ValueError:
        return False


def _coerce_int(value: object, name: str) -> int:
    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        raise ValueError(f"column '{name}' must be an integer, got {value!r}")


def _resolve_level(raw: dict, row_num: int) -> int:
    """Resolve level from 'difficulty' (preferred) or 'level' column."""
    # 'difficulty' wins if present
    if "difficulty" in raw and raw["difficulty"] not in (None, ""):
        val = str(raw["difficulty"]).strip().lower()
        if val in DIFFICULTY_MAP:
            return DIFFICULTY_MAP[val]
        # maybe a raw integer string
        try:
            lvl = int(val)
            if lvl in VALID_LEVELS:
                return lvl
        except ValueError:
            pass
        raise ValueError(
            f"row {row_num}: 'difficulty' must be easy/medium/hard or 1/2/3, got {raw['difficulty']!r}"
        )
    # fall back to 'level'
    raw_level = raw.get("level", OPTIONAL_DEFAULTS["level"])
    level = _coerce_int(raw_level, "level")
    if level not in VALID_LEVELS:
        raise ValueError(f"row {row_num}: 'level' must be 1, 2, or 3, got {raw_level!r}")
    return level


def _clean_row(raw: dict, row_num: int) -> dict:
    """Return a cleaned row dict or raise ValueError with a row-specific message."""
    stem = str(raw.get("stem", "") or "").strip()
    key  = str(raw.get("key",  "") or "").strip()

    if not stem:
        raise ValueError(f"row {row_num}: 'stem' is empty")
    if not key:
        raise ValueError(f"row {row_num}: 'key' is empty")

    topic = str(raw.get("topic", OPTIONAL_DEFAULTS["topic"]) or OPTIONAL_DEFAULTS["topic"]).strip() or "custom"
    level = _resolve_level(raw, row_num)

    if "numeric" in raw and raw["numeric"] not in (None, ""):
        # Explicit column → respect the user's setting
        numeric = _coerce_bool(raw["numeric"])
    else:
        # Not provided → auto-detect from the key value
        numeric = _infer_numeric(key)

    split = str(raw.get("split", OPTIONAL_DEFAULTS["split"]) or "train").strip().lower()
    if split not in VALID_SPLITS:
        raise ValueError(f"row {row_num}: 'split' must be one of {sorted(VALID_SPLITS)}, got {split!r}")

    solution_steps = str(raw.get("solution_steps", "") or "").strip()

    return {
        "stem": stem,
        "key": key,
        "topic": topic,
        "level": level,
        "numeric": numeric,
        "split": split,
        "solution_steps": solution_steps,
        # Missing fields can be filled by labeling.label_rows.
        "topic_missing": not str(raw.get("topic") or "").strip(),
        "level_missing": all(raw.get(c) in (None, "") for c in ("difficulty", "level")),
    }


def _normalise_for_fuzzy(text: str) -> str:
    """Collapse case, whitespace, LaTeX delimiters, and trailing punctuation."""
    text = unicodedata.normalize("NFKC", text)          # unicode canonical form
    text = text.lower()
    text = re.sub(r"[\$\\\(\)\[\]]", "", text)          # LaTeX delimiters $ \ ( ) [ ]
    text = re.sub(r"\\[a-zA-Z]+", "", text)             # \commands like \frac \mathbb
    text = re.sub(r"[{}]", "", text)                     # curly braces
    text = re.sub(r"[?.,;:]+$", "", text.strip())       # trailing punctuation (not ! — could be factorial)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _make_id(stem: str, key: str) -> str:
    """Exact identity — SHA-256 of raw stem + key."""
    payload = f"{stem}\x00{key}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _make_fuzzy_id(stem: str, key: str) -> str:
    """Near-duplicate identity — SHA-256 of normalised stem + normalised key."""
    payload = f"{_normalise_for_fuzzy(stem)}\x00{_normalise_for_fuzzy(key)}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ── Main validation entry point ───────────────────────────────────────────

def validate(data: bytes, filename: str) -> ValidationResult:
    result = ValidationResult(ok=False)

    # File size
    if len(data) > MAX_FILE_BYTES:
        result.errors.append(
            f"File too large ({len(data) / 1048576:.1f} MB). "
            f"The maximum is {MAX_FILE_BYTES // 1048576} MB; split the file and upload it in parts."
        )
        return result

    # Parse
    try:
        if filename.lower().endswith(".jsonl"):
            raw_rows = _parse_jsonl(data)
        elif filename.lower().endswith(".csv"):
            raw_rows = _parse_csv(data)
        else:
            result.errors.append("Unsupported format. Upload a .csv or .jsonl file.")
            return result
    except ValueError as exc:
        result.errors.append(f"Parse error: {exc}")
        return result

    if not raw_rows:
        result.errors.append("File is empty — no rows found.")
        return result

    if len(raw_rows) > MAX_ROWS:
        result.errors.append(f"Too many rows ({len(raw_rows):,}). Max is {MAX_ROWS:,}.")
        return result

    # Check required columns exist
    first = raw_rows[0]
    missing = REQUIRED_COLUMNS - set(first.keys())
    if missing:
        result.errors.append(
            f"Missing required columns: {sorted(missing)}. "
            f"Found columns: {sorted(first.keys())}."
        )
        return result

    # Clean and validate each row
    cleaned: list[dict] = []
    seen_ids: set[str] = set()
    for i, raw in enumerate(raw_rows, 1):
        try:
            row = _clean_row(raw, i)
        except ValueError as exc:
            result.errors.append(str(exc))
            if len(result.errors) >= 10:
                result.errors.append("… (stopping after 10 errors)")
                break
            continue
        qid = _make_id(row["stem"], row["key"])
        if qid in seen_ids:
            result.n_duplicates_skipped += 1
            continue
        seen_ids.add(qid)
        row["id"] = qid
        cleaned.append(row)

    if result.errors:
        return result

    if len(cleaned) < MIN_ROWS_RECOMMENDED:
        result.warnings.append(
            f"Only {len(cleaned)} unique row(s). "
            f"At least {MIN_ROWS_RECOMMENDED} is recommended for meaningful evaluation."
        )

    # Detect fuzzy near-duplicates (warn only — do not auto-reject)
    fuzzy_seen: dict[str, dict] = {}   # fuzzy_id → first row
    fuzzy_pairs: list[FuzzyPair] = []
    for row in cleaned:
        fid = _make_fuzzy_id(row["stem"], row["key"])
        if fid in fuzzy_seen:
            first = fuzzy_seen[fid]
            fuzzy_pairs.append(FuzzyPair(
                row_a=first,
                row_b=row,
                same_key=(_normalise_for_fuzzy(first["key"]) == _normalise_for_fuzzy(row["key"])),
            ))
        else:
            fuzzy_seen[fid] = row
    result.fuzzy_pairs = fuzzy_pairs

    # Check which IDs already exist in DB
    engine = get_engine()
    with engine.connect() as conn:
        existing = {
            row[0]
            for row in conn.execute(
                text("SELECT id FROM questions WHERE source='uploaded'")
            ).all()
        }
    already = [r for r in cleaned if r["id"] in existing]
    result.n_already_in_db = len(already)
    if already:
        result.warnings.append(
            f"{len(already)} row(s) are already in the database and will be skipped on import."
        )

    result.ok = True
    result.rows = cleaned
    return result


# ── Import ────────────────────────────────────────────────────────────────

def import_rows(rows: list[dict], replace: bool = False, filename: str | None = None) -> int:
    """Insert validated rows into questions table. Returns number of new rows inserted."""
    engine = get_engine()
    inserted = 0
    with engine.begin() as conn:
        for row in rows:
            band = BAND_MAP[row["level"]]
            if replace:
                conn.execute(text("DELETE FROM questions WHERE id = :id"), {"id": row["id"]})
            try:
                conn.execute(
                    text(
                        """
                        INSERT OR IGNORE INTO questions (
                            id, topic, level, band, split, template, params_json,
                            stem, code, lang, input_files_json, key, numeric, source,
                            upload_filename
                        ) VALUES (
                            :id, :topic, :level, :band, :split, :template, :params,
                            :stem, :code, :lang, :files, :key, :numeric, 'uploaded',
                            :fname
                        )
                        """
                    ),
                    {
                        "id": row["id"],
                        "topic": row["topic"],
                        "level": row["level"],
                        "band": band,
                        "split": row["split"],
                        "template": f"uploaded_{row['topic']}",
                        "params": json.dumps({
                            "solution_steps": row.get("solution_steps", ""),
                            **({"label_source": row["labels"]} if row.get("labels") else {}),
                        }),
                        "stem": row["stem"],
                        "code": "",
                        "lang": "python",
                        "files": "{}",
                        "key": row["key"],
                        "numeric": int(row["numeric"]),
                        "fname": filename or "",
                    },
                )
                inserted += conn.execute(text("SELECT changes()")).scalar() or 0
            except Exception:
                pass
    return inserted


def uploaded_topics() -> list[str]:
    """Distinct topics already used by uploaded questions (labels to reuse)."""
    with get_engine().connect() as conn:
        return list(
            conn.execute(
                text("SELECT DISTINCT topic FROM questions "
                "WHERE source='uploaded' AND topic != 'custom' ORDER BY 1")
            ).scalars()
        )


def uploaded_ids() -> set[str]:
    with get_engine().connect() as conn:
        return set(conn.execute(text("SELECT id FROM questions WHERE source='uploaded'")).scalars())


def delete_uploaded() -> int:
    """Remove all uploaded questions. Returns count deleted."""
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM questions WHERE source='uploaded'"))
        return conn.execute(text("SELECT changes()")).scalar() or 0

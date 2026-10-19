"""Allowlisted subprocess runner. Not a production isolation boundary."""

from __future__ import annotations

import ast
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from frontier_lens.config import PYTHON_BIN, TIMEOUT_SEC

ALLOWED_PY_IMPORTS = frozenset({"math", "itertools", "collections", "decimal"})
BANNED_PY_CALLS = frozenset({"open", "exec", "eval", "__import__", "compile", "input"})
ALLOWED_SHELL = frozenset(
    {"cat", "cut", "sort", "uniq", "wc", "head", "tail", "tr", "grep", "printf", "awk"}
)
# Banned only outside single quotes, so an awk program can contain them.
_BANNED_OUTSIDE_QUOTES = ("$(", "`", ";", "&&", "||", ">", "<", "$")


@dataclass
class SandboxResult:
    ok: bool
    stdout: str
    stderr: str
    reason: str | None = None


class SandboxError(Exception):
    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        super().__init__(detail or reason)


def validate_python(code: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise SandboxError("unsafe", f"syntax: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_PY_IMPORTS:
                    raise SandboxError("unsafe", f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_PY_IMPORTS:
                raise SandboxError("unsafe", f"from {node.module}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise SandboxError("unsafe", f"dunder attr {node.attr}")
        elif isinstance(node, ast.Name) and node.id.startswith("__"):
            raise SandboxError("unsafe", f"dunder name {node.id}")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in BANNED_PY_CALLS:
                raise SandboxError("unsafe", f"call {func.id}")


def _outside_single_quotes(command: str) -> str:
    parts: list[str] = []
    quoted = False
    for ch in command:
        if ch == "'":
            quoted = not quoted
        elif not quoted:
            parts.append(ch)
    if quoted:
        raise SandboxError("unsafe", "unclosed quote")
    return "".join(parts)


def _split_pipeline(command: str) -> list[str]:
    """Split a shell pipeline on ``|`` characters that are OUTSIDE single quotes.

    A naive ``command.split("|")`` would break valid awk programs that use
    ``|`` inside single-quoted strings, e.g.::

        awk '{gsub(/,/,"|"); print}' data.csv

    This splitter only cuts on ``|`` characters in the unquoted regions.
    """
    stages: list[str] = []
    current: list[str] = []
    in_sq = False
    for ch in command:
        if ch == "'":
            in_sq = not in_sq
            current.append(ch)
        elif ch == "|" and not in_sq:
            stages.append("".join(current))
            current = []
        else:
            current.append(ch)
    stages.append("".join(current))
    return stages


def validate_shell(command: str) -> None:
    outside = _outside_single_quotes(command)
    if any(tok in outside for tok in _BANNED_OUTSIDE_QUOTES):
        raise SandboxError("unsafe", "banned shell metacharacter")
    stages = _split_pipeline(command)
    if not stages:
        raise SandboxError("unsafe", "empty command")
    for raw in stages:
        try:
            argv = shlex.split(raw)
        except ValueError as exc:
            raise SandboxError("unsafe", str(exc)) from exc
        if not argv:
            raise SandboxError("unsafe", "empty pipeline stage")
        if argv[0] not in ALLOWED_SHELL:
            raise SandboxError("unsafe", f"command {argv[0]}")
        for arg in argv[1:]:
            if arg.startswith("/") and arg not in ("/dev/null",):
                raise SandboxError("unsafe", f"absolute path {arg}")


def _run(argv: list[str], cwd: Path, extra_env: dict[str, str] | None = None) -> SandboxResult:
    env = {"LC_ALL": "C", "PATH": "/usr/bin:/bin:/usr/local/bin"}
    if extra_env:
        env.update(extra_env)
    try:
        proc = subprocess.run(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SEC,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return SandboxResult(False, "", "", "timeout")
    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    if proc.returncode != 0:
        return SandboxResult(False, stdout, stderr, "exec_error")
    return SandboxResult(True, stdout, stderr, None)


def run_python(code: str, input_files: dict[str, str] | None = None) -> SandboxResult:
    validate_python(code)
    with tempfile.TemporaryDirectory(prefix="fl-py-") as tmp:
        cwd = Path(tmp)
        for name, content in (input_files or {}).items():
            if "/" in name or name.startswith("."):
                raise SandboxError("unsafe", f"bad filename {name}")
            (cwd / name).write_text(content, encoding="utf-8")
        script = cwd / "snippet.py"
        script.write_text(code, encoding="utf-8")
        return _run([PYTHON_BIN, "-I", str(script)], cwd)


def run_shell(command: str, input_files: dict[str, str] | None = None) -> SandboxResult:
    validate_shell(command)
    stages = [shlex.split(part) for part in command.split("|")]
    with tempfile.TemporaryDirectory(prefix="fl-sh-") as tmp:
        cwd = Path(tmp)
        for name, content in (input_files or {}).items():
            if "/" in name or name.startswith("."):
                raise SandboxError("unsafe", f"bad filename {name}")
            (cwd / name).write_text(content, encoding="utf-8")
        pipeline = " | ".join(shlex.join(stage) for stage in stages)
        return _run(["/bin/sh", "-c", pipeline], cwd)


def _require_single_line(result: SandboxResult) -> SandboxResult:
    """Fail a successful run whose stdout is not exactly one non-empty line."""
    if not result.stdout:
        return SandboxResult(False, "", "prints nothing", "empty_output")
    if "\n" in result.stdout:
        return SandboxResult(False, result.stdout, "more than one line", "multi_line_output")
    return result


def execute(
    lang: str,
    code: str,
    input_files: dict[str, str] | None = None,
    twice: bool = True,
    *,
    single_line: bool = False,
) -> SandboxResult:
    """Run *code*; with ``single_line=True`` its output must be one non-empty line.

    Generated questions use ``single_line=True``: the stdout is the gold key, which the
    prompts promise is a single bare value, so empty or multi-line output is rejected
    (``empty_output`` / ``multi_line_output``) instead of becoming a key nobody can match.
    """
    result = _execute(lang, code, input_files, twice)
    return _require_single_line(result) if single_line and result.ok else result


def _execute(
    lang: str, code: str, input_files: dict[str, str] | None, twice: bool
) -> SandboxResult:
    runner = run_python if lang == "python" else run_shell
    try:
        first = runner(code, input_files)
    except SandboxError as exc:
        return SandboxResult(False, "", str(exc), exc.reason)
    if not first.ok:
        return first
    if not twice:
        return first
    try:
        second = runner(code, input_files)
    except SandboxError as exc:
        return SandboxResult(False, "", str(exc), exc.reason)
    if not second.ok:
        return second
    if first.stdout != second.stdout:
        return SandboxResult(False, first.stdout, "mismatch", "non_deterministic")
    return first

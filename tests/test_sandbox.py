import pytest

from frontier_lens.sandbox import SandboxError, execute, validate_python, validate_shell


def test_python_ok():
    r = execute("python", "print(1+1)\n", twice=True)
    assert r.ok
    assert r.stdout == "2"


def test_bans_open():
    with pytest.raises(SandboxError):
        validate_python("open('x')\n")


def test_bans_import():
    with pytest.raises(SandboxError):
        validate_python("import os\nprint(1)\n")


def test_bans_sed():
    with pytest.raises(SandboxError):
        validate_shell("sed 's/a/b/' f.txt")


def test_bans_unquoted_redirect():
    with pytest.raises(SandboxError):
        validate_shell("cat data.csv > out.txt")


def test_allows_quoted_awk():
    cmd = "cat data.csv | awk -F, '$3+0 >= 2 {s+=$3} END {print s}'"
    validate_shell(cmd)
    files = {"data.csv": "a,b,1\na,b,4\na,c,3\n"}
    r = execute("shell", cmd, files, twice=True)
    assert r.ok
    assert r.stdout == "7"


def test_allows_pipeline():
    r = execute("shell", "printf 'a\\nb\\na\\n' | sort -u | wc -l", twice=True)
    assert r.ok
    assert r.stdout == "2"

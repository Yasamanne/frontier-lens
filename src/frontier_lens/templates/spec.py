from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ItemSpec:
    topic: str
    level: int
    template: str
    params: dict
    stem: str
    code: str
    lang: str
    input_files: dict[str, str] = field(default_factory=dict)
    numeric: bool = True

    @property
    def band(self) -> str:
        return {1: "harness", 2: "standard", 3: "hard"}[self.level]


def shell_stem(files: dict[str, str], command: str, question: str) -> str:
    blocks = []
    for name, content in files.items():
        blocks.append(f"File {name}:\n{content.rstrip()}")
    blocks.append(f"Command:\n{command}")
    blocks.append(f"Question: {question}")
    return "\n\n".join(blocks)


def python_stem(code: str, question: str) -> str:
    return f"Python:\n{code.rstrip()}\n\nQuestion: {question}"


def hidden_stem(question: str) -> str:
    """Problem text only. Verifier code is executed for the key, not shown."""
    return question.strip()

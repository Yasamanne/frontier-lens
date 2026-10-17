"""Procedural Python state items.

Difficulty is mutation length plus aliasing and cross-container copies,
not a named language gotcha. The stem is the program. The printed value
is a rolling checksum mod 1000.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from frontier_lens.templates.procedural import bank
from frontier_lens.templates.spec import ItemSpec, python_stem

TOPIC = "python_state"
_KINDS = ("list", "dict", "set", "str")
_LETTERS = "abcdefghij"


@dataclass
class Obj:
    kind: str
    data: object


class _Prog:
    def __init__(self, rng: random.Random, alias_p: float) -> None:
        self.rng = rng
        self.alias_p = alias_p
        self.objs: dict[int, Obj] = {}
        self.bindings: dict[str, int] = {}
        self.lines: list[str] = []
        self._next = 0
        self._guards = 0

    def add(self, kind: str, data: object, name: str) -> int:
        oid = self._next
        self._next += 1
        self.objs[oid] = Obj(kind, data)
        self.bindings[name] = oid
        return oid

    def names_for(self, oid: int) -> list[str]:
        return [name for name, bound in self.bindings.items() if bound == oid]

    def pick_name(self, oid: int) -> str:
        names = self.names_for(oid)
        aliases = [name for name in names if name.startswith("a")]
        bases = [name for name in names if not name.startswith("a")]
        if aliases and (not bases or self.rng.random() < self.alias_p):
            return self.rng.choice(aliases)
        if bases:
            return bases[0]
        return aliases[0]

    def live_oids(self) -> list[int]:
        return list(dict.fromkeys(self.bindings.values()))

    def oids_of(self, kind: str) -> list[int]:
        live = set(self.bindings.values())
        return [oid for oid, obj in self.objs.items() if oid in live and obj.kind == kind]

    def emit_init(self, containers: int) -> None:
        rot = self.rng.randrange(len(_KINDS))
        for i in range(containers):
            kind = _KINDS[(i + rot) % len(_KINDS)]
            name = f"c{i}"
            if kind == "list":
                data = [self.rng.randrange(0, 8) for _ in range(self.rng.randrange(3, 6))]
                self.lines.append(f"{name} = {data}")
                self.add(kind, data, name)
            elif kind == "dict":
                keys = self.rng.sample(list("abcd"), self.rng.randrange(2, 4))
                data = {key: self.rng.randrange(0, 6) for key in keys}
                body = ", ".join(f"{key!r}: {data[key]}" for key in sorted(data))
                self.lines.append(f"{name} = {{{body}}}")
                self.add(kind, data, name)
            elif kind == "set":
                elems = sorted(self.rng.sample(range(8), 3))
                inner = ", ".join(str(n) for n in elems)
                self.lines.append(f"{name} = set([{inner}])")
                self.add(kind, set(elems), name)
            else:
                data = "".join(self.rng.choice(_LETTERS) for _ in range(self.rng.randrange(4, 7)))
                self.lines.append(f"{name} = {data!r}")
                self.add(kind, data, name)
        n_alias = 0 if self.alias_p <= 0 else math.ceil(self.alias_p * containers - 1e-12)
        n_alias = min(n_alias, containers)
        chosen = self.rng.sample([f"c{i}" for i in range(containers)], n_alias) if n_alias else []
        for i, cname in enumerate(chosen):
            aname = f"a{i}"
            self.lines.append(f"{aname} = {cname}")
            self.bindings[aname] = self.bindings[cname]

    def _list_op(self, name: str, obj: Obj, kind: str) -> bool:
        data: list = obj.data  # type: ignore[assignment]
        if kind == "append":
            value = self.rng.randrange(0, 8)
            data.append(value)
            self.lines.append(f"{name}.append({value})")
            return True
        if kind == "pop" and data:
            data.pop()
            self.lines.append(f"{name}.pop()")
            return True
        if kind == "assign" and data:
            idx = self.rng.randrange(len(data))
            value = self.rng.randrange(0, 8)
            data[idx] = value
            self.lines.append(f"{name}[{idx}] = {value}")
            return True
        if kind == "slice" and data:
            lo = self.rng.randrange(len(data))
            hi = self.rng.randrange(lo + 1, len(data) + 1)
            value = self.rng.randrange(0, 8)
            data[lo:hi] = [value]
            self.lines.append(f"{name}[{lo}:{hi}] = [{value}]")
            return True
        if kind == "while" and len(data) >= 1:
            cap = self.rng.randrange(1, 4)
            threshold = self.rng.randrange(len(data))
            guard = f"g{self._guards}"
            self._guards += 1
            self.lines.append(f"{guard} = 0")
            self.lines.append(f"while len({name}) > {threshold} and {guard} < {cap}:")
            self.lines.append(f"    {name}.pop()")
            self.lines.append(f"    {guard} = {guard} + 1")
            ticks = 0
            while len(data) > threshold and ticks < cap:
                data.pop()
                ticks += 1
            return True
        if kind == "extend":
            others = self.oids_of("list")
            if not others:
                return False
            src_oid = self.rng.choice(others)
            extra = list(self.objs[src_oid].data)  # type: ignore[arg-type]
            data.extend(extra)
            self.lines.append(f"{name}.extend({self.pick_name(src_oid)})")
            return True
        if kind == "insert":
            idx = self.rng.randrange(len(data) + 1)
            value = self.rng.randrange(0, 8)
            data.insert(idx, value)
            self.lines.append(f"{name}.insert({idx}, {value})")
            return True
        if kind == "reverse" and len(data) >= 2:
            data.reverse()
            self.lines.append(f"{name}.reverse()")
            return True
        if kind == "copy_from" and data:
            others = [oid for oid in self.oids_of("list") if self.objs[oid].data]
            if not others:
                return False
            src_oid = self.rng.choice(others)
            src: list = self.objs[src_oid].data  # type: ignore[assignment]
            idx = self.rng.randrange(len(data))
            j = self.rng.randrange(len(src))
            data[idx] = src[j]
            self.lines.append(f"{name}[{idx}] = {self.pick_name(src_oid)}[{j}]")
            return True
        return False

    def _dict_op(self, name: str, obj: Obj, kind: str) -> bool:
        data: dict = obj.data  # type: ignore[assignment]
        if kind == "set":
            key = self.rng.choice(list("abcd"))
            value = self.rng.randrange(0, 8)
            data[key] = value
            self.lines.append(f"{name}[{key!r}] = {value}")
            return True
        if kind == "dict_del" and data:
            key = self.rng.choice(sorted(data))
            del data[key]
            self.lines.append(f"del {name}[{key!r}]")
            return True
        if kind == "from_list":
            lists = [oid for oid in self.oids_of("list") if self.objs[oid].data]
            if not lists:
                return False
            src_oid = self.rng.choice(lists)
            src: list = self.objs[src_oid].data  # type: ignore[assignment]
            key = self.rng.choice(list("abcd"))
            j = self.rng.randrange(len(src))
            data[key] = src[j]
            self.lines.append(f"{name}[{key!r}] = {self.pick_name(src_oid)}[{j}]")
            return True
        return False

    def _set_op(self, name: str, obj: Obj, kind: str) -> bool:
        data: set = obj.data  # type: ignore[assignment]
        if kind == "add":
            value = self.rng.randrange(0, 8)
            data.add(value)
            self.lines.append(f"{name}.add({value})")
            return True
        if kind == "discard":
            value = self.rng.randrange(0, 8)
            data.discard(value)
            self.lines.append(f"{name}.discard({value})")
            return True
        if kind == "set_update":
            vals = [self.rng.randrange(0, 8) for _ in range(self.rng.randrange(1, 4))]
            data.update(vals)
            self.lines.append(f"{name}.update({vals})")
            return True
        return False

    def _str_op(self, name: str, obj: Obj, kind: str) -> bool:
        data: str = obj.data  # type: ignore[assignment]
        if kind != "str_slice" or not data:
            return False
        i = self.rng.randrange(len(data))
        j = self.rng.randrange(i + 1, len(data) + 1)
        new = data[i:j] + data[:1]
        self.lines.append(f"{name} = {name}[{i}:{j}] + {name}[:1]")
        oid = self._next
        self._next += 1
        self.objs[oid] = Obj("str", new)
        self.bindings[name] = oid
        return True

    def apply(self, oid: int, kind: str) -> bool:
        obj = self.objs[oid]
        name = self.pick_name(oid)
        if obj.kind == "list":
            return self._list_op(name, obj, kind)
        if obj.kind == "dict":
            return self._dict_op(name, obj, kind)
        if obj.kind == "set":
            return self._set_op(name, obj, kind)
        return self._str_op(name, obj, kind)

    def random_step(self) -> None:
        oid = self.rng.choice(self.live_oids())
        kind = self.objs[oid].kind
        options = {
            "list": [
                "append",
                "pop",
                "assign",
                "slice",
                "while",
                "extend",
                "insert",
                "reverse",
                "copy_from",
            ],
            "dict": ["set", "dict_del", "from_list"],
            "set": ["add", "discard", "set_update"],
            "str": ["str_slice"],
        }[kind]
        self.rng.shuffle(options)
        for op in options:
            if self.apply(oid, op):
                return
        if kind == "list":
            self.apply(oid, "append")
        elif kind == "dict":
            self.apply(oid, "set")
        elif kind == "set":
            self.apply(oid, "add")
        else:
            name = self.pick_name(oid)
            self._rebind(name, str(self.objs[oid].data) + "z")
            self.lines.append(f"{name} = {name} + 'z'")

    def _rebind(self, name: str, data: str) -> None:
        oid = self._next
        self._next += 1
        self.objs[oid] = Obj("str", data)
        self.bindings[name] = oid

    def forced(self, kind: str) -> bool:
        mapping = {
            "while": "list",
            "dict_del": "dict",
            "set_update": "set",
            "str_slice": "str",
        }
        oids = self.oids_of(mapping[kind])
        if not oids:
            return False
        self.rng.shuffle(oids)
        for oid in oids:
            if self.apply(oid, kind):
                return True
        return False

    def maybe_alias(self) -> None:
        if self.alias_p <= 0:
            return
        cnames = [name for name in self.bindings if name.startswith("c")]
        if not cnames:
            return
        src = self.rng.choice(cnames)
        aliases = [name for name in self.bindings if name.startswith("a")]
        if aliases and self.rng.random() < 0.55:
            dest = self.rng.choice(aliases)
        else:
            dest = f"a{len(aliases)}"
        self.lines.append(f"{dest} = {src}")
        self.bindings[dest] = self.bindings[src]

    def checksum(self) -> None:
        self.lines.append("acc = 0")
        for name in sorted(self.bindings):
            obj = self.objs[self.bindings[name]]
            if obj.kind == "list":
                self.lines.append(f"for x in {name}:")
                self.lines.append("    acc = (acc * 31 + x) % 1000")
            elif obj.kind == "dict":
                self.lines.append(f"for k in sorted({name}):")
                self.lines.append(f"    acc = (acc * 31 + {name}[k]) % 1000")
            elif obj.kind == "set":
                self.lines.append(f"for x in sorted({name}):")
                self.lines.append("    acc = (acc * 31 + x) % 1000")
            else:
                self.lines.append(f"for ch in {name}:")
                self.lines.append("    acc = (acc * 31 + ord(ch)) % 1000")
        self.lines.append('print("%04d" % (acc % 1000))')


def generate(seed: int, *, steps: int, alias_p: float, containers: int) -> ItemSpec:
    steps = int(steps)
    containers = int(containers)
    alias_p = float(alias_p)
    if steps < 1 or containers < 1:
        raise ValueError("steps and containers must be positive")
    rng = random.Random(seed)
    prog = _Prog(rng, alias_p)
    prog.emit_init(containers)
    pending = []
    if prog.oids_of("list"):
        pending.append("while")
    if prog.oids_of("dict"):
        pending.append("dict_del")
    if prog.oids_of("set"):
        pending.append("set_update")
    if prog.oids_of("str"):
        pending.append("str_slice")
    for i in range(steps):
        did = False
        force_now = pending and (i % 2 == 0 or i >= steps - len(pending))
        if force_now and prog.forced(pending[0]):
            pending.pop(0)
            did = True
        if not did:
            prog.random_step()
        if prog.alias_p > 0 and rng.random() < prog.alias_p * 0.35:
            prog.maybe_alias()
    prog.checksum()
    code = "\n".join(prog.lines) + "\n"
    return ItemSpec(
        topic=TOPIC,
        level=2,
        template=f"{TOPIC}_train",
        params={
            "seed": seed,
            "steps": steps,
            "alias_p": alias_p,
            "containers": containers,
            "answer_space_size": 1000,
        },
        stem=python_stem(code, "what number does this print?"),
        code=code,
        lang="python",
        numeric=True,
    )


def _harness(name: str, code: str) -> ItemSpec:
    return ItemSpec(
        topic=TOPIC,
        level=1,
        template=name,
        params={},
        stem=python_stem(code, "what number does this print?"),
        code=code,
        lang="python",
        numeric=True,
    )


def specs() -> list[ItemSpec]:
    harness = [
        _harness("append_len", "x = [1, 2, 3]\nx.append(5)\nprint(len(x))\n"),
        _harness("sum_list", "x = [2, 3, 5]\nprint(sum(x))\n"),
    ]
    return bank(TOPIC, generate, harness)

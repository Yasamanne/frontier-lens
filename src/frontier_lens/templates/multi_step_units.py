"""Procedural decimal chains: conversions, brackets, rounding, compound fees.

The stem states every applied step. The sandbox program is not shown.
Knob: steps (10..20), counting the final half-up rounding.
"""

from __future__ import annotations

import random

from frontier_lens.templates.procedural import bank
from frontier_lens.templates.spec import ItemSpec

TOPIC = "multi_step_units"


class _Chain:
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        tenths = rng.randrange(15, 80)
        self.kg_s = f"{tenths // 10}.{tenths % 10}"
        self.dollars = rng.randrange(25, 160)
        self.unit = "kg"
        self.prose: list[str] = []
        self.code: list[str] = [
            "from decimal import Decimal, ROUND_HALF_UP",
            f"w = Decimal('{self.kg_s}')",
            f"m = Decimal('{self.dollars}')",
        ]

    def _fee(self) -> str:
        return str(self.rng.randrange(3, 12))

    def apply(self, kind: str) -> bool:
        if kind == "to_grams":
            if self.unit != "kg":
                return False
            self.unit = "g"
            self.prose.append("Convert the weight from kilograms to grams.")
            self.code.append("w = w * Decimal('1000')")
            return True
        if kind == "to_kg":
            if self.unit != "g":
                return False
            self.unit = "kg"
            self.prose.append("Convert the weight from grams to kilograms.")
            self.code.append("w = w / Decimal('1000')")
            return True
        if kind == "shipping":
            if self.unit == "g":
                edges = ("2000", "8000")
                nouns = ("2000 g", "8000 g")
            else:
                edges = ("2", "8")
                nouns = ("2 kg", "8 kg")
            fees = [
                str(self.rng.randrange(3, 9)),
                str(self.rng.randrange(9, 16)),
                str(self.rng.randrange(16, 25)),
            ]
            self.prose.append(
                "Add a shipping fee from the weight: "
                f"{nouns[0]} or less costs {fees[0]} dollars, "
                f"{nouns[1]} or less costs {fees[1]} dollars, "
                f"otherwise {fees[2]} dollars."
            )
            self.code.append(
                f"if w <= Decimal('{edges[0]}'):\n"
                f"    m = m + Decimal('{fees[0]}')\n"
                f"elif w <= Decimal('{edges[1]}'):\n"
                f"    m = m + Decimal('{fees[1]}')\n"
                "else:\n"
                f"    m = m + Decimal('{fees[2]}')"
            )
            return True
        if kind == "tax":
            cap = self.rng.choice((20, 40, 50))
            low = self.rng.choice((4, 5, 6))
            high = self.rng.choice((8, 9, 10))
            self.prose.append(
                "Add tax on the current dollar amount: "
                f"{low} percent of the first {cap} dollars and {high} percent of the rest."
            )
            self.code.append(
                f"_cap = Decimal('{cap}')\n"
                "_base = m if m <= _cap else _cap\n"
                "_rest = m - _base\n"
                f"m = m + _base * Decimal('{low}') / Decimal('100') + "
                f"_rest * Decimal('{high}') / Decimal('100')"
            )
            return True
        if kind == "per_kg":
            rate = self._fee()
            if self.unit == "kg":
                self.prose.append(f"Add {rate} dollars for each kilogram of the current weight.")
                self.code.append(f"m = m + w * Decimal('{rate}')")
            else:
                self.prose.append(
                    f"Add {rate} dollars for each kilogram of the current weight, "
                    "using 1000 g = 1 kg."
                )
                self.code.append(f"m = m + (w / Decimal('1000')) * Decimal('{rate}')")
            return True
        if kind == "round_cents":
            self.prose.append("Round the dollar amount half up to the nearest cent.")
            self.code.append("m = m.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)")
            return True
        if kind == "round_weight":
            if self.unit == "g":
                self.prose.append("Round the weight half up to the nearest gram.")
                self.code.append("w = w.quantize(Decimal('1'), rounding=ROUND_HALF_UP)")
            else:
                self.prose.append("Round the weight half up to the nearest tenth of a kilogram.")
                self.code.append("w = w.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)")
            return True
        if kind == "discount_over":
            cap = self.rng.choice((30, 50, 80, 100, 120))
            pct = self.rng.choice((5, 8, 10, 12, 15))
            self.prose.append(
                f"Subtract {pct} percent of the dollar amount that exceeds {cap} dollars. "
                f"If the amount is {cap} dollars or less, leave it unchanged."
            )
            self.code.append(
                f"_cap = Decimal('{cap}')\n"
                f"_pct = Decimal('{pct}') / Decimal('100')\n"
                "if m > _cap:\n"
                "    m = m - (m - _cap) * _pct"
            )
            return True
        if kind == "surcharge":
            edges = (self.rng.choice((40, 50)), self.rng.choice((90, 120, 150)))
            fees = [
                str(self.rng.randrange(2, 7)),
                str(self.rng.randrange(7, 14)),
                str(self.rng.randrange(14, 22)),
            ]
            self.prose.append(
                "Add a handling surcharge from the current dollar amount: "
                f"{edges[0]} dollars or less costs {fees[0]} dollars, "
                f"{edges[1]} dollars or less costs {fees[1]} dollars, "
                f"otherwise {fees[2]} dollars."
            )
            self.code.append(
                f"if m <= Decimal('{edges[0]}'):\n"
                f"    m = m + Decimal('{fees[0]}')\n"
                f"elif m <= Decimal('{edges[1]}'):\n"
                f"    m = m + Decimal('{fees[1]}')\n"
                "else:\n"
                f"    m = m + Decimal('{fees[2]}')"
            )
            return True
        if kind == "rebate":
            threshold = self.rng.choice((50, 80, 100, 150, 180))
            rebate = self.rng.randrange(4, 16)
            extra = self.rng.randrange(2, 9)
            self.prose.append(
                f"If the current dollar amount is strictly greater than {threshold} dollars, "
                f"subtract {rebate} dollars; otherwise add {extra} dollars."
            )
            self.code.append(
                f"if m > Decimal('{threshold}'):\n"
                f"    m = m - Decimal('{rebate}')\n"
                "else:\n"
                f"    m = m + Decimal('{extra}')"
            )
            return True
        if kind == "duty":
            pct = self.rng.choice((3, 4, 5, 6, 7))
            flat = self.rng.randrange(2, 9)
            self.prose.append(
                f"Add an import duty of {pct} percent of the current dollar amount, "
                f"then add {flat} dollars."
            )
            self.code.append(
                f"m = m + m * Decimal('{pct}') / Decimal('100') + Decimal('{flat}')"
            )
            return True
        if kind == "add":
            fee = self._fee()
            self.prose.append(f"Add {fee} dollars.")
            self.code.append(f"m = m + Decimal('{fee}')")
            return True
        return False

    def force(self, kind: str) -> None:
        if self.apply(kind):
            return
        fallbacks = [
            "tax",
            "discount_over",
            "surcharge",
            "rebate",
            "duty",
            "per_kg",
            "shipping",
            "round_weight",
        ]
        self.rng.shuffle(fallbacks)
        for fallback in fallbacks:
            if self.apply(fallback):
                return
        self.apply("add")

    _QUESTION_POOL = [
        "what is the final dollar amount?",
        "what dollar amount results after applying all steps?",
        "what is the dollar amount at the end of the calculation?",
        "after completing every step, what is the value of the dollar amount?",
    ]

    def stem(self, distractors: list[str], ask_step: int | None = None) -> str:
        blocks = [f"{i + 1}. {text}" for i, text in enumerate(self.prose)]
        for sentence in distractors:
            blocks.insert(self.rng.randrange(len(blocks) + 1), sentence)
        start = f"Start with {self.kg_s} kg of goods worth {self.dollars} dollars."
        body = "\n".join(blocks)
        if ask_step is not None:
            question = f"what is the dollar amount immediately after step {ask_step} is applied?"
        else:
            question = self.rng.choice(self._QUESTION_POOL)
        return f"{start}\n{body}\n\nQuestion: {question}"


def generate(seed: int, *, steps: int) -> ItemSpec:
    steps = int(steps)
    if steps < 2:
        raise ValueError("steps must be at least 2")
    rng = random.Random(seed)
    chain = _Chain(rng)
    pool = (
        "to_grams",
        "to_kg",
        "shipping",
        "tax",
        "per_kg",
        "discount_over",
        "surcharge",
        "rebate",
        "duty",
        "round_weight",
    )
    kinds = ["to_grams"]
    while len(kinds) < steps - 1:
        kinds.append(rng.choice(pool))
    for kind in kinds:
        chain.force(kind)
    chain.force("round_cents")
    distractors = []
    if rng.random() < 0.6:
        kg = rng.randrange(20, 80) / 10
        dollars = rng.randrange(40, 200)
        distractors.append(
            f"A separate quote for a different shipment lists {kg:.1f} kg at {dollars} dollars."
        )
    if len(chain.prose) != steps:
        raise RuntimeError(f"built {len(chain.prose)} steps, wanted {steps}")

    # Intermediate-step question: 35% of the time ask about state after step k
    # (only when there are enough steps to make it interesting)
    ask_step = None
    if steps >= 5 and rng.random() < 0.35:
        ask_step = rng.randint(2, steps - 1)

    if ask_step is not None:
        # code[:3] = from-import + w init + m init; then one entry per step
        code_lines = chain.code[:3 + ask_step]
        code = "\n".join([*code_lines, "print(m)"]) + "\n"
    else:
        code = "\n".join([*chain.code, "print(m)"]) + "\n"

    return ItemSpec(
        topic=TOPIC,
        level=2,
        template=f"{TOPIC}_train",
        params={"seed": seed, "steps": steps, "answer_space_size": 100_000},
        stem=chain.stem(distractors, ask_step=ask_step),
        code=code,
        lang="python",
        numeric=True,
    )


def _harness(name: str, stem: str, code: str) -> ItemSpec:
    return ItemSpec(
        topic=TOPIC,
        level=1,
        template=name,
        params={},
        stem=stem,
        code=code,
        lang="python",
        numeric=True,
    )


def specs() -> list[ItemSpec]:
    harness = [
        _harness(
            "km_m",
            "A path is 2.5 km long.\n\nQuestion: how many meters is that?",
            "from decimal import Decimal\nprint(Decimal('2.5') * Decimal('1000'))\n",
        ),
        _harness(
            "min_sec",
            "A wait is 4 minutes.\n\nQuestion: how many seconds is that?",
            "from decimal import Decimal\nprint(Decimal('4') * Decimal('60'))\n",
        ),
    ]
    return bank(TOPIC, generate, harness)

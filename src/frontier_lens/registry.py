from __future__ import annotations

from dataclasses import dataclass

import yaml
from sqlalchemy import text

from frontier_lens.config import MODELS_YAML
from frontier_lens.db import get_engine


@dataclass
class ModelSpec:
    id: str
    role: str
    family: str
    provider: str
    base_url: str | None
    model: str
    quant: str | None
    thinking: bool
    license: str
    api_key_env: str | None


def load_yaml() -> list[ModelSpec]:
    raw = yaml.safe_load(MODELS_YAML.read_text())
    specs = []
    policy_families: set[str] = set()
    reference_families: set[str] = set()
    generator_families: set[str] = set()
    allowed = {"policy", "reference", "generator"}
    for row in raw:
        if not row.get("license"):
            raise ValueError(f"{row.get('id')} missing license")
        spec = ModelSpec(
            id=row["id"],
            role=row["role"],
            family=row["family"],
            provider=row["provider"],
            base_url=row.get("base_url"),
            model=row["model"],
            quant=row.get("quant"),
            thinking=bool(row.get("thinking", False)),
            license=row["license"],
            api_key_env=row.get("api_key_env"),
        )
        if spec.role not in allowed:
            raise ValueError(f"bad role {spec.role}")
        if spec.role == "policy":
            policy_families.add(spec.family)
        elif spec.role == "reference":
            reference_families.add(spec.family)
        else:
            generator_families.add(spec.family)
        specs.append(spec)
    if not policy_families:
        raise ValueError("models.yaml needs a role: policy entry")
    overlap = generator_families & (policy_families | reference_families)
    if overlap:
        raise ValueError(
            f"generator family overlaps policy or reference: {overlap}"
        )
    return specs


def policy_spec() -> ModelSpec:
    specs = load_yaml()
    return next(s for s in specs if s.role == "policy")


def contest_specs() -> list[ModelSpec]:
    return [s for s in load_yaml() if s.role in {"policy", "reference"}]


def sync_models() -> list[ModelSpec]:
    specs = load_yaml()
    engine = get_engine()
    with engine.begin() as conn:
        for spec in specs:
            conn.execute(
                text(
                    """
                    INSERT OR REPLACE INTO models (
                        id, role, family, provider, base_url, model, quant, thinking, license, api_key_env
                    ) VALUES (
                        :id, :role, :family, :provider, :base_url, :model, :quant, :thinking, :license, :api_key_env
                    )
                    """
                ),
                {
                    "id": spec.id,
                    "role": spec.role,
                    "family": spec.family,
                    "provider": spec.provider,
                    "base_url": spec.base_url,
                    "model": spec.model,
                    "quant": spec.quant,
                    "thinking": int(spec.thinking),
                    "license": spec.license,
                    "api_key_env": spec.api_key_env,
                },
            )
    return specs

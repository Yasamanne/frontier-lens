from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR / 'frontier_lens.db'}")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
SPEND_CAP_USD = float(os.getenv("SPEND_CAP_USD", "5"))
SME_NAME = os.getenv("SME_NAME", "yasaman")
MODELS_YAML = ROOT / "models.yaml"
FIXTURES_PATH = ROOT / "fixtures" / "recorded_run.json"
PYTHON_BIN = os.getenv("FRONTIER_LENS_PYTHON", sys.executable)
TIMEOUT_SEC = 2.0
SPLIT_RNG_SEED = 42

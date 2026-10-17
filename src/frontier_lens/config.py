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
LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)
PYTHON_BIN = os.getenv("FRONTIER_LENS_PYTHON", sys.executable)
TIMEOUT_SEC = 2.0
SPLIT_RNG_SEED = 42
POLICY_K = int(os.getenv("POLICY_K", "8"))
REFERENCE_K = int(os.getenv("REFERENCE_K", "3"))
# Max model calls in flight at once (process-wide). 1 = sequential.
MAX_CONCURRENCY = max(1, int(os.getenv("MAX_CONCURRENCY", "16")))
AUDIT_SAMPLE_N = int(os.getenv("AUDIT_SAMPLE_N", "50"))
AUDIT_MAX_DEFECT_HI = float(os.getenv("AUDIT_MAX_DEFECT_HI", "0.10"))
LEARNABLE_LO = 0.0
LEARNABLE_HI = 1.0
REQUIRE_LIVE_KNOBS = os.getenv("FRONTIER_LENS_REQUIRE_LIVE_KNOBS", "").lower() in {
    "1",
    "true",
    "yes",
}

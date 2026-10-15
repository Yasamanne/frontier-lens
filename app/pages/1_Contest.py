import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import requests
import streamlit as st

from frontier_lens.config import OLLAMA_BASE_URL, ROOT
from frontier_lens.registry import load_yaml

st.title("Contest")
role = st.session_state.get("role", "Viewer")

try:
    tags = requests.get(OLLAMA_BASE_URL.replace("/v1", "") + "/api/tags", timeout=2).json()
    ollama_ok = True
    ollama_models = [t.get("name") for t in tags.get("models", [])]
except Exception:
    ollama_ok = False
    ollama_models = []

st.sidebar.write("Ollama", "ok" if ollama_ok else "down")

specs = load_yaml()
eval_ids = [m.id for m in specs if m.role == "evaluated"]
chosen = st.multiselect("Models", eval_ids, default=eval_ids[:1])
levels = st.multiselect("Levels", [1, 2, 3], default=[1, 2, 3])
k = st.number_input("k samples", 1, 5, 3)
mode = st.radio("Mode", ["replay", "live"])
if role == "Viewer":
    st.info("Viewer: replay only.")
    mode = "replay"

if st.button("Evaluate") and chosen:
    cmd = [
        sys.executable,
        "-m",
        "frontier_lens.contest",
        f"--{mode}",
        "--k",
        str(k),
        "--models",
        *chosen,
        "--levels",
        *[str(lv) for lv in levels],
    ]
    st.write("Running:", " ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    st.session_state["contest_proc"] = proc.pid
    out, _ = proc.communicate()
    st.code(out[-4000:] if out else "(no output)")

st.caption("Cached by run rows in SQLite. Re-run creates a new eval_runs id.")
if ollama_models:
    st.write("Ollama tags:", ollama_models)

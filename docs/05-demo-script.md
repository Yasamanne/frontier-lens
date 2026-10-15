# Demo script (45 minutes)

Lead with the finding, not the stack.

1. **Problem and prior art (6 min)** — stale static sets; Dynabench, Terminal-Bench; what is different (cell-targeted generation, executed keys, SME gate, provenance). Open-weight only.
2. **Finding (10 min)** — heatmap with n and CI; L1 harness check; experiment paragraph in `results/experiment.md`.
3. **Walkthrough (10 min)** — a template, sandbox, one-shot vs agent, blind SME, metrics. If Ollama is up: Evaluate a model from the dropdown.
4. **Tradeoffs (7 min)** — execution-verified single-turn vs agentic tasks; open-weight only; Ollama GGUF vs hosted API vs vLLM-on-GPU-later; SQLite vs Postgres; recorded vs invented fixtures; CIs vs pretty charts.
5. **A real miss (6 min)** — saturation, BSD/GNU key mismatch, unparsable small-model output, or generator junk.
6. **Next (6 min)** — k=5, more items, second annotator, evaluated-agent track, text-vs-image modality gap, contamination checks.

If live models are down, replay fixtures. Do not invent scores.

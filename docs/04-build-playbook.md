# Build playbook

Work only in `/Users/yasamanemami/Documents/GitHub/frontier-lens`. Branch `master`. Do not clone again.

1. Design docs (this folder) — commit `docs: vision, prior-art survey, item design, roster, and experiment design`
2. `uv` project, `.gitignore`, LICENSE, SQLite schema, sandbox — commit `chore: uv project, SQLite schema, and sandboxed executor`
3. Templates L1–L3, seed with executed keys and holdout split — commit `feat: templates and execution-verified seed items (L1-L3) with holdout split`
4. `models.yaml`, adapters, contest CLI, record fixtures when a live run exists — commit `feat: provider adapters, model registry, and live contest CLI; record fixtures`
5. Per-item stats, Wilson CIs, heatmap — commit `feat: per-item stats, Wilson CIs, and heatmap`
6. One-shot (then agent), execute + auto-reject, blind SME — commit `feat: one-shot and agent generators, execution check, and blind SME review`
7. Streamlit pages, model dropdown, background eval — commit `feat: Streamlit pages, model dropdown with background evaluation`
8. New-vs-baseline, discrimination, junk, one-shot vs agent — commit `feat: new-vs-baseline, discrimination, junk, and one-shot vs agent metrics`
9. Tests + light CI — commit `test: parser, sandbox rejections, key re-execution, CI math, promotion gate; light CI`
10. README experiment numbers, licenses, screenshots — commit `docs: README experiment numbers, roster and licenses, prior art, screenshots`

Push: `git push -u origin master`.

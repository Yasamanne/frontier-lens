#!/bin/sh
set -e

# ── Required env var check ────────────────────────────────────────────────
MISSING=""

if [ -z "$OPENROUTER_API_KEY" ]; then
    MISSING="$MISSING OPENROUTER_API_KEY"
fi

if [ -n "$MISSING" ]; then
    echo ""
    echo "⚠️  WARNING: the following env vars are not set:$MISSING"
    echo "   Live LLM calls will fail with 401."
    echo "   Fixture-replay and read-only (Viewer) mode will still work."
    echo "   Run with:  docker run --env-file .env frontier-lens"
    echo ""
fi

exec streamlit run app/Home.py \
    --server.port="${PORT:-8501}" \
    --server.address=0.0.0.0 \
    --server.headless=true

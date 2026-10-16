#!/usr/bin/env bash
# Starts the internal FastAPI backend, waits until it is healthy (this warms
# the BGE-M3 retriever before any user request), then execs Streamlit as PID 1
# on the platform-provided $PORT.
set -euo pipefail

API_PORT=8000
UI_PORT="${PORT:-8501}"

echo "[entrypoint] starting FastAPI on 127.0.0.1:${API_PORT}"
python -m uvicorn mmrag.api.app:app --host 127.0.0.1 --port "${API_PORT}" &
API_PID=$!

healthy=0
for i in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1; then
        echo "[entrypoint] API healthy after ${i}s"
        healthy=1
        break
    fi
    if ! kill -0 "${API_PID}" 2>/dev/null; then
        echo "[entrypoint] FATAL: API process exited" >&2
        exit 1
    fi
    sleep 1
done
if [ "${healthy}" -ne 1 ]; then
    echo "[entrypoint] FATAL: API never became healthy in 180s" >&2
    exit 1
fi

echo "[entrypoint] starting Streamlit UI on 0.0.0.0:${UI_PORT}"
exec python -m streamlit run src/mmrag/ui/streamlit_app.py \
    --server.address 0.0.0.0 \
    --server.port "${UI_PORT}" \
    --server.headless true \
    --browser.gatherUsageStats false

# syntax=docker/dockerfile:1
# Multimodal Hybrid RAG — production image.
# Bakes the prebuilt index, rendered pages, figure assets and the local
# BGE-M3 / bge-reranker model copies so first boot needs no downloads.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    STREAMLIT_BROWSER_GATHER_USAGESTATS=false

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# --- pinned runtime dependencies (torch CPU wheels from the pytorch index) ---
COPY requirements.linux.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.linux.txt

# --- project (editable install keeps PROJECT_ROOT=/app for data/models paths) ---
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps -e .

# --- baked artifacts: index, rendered page PNGs, figure crops, eval results ---
COPY data/index ./data/index
COPY data/pages ./data/pages
COPY data/assets ./data/assets
COPY data/results ./data/results

# --- local model copies (~4.4 GB) ---
COPY models ./models

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
# strip CRLF in case the file was checked out on Windows
RUN sed -i 's/\r$//' /usr/local/bin/docker-entrypoint.sh \
    && chmod +x /usr/local/bin/docker-entrypoint.sh

# UI is the public surface; API stays on 127.0.0.1:8000 inside the container.
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD sh -c 'curl -sf http://127.0.0.1:${PORT:-8501}/ >/dev/null && curl -sf http://127.0.0.1:8000/health >/dev/null' || exit 1

ENTRYPOINT ["docker-entrypoint.sh"]

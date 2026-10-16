# syntax=docker/dockerfile:1
# Multimodal Hybrid RAG — production image (fetch-based so it builds from a
# clean git clone: data artifacts come from the repo's data-v1 release and
# model weights from Hugging Face during the build).
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_BROWSER_GATHER_USAGESTATS=false

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# --- pinned runtime dependencies (torch CPU wheels from the pytorch index) ---
COPY requirements.linux.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.linux.txt

# --- project (editable install keeps PROJECT_ROOT=/app for data/models paths) ---
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps -e .

# --- baked artifacts: vector index, rendered page PNGs, figure crops, eval
#     results (~734 MB uncompressed) from the repo's data-v1 GitHub release ---
ARG DATA_BUNDLE_URL=https://github.com/kaix-404/multi_rag/releases/download/data-v1/data-bundle.tar.gz
RUN curl -fL --retry 3 --retry-delay 5 -o /tmp/data-bundle.tar.gz "$DATA_BUNDLE_URL" \
    && tar -xzf /tmp/data-bundle.tar.gz -C /app \
    && rm /tmp/data-bundle.tar.gz

# --- model weights (~3.4 GB) vendored under /app/models/<folder> exactly as
#     config.py's vendored-model preference expects; needs no HF login ---
RUN hf download BAAI/bge-m3 --local-dir models/bge-m3 \
    && hf download BAAI/bge-reranker-large --local-dir models/bge-reranker-large

# offline for all runtime model loads (set AFTER the download layer)
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
# strip CRLF in case the file was checked out on Windows
RUN sed -i 's/\r$//' /usr/local/bin/docker-entrypoint.sh \
    && chmod +x /usr/local/bin/docker-entrypoint.sh

# UI is the public surface (Render injects $PORT); API stays on 127.0.0.1:8000.
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD sh -c 'curl -sf http://127.0.0.1:${PORT:-8501}/ >/dev/null && curl -sf http://127.0.0.1:8000/health >/dev/null' || exit 1

ENTRYPOINT ["docker-entrypoint.sh"]

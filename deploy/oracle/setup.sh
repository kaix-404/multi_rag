#!/usr/bin/env bash
# One-time (idempotent) bootstrap of the Multimodal Hybrid RAG demo on an
# Oracle Cloud Always Free A1 (ARM/ampere) instance.
#
#   Run from the repo root:  bash deploy/oracle/setup.sh
#
# Prerequisites: repo source is at /home/opc/multi_rag with
# data/{index,pages,assets} already transferred, then run this once.
set -euo pipefail

cd "$(dirname "$0")/../.."
echo "==> repo: $(pwd)"

if [ ! -d data/index ] || [ ! -d data/pages ]; then
  echo "==> fetching data bundle (index/pages/assets, 646 MB) from the data-v1 release"
  curl -fL --retry 3 --retry-delay 5 -o /tmp/data-bundle.tar.gz \
      https://github.com/kaix-404/multi_rag/releases/download/data-v1/data-bundle.tar.gz
  tar -xzf /tmp/data-bundle.tar.gz -C .
  rm -f /tmp/data-bundle.tar.gz
fi

echo "==> system packages"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    build-essential curl ca-certificates cron
sudo systemctl enable --now cron 2>/dev/null || true

echo "==> uv (python toolchain)"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="${HOME}/.local/bin:${PATH}"

echo "==> python 3.13 venv"
if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.13 .venv
fi

echo "==> pip dependencies (pinned ARM requirements)"
if ! uv pip install --python .venv/bin/python -r deploy/oracle/requirements.oracle.txt; then
  # pypdfium2 (transitive pdfplumber dep, ingest-only) has no aarch64 wheel;
  # if its source build fails, skip it - query serving never imports it.
  echo "WARN: pinned install failed, retrying without pypdfium2"
  grep -v '^pypdfium2==' deploy/oracle/requirements.oracle.txt > /tmp/req-no-pdfium.txt
  uv pip install --python .venv/bin/python -r /tmp/req-no-pdfium.txt
fi
uv pip install --python .venv/bin/python --no-deps -e .
chmod +x docker-entrypoint.sh

echo "==> models from Hugging Face (~3.4 GB)"
mkdir -p models
.venv/bin/hf download BAAI/bge-m3 --local-dir models/bge-m3
.venv/bin/hf download BAAI/bge-reranker-large --local-dir models/bge-reranker-large

echo "==> weekly keep-alive (Always Free reclaims persistently idle instances)"
if command -v crontab >/dev/null 2>&1; then
  ( crontab -l 2>/dev/null | grep -v 'oracle-keepalive' || true
    echo '17 6 * * 1 sh -c "for i in 1 2 3 4 5 6 7 8 9 10 11 12; do dd if=/dev/zero of=/dev/null bs=1M count=500 2>/dev/null; done" # oracle-keepalive'
  ) | crontab -
fi

echo "==> systemd service"
sudo cp deploy/oracle/mmrag.service /etc/systemd/system/mmrag.service
sudo systemctl daemon-reload
sudo systemctl enable --now mmrag.service
sleep 8
systemctl --no-pager -l status mmrag.service | head -20 || true

echo "==> firewall (open 8501 only if ufw is active)"
if command -v ufw >/dev/null 2>&1 && sudo ufw status | grep -q 'Status: active'; then
  sudo ufw allow 8501/tcp
fi

echo "==> done. UI should answer at http://<public-ip>:8501"

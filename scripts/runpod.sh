#!/usr/bin/env bash
# Runpod (Linux, NVIDIA) one-shot: install uv, sync the locked env, run tests, ingest, evaluate.
# Usage: bash scripts/runpod.sh [run_id]   (from the repo root; needs network for HF downloads)
set -euo pipefail
RUN_ID="${1:-dev-$(date -u +%Y%m%d-%H%M%S)}"
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
uv sync --python 3.11 --frozen
uv run python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'available',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
uv run pytest -q
uv run jevc ingest
uv run jevc run --run-id "$RUN_ID"
echo "report: reports/$RUN_ID/report.md"

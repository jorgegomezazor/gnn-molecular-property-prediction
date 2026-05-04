#!/usr/bin/env bash

# Resolve python: prefer venv python, then python3, then python
if [[ -f ".venv/Scripts/python.exe" ]]; then
  PYTHON=".venv/Scripts/python.exe"
elif command -v python3 &>/dev/null; then
  PYTHON="python3"
else
  PYTHON="python"
fi
# scripts/run_all.sh
#
# Trains all four models sequentially with the same seed.
# Usage:  bash scripts/run_all.sh [--device cuda] [--seed 42]
#
# Results are written to results/<model>_seed42/

set -e

DEVICE="cpu"
SEED=42

# Parse optional arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --device) DEVICE="$2"; shift 2 ;;
    --seed)   SEED="$2";   shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "======================================================"
echo "  Training all models  |  device=$DEVICE  seed=$SEED"
echo "======================================================"

MODELS=("mpnn" "gin" "gat" "graph_transformer")

for MODEL in "${MODELS[@]}"; do
  echo ""
  echo ">>> Starting $MODEL ..."
  "$PYTHON" scripts/train.py \
    --config "configs/${MODEL}.yaml" \
    --device "$DEVICE" \
    --seed "$SEED"
  echo ">>> Finished $MODEL."
done

echo ""
echo "======================================================"
echo "  All models trained. Running comparison analysis ..."
echo "======================================================"

"$PYTHON" scripts/analyze.py --results_dir results/ --seed "$SEED"

echo "Done. Figures saved to results/figures/"

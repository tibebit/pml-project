#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python scripts/01_check_extracted_vocalizations.py
python scripts/02_build_dataset.py
python scripts/03_explore_dataset.py
python scripts/04_run_baselines.py
python scripts/05_final_evaluation.py \
  --seed 2026 \
  --draws 1000 \
  --tune 1000 \
  --chains 4 \
  --target-accept 0.95 \
  --prior-draws 500 \
  --bootstrap-repeats 1000

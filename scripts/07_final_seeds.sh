#!/usr/bin/env bash
set -euo pipefail
source scripts/runtime_env.sh
for seed in 42 3407 2026; do
  echo "Final-seed protocol seed=$seed"
  echo "Run frozen C0/C1/C4-A configurations only after validation selection is complete."
done

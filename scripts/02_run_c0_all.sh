#!/usr/bin/env bash
set -euo pipefail
source scripts/runtime_env.sh
for d in tcmp300 med117 leafsnap plantnet300k_v2 indian_medicinal; do
  python tools/train.py --dataset "configs/datasets/${d}.yaml" --experiment configs/experiments/c0_linear.yaml --seed "${SEED:-42}" "$@"
done

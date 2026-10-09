#!/usr/bin/env bash
set -euo pipefail
STAGE="$1"
EXP="$2"
source scripts/runtime_env.sh
for d in tcmp300 med117 leafsnap plantnet300k_v2 indian_medicinal; do
  echo "===== $STAGE :: $d ====="
  python tools/train.py \
    --dataset "configs/datasets/${d}.yaml" \
    --experiment "$EXP" \
    --seed "${SEED:-42}" "$@"
done

#!/usr/bin/env bash
set -euo pipefail
source scripts/runtime_env.sh
SEED="${SEED:-42}"
for d in tcmp300 med117 leafsnap plantnet300k_v2 indian_medicinal; do
  C1="outputs/paper/${d}/c1_lora/seed_${SEED}/best.pth"
  if [[ ! -f "$C1" ]]; then
    echo "ERROR: missing selected C1 checkpoint: $C1" >&2
    exit 2
  fi
  python tools/train.py \
    --dataset "configs/datasets/${d}.yaml" \
    --experiment configs/experiments/c4a_bsprc.yaml \
    --seed "$SEED" \
    --init-checkpoint "$C1" "$@"
done

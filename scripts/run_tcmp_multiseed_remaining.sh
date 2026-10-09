#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/MorphoTaxa_FM_PaperReady_v1
source scripts/runtime_env.sh

mkdir -p outputs/logs

for SEED in 3407 2026
do
    echo "======================================"
    echo "SEED=${SEED}: C0"
    echo "======================================"

    python tools/train.py \
      --dataset configs/datasets/tcmp300.yaml \
      --experiment configs/experiments/c0_linear.yaml \
      --seed "${SEED}" \
      --execute \
      2>&1 | tee \
      "outputs/logs/tcmp_c0_seed${SEED}.log"

    C0_CKPT="outputs/paper/tcmp300_canonical_2026/c0_linear/seed_${SEED}/best.pth"

    test -f "${C0_CKPT}" || {
        echo "ERROR: missing C0 checkpoint: ${C0_CKPT}"
        exit 1
    }

    echo "======================================"
    echo "SEED=${SEED}: C0-CONTINUE"
    echo "======================================"

    python tools/train.py \
      --dataset configs/datasets/tcmp300.yaml \
      --experiment configs/experiments/c0_continue.yaml \
      --seed "${SEED}" \
      --init-checkpoint "${C0_CKPT}" \
      --execute \
      2>&1 | tee \
      "outputs/logs/tcmp_c0_continue_seed${SEED}.log"

    echo "======================================"
    echo "SEED=${SEED}: C1"
    echo "======================================"

    python tools/train.py \
      --dataset configs/datasets/tcmp300.yaml \
      --experiment configs/experiments/c1_lora.yaml \
      --seed "${SEED}" \
      --init-checkpoint "${C0_CKPT}" \
      --execute \
      2>&1 | tee \
      "outputs/logs/tcmp_c1_seed${SEED}.log"

    C1_CKPT="outputs/paper/tcmp300_canonical_2026/c1_lora/seed_${SEED}/best.pth"

    test -f "${C1_CKPT}" || {
        echo "ERROR: missing C1 checkpoint: ${C1_CKPT}"
        exit 1
    }

    echo "======================================"
    echo "SEED=${SEED}: C4-A"
    echo "======================================"

    python tools/train.py \
      --dataset configs/datasets/tcmp300.yaml \
      --experiment configs/experiments/c4a_bsprc.yaml \
      --seed "${SEED}" \
      --init-checkpoint "${C1_CKPT}" \
      --execute \
      2>&1 | tee \
      "outputs/logs/tcmp_c4a_seed${SEED}.log"

done

echo "======================================"
echo "ALL REMAINING TCMP SEEDS COMPLETE"
echo "======================================"

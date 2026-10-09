#!/usr/bin/env bash
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/pretrained/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TORCH_HOME="${TORCH_HOME:-/root/autodl-tmp/pretrained/torch}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=1
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
mkdir -p outputs/paper outputs/audit

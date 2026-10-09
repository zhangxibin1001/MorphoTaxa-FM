#!/usr/bin/env bash
set -euo pipefail
source scripts/runtime_env.sh
cat <<'EOF'
C1 ablation protocol (validation-only selection):
  blocks: 4, 8, 12
  ranks: 4, 8, 16
  targets: qkv ; qkv+proj
C4-A ablation:
  alpha_max: 0.1, 0.2, 0.3
  lambda_alpha: 0, 1e-4, 1e-3, 1e-2
  variants: global-unbounded, classwise-unbounded, classwise-bounded
Implement sweep runner after the verified trainer is complete. Configs are in configs/sweeps/.
EOF

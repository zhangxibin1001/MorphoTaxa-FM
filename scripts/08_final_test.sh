#!/usr/bin/env bash
set -euo pipefail
cat <<'EOF'
FINAL TEST LOCK
Do not run this stage until:
  [ ] dataset manifests are frozen
  [ ] all hyperparameters are selected on validation data
  [ ] three final seeds are complete
  [ ] config files are copied into the final run directory
  [ ] a git commit identifies the frozen code
Then run exactly one final evaluation pass per frozen checkpoint and save logits, targets, sample IDs, metrics, and checksums.
EOF

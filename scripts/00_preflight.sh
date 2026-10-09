#!/usr/bin/env bash
set -euo pipefail
source scripts/runtime_env.sh
python -m compileall -q src tools
pytest -q
python tools/preflight/environment_preflight.py
python tools/preflight/model_preflight.py

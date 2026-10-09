#!/usr/bin/env bash
set -euo pipefail
if [[ -d .git ]]; then
  echo "Git repository already initialized."
  exit 0
fi
git init
git add .
git commit -m "Initialize MorphoTaxa-FM PaperReady v1"
echo "Repository initialized. Start Codex from: $(pwd)"

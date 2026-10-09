#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

run_audit () {
    local cfg="$1"

    python - "$cfg" <<'PY'
import subprocess
import sys
from pathlib import Path
import yaml

cfg_path = Path(sys.argv[1])

cfg = yaml.safe_load(
    cfg_path.read_text(encoding="utf-8")
)

d = cfg["dataset"]

name = d["name"]
root = d["path"]
expected = d.get("expected_classes")

cmd = [
    sys.executable,
    "tools/data/audit_dataset.py",
    "--root", root,
    "--name", name,
    "--out", "outputs/audit",
]

if expected is not None:
    cmd += ["--expected-classes", str(expected)]

print()
print("=" * 100)
print("DATASET :", name)
print("ROOT    :", root)
print("CONFIG  :", cfg_path)
print("=" * 100)

subprocess.run(cmd, check=True)
PY
}

run_audit configs/datasets/tcmp300.yaml
run_audit configs/datasets/med117.yaml
run_audit configs/datasets/leafsnap.yaml
run_audit configs/datasets/plantnet300k_v2.yaml
run_audit configs/datasets/indian_medicinal.yaml

echo
echo "Audit complete."

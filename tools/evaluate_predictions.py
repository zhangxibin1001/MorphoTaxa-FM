#!/usr/bin/env python
from __future__ import annotations

import argparse, json
import numpy as np
from morphotaxa.metrics import classification_metrics
from morphotaxa.calibration import expected_calibration_error, nll, multiclass_brier


def main():
    ap = argparse.ArgumentParser(description="Evaluate saved logits/targets without re-running the model.")
    ap.add_argument("--logits", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    logits = np.load(args.logits)
    y = np.load(args.targets)
    result = classification_metrics(logits, y)
    result.update({"ece": expected_calibration_error(logits, y), "nll": nll(logits, y), "brier": multiclass_brier(logits, y)})
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        from pathlib import Path
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()

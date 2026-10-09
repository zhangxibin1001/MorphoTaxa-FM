#!/usr/bin/env python
from __future__ import annotations

import argparse, json
from pathlib import Path
from morphotaxa.config import load_experiment
from morphotaxa.reproducibility import seed_everything
from morphotaxa.engine.trainer import train_experiment


def main():
    ap = argparse.ArgumentParser(description="Train C0/C1/C4-A using train+val only; test is never instantiated here.")
    ap.add_argument("--base", default="configs/base.yaml")
    ap.add_argument("--model", default="configs/models/bioclip25.yaml")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--init-checkpoint", default=None, help="Required for C4-A; selected C1 checkpoint")
    ap.add_argument("--output", default=None)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    cfg = load_experiment(args.base, args.model, args.dataset, args.experiment)
    cfg.setdefault("project", {})["seed"] = args.seed
    seed_everything(args.seed)
    if not args.execute:
        print(json.dumps(cfg, ensure_ascii=False, indent=2))
        print("DRY RUN. Add --execute only after dataset/model preflight passes.")
        return

    name = cfg["dataset"]["name"]
    stage = cfg["experiment"]["name"]
    output = Path(args.output or cfg["project"].get("output_root", "outputs/paper")) / name / stage / f"seed_{args.seed}"
    summary = train_experiment(cfg, output, init_checkpoint=args.init_checkpoint)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse, json
from morphotaxa.models.bioclip25 import load_bioclip25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="local-dir:/root/autodl-tmp/pretrained/huggingface/hub/models--imageomics--bioclip-2.5-vith14/snapshots/6e3d04e3d6522012c88181085c5ae666e14c45cd")
    ap.add_argument("--cache-dir", default="/root/autodl-tmp/pretrained/huggingface")
    ap.add_argument("--online", action="store_true")
    args = ap.parse_args()
    model, _, _ = load_bioclip25(args.model, args.cache_dir, offline=not args.online)
    n = sum(p.numel() for p in model.parameters())
    print(json.dumps({"status": "PASS", "model": args.model, "parameters": n}, indent=2))

if __name__ == "__main__":
    main()

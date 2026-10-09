#!/usr/bin/env python
from __future__ import annotations

import json, platform, sys


def main():
    report = {"python": sys.version.split()[0], "platform": platform.platform(), "status": "PASS", "checks": {}}
    for name in ["numpy", "yaml", "PIL", "sklearn"]:
        try:
            mod = __import__(name)
            report["checks"][name] = getattr(mod, "__version__", "installed")
        except Exception as e:
            report["checks"][name] = f"FAIL: {e}"
            report["status"] = "FAIL"
    try:
        import torch
        report["checks"]["torch"] = torch.__version__
        report["checks"]["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            report["checks"]["gpu"] = torch.cuda.get_device_name(0)
    except Exception as e:
        report["checks"]["torch"] = f"OPTIONAL/FAIL: {e}"
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)

if __name__ == "__main__":
    main()

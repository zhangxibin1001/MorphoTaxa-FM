from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--logits",
        required=True,
    )

    ap.add_argument(
        "--targets",
        required=True,
    )

    ap.add_argument(
        "--bins",
        required=True,
    )

    ap.add_argument(
        "--out",
        required=True,
    )

    args = ap.parse_args()

    logits = np.load(args.logits)
    targets = np.load(args.targets).astype(int)

    assert logits.ndim == 2
    assert logits.shape[0] == targets.shape[0]
    assert logits.shape[1] == 1000

    preds = logits.argmax(axis=1)

    obj = json.loads(
        Path(args.bins).read_text(
            encoding="utf-8"
        )
    )

    groups = obj["classes"]

    result = {
        "num_samples": int(len(targets)),
        "num_classes": 1000,
        "overall": {
            "accuracy": float(
                accuracy_score(
                    targets,
                    preds,
                )
            ),
            "macro_precision": float(
                precision_score(
                    targets,
                    preds,
                    labels=list(range(1000)),
                    average="macro",
                    zero_division=0,
                )
            ),
            "macro_recall": float(
                recall_score(
                    targets,
                    preds,
                    labels=list(range(1000)),
                    average="macro",
                    zero_division=0,
                )
            ),
            "macro_f1": float(
                f1_score(
                    targets,
                    preds,
                    labels=list(range(1000)),
                    average="macro",
                    zero_division=0,
                )
            ),
        },
        "groups": {},
    }

    for group in [
        "head",
        "medium",
        "few",
    ]:
        labels = [
            int(x)
            for x in groups[group]
        ]

        label_set = set(labels)

        # Macro class metrics are computed over the
        # complete validation set. This preserves
        # false positives entering these classes
        # from other frequency groups.
        macro_p = precision_score(
            targets,
            preds,
            labels=labels,
            average="macro",
            zero_division=0,
        )

        macro_r = recall_score(
            targets,
            preds,
            labels=labels,
            average="macro",
            zero_division=0,
        )

        macro_f1 = f1_score(
            targets,
            preds,
            labels=labels,
            average="macro",
            zero_division=0,
        )

        # Accuracy only for examples whose true
        # class belongs to this frequency group.
        mask = np.array([
            int(y) in label_set
            for y in targets
        ])

        group_acc = accuracy_score(
            targets[mask],
            preds[mask],
        )

        result["groups"][group] = {
            "num_classes":
                len(labels),

            "num_val_samples":
                int(mask.sum()),

            "accuracy":
                float(group_acc),

            "macro_precision":
                float(macro_p),

            "macro_recall":
                float(macro_r),

            "macro_f1":
                float(macro_f1),
        }

    out = Path(args.out)

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    print()
    print(
        "PLANTNET LONG-TAIL ANALYSIS: PASS"
    )


if __name__ == "__main__":
    main()

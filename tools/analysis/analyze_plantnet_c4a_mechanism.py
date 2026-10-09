from pathlib import Path
import json

import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from sklearn.metrics import (
    precision_recall_fscore_support,
)


C1_DIR = Path(
    "outputs/paper/"
    "plantnet300k_v2_official_2026/"
    "c1_lora/seed_42"
)

C4_DIR = Path(
    "outputs/paper/"
    "plantnet300k_v2_official_2026/"
    "c4a_bsprc/seed_42"
)

ALPHA_CSV = Path(
    "outputs/paper_freeze/"
    "plantnet_v2_c4a_seed42/"
    "alpha_analysis.csv"
)

MANIFEST = Path(
    "data_protocols/plantnet300k_v2/"
    "manifest.csv"
)

OUT = Path(
    "outputs/paper_freeze/"
    "plantnet_v2_c4a_seed42/"
    "mechanism_analysis.csv"
)


def load_pred(folder):
    logits = np.load(
        folder / "val_logits.npy"
    )

    targets = np.load(
        folder / "val_targets.npy"
    ).astype(int)

    return (
        logits.argmax(axis=1),
        targets,
    )


pred_c1, y1 = load_pred(C1_DIR)
pred_c4, y4 = load_pred(C4_DIR)

assert np.array_equal(
    y1,
    y4,
)

targets = y1

labels = np.arange(1000)


_, _, f1_c1, support = (
    precision_recall_fscore_support(
        targets,
        pred_c1,
        labels=labels,
        zero_division=0,
    )
)

_, _, f1_c4, _ = (
    precision_recall_fscore_support(
        targets,
        pred_c4,
        labels=labels,
        zero_division=0,
    )
)


alpha_df = pd.read_csv(
    ALPHA_CSV
)

assert len(alpha_df) == 1000


manifest = pd.read_csv(
    MANIFEST
)

train_counts = (
    manifest[
        manifest["split"] == "train"
    ]
    .groupby("label")
    .size()
    .reindex(
        labels,
        fill_value=0,
    )
)


df = alpha_df.copy()

df["train_count"] = (
    train_counts.values
)

df["val_support"] = support

df["f1_c1"] = f1_c1
df["f1_c4a"] = f1_c4

df["delta_f1"] = (
    df["f1_c4a"]
    - df["f1_c1"]
)

df["saturated_019"] = (
    df["abs_alpha"] >= 0.19
)

df["saturated_0199"] = (
    df["abs_alpha"] >= 0.199
)


valid = (
    df["val_support"] > 0
)


def corr(x, y, mask):
    r, p = spearmanr(
        df.loc[mask, x],
        df.loc[mask, y],
    )

    return float(r), float(p)


print(
    "===== SATURATION ====="
)

for group in [
    "head",
    "medium",
    "few",
]:
    g = df[
        df["group"] == group
    ]

    print(
        group,
        "n=", len(g),
        "sat>=0.19=",
        int(
            g["saturated_019"].sum()
        ),
        f"({100*g['saturated_019'].mean():.2f}%)",
        "sat>=0.199=",
        int(
            g["saturated_0199"].sum()
        ),
        f"({100*g['saturated_0199'].mean():.2f}%)",
    )


print(
    "\n===== PER-CLASS C4A EFFECT ====="
)

for group in [
    "head",
    "medium",
    "few",
]:
    mask = (
        (df["group"] == group)
        & valid
    )

    d = df.loc[
        mask,
        "delta_f1",
    ]

    print(
        group,
        "supported=",
        int(mask.sum()),
        "mean_delta_f1=",
        float(d.mean()),
        "median_delta_f1=",
        float(d.median()),
        "improved=",
        int((d > 0).sum()),
        "degraded=",
        int((d < 0).sum()),
        "unchanged=",
        int((d == 0).sum()),
    )


print(
    "\n===== SPEARMAN ====="
)

tests = [
    (
        "alpha_vs_deltaF1",
        "alpha",
        "delta_f1",
    ),
    (
        "absAlpha_vs_deltaF1",
        "abs_alpha",
        "delta_f1",
    ),
    (
        "trainCount_vs_absAlpha",
        "train_count",
        "abs_alpha",
    ),
    (
        "trainCount_vs_deltaF1",
        "train_count",
        "delta_f1",
    ),
]

for name, x, y in tests:
    r, p = corr(
        x,
        y,
        valid,
    )

    print(
        name,
        "rho=",
        r,
        "p=",
        p,
    )


print(
    "\n===== GROUP MEANS ====="
)

summary = (
    df.groupby("group")
    .agg(
        classes=(
            "label",
            "count",
        ),
        train_count_mean=(
            "train_count",
            "mean",
        ),
        abs_alpha_mean=(
            "abs_alpha",
            "mean",
        ),
        alpha_mean=(
            "alpha",
            "mean",
        ),
        delta_f1_mean=(
            "delta_f1",
            "mean",
        ),
        saturation_019=(
            "saturated_019",
            "mean",
        ),
    )
)

print(summary)


OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_csv(
    OUT,
    index=False,
)

print()
print(
    "saved:",
    OUT
)

print(
    "PLANTNET C4A MECHANISM ANALYSIS: PASS"
)

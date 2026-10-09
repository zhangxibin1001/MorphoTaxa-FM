from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


CKPT = Path(
    "outputs/paper/"
    "plantnet300k_v2_official_2026/"
    "c4a_bsprc/seed_42/best.pth"
)

CLASSES = Path(
    "data_protocols/plantnet300k_v2/"
    "classes.csv"
)

BINS = Path(
    "data_protocols/plantnet300k_v2/"
    "longtail_bins.json"
)

OUT = Path(
    "outputs/paper_freeze/"
    "plantnet_v2_c4a_seed42/"
    "alpha_analysis.csv"
)


ckpt = torch.load(
    CKPT,
    map_location="cpu",
    weights_only=False,
)

state = (
    ckpt.get("model")
    or ckpt.get("state_dict")
    or ckpt.get("model_state_dict")
)

if state is None:
    raise RuntimeError(
        "Cannot find model state dict"
    )

candidate_keys = [
    k
    for k in state
    if k.endswith(
        "semantic_head.raw_alpha"
    )
]

print(
    "alpha keys =",
    candidate_keys,
)

assert len(candidate_keys) == 1

raw = (
    state[candidate_keys[0]]
    .detach()
    .float()
    .cpu()
    .numpy()
)

assert raw.shape == (1000,)

alpha = (
    0.2
    * np.tanh(raw)
)

classes = pd.read_csv(
    CLASSES
)

assert len(classes) == 1000

bins = json.loads(
    BINS.read_text(
        encoding="utf-8"
    )
)

label_to_group = {}

for group in [
    "head",
    "medium",
    "few",
]:
    for label in bins[
        "classes"
    ][group]:
        label_to_group[int(label)] = group

assert len(label_to_group) == 1000


df = classes.copy()

df["alpha"] = alpha
df["abs_alpha"] = np.abs(alpha)

df["group"] = [
    label_to_group[int(x)]
    for x in df["label"]
]


OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_csv(
    OUT,
    index=False,
)


print("\n===== GLOBAL =====")

print(
    "min =",
    float(alpha.min()),
)

print(
    "max =",
    float(alpha.max()),
)

print(
    "mean =",
    float(alpha.mean()),
)

print(
    "std =",
    float(alpha.std()),
)

print(
    "mean_abs =",
    float(
        np.abs(alpha).mean()
    ),
)

print(
    "positive =",
    int(
        (alpha > 0).sum()
    ),
)

print(
    "negative =",
    int(
        (alpha < 0).sum()
    ),
)

print(
    "zero =",
    int(
        (alpha == 0).sum()
    ),
)


print(
    "\n===== HEAD / MEDIUM / FEW ====="
)

summary = (
    df.groupby("group")
    .agg(
        n=("label", "count"),
        alpha_mean=("alpha", "mean"),
        alpha_std=("alpha", "std"),
        abs_alpha_mean=(
            "abs_alpha",
            "mean",
        ),
        abs_alpha_median=(
            "abs_alpha",
            "median",
        ),
        positive=(
            "alpha",
            lambda x: int(
                (x > 0).sum()
            ),
        ),
        negative=(
            "alpha",
            lambda x: int(
                (x < 0).sum()
            ),
        ),
    )
)

print(summary)


print(
    "\n===== TOP 20 |ALPHA| ====="
)

print(
    df.sort_values(
        "abs_alpha",
        ascending=False,
    )[
        [
            "label",
            "class_name",
            "group",
            "alpha",
            "abs_alpha",
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)


print()
print(
    "saved:",
    OUT,
)

print(
    "PLANTNET C4A ALPHA ANALYSIS: PASS"
)

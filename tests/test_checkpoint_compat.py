import torch
from torch import nn

from morphotaxa.engine.checkpoint_compat import (
    load_init_checkpoint_compatible,
)
from morphotaxa.models.lora import (
    inject_lora_last_blocks,
)


class Block(nn.Module):
    def __init__(self):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            32,
            4,
            batch_first=True,
        )


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.resblocks = nn.ModuleList(
            [Block() for _ in range(4)]
        )


class Visual(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = Transformer()


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = nn.Module()
        self.backbone.visual = Visual()

        self.classifier = nn.Linear(
            32,
            3,
        )


def test_c0_to_c1_checkpoint_mapping(tmp_path):
    torch.manual_seed(123)

    # C0 model.
    c0 = Model()

    ckpt = tmp_path / "c0.pth"

    torch.save(
        {
            "model": c0.state_dict(),
            "epoch": 17,
            "val": {
                "macro_f1": 0.93
            },
        },
        ckpt,
    )

    # C1 starts from an identical base architecture.
    torch.manual_seed(999)
    c1 = Model()

    for p in c1.backbone.parameters():
        p.requires_grad = False

    inject_lora_last_blocks(
        c1.backbone,
        last_n_blocks=2,
        rank=4,
        alpha=8.0,
        dropout=0.0,
        targets=("qkv", "proj"),
    )

    report = load_init_checkpoint_compatible(
        c1,
        ckpt,
    )

    # 2 blocks x QKV/output projection.
    assert report["remapped_count"] == 4

    # Classifier must come from C0.
    assert torch.equal(
        c0.classifier.weight,
        c1.classifier.weight,
    )

    assert torch.equal(
        c0.classifier.bias,
        c1.classifier.bias,
    )

    # Original attention weights must also exactly match C0.
    for i in (2, 3):
        c0_attn = (
            c0.backbone.visual
            .transformer.resblocks[i].attn
        )

        c1_attn = (
            c1.backbone.visual
            .transformer.resblocks[i].attn
        )

        assert torch.equal(
            c0_attn.in_proj_weight,
            c1_attn.in_proj_weight,
        )

        assert torch.equal(
            c0_attn.out_proj.weight,
            c1_attn.out_proj.weight,
        )

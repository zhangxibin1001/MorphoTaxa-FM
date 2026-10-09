import torch
from torch import nn

from morphotaxa.models.lora import inject_lora_last_blocks


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            32, 4, batch_first=True
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
        self.visual = Visual()


def test_openclip_lora_identity():
    torch.manual_seed(42)

    model = Model()

    for p in model.parameters():
        p.requires_grad = False

    attn = model.visual.transformer.resblocks[-1].attn

    qkv0 = attn.in_proj_weight.detach().clone()
    proj0 = attn.out_proj.weight.detach().clone()

    replaced = inject_lora_last_blocks(
        model,
        last_n_blocks=2,
        rank=4,
        alpha=8.0,
        dropout=0.0,
        targets=("qkv", "proj"),
    )

    assert len(replaced) == 4

    # B=0 -> exact epoch-0 identity.
    assert torch.equal(
        qkv0,
        attn.in_proj_weight.detach()
    )
    assert torch.equal(
        proj0,
        attn.out_proj.weight.detach()
    )

    trainable = {
        n: p
        for n, p in model.named_parameters()
        if p.requires_grad
    }

    assert trainable
    assert any("lora_a" in n for n in trainable)
    assert any("lora_b" in n for n in trainable)

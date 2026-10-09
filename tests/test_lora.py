import torch
from torch import nn
from morphotaxa.models.lora import LoRALinear


def test_lora_identity_at_init():
    base = nn.Linear(4, 3)
    x = torch.randn(2, 4)
    expected = base(x).detach()
    lora = LoRALinear(base, rank=2, alpha=4)
    got = lora(x)
    assert torch.allclose(got, expected)
    assert not any(p.requires_grad for p in lora.base.parameters())

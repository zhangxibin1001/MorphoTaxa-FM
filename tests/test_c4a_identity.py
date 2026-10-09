import torch
from morphotaxa.models.c4a import BoundedSemanticResidual


def test_c4a_is_identity_at_initialization():
    head = BoundedSemanticResidual(num_classes=3, alpha_max=0.2, kappa=1.0)
    logits = torch.randn(4, 3)
    features = torch.randn(4, 5)
    proto = torch.randn(3, 5)
    out = head(logits, features, proto)
    assert torch.equal(out, logits)
    assert torch.all(head.bounded_alpha == 0)


def test_c4a_alpha_is_bounded():
    head = BoundedSemanticResidual(num_classes=2, alpha_max=0.2)
    with torch.no_grad():
        head.raw_alpha[:] = torch.tensor([100.0, -100.0])
    assert float(head.bounded_alpha.detach().max()) <= 0.200001
    assert float(head.bounded_alpha.detach().min()) >= -0.200001

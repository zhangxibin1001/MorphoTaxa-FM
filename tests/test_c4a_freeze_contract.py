from morphotaxa.config import load_experiment
from morphotaxa.engine.trainer import build_training_state


def test_c4a_only_semantic_alpha_is_trainable():
    cfg = load_experiment(
        "configs/base.yaml",
        "configs/models/bioclip25.yaml",
        "configs/datasets/leafsnap_cd184_lab2field.yaml",
        "configs/experiments/c4a_smoke_portable.yaml",
    )

    model, _, _, class_names, _ = build_training_state(
        cfg,
        init_checkpoint=(
            "outputs/paper/leafsnap_cd184_lab2field/"
            "c4a_smoke_portable/seed_42/best.pth"
        ),
    )

    trainable = {
        name: p
        for name, p in model.named_parameters()
        if p.requires_grad
    }

    assert set(trainable) == {
        "semantic_head.raw_alpha"
    }

    assert trainable[
        "semantic_head.raw_alpha"
    ].numel() == len(class_names)

    assert len(class_names) == 184

    for name, p in model.named_parameters():
        if "lora" in name.lower():
            assert not p.requires_grad

        if name.startswith("classifier."):
            assert not p.requires_grad

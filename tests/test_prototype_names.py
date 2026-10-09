from morphotaxa.models.prototypes import (
    clean_semantic_class_name,
)


def test_clean_tcmp_semantic_name():
    assert (
        clean_semantic_class_name(
            "001.Veronica persica Poir."
        )
        == "Veronica persica Poir."
    )

    assert (
        clean_semantic_class_name(
            "300.Callicarpa bodinieri H. Lév."
        )
        == "Callicarpa bodinieri H. Lév."
    )


def test_clean_underscore_name():
    assert (
        clean_semantic_class_name(
            "abies_concolor"
        )
        == "abies concolor"
    )

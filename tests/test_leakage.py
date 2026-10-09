from morphotaxa.data.manifest import ManifestRow
from morphotaxa.data.leakage import exact_cross_split_leaks


def row(path, split, h):
    return ManifestRow(path, split, "a", 0, 1, h)


def test_cross_split_duplicate_detected():
    rows = [row("train/a/1.jpg", "train", "same"), row("test/a/2.jpg", "test", "same")]
    leaks = exact_cross_split_leaks(rows)
    assert len(leaks) == 1
    assert set(leaks[0].splits) == {"train", "test"}

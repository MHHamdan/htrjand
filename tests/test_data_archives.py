"""Protocol tests that need the archives; skipped when they are absent."""
import hashlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from htrjand import data as D      # noqa: E402

OFFICIAL_COUNTS = {           # official partitions (docs/DATA_SPLITS.md)
    "iam": (6161, 1840, 1861), "rimes": (10192, 1134, 778), "bentham": (9194, 1415, 860),
    "saintgall": (468, 235, 707), "washington": (325, 168, 163),
}


def _have(ds):
    try:
        D.archive_path(ds)
        return True
    except FileNotFoundError:
        return False


@pytest.mark.parametrize("ds,counts", OFFICIAL_COUNTS.items())
def test_split_sizes_and_geometry(ds, counts):
    if not _have(ds):
        pytest.skip(f"{ds} archive not built")
    import h5py
    with h5py.File(D.archive_path(ds)) as h:
        for sp, n in zip(("train", "valid", "test"), counts):
            assert h[sp]["dt"].shape == (n, 68, 864) and h[sp]["dt"].dtype == np.uint8


@pytest.mark.parametrize("ds", ["washington", "saintgall", "iam"])
def test_no_image_in_two_splits(ds):
    if not _have(ds):
        pytest.skip(f"{ds} archive not built")
    hashes = {}
    for sp in ("train", "valid", "test"):
        imgs, _ = D.read_split(ds, sp)
        hashes[sp] = {hashlib.md5(x.tobytes()).hexdigest() for x in imgs}
    assert not (hashes["train"] & hashes["test"])
    assert not (hashes["train"] & hashes["valid"])
    assert not (hashes["valid"] & hashes["test"])


@pytest.mark.parametrize("ds", list(OFFICIAL_COUNTS))
def test_charset_is_train_only_and_ctc_feasible(ds):
    if not _have(ds):
        pytest.skip(f"{ds} archive not built")
    tok = D.Tokenizer(D.build_charset(ds))
    _, train = D.read_split(ds, "train")
    assert set("".join(train)) == set(tok.charset)
    for sp in ("valid", "test"):
        _, labels = D.read_split(ds, sp)
        need = max(len(t) + sum(a == b for a, b in zip(t, t[1:])) for t in labels)
        assert need <= 108, f"{ds} {sp}: a target needs {need} > 108 frames"

#!/usr/bin/env python3
"""Build the fixed-canvas HDF5 line archives (68 x 864, or 128 x 1632 with --size) from
the raw corpora, on the official partitions.

    python -m htrjand.prepare_data --dataset iam --raw-dir raw --out-dir data
    # likewise rimes, bentham, saintgall, washington, iam_aachen

Image recipe (Section III-A of the paper gives only the size; the recipe is the
historical builder, `git show dda617e:src/preprocess/preprocess-Copy1.py`,
which the audit matched pixel-exact against the pre-2026 archives): read in
grayscale (RIMES lines are cropped from the page with the XML box); scale by
s = min(864 / w, 68 / h) with cv2.INTER_AREA so the aspect ratio is kept;
centre-pad with white on both axes to 68 x 864; invert when the mean exceeds
128, so every line is stored ink-bright on a dark ground. Per split the
archive holds `dt` (uint8, N x 68 x 864) and `gt` (UTF-8 transcriptions).

Expected raw layout under --raw-dir/<dataset>/ (as distributed by the providers):
  iam/        ascii/lines.txt, lines/, largeWriterIndependentTextLineRecognitionTask/
              (and aachen/{trainset,validationset,testset}.txt for iam_aachen)
  rimes/      training_2011.xml, eval_2011_annotated.xml, training_2011/, eval_2011/
              (validation is the last 10 % of the training lines, in file order)
  bentham/    BenthamDatasetR0-GT/{Partitions,Transcriptions,Images/Lines}
  saintgall/  sets/, ground_truth/transcription.txt, data/line_images_normalized/
  washington/ sets/cv1/, ground_truth/transcription.txt, data/line_images_normalized/

The corpora are not redistributed; each has its own licence.
`--verify-against` compares a build with an existing archive (multiset of
image hashes and labels per split).
"""
import argparse
import functools
import html
import os
import sys
import xml.etree.ElementTree as ET
from functools import partial
from glob import glob
from multiprocessing import Pool

import numpy as np

from ._compat import lazy

cv2 = lazy("cv2", "pip install opencv-python-headless")
h5py = lazy("h5py", "pip install h5py")

from ._compat import REPO_ROOT  # noqa: E402  (source checkout, else the working directory)
INPUT_SIZE = (68, 864)          # H, W
PARTITIONS = ("train", "valid", "test")


# ----------------------------------------------------------------------------- readers
def _iam(src, partition="fki"):
    """IAM lines under one of the two partitions in circulation.

    "fki" is the `largeWriterIndependentTextLineRecognitionTask` shipped with
    the corpus: 6,161 train / 1,840 validation (validationset1 + 2) / 1,861
    test. Every HTR-JAND result to date uses it.

    "aachen" is the RWTH partition distributed with Puigcerver's Laia recipe:
    the SAME 6,161 training lines, but 966 validation and 2,915 test. It is the
    partition Puigcerver, Michael, Dutta, Retsinas, Yousef and TrOCR all report
    on, so it is the only one on which a number of ours is comparable with
    theirs. The two test sets are not nested: 1,313 of the 1,861 FKI test lines
    are in the Aachen test set, 1,468 more of it comes from what FKI calls
    validation, and 134 lines belong to neither FKI split.
    """
    if partition == "aachen":
        pt = os.path.join(src, "aachen")
        rd = lambda f: open(os.path.join(pt, f)).read().split()
        paths = {"train": rd("trainset.txt"), "valid": rd("validationset.txt"),
                 "test": rd("testset.txt")}
    else:
        pt = os.path.join(src, "largeWriterIndependentTextLineRecognitionTask")
        rd = lambda f: open(os.path.join(pt, f)).read().splitlines()
        paths = {"train": rd("trainset.txt"),
                 "valid": rd("validationset1.txt") + rd("validationset2.txt"),
                 "test": rd("testset.txt")}
    gt = {}
    for line in open(os.path.join(src, "ascii", "lines.txt")).read().splitlines():
        if line and line[0] != "#":
            s = line.split(); gt[s[0]] = " ".join(s[8:]).replace("|", " ")
    out = {p: [] for p in PARTITIONS}
    for p in PARTITIONS:
        for lid in paths[p]:
            if lid not in gt:
                continue
            a, b, c = lid.split("-")[:3]
            out[p].append((os.path.join(src, "lines", a, f"{a}-{b}", f"{a}-{b}-{c}.png"), gt[lid]))
    return out


def _rimes(src):
    def pages(xml, sub):
        items = []
        for page in ET.parse(os.path.join(src, xml)).getroot():
            f = page.attrib["FileName"]
            for ln in page.iter("Line"):
                text = " ".join(html.unescape(ln.attrib.get("Value", "")).split())
                box = tuple(abs(int(ln.attrib[k])) for k in ("Top", "Bottom", "Left", "Right"))
                items.append(((os.path.join(src, sub, f), box), text))
        return items
    tr = pages("training_2011.xml", "training_2011")
    cut = int(len(tr) * 0.9)
    return {"train": tr[:cut], "valid": tr[cut:], "test": pages("eval_2011_annotated.xml", "eval_2011")}


def _bentham(src):
    src = os.path.join(src, "BenthamDatasetR0-GT")
    pt = os.path.join(src, "Partitions")
    rd = lambda f: open(os.path.join(pt, f)).read().splitlines()
    paths = {"train": rd("TrainLines.lst"), "valid": rd("ValidationLines.lst"), "test": rd("TestLines.lst")}
    tdir = os.path.join(src, "Transcriptions")
    gt = {}
    for x in os.listdir(tdir):
        text = " ".join(open(os.path.join(tdir, x)).read().splitlines())
        gt[os.path.splitext(x)[0]] = " ".join(html.unescape(text).replace("<gap/>", "").split())
    img = os.path.join(src, "Images", "Lines")
    return {p: [(os.path.join(img, f"{l}.png"), gt[l]) for l in paths[p]] for p in PARTITIONS}


def _saintgall(src):
    pt = os.path.join(src, "sets")
    rd = lambda f: open(os.path.join(pt, f)).read().splitlines()
    paths = {"train": rd("train.txt"), "valid": rd("valid.txt"), "test": rd("test.txt")}
    gt = {}
    for line in open(os.path.join(src, "ground_truth", "transcription.txt")).read().splitlines():
        s = line.split(); gt[s[0]] = s[1].replace("-", "").replace("|", " ")
    img = os.path.join(src, "data", "line_images_normalized")
    out = {p: [] for p in PARTITIONS}
    for p in PARTITIONS:
        for page in paths[p]:
            for f in sorted(glob(os.path.join(img, f"{page}*"))):
                lid = os.path.splitext(os.path.basename(f))[0]
                out[p].append((os.path.join(img, f"{lid}.png"), gt[lid]))
    return out


def _washington(src):
    pt = os.path.join(src, "sets", "cv1")
    rd = lambda f: open(os.path.join(pt, f)).read().splitlines()
    paths = {"train": rd("train.txt"), "valid": rd("valid.txt"), "test": rd("test.txt")}
    gt = {}
    for line in open(os.path.join(src, "ground_truth", "transcription.txt")).read().splitlines():
        s = line.split()
        t = s[1].replace("-", "").replace("|", " ")
        # Same mapping as the archives were built with. NB: the original code
        # dropped the results of the s_sq/s_et/s_bl/s_br/s_qt/s_GW replacements
        # (no assignment); the final generic strip then turned e.g. "s_sq" into
        # "sq". Reproduced verbatim so the archive is byte-identical.
        t = t.replace("s_pt", ".").replace("s_cm", ",").replace("s_mi", "-").replace("s_qo", ":")
        t = t.replace("s_", "")
        gt[s[0]] = t
    img = os.path.join(src, "data", "line_images_normalized")
    return {p: [(os.path.join(img, f"{l}.png"), gt[l]) for l in paths[p]] for p in PARTITIONS}


READERS = {"iam": _iam, "rimes": _rimes, "bentham": _bentham,
           "saintgall": _saintgall, "washington": _washington,
           # same raw corpus, the other partition (see _iam)
           "iam_aachen": partial(_iam, partition="aachen")}


def _raw_subdir(dataset):
    """Directory under --raw-dir holding a dataset's raw files. The two IAM
    partitions are two readings of one corpus, not two downloads."""
    return "iam" if dataset == "iam_aachen" else dataset


# ----------------------------------------------------------------------------- image recipe
def preprocess(item, size=INPUT_SIZE):
    if isinstance(item, tuple):
        path, (top, bottom, left, right) = item
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        img = img[top:bottom, left:right]
    else:
        img = cv2.imread(item, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(item)
    th, tw = size
    h, w = img.shape
    s = min(tw / w, th / h)
    nw, nh = int(w * s), int(h * s)
    if nw == 0 or nh == 0:
        raise ValueError(f"degenerate line image {item}: {img.shape}")
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    dh, dw = th - nh, tw - nw
    img = cv2.copyMakeBorder(img, dh // 2, dh - dh // 2, dw // 2, dw - dw // 2,
                             cv2.BORDER_CONSTANT, value=255)
    if img.mean() > 128:                     # dark ink on light paper -> invert
        img = 255 - img
    return img.astype(np.uint8)


def build(dataset, raw_dir, out, limit=None, workers=None, size=INPUT_SIZE):
    parts = READERS[dataset](os.path.join(raw_dir, _raw_subdir(dataset)))
    for p in PARTITIONS:
        n0 = len(parts[p])
        parts[p] = [(im, g) for im, g in parts[p] if g]          # drop empty labels, as before
        if n0 != len(parts[p]):
            print(f"  {p}: dropped {n0 - len(parts[p])} line(s) with empty transcription", file=sys.stderr)
        if limit:
            parts[p] = parts[p][:limit]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with h5py.File(out, "w") as hf, Pool(workers) as pool:
        for p in PARTITIONS:
            items = [im for im, _ in parts[p]]
            imgs = pool.map(functools.partial(preprocess, size=tuple(size)), items, chunksize=32) if items else []
            dt = np.stack(imgs) if imgs else np.zeros((0,) + tuple(size), np.uint8)
            hf.create_dataset(f"{p}/dt", data=dt, compression="gzip", compression_opts=9)
            hf.create_dataset(f"{p}/gt", data=[g.encode("utf-8") for _, g in parts[p]],
                              dtype=h5py.special_dtype(vlen=str), compression="gzip", compression_opts=9)
            print(f"  {p}: {len(items)} lines", file=sys.stderr)
    return out


def verify(built, reference):
    """Compare (image, label) pairs. Index-wise equality is reported, but the
    verdict is order-independent: the original Saint Gall archive was written in
    filesystem glob order, which this script replaces with sorted order."""
    import hashlib
    from collections import Counter
    ok = True
    with h5py.File(built) as a, h5py.File(reference) as b:
        for p in PARTITIONS:
            n = a[p]["dt"].shape[0]
            if n == 0:
                continue
            full = b[p]["dt"].shape[0]
            dec = lambda x: x.decode() if isinstance(x, bytes) else x
            pa = [(hashlib.md5(x.tobytes()).hexdigest(), dec(g)) for x, g in zip(a[p]["dt"][:], a[p]["gt"][:])]
            if n == full:                      # full build: compare as multisets
                pb = [(hashlib.md5(x.tobytes()).hexdigest(), dec(g)) for x, g in zip(b[p]["dt"][:], b[p]["gt"][:])]
                idx = sum(x == y for x, y in zip(pa, pb))
                same = Counter(pa) == Counter(pb)
                print(f"  {p}: {n} lines | identical at same index {idx}/{n} | identical as a set: {same}")
            else:                              # partial build (--limit): the first n pairs must match
                pb = [(hashlib.md5(x.tobytes()).hexdigest(), dec(g)) for x, g in zip(b[p]["dt"][:n], b[p]["gt"][:n])]
                idx = sum(x == y for x, y in zip(pa, pb))
                same = idx == n
                print(f"  {p}: first {n} of {full} lines | identical at same index {idx}/{n}")
            ok &= same
    print("VERIFY:", "PASS (every image and label reproduced)" if ok else "FAIL")
    return ok




def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(READERS))
    ap.add_argument("--raw-dir", default=os.path.join(REPO_ROOT, "raw"))
    ap.add_argument("--out-dir", default=os.environ.get("HTRJAND_DATA_DIR") or os.path.join(REPO_ROOT, "data"))
    ap.add_argument("--out", default=None, help="explicit output file (default <out-dir>/<dataset>.hdf5)")
    ap.add_argument("--limit", type=int, default=None, help="max lines per split (smoke tests)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--size", type=int, nargs=2, default=list(INPUT_SIZE), metavar=("H", "W"),
                    help="canvas (paper: 68 864); the enhancement runs also use 128 1632")
    ap.add_argument("--verify-against", default=None, help="existing archive to compare with")
    a = ap.parse_args(argv)
    out = a.out or os.path.join(a.out_dir, f"{a.dataset}.hdf5")
    if os.path.exists(out) and not a.overwrite:
        sys.exit(f"{out} exists; pass --overwrite or choose --out")
    print(f"building {out} from {os.path.join(a.raw_dir, _raw_subdir(a.dataset))}", file=sys.stderr)
    build(a.dataset, a.raw_dir, out, a.limit, a.workers, a.size)
    if a.verify_against:
        sys.exit(0 if verify(out, a.verify_against) else 1)


if __name__ == "__main__":
    main()

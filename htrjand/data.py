"""Datasets, tokenizer and augmentation of HTR-JAND.

Archives are the official-partition HDF5 files built by `prepare_data.py`
(68 x 864, uint8, ink bright on a dark ground, one `train/valid/test` group
each with `dt` images and `gt` UTF-8 transcriptions). Only the official
partitions are used (docs/DATA_SPLITS.md).

Where the archives live is configurable: `--data-dir` on every command line,
else the environment variable `HTRJAND_DATA_DIR`, else `<repo>/data`.
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from ._compat import lazy

cv2 = lazy("cv2", "pip install opencv-python-headless")
h5py = lazy("h5py", "pip install h5py")

from ._compat import REPO_ROOT  # noqa: E402  (source checkout, else the working directory)

#: official writer-independent partitions (docs/DATA_SPLITS.md)
OFFICIAL = {
    "iam": "iam.hdf5",                 # FKI large writer-independent task: 6,161 / 1,840 / 1,861
    "iam_aachen": "iam_aachen.hdf5",   # same corpus, RWTH Aachen partition: 6,161 / 966 / 2,915
    "rimes": "rimes.hdf5",             # ICDAR 2011: 10,192 / 1,134 / 778
    "bentham": "bentham.hdf5",         # ICFHR 2014 R0: 9,194 / 1,415 / 860
    "saintgall": "saintgall.hdf5",     # 468 / 235 / 707
    # same lines; the raw "pt" special letter (punctuation dot) removed. It is annotated on the
    # official TRAINING pages only (716 tokens; 0 on validation / test pages), so a model trained
    # with it inserts "pt" that no evaluation reference contains.
    "saintgall_nopt": "saintgall_nopt.hdf5",
    "washington": "washington.hdf5",   # cv1: 325 / 168 / 163
    "combined": "cleaned_combined.hdf5",  # concatenation of the five, no overlap
}

INPUT_SIZE = (68, 864)   # base-configuration canvas; the released models use 128 x 1632

#: the two input scalings the package can apply to a stored uint8 line
NORMALISATIONS = ("paper", "unit")


def data_dir(explicit=None):
    """Resolve the archive directory: explicit argument > $HTRJAND_DATA_DIR > repo/data."""
    d = explicit or os.environ.get("HTRJAND_DATA_DIR") or os.path.join(REPO_ROOT, "data")
    return os.path.abspath(d)


def archive_path(dataset, root=None):
    if dataset.endswith(".hdf5") and os.path.exists(dataset):   # e.g. a synthetic archive
        return dataset
    if dataset not in OFFICIAL:
        raise KeyError(f"unknown dataset {dataset!r}; choose from {sorted(OFFICIAL)}")
    p = os.path.join(data_dir(root), OFFICIAL[dataset])
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{p} not found. Build it with `python -m htrjand.prepare_data "
            f"--dataset {dataset}` or point --data-dir / $HTRJAND_DATA_DIR at the archives.")
    return p


def resolve_for(dataset, input_size):
    """The archive of `dataset` at a checkpoint's input size: the official 68 x 864
    archive, or <data>/<dataset>_<H>x<W>.hdf5 built by prepare_data --size H W."""
    if tuple(input_size) == INPUT_SIZE or dataset.endswith(".hdf5"):
        return dataset
    return os.path.join(data_dir(), f"{dataset}_{input_size[0]}x{input_size[1]}.hdf5")


def normalise(img_u8, how="paper"):
    """uint8 (H, W) -> float32 (H, W).

    "paper": per-image min-max normalisation to [-1, 1]. A blank image
             (max == min) maps to -1 everywhere.
    "unit":  the historical loader's img / 255 -> [0, 1].
    """
    x = img_u8.astype(np.float32)
    if how == "paper":
        lo, hi = float(x.min()), float(x.max())
        return (2.0 * (x - lo) / (hi - lo) - 1.0) if hi > lo else np.full_like(x, -1.0)
    if how == "unit":
        return x / 255.0
    raise ValueError(how)


class Tokenizer:
    """Character tokenizer with a dedicated CTC blank.

    Index layout: 0 = [PAD] (target padding only, never emitted), 1 = [UNK]
    (a character outside the training charset), 2.. = the sorted charset,
    and BLANK = the last index. The original code set BLANK = vocab_size - 1
    on top of the character list, which aliased the blank onto the last real
    character (audit Section 4.3); here no character is unreachable.
    """

    def __init__(self, chars):
        self.chars = ["[PAD]", "[UNK]"] + sorted(set(chars))
        self.char_to_index = {c: i for i, c in enumerate(self.chars)}
        self.index_to_char = {i: c for i, c in enumerate(self.chars)}
        self.PAD, self.UNK = 0, 1
        self.BLANK = len(self.chars)
        self.vocab_size = len(self.chars) + 1

    @property
    def charset(self):
        return "".join(self.chars[2:])

    def encode(self, text):
        return [self.char_to_index.get(c, self.UNK) for c in text]

    def decode(self, indices):
        """Indices -> text. A predicted [UNK] becomes U+FFFD (one character), so
        it scores as one substitution rather than as the five-letter literal."""
        return "".join("\ufffd" if int(i) == self.UNK else self.index_to_char[int(i)]
                       for i in indices if int(i) not in (self.PAD, self.BLANK))

    def decode_batch(self, batch):
        return [self.decode(x) for x in batch]

    def ctc_greedy_decode(self, logits):
        """(B, T, V) logits or log-probabilities -> list[str]: argmax per frame,
        collapse repeats, drop blanks (Graves et al., 2006)."""
        best = torch.argmax(logits, dim=-1).cpu().numpy()
        out = []
        for seq in best:
            prev, buf = None, []
            for idx in seq:
                if idx != prev and idx != self.BLANK and idx != self.PAD:
                    buf.append("\ufffd" if idx == self.UNK else self.index_to_char[int(idx)])
                prev = idx
            out.append("".join(buf))
        return out


def read_split(dataset, split, root=None):
    with h5py.File(archive_path(dataset, root), "r") as h:
        images = h[split]["dt"][:]
        labels = [g.decode("utf-8") for g in h[split]["gt"][:]]
    return images, labels


def build_charset(dataset, root=None):
    """Charset from the TRAIN transcriptions only, so neither the validation
    nor the test split can leak a character into the model's output layer."""
    _, labels = read_split(dataset, "train", root)
    return "".join(sorted(set("".join(labels))))


# --------------------------------------------------------------------------- augmentation
# The training procedure applies a composition of augmentation transforms but
# names none of them. The pipeline below is the one the 2026 reimplementation
# used for its clean runs and is kept as the declared assumption A-PRE-03:
# a small affine jitter, then photometric jitter, each photometric step with
# probability 1/2. Images are float32 in [0, 1] (ink bright) at this point; the
# affine border is background (0).

AUG_SETS = ("none", "base", "eg", "egb", "strong")
# "base": affine + photometric jitter (REPOSITORY_PRIOR). The stronger sets add,
# with p = 1/2 each, the recipes the 2026-08 reimplementation measured on IAM:
#   eg     = base + projective ("geometric") and elastic distortion (Kizilirmak et al.)
#   egb    = eg + ink blots (Shonenkov et al.)
#   strong = egb with height-relative blots and morphological stroke-width jitter
# Images are float32 in [0, 1], ink bright on a dark ground (border value 0).


def _elastic(img, rng, alpha_range=(4.0, 10.0), ds=8, sigma_cells=1.0):
    """Elastic distortion with a COARSE displacement field.

    The random field is generated at 1/ds resolution, smoothed there, then
    bilinearly upsampled to full resolution. Upsampling a smoothed coarse grid
    is what makes the field smooth, so this replaces the usual full-resolution
    Gaussian blur at ~10x lower cost (measured 0.34 ms vs 3.55 ms per line) and
    is why the full-resolution variant was not used.

    alpha is the peak displacement in pixels, drawn per sample.
    """
    h, w = img.shape
    hs, ws = max(h // ds, 2), max(w // ds, 2)
    alpha = rng.uniform(*alpha_range)
    fx = gaussian_blur_small(rng.uniform(-1, 1, (hs, ws)).astype(np.float32), sigma_cells)
    fy = gaussian_blur_small(rng.uniform(-1, 1, (hs, ws)).astype(np.float32), sigma_cells)
    dx = cv2.resize(fx, (w, h), interpolation=cv2.INTER_LINEAR) * alpha
    dy = cv2.resize(fy, (w, h), interpolation=cv2.INTER_LINEAR) * alpha
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, gx + dx, gy + dy, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)


def gaussian_blur_small(a, sigma):
    k = max(3, int(2 * round(3 * sigma) + 1))
    return cv2.GaussianBlur(a, (k, k), sigma, borderType=cv2.BORDER_REPLICATE)


def _perspective(img, rng, mx=0.010, my=0.040):
    """Geometric (projective) distortion: each corner is jittered independently,
    horizontally by up to mx*w and vertically by up to my*h. The vertical
    fraction is the larger one because a line image is wide and flat, so an
    equal relative jitter on both axes would be invisible vertically and
    extreme horizontally."""
    h, w = img.shape
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jit = rng.uniform(-1, 1, (4, 2)).astype(np.float32) * np.float32([w * mx, h * my])
    M = cv2.getPerspectiveTransform(src, src + jit)
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)


def _blot(img, rng, n_range=(1, 4), r_range=(3, 10)):
    """Ink blot: a few filled ellipses of ink drawn over the line, simulating
    the blots and show-through that occlude glyphs in real manuscripts. Ink is
    bright in this representation, so the blots are drawn at full intensity."""
    h, w = img.shape
    out = img.copy()
    for _ in range(int(rng.integers(n_range[0], n_range[1] + 1))):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        ax, ay = int(rng.integers(*r_range)), int(rng.integers(*r_range))
        cv2.ellipse(out, (cx, cy), (ax, ay), float(rng.uniform(0, 180)),
                    0, 360, 1.0, -1)
    return out


def _blot_scaled(img, rng, n_range=(1, 11), hfrac=(0.25, 0.65), wfrac=(0.06, 0.30),
                 alpha=0.95):
    r"""Occlusion at the strength \citet{shonenkov2021stackmix} publish, and
    sized relative to the line height rather than in absolute pixels.

    The absolute radii of `_blot` (3-10 px) were chosen for a 68-row canvas and
    would cover a quarter as much of a 128-row one, so an experiment that
    changed the canvas would silently have weakened its own augmentation. Here
    the semi-axes are fractions of the image height and the count reaches the
    published 11, and the blot is composited with transparency rather than
    painted opaque, so some ink survives underneath as it does on a real page.
    """
    h, w = img.shape
    out = img.copy()
    layer = np.zeros_like(img)
    for _ in range(int(rng.integers(n_range[0], n_range[1] + 1))):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        ay = max(1, int(rng.uniform(*hfrac) * h * 0.5))
        ax = max(1, int(rng.uniform(*wfrac) * h * 0.5))
        cv2.ellipse(layer, (cx, cy), (ax, ay), float(rng.uniform(-15, 15)),
                    0, 360, 1.0, -1)
    return out * (1.0 - alpha * layer) + alpha * layer


def _morph(img, rng):
    """Morphological ink-width jitter: dilate to thicken the stroke, erode to
    thin it. Ink is bright here, so cv2.dilate grows the writing. A broad nib
    and a fine one are the same text, and no other term in this pipeline moves
    stroke width; kernels are kept to 2-3 px because a larger one closes the
    counters of e, a and o and destroys the character rather than varying it."""
    k = int(rng.integers(2, 4))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return (cv2.dilate(img, ker) if rng.random() < 0.5 else cv2.erode(img, ker))



def augment(img, rng, aug_set="base"):
    if aug_set == "none":
        return img
    h, w = img.shape
    ang = rng.uniform(-1.5, 1.5)                 # degrees
    sc = rng.uniform(0.95, 1.0)
    sh = rng.uniform(-0.1, 0.1)                  # horizontal shear, radians
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, sc)
    M[0, 1] += sh
    M[0, 2] -= sh * h / 2.0
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    if aug_set in ("eg", "egb", "strong"):
        if rng.random() < 0.5:
            img = _perspective(img, rng)
        if rng.random() < 0.5:
            img = _elastic(img, rng)
    if aug_set == "strong" and rng.random() < 0.5:
        img = _morph(img, rng)
    if rng.random() < 0.5:
        img = img * rng.uniform(0.8, 1.2)                        # brightness
    if rng.random() < 0.5:
        m = float(img.mean())
        img = (img - m) * rng.uniform(0.8, 1.2) + m              # contrast
    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (3, 3), rng.uniform(0.5, 1.5))
    if rng.random() < 0.5:
        img = img + rng.normal(0.0, 0.02, img.shape).astype(np.float32)
    if aug_set in ("egb", "strong") and rng.random() < 0.5:
        clipped = np.clip(img, 0.0, 1.0).astype(np.float32)
        img = _blot_scaled(clipped, rng) if aug_set == "strong" else _blot(clipped, rng)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


class LineDataset(Dataset):
    """One text line per item: (image float32 (1, 68, 864), target int64 (L,), L).

    Augmentation is applied on the training split only, on the [0, 1] image,
    before the input normalisation. The per-item RNG is derived from the
    dataset seed, the item index and the epoch counter that the trainer sets
    with `set_epoch`, so two workers never draw the same stream and a run is
    repeatable for a given seed.
    """

    def __init__(self, dataset, split, tokenizer, augment=False, seed=0,
                 aug_set="base", normalisation="paper", root=None, limit=None, fold=None):
        self.images, self.labels = read_split(dataset, split, root)
        if fold is not None:
            # ("train", k, K) keeps the K-fold complement of fold k; ("holdout", k, K)
            # keeps fold k only. Used for cross-decoding the training split when
            # building post-OCR correction pairs (postprocess.py).
            mode, k, K = fold
            idx = np.arange(len(self.labels))
            keep = (idx % K == k) if mode == "holdout" else (idx % K != k)
            self.images = self.images[keep]
            self.labels = [l for l, m in zip(self.labels, keep) if m]
        if limit:
            self.images, self.labels = self.images[:limit], self.labels[:limit]
        self.tokenizer = tokenizer
        self.do_augment = bool(augment) and split == "train" and aug_set != "none"
        self.aug_set = aug_set
        self.normalisation = normalisation
        self.seed, self.epoch = seed, 0
        self.split = split

    def set_epoch(self, e):
        self.epoch = int(e)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = self.images[i].astype(np.float32) / 255.0
        if self.do_augment:
            rng = np.random.default_rng(
                (self.seed * 1_000_003 + self.epoch * 7_919 + i) % (2 ** 32))
            img = augment(img, rng, self.aug_set)
        img = normalise((img * 255.0).astype(np.uint8), self.normalisation) \
            if self.normalisation == "paper" else img
        enc = self.tokenizer.encode(self.labels[i])
        return (torch.from_numpy(np.expand_dims(img, 0)),
                torch.tensor(enc, dtype=torch.long), len(enc))


def collate(batch):
    """Pad targets to the longest in the batch (PAD = 0, never emitted)."""
    imgs, tgts, lens = zip(*batch)
    L = max(lens)
    out = torch.zeros(len(tgts), L, dtype=torch.long)
    for k, t in enumerate(tgts):
        out[k, : len(t)] = t
    return torch.stack(imgs), out, torch.tensor(lens, dtype=torch.long)

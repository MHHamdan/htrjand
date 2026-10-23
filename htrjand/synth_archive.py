"""Build a synthetic-line archive (synthetic-line generation).

    python -m htrjand.synth_archive --fonts synth/gfonts --coverage synth/font_coverage.json \
        --text en=synth/text_en.txt:0.55 fr=synth/text_fr.txt:0.35 la=synth/text_la.txt:0.10 \
        --n 200000 --out synth/synth_200k.hdf5

Differences from the basic synthetic-line generation, all on the data side: RandomText draws
real sentences (Wikipedia en/fr/la, checked for zero verbatim overlap with every
official test split) instead of random character strings, and the rendering adds
handwriting-like variation (slant, rotation, word spacing and baseline jitter,
stroke width, blur, ink and paper levels). Every line then goes through the
archive recipe of prepare_data.preprocess (aspect-preserving scale into 68 x 864,
centre pad, invert), so synthetic and real lines share one format. The archive has
one split, "train", with the same dt / gt layout as the real archives.
"""
import argparse
import json
import os
import random
import unicodedata
from multiprocessing import Pool

import cv2
import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .prepare_data import INPUT_SIZE

_MAP = {"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-",
        "«": '"', "»": '"', "…": "...", " ": " ", "æ": "ae", "œ": "œ"}

_G = {}


def _init(fonts, coverage, texts, charset, seed, size=INPUT_SIZE):
    _G.update(fonts=fonts, coverage=coverage, texts=texts, charset=set(charset), seed=seed, size=tuple(size))


def clean(t, allowed):
    t = "".join(_MAP.get(c, c) for c in t)
    t = unicodedata.normalize("NFC", t)
    return t if t and all(c in allowed for c in t) else None


def to_archive(img, size=INPUT_SIZE):
    """prepare_data.preprocess on an in-memory dark-on-light array."""
    th, tw = size
    h, w = img.shape
    s = min(tw / w, th / h)
    nw, nh = max(int(w * s), 1), max(int(h * s), 1)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    dh, dw = th - nh, tw - nw
    img = cv2.copyMakeBorder(img, dh // 2, dh - dh // 2, dw // 2, dw - dw // 2,
                             cv2.BORDER_CONSTANT, value=255)
    if img.mean() > 128:
        img = 255 - img
    return img.astype(np.uint8)


def render(text, font_path, rng):
    pt = rng.randint(30, 56)
    font = ImageFont.truetype(font_path, pt)
    space = font.getlength(" ") * rng.uniform(0.8, 2.0)
    words = text.split(" ")
    widths = [font.getlength(w) for w in words]
    W = int(sum(widths) + space * (len(words) - 1) + 2 * pt) + 10
    H = int(pt * 2.2)
    ink = rng.randint(0, 70)
    canvas = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(canvas)
    x = pt * 0.5
    base = H * 0.25
    for w, wd in zip(words, widths):
        d.text((x, base + rng.uniform(-0.06, 0.06) * pt), w, font=font, fill=ink)
        x += wd + space * rng.uniform(0.7, 1.3)
    a = np.array(canvas)
    # slant (shear) and a small rotation
    sh = rng.uniform(-0.35, 0.35)
    rot = np.deg2rad(rng.uniform(-1.5, 1.5))
    M = np.array([[1, sh, -sh * H / 2], [0, 1, 0]], np.float32)
    R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]], np.float32)
    M[:, :2] = R @ M[:, :2]
    a = cv2.warpAffine(a, M, (W + int(abs(sh) * H), H), borderValue=255, flags=cv2.INTER_LINEAR)
    # stroke width
    r = rng.random()
    if r < 0.25:
        a = cv2.erode(a, np.ones((2, 2), np.uint8))           # thicker ink
    elif r < 0.35:
        a = cv2.dilate(a, np.ones((2, 2), np.uint8))          # thinner ink
    # tight crop with a small random margin, like a segmented line
    ys, xs = np.where(a < 200)
    if len(ys) == 0:
        return None
    top = max(ys.min() - rng.randint(1, 8), 0)
    bot = min(ys.max() + rng.randint(1, 8), a.shape[0])
    lef = max(xs.min() - rng.randint(2, 20), 0)
    rig = min(xs.max() + rng.randint(2, 20), a.shape[1])
    a = a[top:bot, lef:rig].astype(np.float32)
    # paper level, blur, noise
    paper = rng.uniform(200, 255)
    a = paper - (255 - a) * (paper - ink) / max(255 - ink, 1)
    if rng.random() < 0.5:
        a = cv2.GaussianBlur(a, (0, 0), rng.uniform(0.3, 1.0))
    if rng.random() < 0.5:
        a = a + np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, rng.uniform(2, 10), a.shape)
    return to_archive(np.clip(a, 0, 255).astype(np.uint8), _G.get("size", INPUT_SIZE))


def _one(i):
    rng = random.Random(_G["seed"] * 1_000_003 + i)
    langs, weights = zip(*[(k, v[1]) for k, v in _G["texts"].items()])
    for _ in range(50):
        lang = rng.choices(langs, weights)[0]
        lines = _G["texts"][lang][0]
        font = rng.choice(_G["fonts"])
        t = clean(lines[rng.randrange(len(lines))].strip(), _G["charset"] & set(_G["coverage"][font]) | {" "})
        if not t or len(t) < 3:
            continue
        if rng.random() < 0.08:                       # digits and numbers appear in every corpus
            t = t + " " + str(rng.randint(0, 9999))
        img = render(t, font, rng)
        if img is not None:
            return img, t
    raise RuntimeError("no renderable line after 50 draws")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fonts", required=True)
    ap.add_argument("--coverage", required=True, help="json font path -> covered characters")
    ap.add_argument("--text", nargs="+", required=True, metavar="LANG=PATH:WEIGHT")
    ap.add_argument("--charset-dataset", default="combined")
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--size", type=int, nargs=2, default=list(INPUT_SIZE), metavar=("H", "W"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from .data import build_charset
    charset = build_charset(a.charset_dataset)
    coverage = json.load(open(a.coverage))
    fonts = sorted(coverage)
    texts = {}
    for spec in a.text:
        lang, rest = spec.split("=", 1)
        path, w = rest.rsplit(":", 1)
        texts[lang] = ([l.rstrip("\n") for l in open(path, encoding="utf-8") if l.strip()], float(w))
    imgs = np.zeros((a.n, *a.size), np.uint8)
    gts = []
    with Pool(a.workers, _init, (fonts, coverage, texts, charset, a.seed, a.size)) as p:
        for i, (img, t) in enumerate(p.imap(_one, range(a.n), chunksize=64)):
            imgs[i] = img
            gts.append(t)
            if (i + 1) % 20000 == 0:
                print(f"{i + 1}/{a.n}", flush=True)
    with h5py.File(a.out, "w") as h:
        g = h.create_group("train")
        g.create_dataset("dt", data=imgs)
        g.create_dataset("gt", data=[t.encode("utf-8") for t in gts])
    print(f"wrote {a.n} lines to {a.out}")


if __name__ == "__main__":
    main()

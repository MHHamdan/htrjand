"""Synthetic lines (Algorithm 2) and the curriculum ratio (Eq. 33) of the paper.

Algorithm 2 (SDG): n = |D| * r / (1 - r) lines; for each, random text from the
charset C, a random font from F, render, augment. Eq. (33):
r_s(e) = min(r_max, r_0 + (e / E) (r_max - r_0)), r_0 = 0.1, r_max = 0.4.

What the paper leaves open (register A-07, A-24, A-25): the font set F, the
text model behind RandomText, how the synthetic share is drawn per epoch, and
the five stages of Algorithm 3. This module implements the two equations as
written, with RandomText = words of 1-10 characters drawn uniformly from C
(space-separated, 1-8 words), rendered with PIL on a white canvas and then put
through the same archive recipe as real lines. It is an optional training-data
source (`train.py --synthetic-fonts DIR`); no reported run of this package
uses it, and no historical result traces to synthetic data either.
"""
import os
import random

import numpy as np

from .prepare_data import INPUT_SIZE


def synthetic_ratio(epoch, total_epochs, r0=0.1, r_max=0.4):
    """Eq. (33). With e in [0, E] the min never binds; it is kept as written."""
    return min(r_max, r0 + (epoch / max(total_epochs, 1)) * (r_max - r0))


def n_synthetic(n_real, ratio):
    """Alg. 2 line 1: n = |D| r / (1 - r)."""
    return int(n_real * ratio / (1.0 - ratio))


def random_text(charset, rng, max_words=8, max_word=10):
    chars = [c for c in charset if not c.isspace()]
    words = [''.join(rng.choice(chars) for _ in range(rng.randint(1, max_word)))
             for _ in range(rng.randint(1, max_words))]
    return " ".join(words)


def list_fonts(font_dir):
    out = []
    for root, _, files in os.walk(font_dir):
        out += [os.path.join(root, f) for f in files if f.lower().endswith((".ttf", ".otf"))]
    return sorted(out)


def render_line(text, font_path, rng, size=INPUT_SIZE):
    """Render dark text on a white canvas, then apply the archive recipe
    (aspect-preserving scale into 68 x 864, centre pad, invert) so synthetic
    lines are stored exactly like real ones."""
    from PIL import Image, ImageDraw, ImageFont
    import cv2
    from .prepare_data import preprocess
    pt = rng.randint(28, 44)
    font = ImageFont.truetype(font_path, pt)
    l, t, r, b = font.getbbox(text)
    w, h = max(r - l, 1) + 20, max(b - t, 1) + 20
    img = Image.new("L", (w, h), 255)
    ImageDraw.Draw(img).text((10 - l, 10 - t), text, font=font, fill=0)
    arr = np.array(img)
    import tempfile
    tmp = os.path.join(tempfile.gettempdir(), f"htrjand_synth_{os.getpid()}_{rng.randint(0, 1 << 30)}.png")
    cv2.imwrite(tmp, arr)
    try:
        return preprocess(tmp, size)
    finally:
        os.remove(tmp)


def generate(charset, font_dir, n, seed=0):
    """Alg. 2 lines 2-8 (augmentation is applied later by the dataset, like
    for real lines). Returns (images uint8 (n, 68, 864), texts)."""
    fonts = list_fonts(font_dir)
    if not fonts:
        raise FileNotFoundError(f"no .ttf/.otf fonts under {font_dir}")
    rng = random.Random(seed)
    imgs, texts = [], []
    for _ in range(n):
        t = random_text(charset, rng)
        imgs.append(render_line(t, rng.choice(fonts), rng))
        texts.append(t)
    return np.stack(imgs), texts

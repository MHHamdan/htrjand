"""Transcribe line images with a checkpoint of this package.

    python -m htrjand.predict --ckpt runs/iam_paper_teacher_s1/best.pth line.png
    python -m htrjand.predict --ckpt ... --cpu lines_dir/

Each image is prepared exactly as the archives were (prepare_data.preprocess:
grayscale, aspect-preserving scale into 68 x 864, centre pad, invert if the
page is light), normalised as the checkpoint was trained, and decoded greedily.
"""
import argparse
import os
import sys

import numpy as np
import torch

from . import data as D
from . import models as M
from .prepare_data import preprocess

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def load_lines(inputs):
    paths = []
    for p in inputs:
        if os.path.isdir(p):
            paths += sorted(os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith(IMAGE_EXT))
        else:
            paths.append(p)
    return paths


@torch.no_grad()
def transcribe(model, tok, paths, normalisation="paper", device="cpu", batch_size=16):
    out = []
    for i in range(0, len(paths), batch_size):
        imgs = [D.normalise(preprocess(p), normalisation) for p in paths[i:i + batch_size]]
        x = torch.from_numpy(np.stack(imgs)[:, None]).to(device)
        out += tok.ctc_greedy_decode(model(x))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="image files and/or directories")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args(argv)
    device = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else f"cuda:{a.gpu}")
    model, ck = M.from_checkpoint(a.ckpt)
    model.to(device)
    tok = D.Tokenizer(ck["charset"])
    paths = load_lines(a.inputs)
    if not paths:
        sys.exit("no images found")
    for p, t in zip(paths, transcribe(model, tok, paths, ck.get("normalisation", "paper"), device)):
        print(f"{p}\t{t}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Saint Gall archives without the raw "pt" special letter (punctuation dot).

The raw ground truth (raw/saintgall/ground_truth/transcription.txt, README) spells words letter by letter
with two special letters, "pt" (punctuation dot) and "et" ("&"). "pt" is annotated on the official
TRAINING pages only (716 tokens; 0 on validation and test pages), so this removes it everywhere; only
training labels change. Images and line order are copied from the existing archives; labels are mapped
old -> new through the raw tokens, so the result does not depend on archive order.

    python scripts/build_saintgall_nopt.py --data-dir $HTRJAND_DATA_DIR --raw-dir raw
"""
import argparse
import os

import h5py

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=os.environ.get("HTRJAND_DATA_DIR", "data"))
ap.add_argument("--raw-dir", default="raw")
a = ap.parse_args()
old2new = {}
for line in open(os.path.join(a.raw_dir, "saintgall", "ground_truth", "transcription.txt")):
    words = line.split()[1].split("|")
    old = " ".join(w.replace("-", "") for w in words)
    new = " ".join(x for x in ("".join(c for c in w.split("-") if c != "pt") for w in words) if x)
    if old2new.get(old, new) != new:
        raise SystemExit(f"ambiguous label {old!r}")
    old2new[old] = new
for src, dst in (("saintgall.hdf5", "saintgall_nopt.hdf5"), ("saintgall_128x1632.hdf5", "saintgall_nopt_128x1632.hdf5")):
    with h5py.File(os.path.join(a.data_dir, src)) as fi, h5py.File(os.path.join(a.data_dir, dst), "w") as fo:
        for sp in ("train", "valid", "test"):
            gt = [g.decode() if isinstance(g, bytes) else g for g in fi[sp]["gt"][:]]
            new = [old2new[g] for g in gt]
            fo.create_dataset(f"{sp}/dt", data=fi[sp]["dt"][:])
            fo.create_dataset(f"{sp}/gt", data=[g.encode() for g in new])
            print(dst, sp, len(gt), "labels changed:", sum(x != y for x, y in zip(gt, new)))

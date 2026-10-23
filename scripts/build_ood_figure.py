"""Out-of-distribution demo figure: every line in examples/ through the released general Student.

    python scripts/build_ood_figure.py [--model MHamdan/htrjand-student] [--device cpu]
        -> docs/assets/ood_examples.png, docs/assets/ood_examples.json

All twelve demo crops are shown, in file order, with no selection. Each line is decoded with greedy
CTC and with the released Wikipedia character LM of the line's language (English LM for the
English and German lines, French LM for the French lines; no German LM is released). The
"authors' reading" comes from examples/SOURCES.md; it is not ground truth and no error rate is
computed on these images.
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from htrjand import HTRJANDModel  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = dict(balzac="rimes", marieantoinette="rimes")   # French LM; everything else English
LABEL = dict(lincoln="Lincoln, letter (1854)", gettysburg="Lincoln, Gettysburg Address (1864)",
             darwin="Darwin, letter (1881), faint ink", balzac="Balzac, letter (1833), French",
             marieantoinette="Marie-Antoinette, letter (1793), French, bleed-through",
             modern_de="Modern German cursive (2007)", modern_en="Modern English (2006)")


def readings():
    out = {}
    for line in open(os.path.join(ROOT, "examples", "SOURCES.md"), encoding="utf-8"):
        m = re.match(r"\|\s*`(ood_[^`]+\.png)`\s*\|[^|]*\|\s*(.+?)\s*\|\s*$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="hub id or local directory (default: the released Student)")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    model = HTRJANDModel.from_pretrained(*([a.model] if a.model else []), variant="general", device=a.device)
    read = readings()
    rows = []
    for p in sorted(glob.glob(os.path.join(ROOT, "examples", "ood_*.png"))):
        name = os.path.basename(p)
        key = re.match(r"ood_\d+_(.+?)_[ab]\.png", name).group(1)
        corpus = CORPUS.get(key, "iam")
        g = model.transcribe(p, decoder="greedy", return_confidence=True)
        l = model.transcribe(p, decoder="lm", corpus=corpus, return_confidence=True)
        g, l = (g[0] if isinstance(g, list) else g), (l[0] if isinstance(l, list) else l)
        rows.append(dict(file=name, source=LABEL.get(key, key), lm_language="fr" if corpus == "rimes" else "en",
                         reading=read.get(name, ""), greedy=g.text, lm=l.text, confidence=round(float(g.confidence), 3)))
    label = (os.path.basename(os.path.normpath(a.model)) + " (local copy)") if a.model and os.path.isdir(a.model) else (a.model or "default")
    json.dump(dict(model=label, variant="general", rows=rows),
              open(os.path.join(ROOT, "docs", "assets", "ood_examples.json"), "w"), indent=1, ensure_ascii=False)
    fig, axes = plt.subplots(len(rows), 1, figsize=(11, 0.62 * len(rows)))
    for ax, r in zip(axes, rows):
        im = Image.open(os.path.join(ROOT, "examples", r["file"])).convert("L")
        ax.imshow(im, cmap="gray", aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4); s.set_color("#999999")
        ax.set_title(f"{r['source']}   ({r['file']})", fontsize=8, loc="left", color="#333333", pad=2)
        txt = (f"authors' reading: {r['reading']}\n"
               f"greedy CTC:       {r['greedy']}\n"
               f"+ {r['lm_language']} char LM:    {r['lm']}")
        ax.text(1.01, 0.5, txt, transform=ax.transAxes, fontsize=7.5, family="monospace", va="center", ha="left")
    fig.subplots_adjust(left=0.01, right=0.40, top=0.97, bottom=0.01, hspace=0.9)
    out = os.path.join(ROOT, "docs", "assets", "ood_examples.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()

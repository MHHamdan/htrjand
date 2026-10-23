#!/usr/bin/env python3
"""Qualitative figure from results/qualitative_examples.json (run build_qualitative_examples.py first).

    python scripts/fig_qualitative.py
    -> docs/figures/qualitative.{pdf,png,svg}, qualitative_examples_readme.png (public README),
       qualitative_caption.txt

Row choice (fixed): (a) IAM lm_correction, (b) IAM t5_correction, (c) Saint Gall historical,
(d) Washington historical, (e) IAM t5_failure, (f) IAM model_failure. README version (qualitative_examples_readme.png): IAM and Bentham only (IAM-HistDB images may not
be shown publicly): IAM lm_correction, t5_correction, t5_failure, model_failure; Bentham median_cer;
with an attribution footer.
Line image: the 128x1632 archive image (the network input) re-inverted for display and cropped to the
bounding box of non-padding pixels. Text rows: ground truth; HTR-JAND base (68x864);
HTR-JAND Student greedy CTC; + char LM; + char LM + T5 only when T5 changed the line. Error marks come from
the Levenshtein alignment against the ground truth: substituted / inserted characters in red on a red
tint, a deleted reference character as a red wedge under the gap where it is missing.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refined_figs_common as C  # noqa: E402

fk, S = C.figkit()
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

MONO = "DejaVu Sans Mono"
RED, RED_BG = S.PALETTE["red_strong"], S.tint(S.PALETTE["red_strong"], 0.16)
ROWS_PAPER = [("iam", "lm_correction"), ("iam", "t5_correction"), ("saintgall", "historical"),
              ("washington", "historical"), ("iam", "t5_failure"), ("iam", "model_failure")]
# public README: only IAM (CC BY-NC 4.0) and Bentham (CC BY 4.0) rows; no IAM-HistDB images (LICENSING.md)
ROWS_README = [("iam", "lm_correction"), ("iam", "t5_correction"), ("iam", "t5_failure"), ("iam", "model_failure"),
               ("bentham", "median_cer")]
README_FOOTER = ("Line images: IAM Handwriting Database (Marti & Bunke 2002, CC BY-NC 4.0) and Bentham (Sánchez et al. "
                 "2014, CC BY 4.0); test split; used for non-commercial research illustration.")
ATTRIBUTION = ("Line images: IAM Handwriting Database (U. Marti and H. Bunke, IJDAR 5:39-46, 2002; CC BY-NC 4.0); "
               "Saint Gall, IAM-HistDB (A. Fischer et al.), from St. Gallen, Stiftsbibliothek, Cod. Sang. 562, p. 30 "
               "(www.e-codices.ch; CC BY-NC 4.0); Washington, IAM-HistDB (A. Fischer et al.), George Washington Papers, "
               "Library of Congress; official test splits, used for non-commercial research.")
RULE_NAME = dict(lm_correction="largest raw CTC → char-LM CER reduction",
                 t5_correction="largest char-LM → T5 CER reduction",
                 historical="median raw-CTC CER line",
                 t5_failure="largest char-LM → T5 CER increase (over-correction)",
                 model_failure="highest final CER, reference ≥ 20 chars",
                 median_cer="median raw-CTC CER line")


def find(samples, ds, rule):
    hit = [s for s in samples if s["dataset"] == ds and rule in s["rules"]]
    if len(hit) != 1:
        raise SystemExit(f"rule {rule} on {ds}: {len(hit)} samples")
    return hit[0]


def crop(img):
    """Archive image is ink-bright, padding exactly 0: keep the bounding box of nonzero pixels."""
    rows, cols = np.where(img.max(1) > 0)[0], np.where(img.max(0) > 0)[0]
    return img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def marked(d, x, y, ref, hyp, fs, adv):
    """Draw hyp at (x, y) (baseline-centre y) in monospace with error marks; returns end x."""
    ops = C.align(ref, hyp) if ref is not None else [("=", c, c) for c in hyp]
    h = fs * 0.3528 * 1.05
    cx = x
    for op, a, b in ops:
        if op == "D":
            w = 0.55
            d.ax.add_patch(Polygon([(cx - w, y - h * 0.62), (cx + w, y - h * 0.62), (cx, y - h * 0.25)],
                                   closed=True, fc=RED, ec="none", zorder=7))
            continue
        if op in ("S", "I"):
            d.ax.add_patch(Rectangle((cx, y - h / 2), adv, h, fc=RED_BG, ec="none", zorder=5))
        d.ax.text(cx + adv / 2, y, b, fontsize=fs, family=MONO, ha="center", va="center",
                  color=RED if op in ("S", "I") else S.INK, zorder=6)
        cx += adv
    return cx


def build(rows, out_base, width=160.0, fs=6.0, formats=("pdf", "png", "svg"), dpi=200, footer=None):
    data = json.load(open(os.path.join(C.RES, "qualitative_examples.json")))
    samples = [find(data["samples"], ds, r) for ds, r in rows]
    adv = 0.6021 * fs * 0.3528                         # DejaVu Sans Mono advance (mm)
    lab_w, line_h, img_h, head_h, gap = 41.0, 2.95, 6.2, 3.6, 2.6
    blocks = []
    for s in samples:
        p = s["predictions"]
        tr = [("Ground truth", s["gt"], None, None),
              (C.BASE_LABEL, p["base"]["text"], s["gt"], p["base"]["cer"]),
              ("HTR-JAND Student, greedy CTC", p["raw CTC"]["text"], s["gt"], p["raw CTC"]["cer"]),
              ("+ char LM", p["char LM"]["text"], s["gt"], p["char LM"]["cer"])]
        if p["T5 on char LM"]["text"] != p["char LM"]["text"]:
            tr.append(("+ char LM + T5", p["T5 on char LM"]["text"], s["gt"], p["T5 on char LM"]["cer"]))
        blocks.append((s, tr))
    H = sum(head_h + img_h + 1.2 + line_h * len(tr) + gap for _, tr in blocks) + 1.0 + (3.2 if footer else 0.0)
    d = fk.Diagram(width, H, font_size=fs)
    y = H - 0.6
    letters = "abcdefgh"
    for k, ((s, tr), (ds, rule)) in enumerate(zip(blocks, rows)):
        sid = s["sample_id"] or f"test #{s['archive_index']}"
        d.text(0.5, y - head_h / 2, f"({letters[k]})", size=fs + 0.6, weight="bold", ha="left")
        d.text(5.0, y - head_h / 2, f"{C.NAMES[ds]} — {RULE_NAME[rule]}", size=fs + 0.3,
               weight="semibold", ha="left")
        d.text(width - 0.5, y - head_h / 2, f"{sid}  (test index {s['archive_index']})", size=fs - 0.4,
               color=S.MUTED, ha="right")
        y -= head_h
        im = crop(C.test_image(ds, s["archive_index"]))
        disp = 255 - im
        w = min(width - lab_w - 1.0, img_h * disp.shape[1] / disp.shape[0])
        d.image(np.dstack([disp] * 3).astype(np.uint8), lab_w, y - img_h, w, img_h, border=S.tint(S.INK, 0.35), lw=0.3)
        d.text(lab_w - 1.5, y - img_h / 2, "Line image", size=fs - 0.3, color=S.MUTED, ha="right")
        y -= img_h + 1.2
        for label, text, ref, c in tr:
            yc = y - line_h / 2
            d.text(lab_w - 1.5, yc, label, size=fs - 0.3, color=S.INK if ref is None else S.MUTED, ha="right")
            marked(d, lab_w, yc, ref, text, fs, adv)
            if c is not None:
                d.text(width - 0.5, yc, f"{100 * c:.1f}", size=fs - 0.3, color=S.MUTED, ha="right", family=MONO)
            else:
                d.text(width - 0.5, yc, "CER %", size=fs - 0.6, color=S.MUTED, ha="right")
            y -= line_h
        y -= gap
        if k < len(blocks) - 1:
            d.line((0.5, y + gap / 2), (width - 0.5, y + gap / 2), color=S.tint(S.INK, 0.18), lw=0.3)
    if footer:
        d.text(0.5, 1.6, footer, size=5.0, color=S.MUTED, ha="left")
    paths = d.save(out_base, formats=formats, dpi=dpi)
    return paths, samples


def caption(samples):
    lines = ["Qualitative test-set examples, chosen by fixed rules (scripts/build_qualitative_examples.py; "
             "results/qualitative_examples.json). Rows: (a) IAM, largest per-line CER reduction from raw CTC to "
             "the character LM; (b) IAM, largest reduction from the character LM to the gated T5 corrector; "
             "(c) Saint Gall and (d) Washington, the line at the median raw-CTC CER of the corpus; (e) IAM, "
             "largest CER increase caused by T5; (f) IAM, highest CER of the validation-selected path among "
             "references of at least 20 characters (ties: lowest archive index). Each block shows the 128×1632 "
             "network input (re-inverted, padding cropped), the reference, the HTR-JAND base "
             "Student (68×864 input, five-corpus training, greedy CTC), and the HTR-JAND Student (128×1632, seed 1) with "
             "greedy CTC, with the character "
             "LM, and with T5 when T5 changed the line. Red characters on a tint are substitutions or "
             "insertions with respect to the reference; a red wedge marks a deleted reference character. "
             "Numbers on the right are per-line CER (%). " + ATTRIBUTION, "", "Rows:"]
    for k, s in enumerate(samples):
        p = s["predictions"]
        lines.append(f"({'abcdef'[k]}) {s['dataset']} test#{s['archive_index']} {s['sample_id']} rules={s['rules']} "
                     f"base {100*p['base']['cer']:.1f} raw {100*p['raw CTC']['cer']:.1f} "
                     f"LM {100*p['char LM']['cer']:.1f} T5 {100*p['T5 on char LM']['cer']:.1f}")
    return "\n".join(lines) + "\n"


def main():
    os.makedirs(C.OUT, exist_ok=True)
    paths, samples = build(ROWS_PAPER, os.path.join(C.OUT, "qualitative"))
    open(os.path.join(C.OUT, "qualitative_caption.txt"), "w").write(caption(samples))
    p2, _ = build(ROWS_README, os.path.join(C.OUT, "qualitative_examples_readme"), width=165.0, formats=("png",), dpi=200,
                  footer=README_FOOTER)
    print("\n".join(map(str, paths + p2)))


if __name__ == "__main__":
    main()

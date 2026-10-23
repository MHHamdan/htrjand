#!/usr/bin/env python3
"""Recognition pipeline on one real IAM test line: original image -> 128x1632 network input -> raw CTC
-> char LM -> T5 (and the ground truth), with the actual strings.

    python scripts/fig_pipeline.py      -> docs/figures/pipeline.{pdf,png,svg}

Line (fixed rule): the IAM t5_correction sample of results/qualitative_examples.json (the only IAM
rule sample on which raw CTC, char LM and T5 each change the text). Strings: raw CTC from
results/ft_iam_c128_student_kd_s1_iam_test_preds.tsv, char LM from
results/ens_iam_ft_iam_c128_student_kd_s1_wiki_test_preds.tsv, T5 (candidate, confidence, final) from
results/t5_iam_ftstudent_lm_t5_ext_ctx1_test_preds.tsv; decoder / corrector settings from the
matching result JSONs (chosen on validation).
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refined_figs_common as C  # noqa: E402
from fig_qualitative import MONO, marked  # noqa: E402

fk, S = C.figkit()

COL = dict(pre=S.OKABE_ITO["sky"], net=S.OKABE_ITO["blue"], lm=S.OKABE_ITO["orange"], t5=S.OKABE_ITO["purple"])


def main():
    q = json.load(open(os.path.join(C.RES, "qualitative_examples.json")))
    s = next(x for x in q["samples"] if x["dataset"] == "iam" and "t5_correction" in x["rules"])
    i = s["archive_index"]
    t5 = C.read_tsv(os.path.join(C.RES, "t5_iam_ftstudent_lm_t5_ext_ctx1_test_preds.tsv"))[i]
    t5j = json.load(open(os.path.join(C.RES, "t5_iam_ftstudent_lm_t5_ext_ctx1.json")))
    lmj = json.load(open(os.path.join(C.RES, "ens_iam_ft_iam_c128_student_kd_s1_wiki.json")))
    raw = cv2.imread(os.path.join(C.RAW, s["sample_source"]["file"]), cv2.IMREAD_GRAYSCALE)
    net = 255 - C.test_image("iam", i)
    p = s["predictions"]
    assert t5["raw"] == p["char LM"]["text"] and t5["final"] == p["T5 on char LM"]["text"]

    W, H, fs = 160.0, 84.0, 6.4
    d = fk.Diagram(W, H, font_size=fs)
    adv = 0.6021 * fs * 0.3528
    x0, bw = 1.0, 47.0             # stage boxes
    xo = x0 + bw + 6.0             # outputs
    # row 1: original image
    y = H - 2.0
    d.text(x0, y - 1.5, f"Original line image  {s['sample_id']}.png  ({raw.shape[1]}×{raw.shape[0]} px)",
           size=fs, weight="semibold", ha="left")
    ih = 9.0
    iw = ih * raw.shape[1] / raw.shape[0]
    b_raw = d.image(np.dstack([raw] * 3), x0, y - 4.0 - ih, iw, ih, border=S.tint(S.INK, 0.4), lw=0.3)
    # preprocessing box
    yb = y - 4.0 - ih - 15.0
    pre = d.box(x0, yb, bw, 11.0, "Preprocessing", "grayscale; scale s = min(1632/w, 128/h)\n(INTER_AREA); centre-pad; invert;\nper-image min–max to [−1, 1]",
                color=COL["pre"], fill=0.10, title_size=fs, sub_size=5.6)
    d.arrow((x0 + bw / 2, b_raw.y), pre.n)
    nw = W - xo - 0.5
    nh = nw * 128 / 1632
    b_net = d.image(np.dstack([net] * 3), xo, yb - 0.5, nw, nh, border=S.INK, lw=0.4)
    d.text(xo, yb + nh + 0.6, "Network input, 128 × 1632 canvas (shown re-inverted; border = canvas)", size=5.8,
           color=S.MUTED, ha="left", va="bottom")
    d.arrow(pre.e, (xo, pre.c[1]))
    # stages
    rows = [("net", "HTR-JAND Student (seed 1)", "806,287 deployed parameters\ngreedy CTC decoding",
             "Raw CTC", p["raw CTC"]),
            ("lm", "Character LM decoding (external)", f"CTC prefix beam search + {lmj['lm']['order']}-gram char LM\n"
             f"α = {lmj['chosen']['alpha']}, β = {lmj['chosen']['beta']} (chosen on validation)", "+ char LM", p["char LM"]),
            ("t5", "Gated T5-small corrector (external)", f"detect: {t5j['selected']['detect'].upper()} words; context ±{t5j['context_window']} line\n"
             f"apply if confidence > θ = {t5j['selected']['theta']}  (here {float(t5['t5_confidence']):.4f})",
             "+ char LM + T5", p["T5 on char LM"])]
    yy = yb - 6.0
    prev = (xo + nw / 2, b_net.y)
    prev_box = None
    for k, (key, title, sub, lab, pr) in enumerate(rows):
        yy -= 12.5
        bx = d.box(x0, yy, bw, 10.0, title, sub, color=COL[key], fill=0.10, ls="-" if key == "net" else (0, (3, 1.6)),
                   title_size=fs - 0.2, sub_size=5.5)
        if k == 0:
            d.arrow((x0 + bw / 2, yb), bx.n)
        else:
            d.arrow(prev_box.s, bx.n)
        prev_box = bx
        yc = bx.c[1]
        d.text(xo, yc + 2.4, lab, size=5.8, color=S.MUTED, ha="left")
        marked(d, xo, yc - 1.2, s["gt"], pr["text"], fs + 0.6, 0.6021 * (fs + 0.6) * 0.3528)
        d.text(W - 0.5, yc - 1.2, f"CER {100 * pr['cer']:.1f} %", size=5.8, color=S.MUTED, ha="right")
        d.arrow(bx.e, (xo - 1.0, yc - 1.2), color=S.tint(S.INK, 0.5), lw=0.4)
    yy -= 7.0
    d.text(xo, yy + 2.4, "Ground truth", size=5.8, color=S.MUTED, ha="left")
    marked(d, xo, yy - 1.2, None, s["gt"], fs + 0.6, 0.6021 * (fs + 0.6) * 0.3528)
    d.text(x0, yy - 1.2, f"IAM test line {i}. Red: substituted / inserted\ncharacters; red wedge: deleted character.",
           size=5.4, color=S.MUTED, ha="left")
    d.save(os.path.join(C.OUT, "pipeline"), formats=("pdf", "png", "svg"), dpi=200)
    print(s["sample_id"], [(r[3], r[4]["text"]) for r in rows])


if __name__ == "__main__":
    main()

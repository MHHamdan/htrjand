#!/usr/bin/env python3
"""Static architecture figure of the HTR-JAND Student (and its training-only / decoding context).

    python scripts/verify_architecture.py     # shapes + parameter counts from the real checkpoints
    python scripts/fig_architecture.py        # -> docs/figures/architecture.{pdf,png,svg}

Every number drawn is read from docs/figures/architecture_shapes.json (written by
verify_architecture.py from runs/ft_iam_c128_student_kd_s1, c128_student_kd_s1, c128_syn_teacher_s1)
or from the decoder / corrector result JSONs; nothing is typed by hand except the wording.
The thumbnail is the real network input of IAM test line 0 (licensed image; keep internal).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refined_figs_common as C  # noqa: E402

fk, S = C.figkit()
OK = S.OKABE_ITO
COL = dict(pre=OK["sky"], cnn=OK["blue"], rnn="#1C2E62", att=OK["purple"], head=S.PALETTE["gray_2"],
           train=OK["vermillion"], dec=OK["green"])


def main():
    A = json.load(open(os.path.join(C.OUT, "architecture_shapes.json")))
    st, te, par = A["student"], A["teacher"], A["student_parent_kd"]
    lmj = C.J("ens_iam_ft_iam_c128_student_kd_s1_wiki.json")
    orders = sorted({C.J(f"{C.SEED1[ds]['lm']}.json")["lm"]["order"] for ds in C.CORPORA})
    t5j = C.J("t5_iam_ftstudent_lm_t5_ext_ctx1.json")
    cfg, tcfg = st["cfg"], te["cfg"]
    hk = {n.split(" (")[0]: o for n, i, o in st["hooks"]}
    T, V, d, cd = st["frames"], st["vocab_incl_blank"], st["d_model"], st["column_dim"]
    H_in, W_in = st["input"][2], st["input"][3]
    blocks = [o for n, i, o in st["hooks"] if n.startswith("cnn block")]
    tch = tcfg["channels"]
    pa = par["train_args"]

    W, H, fs = 170.0, 104.0, 6.6
    dg = fk.Diagram(W, H, font_size=fs)
    sub = 5.6
    # ------------------------------------------------------------------ recognizer panel
    P = dg.panel(1.0, 50.0, W - 2.0, 52.0, COL["cnn"], fill=0.035)
    dg.tag(3.0, 99.5 - 1.4, f"Recognizer — HTR-JAND Student, {st['params_deployed']:,} parameters at inference",
           COL["cnn"], ha="left", va="center", size=6.6)
    # row 1
    y1 = 82.0
    img = 255 - C.test_image("iam", 0)
    tw = 34.0
    th = tw * H_in / W_in
    thumb = dg.image(np.dstack([img] * 3).astype(np.uint8), 4.0, y1 - th / 2, tw, th, border=S.INK, lw=0.35)
    dg.text(4.0 + tw / 2, y1 + th / 2 + 2.0, f"text-line image", size=sub, ha="center")
    dg.text(4.0 + tw / 2, y1 - th / 2 - 2.0, f"{H_in} × {W_in} (H × W), 1 channel", size=sub, color=S.MUTED)
    pre = dg.box(42.0, y1 - 5.0, 21.0, 10.0, "Preprocess", "aspect-kept resize,\ncentre-pad, invert,\nmin–max to [−1, 1]",
                 color=COL["pre"], fill=0.12, title_size=fs - 0.2, sub_size=5.2)
    dg.arrow(thumb.e, pre.west)
    # CNN cuboids
    xs = 68.0
    sizes = [(9.0, 12.0, 2.0), (7.5, 9.0, 3.0), (6.0, 5.0, 4.4)]
    cubes = []
    for k, ((w, h, dep), shp) in enumerate(zip(sizes, blocks)):
        c = dg.cuboid(xs, y1 - h / 2 - 1.0, w, h, dep, COL["cnn"])
        dg.text(xs + w / 2 + 1.0, y1 + 9.2, f"{shp[1]}@{shp[2]}×{shp[3]}", size=5.2, color=S.INK)
        dg.text(xs + w / 2 + 1.0, y1 + 7.0, "pool " + "×".join(map(str, cfg["pools"][k])), size=5.0, color=S.MUTED)
        cubes.append(c)
        xs += w + dep * 0.55 + 6.0
    dg.text((68.0 + xs - 6.0) / 2 + 1.0, y1 + 13.0, f"{len(blocks)} × [gated conv 3×3 → BN → ReLU → max-pool → SE (r = {cfg['se_reduction']})]; C@H×W",
            size=5.6, ha="center")
    dg.arrow(pre.e, (68.0, y1))
    # columns
    cx = xs + 1.0
    cols = dg.cells(cx, y1 - 0.75, 14, 1, 1.5, [S.tint(COL["cnn"], 0.25 + 0.04 * (i % 5)) for i in range(14)], lw=0.25)
    dg.text(cx + 10.5, y1 + 3.0, "columns", size=5.6)
    dg.text(cx + 10.5, y1 - 4.0, f"{T} × {cd}\n({blocks[-1][1]} ch × {blocks[-1][2]} rows)", size=5.2, color=S.MUTED)
    dg.arrow((cubes[-1].x1 + 2.6, y1), (cx - 0.3, y1))
    # aux head (training only), attached to the columns
    ax_ = dg.box(cx + 26.0, y1 - 5.0, 22.0, 10.0, "Aux. CTC head", f"linear {cd} → {V}\nloss weight δ = {pa['aux_weight']}",
                 color=COL["train"], fill=0.08, ls=(0, (3, 1.6)), title_size=fs - 0.4, sub_size=5.2)
    dg.arrow((cx + 21.3, y1), ax_.west, color=COL["train"], ls=(0, (3, 1.6)))
    dg.text(ax_.c[0], ax_.y1 + 1.6, "training only", size=5.2, color=COL["train"])
    # row 2
    y2 = 61.0
    dg.arrow((cx + 10.5, y1 - 7.4), (cx + 10.5, 73.5), (10.0 + 13.0, 73.5), (10.0 + 13.0, y2 + 5.6), r=1.2)
    lst = dg.box(6.0, y2 - 5.5, 34.0, 11.0, f"{cfg['lstm_layers']} × BiLSTM + LayerNorm",
                 f"{cfg['lstm_hidden']} units per direction\n{T} × {cd} → {T} × {d}", color=COL["rnn"], fill=0.10,
                 title_size=fs - 0.2, sub_size=5.4)
    dg.stack_shadow(lst, n=2, step=(0.8, 0.8), color=COL["rnn"], r=1.0)
    mh = dg.box(50.0, y2 + 1.2, 33.0, 8.4, f"MHSA ({cfg['heads']} head)", f"softmax($QK^{{T}}/\\sqrt{{d}}$)$V$; weights {T} × {T}",
                color=COL["att"], fill=0.10, title_size=fs - 0.4, sub_size=4.9)
    px = dg.box(50.0, y2 - 9.6, 33.0, 8.4, "Proxima", f"one learned query; weights 1 × {T};\nsummary broadcast to all frames",
                color=COL["att"], fill=0.10, title_size=fs - 0.4, sub_size=5.0)
    dg.arrow(lst.e, (45.0, y2), (45.0, mh.c[1]), mh.west, r=1.0)
    dg.arrow((45.0, y2), (45.0, px.c[1]), px.west, r=1.0)
    cb = dg.box(91.0, y2 - 6.0, 32.0, 12.0, "Combined Attention", f"$O$ = LN($W_f\\,[A\\,;P] + b + X$)\n{T} × {2*d} → {T} × {d}",
                color=COL["att"], fill=0.16, title_size=fs - 0.2, sub_size=5.4)
    dg.arrow(mh.e, (87.0, mh.c[1]), (87.0, y2 + 2.0), (cb.x, y2 + 2.0), r=1.0)
    dg.arrow(px.e, (87.0, px.c[1]), (87.0, y2 - 2.0), (cb.x, y2 - 2.0), r=1.0)
    # residual X
    dg.arrow((44.0, y2 + 0.0), (44.0, 71.2), (107.0, 71.2), (107.0, cb.y1), color=S.MUTED, lw=0.45, r=1.0)
    dg.text(108.5, 69.4, "residual $X$", size=5.0, color=S.MUTED, ha="left")
    lin = dg.box(129.0, y2 - 5.0, 13.0, 10.0, "Linear", f"{d} → {V}", color=COL["head"], fill=0.10, title_size=fs - 0.4, sub_size=5.4)
    dg.arrow(cb.e, lin.west)
    post = dg.box(147.0, y2 - 6.0, 20.0, 12.0, "CTC", f"posteriors\n{T} × {V}", color=COL["head"], fill=0.16,
                  title_size=fs - 0.2, sub_size=5.4)
    dg.arrow(lin.e, post.west)
    dg.text(post.c[0], post.y - 2.2, f"V = {st['charset_size']} chars\n+ PAD, UNK, blank", size=4.9, color=S.MUTED)

    # ------------------------------------------------------------------ training-only panel
    Tp = dg.panel(1.0, 1.0, 98.0, 43.0, COL["train"], fill=0.04, ls=(0, (4, 2)))
    dg.tag(3.0, 41.6, "Training only (not deployed)", COL["train"], ha="left", va="center", size=6.4)
    teacher = dg.box(4.0, 22.0, 44.0, 14.0, "Teacher (frozen)",
                     f"{len(tch)} gated conv + SE blocks, {tch[0]}–{tch[-1]} ch\n{tcfg['lstm_layers']} × BiLSTM {tcfg['lstm_hidden']}/dir, MHSA {tcfg['heads']} heads\n"
                     f"{te['params_deployed']:,} params (same input, {te['frames']} frames)",
                     color=COL["train"], fill=0.10, title_size=fs - 0.2, sub_size=5.2)
    kd = dg.box(54.0, 22.0, 42.0, 14.0, "Distillation loss",
                f"$\\tau^2$ KL(softmax($z_T/\\tau$) ‖ softmax($z_S/\\tau$)),\nframe-wise, τ = {pa['tau']:g}; weight γ {pa['gamma'][0]} → {pa['gamma'][1]}",
                color=COL["train"], fill=0.10, title_size=fs - 0.2, sub_size=5.2)
    dg.arrow(teacher.e, kd.west, color=COL["train"])
    dg.text(51.0, 30.6, "$z_T$", size=5.6, color=COL["train"])
    syn = dg.box(4.0, 6.5, 44.0, 11.0, "Synthetic lines mixed in",
                 f"share r = {pa['synth_ratio'][0]} → {pa['synth_ratio'][1]} of each epoch (linear schedule);\n"
                 "rendered fonts, Wikipedia text", color=COL["train"], fill=0.06, title_size=fs - 0.4, sub_size=5.1)
    dg.text(54.0, 15.0, f"$L = \\alpha L_{{CTC}} + \\gamma L_{{KD}} + \\delta L_{{aux}}$,  α {pa['alpha'][0]} → {pa['alpha'][1]}, δ = {pa['aux_weight']}",
            size=5.3, ha="left")
    dg.text(54.0, 9.5, f"KD + synthetic: five-corpus stage, {pa['epochs']} epochs.\n"
            f"Per-corpus fine-tuning, {st['train_args']['epochs']} epochs: $L_{{CTC}} + \\delta L_{{aux}}$.", size=5.1, color=S.MUTED, ha="left")
    # student logits -> KD
    dg.arrow((post.c[0] - 4.0, post.y - 5.0), (post.c[0] - 4.0, 47.0), (kd.c[0] + 8.0, 47.0), (kd.c[0] + 8.0, kd.y1),
             color=COL["train"], ls=(0, (3, 1.6)), r=1.2)
    dg.text(kd.c[0] + 10.0, 38.5, "$z_S$", size=5.6, color=COL["train"], ha="left")

    # ------------------------------------------------------------------ decoding panel
    Dp = dg.panel(103.0, 1.0, W - 104.0, 43.0, COL["dec"], fill=0.04)
    dg.tag(105.0, 41.6, "Decoding (outside the network)", COL["dec"], ha="left", va="center", size=6.4)
    gr = dg.box(106.0, 27.0, 26.0, 9.0, "Greedy CTC", "best path, collapse,\ndrop blanks", color=COL["dec"], fill=0.08,
                title_size=fs - 0.4, sub_size=5.0)
    lm = dg.box(137.0, 27.0, 30.0, 9.0, "Prefix beam search", f"+ external char n-gram LM\n(n = {'–'.join(map(str, orders))}; train text + Wikipedia)",
                color=COL["dec"], fill=0.08, title_size=fs - 0.4, sub_size=5.0)
    dg.text(134.5, 31.5, "or", size=5.6, color=S.MUTED)
    t5 = dg.box(122.0, 9.0, 34.0, 11.0, "Optional T5-small corrector", f"gated: OOV words and\nconfidence > θ ({t5j['selected']['theta']} on IAM)",
                color=COL["dec"], fill=0.04, ls=(0, (3, 1.6)), title_size=fs - 0.4, sub_size=5.0)
    dg.arrow((post.c[0] + 4.0, post.y - 5.0), (post.c[0] + 4.0, 39.0), (gr.c[0], 39.0), gr.n, color=COL["dec"], r=1.0)
    dg.arrow((post.c[0] + 4.0, 39.0), (lm.c[0], 39.0), lm.n, color=COL["dec"], r=1.0)
    dg.arrow(lm.s, (lm.c[0], 23.0), (t5.c[0] + 6.0, 23.0), (t5.c[0] + 6.0, t5.y1), color=COL["dec"], r=1.0)
    dg.arrow(gr.s, (gr.c[0], 23.0), (t5.c[0] - 6.0, 23.0), (t5.c[0] - 6.0, t5.y1), color=COL["dec"], r=1.0)
    dg.arrow(t5.e, (W - 4.0, t5.c[1]), color=COL["dec"])
    dg.text(W - 3.0, t5.c[1] - 3.2, "text", size=5.6, ha="right")
    dg.text(106.0, 4.0, "choice of path, α, β, θ made on validation", size=5.0, color=S.MUTED, ha="left")
    dg.save(os.path.join(C.OUT, "architecture"), formats=("pdf", "png", "svg"), dpi=200)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Interpretability figure for the selected IAM Student, from real activations.

    CUDA_VISIBLE_DEVICES=0 python scripts/attention_activations.py   # activations + test-set statistics
    python scripts/fig_attention.py [--index 1849]
    -> docs/figures/attention.{pdf,png,svg}

Line: the IAM lm_correction sample of results/qualitative_examples.json (the pipeline-figure line).
All panels share the frame axis (T = 204 frames; frame f covers pixels [8f, 8f+8) of the 1632-px input).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refined_figs_common as C  # noqa: E402

fk, S = C.figkit()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec  # noqa: E402
from htrjand import data as D  # noqa: E402

RED, BLUE = S.PALETTE["red_strong"], S.PALETTE["blue_main"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=None)
    a = ap.parse_args()
    q = json.load(open(os.path.join(C.RES, "qualitative_examples.json")))
    samp = next(s for s in q["samples"] if s["dataset"] == "iam" and "lm_correction" in s["rules"])
    i = a.index if a.index is not None else samp["archive_index"]
    z = np.load(os.path.join(C.RES, "attention_student_iam_test.npz"), allow_pickle=True)
    st = json.load(open(os.path.join(C.OUT, "attention_stats.json")))
    lp, W, p = z[f"{i}_lp"], z[f"{i}_mha"], z[f"{i}_prox"]
    cA, cP, cX = z[f"{i}_cA"], z[f"{i}_cP"], z[f"{i}_cX"]
    emit, ops, (lo, hi) = z[f"{i}_emit"], list(z[f"{i}_ops"]), tuple(z[f"{i}_ink"])
    hyp = str(z[f"{i}_hyp"])
    import torch
    ck = torch.load(os.path.join(C.ROOT, "runs", "ft_iam_c128_student_kd_s1", "best.pth"), map_location="cpu", weights_only=False)
    tok = D.Tokenizer(ck["charset"])
    gt = C.test_gt("iam")[i]
    img = 255 - C.test_image("iam", i)
    T = lp.shape[0]
    conf = np.exp(lp.max(-1))
    # error flag per emission (S / I = wrong emitted char)
    err = [op for op, _, _ in ops if op != "D"]
    assert len(err) == len(emit)
    sp_id = tok.char_to_index[" "]

    S.use_style(6.2)
    fig = plt.figure(figsize=(160 * S.MM, 112 * S.MM))
    gs = GridSpec(6, 2, figure=fig, width_ratios=[2.35, 1], height_ratios=[1.25, 0.55, 1, 1, 0.9, 0.35],
                  left=0.085, right=0.955, top=0.93, bottom=0.075, wspace=0.3, hspace=0.55)
    axs = [fig.add_subplot(gs[r, 0]) for r in range(6)]
    for ax in axs[1:]:
        ax.sharex(axs[0])
    ext = (0, T, 0, 128 / 8)
    # (a) input
    ax = axs[0]
    ax.imshow(img, cmap="gray", extent=ext, aspect="auto", interpolation="lanczos")
    ax.set_yticks([]); ax.tick_params(labelbottom=False)
    ax.set_title(f"(a) network input, IAM test {i} ({samp['sample_id']})   GT: “{gt}”", loc="left", fontsize=6.2)
    # (b) CTC alignment
    ax = axs[1]
    for (f, c), e in zip(emit, err):
        ch = tok.chars[c]
        col = RED if e != "=" else S.INK
        ax.axvline(f + 0.5, color=S.tint(col, 0.5), lw=0.4)
        ax.text(f + 0.5, 0.5, "␣" if ch == " " else ch, ha="center", va="center", fontsize=5.6, color=col,
                family="DejaVu Sans Mono", bbox=dict(fc="white", ec="none", pad=0.2))
    ax.set_ylim(0, 1); ax.set_yticks([]); ax.tick_params(labelbottom=False)
    for s_ in ("left",):
        ax.spines[s_].set_visible(False)
    ax.set_title(f"(b) greedy CTC emissions at their frames (red: wrong)   “{hyp}”", loc="left", fontsize=6.2)
    # space frames
    spf = [f for (f, c) in emit if c == sp_id]
    def shade(ax):
        for f in spf:
            ax.axvspan(f - 2, f + 3, color=S.tint(S.PALETTE["teal"], 0.18), lw=0, zorder=0)
    # (c) MHSA received attention
    ax = axs[2]
    Wn = W[lo:hi, lo:hi] / W[lo:hi, lo:hi].sum(1, keepdims=True)
    colm = np.zeros(T); colm[lo:hi] = Wn.mean(0)
    shade(ax)
    ax.plot(np.arange(T) + 0.5, colm, color=BLUE, lw=0.8)
    ax.axhline(1 / (hi - lo), color=S.MUTED, lw=0.5, ls=":")
    ax.set_ylabel("mean\nweight", fontsize=5.6); ax.tick_params(labelbottom=False)
    m = st["mha"]
    ax.set_title("(c) MHSA: mean weight received per key frame (teal: emitted space ±2 frames; dotted: uniform)",
                 loc="left", fontsize=6.2)
    # (d) Proxima
    ax = axs[3]
    shade(ax)
    ax.plot(np.arange(T) + 0.5, p, color=S.PALETTE["violet"], lw=0.8)
    ax.axhline(1 / T, color=S.MUTED, lw=0.5, ls=":")
    for f, c in emit:
        if c != sp_id:
            ax.plot(f + 0.5, -0.0015, "|", color=S.INK, ms=2.5, mew=0.5, clip_on=False)
    ax.set_ylabel("weight", fontsize=5.6); ax.tick_params(labelbottom=False)
    pr = st["proxima"]
    ax.set_title("(d) Proxima: learned-query softmax over frames (ticks: character emissions; dotted: uniform)", loc="left", fontsize=6.2)
    # (e) fusion norms
    ax = axs[4]
    xs = np.arange(T) + 0.5
    ax.plot(xs, cX, color=S.PALETTE["gray_2"], lw=0.8, label=r"residual $\|X\|$")
    ax.plot(xs, cA, color=BLUE, lw=0.8, label=r"MHSA term $\|W_f^{A}A\|$")
    ax.plot(xs, cP, color=S.PALETTE["violet"], lw=0.8, label=r"Proxima term $\|W_f^{P}P+b\|$")
    ax.set_ylim(0, max(cX.max(), cA.max(), cP.max()) * 1.6)
    ax.set_ylabel("L2 norm", fontsize=5.6); ax.tick_params(labelbottom=False)
    ax.legend(fontsize=5.2, ncol=3, loc="upper right", handlelength=1.4, borderaxespad=0.1)
    ax.set_title("(e) Combined-Attention fusion O = LN(W$_f$[A;P] + b + X): per-frame L2 norm of each summand", loc="left", fontsize=6.2)
    # (f) confidence strip
    ax = axs[5]
    ax.imshow(conf[None, :], cmap="Greys_r", vmin=0, vmax=1, extent=(0, T, 0, 1), aspect="auto", interpolation="nearest")
    for (f, c), e in zip(emit, err):
        if e != "=":
            ax.plot(f + 0.5, 1.35, "v", color=RED, ms=2.6, clip_on=False)
    ax.set_yticks([]); ax.set_xlabel("frame index (frame f = input columns 8f to 8f+7)")
    ax.set_xlim(lo - 4, hi + 4)
    ax.set_title("(f) per-frame max posterior (black 0, white 1); ▼ wrong emission", loc="left", fontsize=6.2)
    # right: MHSA matrix
    rg = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[:, 1], height_ratios=[1.1, 1], hspace=0.62)
    axm = fig.add_subplot(rg[0])
    im = axm.imshow(Wn, cmap="magma", extent=(lo, hi, hi, lo), interpolation="nearest", vmin=0, vmax=np.percentile(Wn, 99.5))
    axm.set_xlabel("key frame"); axm.set_ylabel("query frame")
    axm.set_title("(g) MHSA attention matrix (1 head)", loc="left", fontsize=6.2)
    cb = fig.colorbar(im, ax=axm, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=5); cb.set_label("weight", fontsize=5.4)
    # right: confidence histogram (test set)
    axh = fig.add_subplot(rg[1])
    zz = None
    c = st["confidence"]
    hist = st.get("confidence_hist")
    if hist:
        e = np.array(hist["edges"])
        axh.stairs(hist["correct"], e, color=S.INK, lw=0.8, label=f"correct (n={c['n_correct_emissions']:,})")
        axh.stairs(hist["error"], e, color=RED, lw=0.8, label=f"wrong (n={c['n_error_emissions']:,})")
        axh.set_yscale("log")
    axh.set_xlabel("max posterior at the emitting frame"); axh.set_ylabel("share of emissions")
    axh.legend(fontsize=5.2, loc="upper left")
    axh.set_title(f"(h) emission confidence, all IAM test lines\n(AUROC {c['auroc_low_conf_detects_error']:.2f} for flagging wrong ones)",
                  loc="left", fontsize=6.2)
    S.save(fig, os.path.join(C.OUT, "attention"), formats=("pdf", "png", "svg"), dpi=200)
    print("line", i, "GT", gt, "| hyp", hyp)


if __name__ == "__main__":
    main()

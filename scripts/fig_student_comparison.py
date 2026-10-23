#!/usr/bin/env python3
"""Student CER per corpus: HTR-JAND base (68x864) vs the HTR-JAND Student (128x1632, seeds 1 and 2), and the
raw CTC -> char LM -> T5 decoding ablation.

    python scripts/fig_student_comparison.py
    -> docs/figures/student_comparison.{pdf,png,svg}, decoding_ablation.{pdf,png,svg},
       student_comparison_data.json (every plotted number with its source)

Sources (test CER; every choice made on validation CER):
  seed 1   results/final_selection.json, "Student | <stage>" entries
  seed 2   results/seed2_student.json; for Saint Gall its "saintgall" entry = the marker-free Student
           nopt_sg_student_s2 ("(marker-free labels)" paths, incl. the T5 arms
           results/t5_saintgall_nopt_s2student_{ctc,lm}_t5_ext_ctx1.json, whose gate kept every line)
  base     HTR-JAND base (68x864): results/combined_final_student_kd_s{1,2}_<corpus>_test.json (tuned five-corpus
           Student, greedy CTC); Saint Gall: that model was trained with the original labels (raw "pt" letter)
Validation-selected path per seed: lowest validation CER among the stages available (exact tie -> the
simpler path), refined_figs_common.pick_path (scripts/refined_figs_common.py).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refined_figs_common as C  # noqa: E402

fk, S = C.figkit()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


def collect():
    sel = C.J("final_selection.json")
    s2 = C.J("seed2_student.json")
    out = {}
    for ds in C.CORPORA:
        seeds = {}
        e = {st: dict(valid=sel[ds][f"Student | {st}"]["v"][0], test=sel[ds][f"Student | {st}"]["t"][0],
                      id=sel[ds][f"Student | {st}"]["id"]) for st in C.STAGES}
        seeds["1"] = e
        if ds == "saintgall":
            # marker-free pooled selection; T5 arms results/t5_saintgall_nopt_s2student_{ctc,lm}_t5_ext_ctx1.json
            suf = " (marker-free labels)"
            seeds["2"] = {st[:-len(suf)]: dict(valid=v["valid"], test=v["test"], id=f"seed2_student.json:saintgall:{st}")
                          for st, v in s2[ds]["paths"].items() if st.endswith(suf)}
            assert s2[ds]["model"] == "nopt_sg_student_s2" and set(seeds["2"]) == set(C.STAGES)
        else:
            seeds["2"] = {st: dict(valid=v["valid"], test=v["test"], id=f"seed2_student.json:{ds}:{st}")
                          for st, v in s2[ds]["paths"].items()}
        for k, e in seeds.items():
            path = C.pick_path({st: v["valid"] for st, v in e.items()})
            e["selected"] = dict(path=path, **{x: e[path][x] for x in ("valid", "test", "id")})
        base = {k: dict(test=C.J(f"{v[ds]}_test.json")["cer"], id=v[ds]) for k, v in C.BASE_SEEDS.items()}
        out[ds] = dict(base=base, base_mean=float(np.mean([b["test"] for b in base.values()])), seeds=seeds)
    return out


def comparison(D, base):
    fig = plt.figure(figsize=(160 * S.MM, 66 * S.MM))
    ax = fig.add_axes([0.065, 0.235, 0.925, 0.63])
    blue, lblue, grey = S.PALETTE["blue_main"], S.tint(S.PALETTE["blue_secondary"], 0.45), S.PALETTE["neutral"]
    w = 0.26
    for i, ds in enumerate(C.CORPORA):
        d = D[ds]
        for off, key, col in ((-w, "base", grey), (0.0, "raw CTC", lblue), (w, "selected", blue)):
            vals = [100 * (d["base"][s] if key == "base" else d["seeds"][s][key])["test"] for s in ("1", "2")]
            m = float(np.mean(vals))
            ax.bar(i + off, m, w * 0.92, color=col, ec="none")
            for s, v, mk in zip(("1", "2"), vals, ("o", "s")):
                ax.plot(i + off + (-0.055 if s == "1" else 0.055), v, mk, ms=2.6, mfc="white", mec=S.INK, mew=0.5, zorder=5)
            ax.text(i + off, max(vals) + 0.12, f"{m:.2f}", ha="center", va="bottom", fontsize=5.6)
        paths = " / ".join(d["seeds"][s]["selected"]["path"].replace("T5 on char LM", "LM+T5").replace("char LM", "LM")
                           for s in ("1", "2"))
        ax.text(i, -0.085, "selected: " + paths, ha="center", va="top", fontsize=5.0, color=S.MUTED, transform=ax.get_xaxis_transform())
    ax.set_xticks(range(len(C.CORPORA)), [C.NAMES[d] + ("†" if d == "saintgall" else "") for d in C.CORPORA])
    ax.tick_params(axis="x", length=0, pad=2.5)
    ax.set_ylabel("Test CER (%)")
    top = max(100 * x["test"] for d in D.values() for x in list(d["base"].values()))
    ax.set_ylim(0, float(np.ceil(top * 1.1)))
    ax.set_xlim(-0.6, len(C.CORPORA) - 0.4)
    ax.yaxis.grid(True, lw=0.3, color=S.tint(S.INK, 0.15)); ax.set_axisbelow(True)
    h = [Patch(fc=grey, label="HTR-JAND base (68×864), greedy CTC"),
         Patch(fc=lblue, label="HTR-JAND Student, greedy CTC"),
         Patch(fc=blue, label="HTR-JAND Student, val.-selected path (char LM ± T5)"),
         Line2D([], [], ls="", marker="o", ms=2.6, mfc="white", mec=S.INK, mew=0.5, label="seed 1"),
         Line2D([], [], ls="", marker="s", ms=2.6, mfc="white", mec=S.INK, mew=0.5, label="seed 2")]
    h = [h[0], h[3], h[1], h[4], h[2]]
    fig.legend(handles=h, loc="upper center", ncol=3, fontsize=5.6, bbox_to_anchor=(0.53, 1.0),
               handlelength=1.6, columnspacing=1.2, frameon=False)
    fig.text(0.065, 0.012, "Bars: mean of seeds 1 and 2 (markers). Student input 128×1632. Under each corpus: selected path for seed 1 / seed 2.\n"
             "LM = CTC prefix beam search + character n-gram LM; LM+T5 = gated T5-small corrector on the LM output.\n"
             "† Saint Gall base model trained with the original labels (raw \"pt\" letter); the Student with marker-free labels.", fontsize=5.2, color=S.MUTED)
    return S.save(fig, base, formats=("pdf", "png", "svg"), dpi=200)


def ablation(D, base):
    stages = ["raw CTC", "char LM", "T5 on char LM"]
    lab = ["CTC", "+LM", "+LM\n+T5"]
    fig, axes = plt.subplots(1, 5, figsize=(160 * S.MM, 46 * S.MM), sharey=False)
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.25, top=0.83, wspace=0.42)
    for ax, ds in zip(axes, C.CORPORA):
        for s, ls, mk in (("1", "-", "o"), ("2", "--", "s")):
            e = D[ds]["seeds"][s]
            xs = [k for k, st in enumerate(stages) if st in e]
            ys = [100 * e[stages[k]]["test"] for k in xs]
            ax.plot(xs, ys, ls=ls, marker=mk, ms=2.8, color=S.PALETTE["blue_main"], lw=0.9, mfc="white", mew=0.7)
        ax.axhline(100 * D[ds]["base_mean"], color=S.PALETTE["gray_2"], lw=0.6, ls=":")
        top = max(100 * D[ds]["base_mean"], max(100 * v["test"] for s in D[ds]["seeds"].values() for k, v in s.items() if k in stages))
        ax.set_ylim(0, top * 1.18)
        ax.set_xticks(range(3), lab, fontsize=5.2)
        ax.set_xlim(-0.35, 2.35)
        ax.set_title(C.NAMES[ds] + ("†" if ds == "saintgall" else ""), fontsize=6.6, pad=3)
        ax.tick_params(labelsize=5.6)
    axes[0].set_ylabel("Test CER (%)")
    h = [Line2D([], [], color=S.PALETTE["blue_main"], marker="o", ms=2.8, mfc="white", lw=0.9, label="HTR-JAND Student, seed 1"),
         Line2D([], [], color=S.PALETTE["blue_main"], marker="s", ms=2.8, mfc="white", lw=0.9, ls="--", label="HTR-JAND Student, seed 2"),
         Line2D([], [], color=S.PALETTE["gray_2"], lw=0.6, ls=":", label="HTR-JAND base (68×864), greedy CTC, seed mean")]
    fig.legend(handles=h, loc="upper center", ncol=3, fontsize=5.8, frameon=False, bbox_to_anchor=(0.53, 1.0))
    fig.text(0.06, 0.02, "† Saint Gall base model trained with the original labels (raw \"pt\" letter); the Student with marker-free labels.",
             fontsize=5.2, color=S.MUTED)
    return S.save(fig, base, formats=("pdf", "png", "svg"), dpi=200)


def main():
    S.use_style(6.4)
    D = collect()
    json.dump(D, open(os.path.join(C.OUT, "student_comparison_data.json"), "w"), indent=1)
    for ds in C.CORPORA:
        r = [100 * D[ds]["seeds"][s]["raw CTC"]["test"] for s in "12"]
        q = [100 * D[ds]["seeds"][s]["selected"]["test"] for s in "12"]
        print(f"{ds:10s} base {100*D[ds]['base_mean']:.2f} raw {r[0]:.2f}/{r[1]:.2f} mean {np.mean(r):.2f}  "
              f"selected {q[0]:.2f}/{q[1]:.2f} mean {np.mean(q):.2f}  paths "
              f"{D[ds]['seeds']['1']['selected']['path']} / {D[ds]['seeds']['2']['selected']['path']}")
    print(comparison(D, os.path.join(C.OUT, "student_comparison")))
    print(ablation(D, os.path.join(C.OUT, "decoding_ablation")))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Paired bootstrap over IAM test lines for the comparisons the report makes.

    python scripts/paired_tests.py [--results results] [--out results/paired_tests.json]

Reads the per-line error counts written by htrjand.evaluate; never
decodes anything. Each comparison is per seed (same lines, same seed), then
averaged over seeds.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from htrjand import metrics as Me   # noqa: E402

COMPARISONS = [
    ("KD tau=2 vs no KD, tuned", "iam_tuned_student_kd_tau2_s{s}", "iam_tuned_student_nokd_s{s}", (1, 2, 3)),
    ("KD tau=4 vs no KD, tuned", "iam_tuned_student_kd_s{s}", "iam_tuned_student_nokd_s{s}", (1, 2, 3)),
    ("KD tau=4 vs no KD, paper default", "iam_paper_student_kd_s{s}", "iam_paper_student_nokd_s{s}", (1, 2, 3)),
    ("full Teacher vs no attention, tuned", "iam_tuned_teacher_s{s}", "iam_tuned_abl_no_attention_s{s}", (1,)),
    ("full Teacher vs plain conv, tuned", "iam_tuned_teacher_s{s}", "iam_tuned_abl_plain_conv_s{s}", (1,)),
    ("full Teacher vs no SE, tuned", "iam_tuned_teacher_s{s}", "iam_tuned_abl_no_se_s{s}", (1,)),
    ("full Teacher vs plain CNN-LSTM, tuned", "iam_tuned_teacher_s{s}", "iam_tuned_abl_cnn_lstm_s{s}", (1,)),
    ("full Teacher vs no attention, paper default", "iam_paper_teacher_s{s}", "iam_abl_no_attention_s{s}", (1,)),
    ("full Teacher vs plain conv, paper default", "iam_paper_teacher_s{s}", "iam_abl_plain_conv_s{s}", (1,)),
    ("full Teacher vs historical model, paper default", "iam_paper_teacher_s{s}", "iam_historical_teacher_s{s}", (1,)),
    ("tuned Teacher vs paper-default Teacher", "iam_tuned_teacher_s{s}", "iam_paper_teacher_s{s}", (1, 2, 3)),
    ("final Teacher vs tuned Teacher", "iam_final_teacher_s{s}", "iam_tuned_teacher_s{s}", (1, 2, 3)),
    ("final KD Student (tau=2) vs final Student", "iam_final_student_kd_s{s}", "iam_final_student_nokd_s{s}", (1, 2, 3)),
    ("final Student vs tuned Student", "iam_final_student_nokd_s{s}", "iam_tuned_student_nokd_s{s}", (1, 2, 3)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    def counts(tag):
        z = np.load(os.path.join(a.results, f"{tag}_test_counts.npz"))
        return z["char_err"], z["char_len"]

    out = []
    for name, ta, tb, seeds in COMPARISONS:
        rows = []
        for s in seeds:
            try:
                ea, cl = counts(ta.format(s=s))
                eb, _ = counts(tb.format(s=s))
            except FileNotFoundError:
                continue
            obs, lo, hi, p = Me.paired_bootstrap(ea, eb, cl)
            rows.append(dict(seed=s, diff_pp=100 * obs, ci95_pp=[100 * lo, 100 * hi], p=p))
        if rows:
            out.append(dict(comparison=name, a=ta, b=tb, per_seed=rows,
                            mean_diff_pp=float(np.mean([r["diff_pp"] for r in rows]))))
            print(f"{name}: mean {out[-1]['mean_diff_pp']:+.2f} pp; " +
                  "; ".join(f"s{r['seed']} {r['diff_pp']:+.2f} [{r['ci95_pp'][0]:+.2f},{r['ci95_pp'][1]:+.2f}] p={r['p']:.3f}" for r in rows))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

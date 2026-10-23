#!/usr/bin/env python3
"""CONTROL C2 (review M2, M3, Q8.2): the existing T5 correctors on recognizers of a different
recipe (the 68x864 reconstruction models, greedy CTC), with the standard validation-selected gate.

Correctors: postproc/t5_ext_ctx1 and postproc/t5_ctx1, unchanged (trained on out-of-fold recognitions
of the TRAINING split only; validation used for their early stopping; test never seen).
Gate (detect mode, apply mode, eps threshold, theta) chosen on VALIDATION by t5post apply; test decoded
once per (recognizer, corrector).  ABLATION / CONTROL - not a selected system.

    python scripts/control_t5_second_family.py --run
    python scripts/control_t5_second_family.py --collect   # results/control_t5_second_family.json
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPORA = ["iam", "rimes", "bentham", "saintgall", "washington"]
CORRECTORS = ["t5_ext_ctx1", "t5_ctx1"]
FAMILIES = {
    "combined_paper_student_kd_s1": "reconstruction Student (literal, 32 units/dir, KD), combined training",
    "combined_paper_teacher_s1": "reconstruction Teacher (literal, 64 units/dir), combined training",
    "combined_final_teacher_s1": "reconstruction Teacher (final recipe, 128 units/dir), combined training",
    "iam_paper_student_kd_s1": "reconstruction Student (literal, 64 units/dir, KD), IAM only",
    "iam_paper_student_nokd_s1": "reconstruction Student (literal, 64 units/dir, no KD), IAM only",
    "iam_paper_teacher_s1": "reconstruction Teacher (literal, 128 units/dir), IAM only",
    "iam_tuned_abl_cnn_lstm_s1": "plain CNN-BiLSTM-CTC ablation (no gating, no SE, no attention), IAM only",
}


def jobs():
    for m in FAMILIES:
        for ds in (CORPORA if m.startswith("combined") else ["iam"]):
            tag = f"{m}_{ds}" if m.startswith("combined") else m
            vp, tp = f"results/controls/{tag}_valid_preds.tsv", f"results/{tag}_test_preds.tsv"
            for c in CORRECTORS:
                out = f"results/controls/t5ctl_{tag}_{c}.json"
                cmd = (f"python -m htrjand.t5post apply --gpu 0 --dataset {ds} --corrector postproc/{c} "
                       f"--tag t5ctl_{tag}_{c} --val-preds {vp} --test-preds {tp} --out {out}")
                yield m, ds, tag, c, vp, tp, out, cmd


def run():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    for m, ds, tag, c, vp, tp, out, cmd in jobs():
        if os.path.exists(os.path.join(ROOT, out)):
            continue
        if not (os.path.exists(os.path.join(ROOT, vp)) and os.path.exists(os.path.join(ROOT, tp))):
            print("MISSING", vp, tp, flush=True)
            continue
        print(f"[{time.strftime('%H:%M:%S')}] {cmd}", flush=True)
        subprocess.run(cmd, shell=True, cwd=ROOT, env=env)
    print("C2 RUN DONE", flush=True)


def collect():
    rows = []
    for m, ds, tag, c, vp, tp, out, cmd in jobs():
        p = os.path.join(ROOT, out)
        if not os.path.exists(p):
            continue
        x = json.load(open(p))
        b, g, u = x["test_raw"]["cer"], x["test_t5"]["cer"], x["test_t5_ungated"]["cer"]
        rows.append(dict(model=m, family=FAMILIES[m], dataset=ds, corrector=c, file=out, command=cmd,
                         valid_base=x["valid_raw"]["cer"], valid_gated=x["selected"]["cer"], gate=
                         {k: x["selected"][k] for k in ("detect", "apply", "eps_thr", "theta")},
                         test_base=b, test_gated=g, test_ungated=u, n_test_changed_normalised=x["n_test_changed"],
                         rel_gated=(g - b) / b, rel_ungated=(u - b) / b,
                         test_base_wer=x["test_raw"]["wer"], test_gated_wer=x["test_t5"]["wer"]))
    summary = {}
    for m in FAMILIES:
        for c in CORRECTORS:
            r = [x for x in rows if x["model"] == m and x["corrector"] == c]
            if r:
                summary[f"{m} | {c}"] = dict(n_corpora=len(r), mean_rel_gated=float(np.mean([x["rel_gated"] for x in r])),
                                            mean_rel_ungated=float(np.mean([x["rel_ungated"] for x in r])),
                                            gate_opened=sum(x["n_test_changed_normalised"] > 0 for x in r))
    res = dict(control="C2 T5 correctors on a second recognizer family (review M2/M3/Q8.2)",
               label="ABLATION / CONTROL - not a selected system", rows=rows, summary=summary,
               leakage="correctors trained on out-of-fold recognitions of the TRAINING split of the combined archive "
                       "(fold models runs/c128_fold{0,1}of2 trained on the complement fold, index % 2); validation pairs "
                       "used only for corrector early stopping; external text gated against verbatim test lines; "
                       "test labels never read before the final metric")
    json.dump(res, open(os.path.join(ROOT, "results", "control_t5_second_family.json"), "w"), indent=1)
    for x in rows:
        print(f"{x['model']:30s} {x['dataset']:10s} {x['corrector']:12s} valid {100*x['valid_base']:6.2f}->{100*x['valid_gated']:6.2f}  "
              f"test {100*x['test_base']:6.2f} -> gated {100*x['test_gated']:6.2f} ({100*x['rel_gated']:+6.1f}%) "
              f"ungated {100*x['test_ungated']:6.2f} ({100*x['rel_ungated']:+6.1f}%)  {x['gate']}")
    for k, v in summary.items():
        print(k, json.dumps(v))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--collect", action="store_true")
    a = ap.parse_args()
    run() if a.run else collect()

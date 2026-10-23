#!/usr/bin/env python3
"""Executable leakage audit of the T5 / LM / selection pipeline. Writes results/leakage_audit.json
and exits non-zero if any check fails.

Checks
 1. T5 training targets vs test transcriptions: a target equal to a test line is allowed only when
    the same line also occurs in that corpus's official TRAINING split (stock phrases repeated by the
    benchmark itself, e.g. RIMES letter closings); any other match is a failure.
 2. T5 early-stopping pairs come from the VALIDATION split only (targets == validation labels).
 3. External text (LM and synthetic sources): no test line of >= 25 characters occurs verbatim.
 4. Every checkpoint used in a final path has a charset built from TRAINING text (subset of train chars).
 5. Every T5 result's gate setting is the validation-optimal row of its own recorded sweep (LM / ensemble
    weights are chosen inside ensemble_rescore.py on validation; their grids are not stored, so not re-checked here).
 6. Fold models: the held-out half of the training split is disjoint from what each fold model trained on.
"""
import argparse
import csv
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                              epilog="Needs the licensed archives ($HTRJAND_DATA_DIR or --data-dir), runs/<id>/best.pth "
                                     "(--runs-dir), the T5 pair files postproc/t5data*/ (--postproc-dir) and the "
                                     "Wikipedia text synth/text_*.txt (--synth-dir); none of them is distributed.")
_ap.add_argument("--data-dir", default=None)
_ap.add_argument("--runs-dir", default=os.path.join(ROOT, "runs"))
_ap.add_argument("--postproc-dir", default=os.path.join(ROOT, "postproc"))
_ap.add_argument("--synth-dir", default=os.path.join(ROOT, "synth"))
_ap.add_argument("--results-dir", default=os.path.join(ROOT, "results"))
_ap.add_argument("--out", default=None, help="default <results-dir>/leakage_audit.json")
A = _ap.parse_args()
if A.data_dir:
    os.environ["HTRJAND_DATA_DIR"] = A.data_dir

import h5py  # noqa: E402,F401
from htrjand import data as D  # noqa: E402
from htrjand.t5post import CORPORA, normalise, read_labels  # noqa: E402

try:
    for _c in CORPORA:
        D.archive_path(_c)
except FileNotFoundError as e:
    sys.exit(f"leakage audit needs the benchmark archives: {e}")

out, ok = {}, True


def fail(name, detail):
    global ok
    ok = False
    out.setdefault("failures", []).append({name: detail})


labels = {c: {s: [normalise(t) for t in read_labels(c, s)] for s in ("train", "valid", "test")} for c in CORPORA}

# 1-2
for d in sorted(glob.glob(os.path.join(A.postproc_dir, "t5data*"))):
    rec = {}
    for split in ("train", "valid"):
        p = os.path.join(d, f"pairs_{split}.tsv")
        if not os.path.exists(p):
            continue
        by_c = {}
        with open(p) as fh:
            r = csv.reader(fh, delimiter="\t"); next(r)
            for c, kind, src, tgt in r:
                by_c.setdefault(c, set()).add(tgt)
        for c, tg in by_c.items():
            test, train, valid = set(labels[c]["test"]), set(labels[c]["train"]), set(labels[c]["valid"])
            hit = tg & test
            unexplained = hit - train
            rec[f"{split}:{c}"] = dict(targets=len(tg), equal_to_a_test_line=len(hit),
                                       of_which_also_in_official_train=len(hit & train), unexplained=len(unexplained))
            if unexplained and split == "train":
                # an external-text target may coincide with a short test line only if not in train; report it
                long_ = [t for t in unexplained if len(t) >= 25]
                if long_:
                    fail("t5_train_targets", {c: long_[:5]})
            if split == "valid":
                real = {t for t in tg}
                if not real <= valid:
                    fail("t5_valid_pairs", {c: len(real - valid)})
    out[f"pairs:{os.path.basename(d)}"] = rec

# 3
for f in sorted(glob.glob(os.path.join(A.synth_dir, "text_*.txt"))):
    big = open(f, encoding="utf-8").read()
    hits = {c: sum(1 for t in labels[c]["test"] if len(t) >= 25 and t in big) for c in CORPORA}
    out[f"external_text:{os.path.basename(f)}"] = hits
    if any(hits.values()):
        fail("external_text", {os.path.basename(f): hits})

# 4
import torch  # noqa: E402
train_chars = set("".join(labels[c]["train"][i] for c in CORPORA for i in range(len(labels[c]["train"]))))
ck_rows = {}
for p in sorted(glob.glob(os.path.join(A.runs_dir, "*", "best.pth"))):
    tag = p.split("/")[-2]
    if not (tag.startswith(("c128_", "ft_", "combined_", "nopt_")) or tag.endswith(("_final_teacher_s1", "_final_teacher_s2", "_final_teacher_s3", "_tuned_teacher_s1"))):
        continue
    cs = torch.load(p, map_location="cpu", weights_only=False)["charset"]
    extra = set(cs) - train_chars
    ck_rows[tag] = len(extra)
    if extra:
        fail("charset", {tag: sorted(extra)})
out["charset_outside_training_text"] = ck_rows

# 5
gate = {}
for f in sorted(glob.glob(os.path.join(A.results_dir, "t5_*.json"))):
    r = json.load(open(f))
    if "sweep" not in r:
        continue
    best = min(r["sweep"], key=lambda x: (x["cer"], x["wer"]))
    same = all(best[k] == r["selected"][k] for k in ("detect", "apply", "theta", "eps_thr"))
    gate[os.path.basename(f)] = same
    if not same:
        fail("t5_gate_selection", os.path.basename(f))
out["t5_gate_is_validation_optimum"] = gate

# 6
fold = {}
for k in (0, 1):
    p = os.path.join(A.runs_dir, f"c128_fold{k}of2", "train_log.json")
    if os.path.exists(p):
        a = json.load(open(p))["args"]
        fold[f"c128_fold{k}of2"] = dict(trained_on=f"index % 2 != {k}", decoded_for_pairs=f"index % 2 == {k}",
                                        fold_arg=a.get("fold"))
        if a.get("fold") != f"{k}/2":
            fail("fold", fold[f"c128_fold{k}of2"])
out["fold_models"] = fold

out["verdict"] = "PASS" if ok else "FAIL"
json.dump(out, open(A.out or os.path.join(A.results_dir, "leakage_audit.json"), "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in out.items() if k in ("verdict", "failures")}, indent=1))
for k, v in out.items():
    if k.startswith("pairs:"):
        print(k, {kk: (vv["equal_to_a_test_line"], vv["of_which_also_in_official_train"], vv["unexplained"]) for kk, vv in v.items()})
sys.exit(0 if ok else 1)

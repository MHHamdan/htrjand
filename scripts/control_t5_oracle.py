#!/usr/bin/env python3
"""CONTROL C3 (review M3, M4, Q8.4, Q8.6): oracle line gate, changed-line accounting and paired
bootstrap for the Student T5 paths (seed 1 ids from results/final_selection.json, seed 2 from the
seed-2 chain).  No decoding: everything is recomputed from the stored T5 candidates
(results/t5_*_t5cache.json) and prediction files.

  oracle   per line min(edit distance of the base line, of the T5 line-mode candidate), i.e. the
           gate that opens exactly where T5 helps.  DIAGNOSTIC UPPER BOUND: it reads the labels of
           the split it is evaluated on (test labels for the test number).
  changed  lines whose final output differs from the base, counted raw (string !=) and after the
           t5post whitespace/typography normalisation (what t5post stores as n_test_changed).
  bootstrap  paired, 10,000 line resamples, seed 20261017, of CER(T5 path) - CER(base) on test;
           exact two-sided sign test on the changed lines that improved vs worsened.
ABLATION / CONTROL - not a selected system.   Writes results/control_t5_oracle.json.
"""
import csv
import json
import os
import sys

import editdistance
import numpy as np
from scipy.stats import binomtest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from htrjand import t5post as T  # noqa: E402

CORPORA = ["iam", "rimes", "bentham", "saintgall", "washington"]
SEED = 20261017
NB = 10000


def j(p):
    return json.load(open(os.path.join(ROOT, p)))


def tsv(p):
    with open(p) as fh:
        r = csv.reader(fh, delimiter="\t")
        hd = next(r)
        return [dict(zip(hd, row)) for row in r]


def paths():
    fs = j("results/final_selection.json")
    for ds in CORPORA:
        st = {k: v for k, v in fs[ds].items() if k.startswith("Student")}
        sel = min(st, key=lambda k: st[k]["v"][0])
        for stage in ("T5 on raw CTC", "T5 on char LM"):
            k = f"Student | {stage}"
            yield ds, 1, stage, st[k]["id"], sel == k
    s2 = j("results/seed2_student.json")
    for ds in CORPORA:
        pre = "saintgall_nopt" if ds == "saintgall" else ds
        selp = s2[ds]["selected_path"]
        for stage, k in (("T5 on raw CTC", "ctc"), ("T5 on char LM", "lm")):
            yield ds, 2, stage, f"t5_{pre}_s2student_{k}_t5_ext_ctx1", selp.startswith(stage)


def ed(a, b):
    return np.array([editdistance.eval(x, y) for x, y in zip(a, b)], dtype=float)


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    fmt_cache = {}
    for ds, seed, stage, tid, selected in paths():
        x = j(f"results/{tid}.json")
        dset = x["dataset"]
        if dset not in fmt_cache:
            fmt_cache[dset] = T.Format([T.normalise(t) for t in T.read_labels(dset, "train")])
        fmt = fmt_cache[dset]
        cache = j(f"results/{tid}_t5cache.json")
        r = dict(corpus=ds, seed=seed, path=stage, id=tid, validation_selected_path=selected,
                 corrector=os.path.basename(x["corrector"]), gate=x["selected"])
        for split, pf in (("valid", x["val_preds"]), ("test", x["test_preds"])):
            refs, hyps, _ = T.read_preds(pf)
            key = [k for k in cache if k.startswith(split + ":") and k.endswith(f":{len(hyps)}")]
            assert len(key) == 1, (tid, split, list(cache))
            cands, confs = cache[key[0]]
            line = [fmt.restore(fmt.restore(c)) for c in cands]
            L = np.array([len(t) for t in refs], dtype=float)
            eb, ec = ed(refs, hyps), ed(refs, line)
            r[split] = dict(base_cer=eb.sum() / L.sum(), ungated_line_cer=ec.sum() / L.sum(),
                            oracle_cer=np.minimum(eb, ec).sum() / L.sum(),
                            oracle_lines_switched=int((ec < eb).sum()), n_lines=len(refs))
        r["valid"]["gated_cer"] = x["selected"]["cer"]
        # final test outputs of the validation-selected gate
        rows_t = tsv(os.path.join(ROOT, f"results/{tid}_test_preds.tsv"))
        refs = [d["ref"] for d in rows_t]; raw = [d["raw"] for d in rows_t]; fin = [d["final"] for d in rows_t]
        L = np.array([len(t) for t in refs], dtype=float)
        eb, ef = ed(refs, raw), ed(refs, fin)
        r["test"]["gated_cer"] = ef.sum() / L.sum()
        assert abs(r["test"]["gated_cer"] - x["test_t5"]["cer"]) < 1e-12
        ch_raw = np.array([a != b for a, b in zip(raw, fin)])
        ch_norm = np.array([T.normalise(a) != T.normalise(b) for a, b in zip(raw, fin)])
        imp, wor = int(((ef < eb) & ch_raw).sum()), int(((ef > eb) & ch_raw).sum())
        r["test_changed"] = dict(raw_string=int(ch_raw.sum()), normalised=int(ch_norm.sum()),
                                 stored_n_test_changed=x["n_test_changed"],
                                 whitespace_or_typography_only=int((ch_raw & ~ch_norm).sum()),
                                 improved=imp, worsened=wor, equal_cer=int(ch_raw.sum()) - imp - wor)
        n = len(refs)
        idx = rng.integers(0, n, size=(NB, n))
        d = (ef[idx].sum(1) - eb[idx].sum(1)) / L[idx].sum(1)
        lo, hi = np.percentile(d, [2.5, 97.5])
        r["paired_bootstrap"] = dict(delta_cer_pp=100 * (ef.sum() - eb.sum()) / L.sum(), ci95_pp=[100 * lo, 100 * hi],
                                     p_two_sided=float(min(1.0, 2 * min((d >= 0).mean(), (d <= 0).mean()))),
                                     n_boot=NB, seed=SEED, note="one rng stream shared by all rows, in row order")
        r["sign_test_p"] = float(binomtest(imp, imp + wor, 0.5).pvalue) if imp + wor else None
        rows.append(r)
        print(f"{ds:10s} s{seed} {stage:14s} sel={int(selected)} valid base {100*r['valid']['base_cer']:.2f} gated "
              f"{100*r['valid']['gated_cer']:.2f} oracle {100*r['valid']['oracle_cer']:.2f} | test base {100*r['test']['base_cer']:.2f} "
              f"gated {100*r['test']['gated_cer']:.2f} ungated {100*r['test']['ungated_line_cer']:.2f} oracle "
              f"{100*r['test']['oracle_cer']:.2f} | changed raw {r['test_changed']['raw_string']} norm {r['test_changed']['normalised']} "
              f"+{imp}/-{wor} | d {r['paired_bootstrap']['delta_cer_pp']:+.3f} [{100*lo:+.3f},{100*hi:+.3f}] sign p {r['sign_test_p']}",
              flush=True)
    json.dump(dict(control="C3 oracle gate, changed lines, paired bootstrap (review M3/M4)",
                   label="ABLATION / CONTROL; oracle rows are DIAGNOSTIC UPPER BOUNDS that read the evaluated split's labels",
                   rows=rows), open(os.path.join(ROOT, "results", "control_t5_oracle.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()

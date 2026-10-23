#!/usr/bin/env python3
"""CONTROL C1 (review M7): LM-source ablation for the validation-selected single Students.

For the seed-1 and seed-2 Students of every corpus, decode with
  (a) an LM fitted on the corpus TRAINING transcriptions only (no --lm-text),
  (b) an LM fitted on the Wikipedia text only (--lm-text ... --lm-no-train-text),
and compare with the existing combined LM (training transcriptions + Wikipedia) and greedy.
Orders / max-lines / alpha grid / beta grid are those of scripts/seed2_student_chain.py; alpha and beta
are chosen on validation inside ensemble_rescore, test is decoded once per configuration.
This is an ABLATION, not a new selected system.

    python scripts/control_lm_source.py --arm train      # runs (a)
    python scripts/control_lm_source.py --arm wiki       # runs (b)
    python scripts/control_lm_source.py --collect        # writes results/control_lm_source.json
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "controls")
LM = dict(iam=("synth/text_en.txt", 400000, 10), rimes=("synth/text_fr.txt", 200000, 9),
          bentham=("synth/text_en.txt", 200000, 9), saintgall=("synth/text_la.txt", 200000, 9),
          washington=("synth/text_en.txt", 200000, 9))
BETAS = "0.0,0.5,1.0,1.5,2.0,3.0,4.0,5.0"
# (corpus, dataset, seed, model, existing combined-LM tag)
MODELS = [
    ("iam", "iam", 1, "ft_iam_c128_student_kd_s1", "iam_ft_iam_c128_student_kd_s1_wiki"),
    ("iam", "iam", 2, "ft_iam_c128_student_s2", "iam_ft_iam_c128_student_s2_wiki"),
    ("rimes", "rimes", 1, "ft_rimes_c128_student_kd_s1", "rimes_ft_rimes_c128_student_kd_s1_wiki"),
    ("rimes", "rimes", 2, "ft_rimes_c128_student_kd_s2", "rimes_ft_rimes_c128_student_kd_s2_wiki"),
    ("bentham", "bentham", 1, "ft_bentham_c128_student_kd_s1", "bentham_ft_bentham_c128_student_kd_s1_wiki"),
    ("bentham", "bentham", 2, "ft_bentham_c128_student_kd_s2", "bentham_ft_bentham_c128_student_kd_s2_wiki"),
    ("saintgall", "saintgall_nopt", 1, "nopt_sg_student_s1", "saintgall_nopt_student_wiki"),
    ("saintgall", "saintgall_nopt", 2, "nopt_sg_student_s2", "saintgall_nopt_student_s2_wiki"),
    ("washington", "washington", 1, "ft_washington_c128_student_s1", "washington_ft_washington_c128_student_s1_wiki"),
    ("washington", "washington", 2, "ft_washington_c128_student_s2", "washington_ft_washington_c128_student_s2_wiki"),
]


def cmd(arm, corpus, ds, model):
    txt, nmax, order = LM[corpus]
    c = (f"python -m htrjand.ensemble_rescore --gpu 0 --dataset {ds} --tag ctl_lm{arm}_{ds}_{model} "
         f"--members runs/{model}/best.pth --order {order} --betas {BETAS} --workers 12 --out-dir results/controls")
    if arm == "wiki":
        c += f" --lm-text {txt} --lm-max-lines {nmax} --lm-no-train-text"
    return c


def j(p):
    return json.load(open(os.path.join(ROOT, p)))


def collect():
    rows = []
    for corpus, ds, seed, model, comb in MODELS:
        r = dict(corpus=corpus, dataset=ds, seed=seed, model=model)
        r["greedy"] = {sp: {k: j(f"results/{model}_{ds}_{sp}.json")[k] for k in ("cer", "wer")} for sp in ("valid", "test")}
        for arm, p in (("train_only", f"results/controls/ens_ctl_lmtrain_{ds}_{model}.json"),
                       ("wiki_only", f"results/controls/ens_ctl_lmwiki_{ds}_{model}.json"),
                       ("combined", f"results/ens_{comb}.json")):
            if not os.path.exists(os.path.join(ROOT, p)):
                r[arm] = None
                continue
            e = j(p)
            r[arm] = dict(file=p, valid=dict(cer=e["chosen"]["cer"], wer=e["chosen"]["wer"]),
                          test=dict(cer=e["test"]["cer"], wer=e["test"]["wer"]), cer_ci95=e.get("cer_ci95"),
                          alpha=e["chosen"]["alpha"], beta=e["chosen"]["beta"], lm=e["lm"])
        rows.append(r)
    sizes = os.path.join(ROOT, "results", "control_lm_sizes.json")
    out = dict(control="C1 LM-source ablation (review M7)", label="ABLATION / CONTROL - not a selected system",
               protocol="alpha in default grid, beta in " + BETAS + ", chosen on validation; test decoded once per configuration; "
                        "gen-alpha 0.2 / gen-beta 0.5 / beam 16 / nbest 8 (ensemble_rescore defaults)",
               commands={arm: [cmd(arm, c, d, m) for c, d, _, m, _ in MODELS] for arm in ("train", "wiki")},
               rows=rows, lm_sizes=json.load(open(sizes)) if os.path.exists(sizes) else None)
    json.dump(out, open(os.path.join(ROOT, "results", "control_lm_source.json"), "w"), indent=1)
    print(f"{'corpus':10s} s  {'greedy':>13s} {'train-only':>13s} {'wiki-only':>13s} {'combined':>13s}   (valid/test CER %)")
    for r in rows:
        f = lambda x: "-" if x is None else f"{100*x['valid']['cer']:5.2f}/{100*x['test']['cer']:5.2f}"
        print(f"{r['corpus']:10s} {r['seed']}  {f(r['greedy']):>13s} {f(r['train_only']):>13s} {f(r['wiki_only']):>13s} {f(r['combined']):>13s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("train", "wiki"))
    ap.add_argument("--collect", action="store_true")
    a = ap.parse_args()
    if a.collect:
        return collect()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    for corpus, ds, seed, model, _ in MODELS:
        if os.path.exists(os.path.join(OUT, f"ens_ctl_lm{a.arm}_{ds}_{model}.json")):
            continue
        c = cmd(a.arm, corpus, ds, model)
        print(f"[{time.strftime('%H:%M:%S')}] {c}", flush=True)
        rc = subprocess.run(c, shell=True, cwd=ROOT, env=env).returncode
        if rc:
            print(f"FAILED rc={rc}", flush=True)
    print(f"ARM {a.arm} DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())

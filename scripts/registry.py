#!/usr/bin/env python3
"""Experiment registry: one row per training run and per decoding / post-processing
experiment, generated from the artefacts (runs/*/train_log.json, results/*.json), never
typed by hand.

    python scripts/registry.py      # writes docs/EXPERIMENT_REGISTRY.md and results/registry.jsonl
    python scripts/registry.py --runs-dir /path/to/runs --results-dir results --out-dir /tmp/reg

The training logs (runs/<id>/train_log.json) are not distributed with the repository; point
--runs-dir at a directory that holds them (the script stops with a message when it is absent).

Fields: id, kind, git commit, configuration, dataset, seed, training source, checkpoint
selection rule, decoder, post-processing, validation CER/WER, test CER/WER (computed once,
after selection), runtime, parameters, status.
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
RESULTS = os.path.join(ROOT, "results")
KEYS = ("preset", "dropout", "dropout_cnn", "dropout_rnn", "weight_decay", "lr", "sched", "epochs",
        "batch_size", "aug_set", "ema", "kd", "tau", "synthetic", "synth_only", "init_ckpt", "fold", "set")


def short_cfg(args):
    out = []
    for k in KEYS:
        v = args.get(k)
        if v in (None, False, [], "", 0.0) and k not in ("dropout",):
            continue
        if isinstance(v, str) and "/" in v and k != "fold":
            v = os.path.basename(v.rstrip("/")) if not v.endswith("best.pth") else v.split("/")[-2]
        out.append(f"{k}={v}")
    return " ".join(out)


def pct(x):
    return "" if x is None else f"{100 * x:.2f}"


def runs():
    rows = []
    for log in sorted(glob.glob(os.path.join(RUNS, "*", "train_log.json"))):
        d = json.load(open(log))
        tag, a = d["tag"], d.get("args", {})
        env = d.get("environment", {})
        hist = d.get("history", [])
        best = min(hist, key=lambda e: e["val_cer"]) if hist else {}
        tests = {}
        for f in glob.glob(os.path.join(RESULTS, f"{tag}*_test.json")):
            r = json.load(open(f))
            if r.get("tag", "").startswith(tag) and "cer" in r:
                tests[r.get("dataset", "?") if r["tag"] != tag else "own"] = (r["cer"], r["wer"])
        ds = os.path.basename(str(a.get("dataset", ""))).replace(".hdf5", "")
        status = "finished" if d.get("finished") else ("stopped/superseded" if tag in STOPPED else "running or killed")
        rows.append(dict(
            id=tag, kind="training", commit=(env.get("git_commit") or "")[:7] + ("+dirty" if env.get("git_dirty") else ""),
            config=short_cfg(a), dataset=ds, seed=a.get("seed"),
            training_source=f"{ds} train split" + (f", fold {a['fold']} complement" if a.get("fold") else "")
            + (" + synthetic (synthetic-ratio schedule)" if a.get("synthetic") and not a.get("synth_only") else "")
            + (" synthetic only" if a.get("synth_only") else ""),
            selection="best validation CER (EMA or raw weights, whichever lower)", decoder="greedy CTC",
            postprocessing="none", valid_cer=best.get("val_cer"), valid_wer=best.get("val_wer"),
            test={k: v for k, v in sorted(tests.items())}, runtime_h=round(d.get("elapsed_s", hist[-1]["elapsed_s"] if hist else 0) / 3600, 2),
            params_deployed=d.get("params_deployed"), status=status))
    return rows


STOPPED = {"c128_syn_teacher_s2_first_attempt"}


def decodes():
    rows = []
    for f in sorted(glob.glob(os.path.join(RESULTS, "ens_*.json"))):
        r = json.load(open(f))
        if "test" not in r:
            continue
        ch = r.get("chosen", {})
        rows.append(dict(
            id=os.path.basename(f)[:-5], kind="decoding", commit="", dataset=r["dataset"],
            config=f"{len(r['members'])} member(s): " + ", ".join(m.split('/')[-2] for m in r["members"]),
            decoder=f"CTC prefix beam 16 + char {r['lm'].get('order')}-gram LM ({r['lm'].get('source')} train"
                    + (" + " + ", ".join(os.path.basename(x) for x in r['lm'].get('extra', [])) if r['lm'].get('extra') else "") + ")",
            selection=f"members/alpha/beta on validation (alpha {ch.get('alpha')}, beta {ch.get('beta')})",
            postprocessing="none", valid_cer=ch.get("cer"), valid_wer=ch.get("wer"),
            test={"own": (r["test"]["cer"], r["test"]["wer"])}, status="finished"))
    for f in sorted(glob.glob(os.path.join(RESULTS, "t5_*.json"))):
        r = json.load(open(f))
        if "test_t5" not in r:
            continue
        s = r["selected"]
        rows.append(dict(
            id=os.path.basename(f)[:-5], kind="post-processing", commit="", dataset=r["dataset"],
            config=f"base {r['base']}; corrector {os.path.basename(r['corrector'])}; context {r['context_window']}",
            decoder="(base path)", postprocessing=f"T5 post-processing: detect {s['detect']}, apply {s['apply']}, theta {s['theta']}",
            selection="detect/apply/theta on validation", valid_cer=s["cer"], valid_wer=s["wer"],
            test={"own": (r["test_t5"]["cer"], r["test_t5"]["wer"])}, status="finished"))
    return rows


def main(argv=None):
    global RUNS, RESULTS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default=RUNS, help="directory of runs/<id>/train_log.json (default <repo>/runs)")
    ap.add_argument("--results-dir", default=RESULTS)
    ap.add_argument("--out-dir", default=None, help="write EXPERIMENT_REGISTRY.md and registry.jsonl here "
                                                   "(default: docs/ and results/ of the repository)")
    a = ap.parse_args(argv)
    RUNS, RESULTS = a.runs_dir, a.results_dir
    if not os.path.isdir(RUNS) or not glob.glob(os.path.join(RUNS, "*", "train_log.json")):
        sys.exit(f"{RUNS}: no runs/<id>/train_log.json found. The training logs are not part of the public "
                 "repository; pass --runs-dir pointing at a directory that holds them.")
    jsonl = os.path.join(a.out_dir, "registry.jsonl") if a.out_dir else os.path.join(RESULTS, "registry.jsonl")
    md = os.path.join(a.out_dir, "EXPERIMENT_REGISTRY.md") if a.out_dir else os.path.join(ROOT, "docs", "EXPERIMENT_REGISTRY.md")
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
    allrows = runs() + decodes()
    with open(jsonl, "w") as fh:
        for r in allrows:
            fh.write(json.dumps(r) + "\n")
    lines = ["# Experiment registry", "",
             "Generated by `scripts/registry.py` from `runs/*/train_log.json` and `results/*.json`. "
             "Validation numbers drove every selection; test numbers were computed once, after selection. "
             "CER / WER in %; `test` lists the test CER / WER per decoded corpus ('own' = the run's own test split).", "",
             "| id | kind | commit | dataset | seed | configuration | training source | selection | decoder | post-processing | valid CER / WER | test CER / WER | runtime h | params | status |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in allrows:
        t = "; ".join(f"{k} {pct(v[0])}/{pct(v[1])}" for k, v in r["test"].items())
        lines.append(f"| {r['id']} | {r['kind']} | {r.get('commit','')} | {r.get('dataset','')} | {r.get('seed','') or ''} | "
                     f"{r.get('config','')} | {r.get('training_source','')} | {r['selection']} | {r['decoder']} | "
                     f"{r['postprocessing']} | {pct(r['valid_cer'])} / {pct(r['valid_wer'])} | {t} | "
                     f"{r.get('runtime_h','')} | {r.get('params_deployed','') or ''} | {r['status']} |")
    lines += ["", "## Events the artefacts do not show", "",
              "- `c128_syn_teacher_s2`: a first attempt was stopped after about one hour (2026-10-14) to give the "
              "128 x 1632 Student its GPU slot; it was rerun from scratch under the same tag, so its log holds only the rerun.",
              "- `+dirty` commits: the run was launched before the code change it used was committed; the change is in the "
              "next commit touching `htrjand/` (synthetic options in 6878706, input-size options in 8d90bf8, "
              "Student pooling override in 6a1a02e).",
              "- Withdrawn selection: `c128_student_s1` + LM has the lower IAM test CER (4.47) but the higher validation CER "
              "(3.62 vs 3.45), so the validation-selected IAM Student is `c128_student_kd_s1` (test 4.68)."]
    open(md, "w").write("\n".join(lines) + "\n")
    print(f"{len(allrows)} rows -> {md}, {jsonl}")


if __name__ == "__main__":
    main()

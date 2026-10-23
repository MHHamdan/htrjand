#!/usr/bin/env python3
"""Per-corpus test results of the multi-corpus (combined) models. Reads
results/combined_*_test.json and results/c128_*_test.json; prints a Markdown table."""
import glob, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPORA = ["iam", "rimes", "bentham", "saintgall", "washington", "combined"]
rows = {}
for f in sorted(glob.glob(os.path.join(ROOT, "results", "combined_*_test.json")) + glob.glob(os.path.join(ROOT, "results", "c128_*_test.json"))):
    r = json.load(open(f))
    rows.setdefault(r["tag"].replace("_" + r["dataset"], "") if r["dataset"] != "combined" else r["tag"], {})[r["dataset"]] = r
print("| model | " + " | ".join(CORPORA) + " |")
print("|---|" + "---:|" * len(CORPORA))
for tag, by in rows.items():
    print(f"| {tag} CER / WER | " + " | ".join(
        f"{100*by[c]['cer']:.2f} / {100*by[c]['wer']:.2f}" if c in by else "—" for c in CORPORA) + " |")

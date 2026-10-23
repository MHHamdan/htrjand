"""Shared helpers for the HTR-JAND paper figures under docs/figures/.

Paths of every artifact the figures read, the character alignment used to mark
errors, the archive/raw readers and the print-scale style (figkit from
figures4papers; set FIGKIT_ROOT if it is not at ~/figures4papers).
"""
import csv
import json
import os
import sys
import unicodedata

import h5py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
DATA = os.environ.get("HTRJAND_DATA_DIR", os.path.join(ROOT, "data"))
RAW = os.environ.get("HTRJAND_RAW_DIR", os.path.join(ROOT, "raw"))
OUT = os.path.join(ROOT, "docs", "figures")
FIGKIT_ROOT = os.environ.get("FIGKIT_ROOT", os.path.expanduser("~/figures4papers"))
sys.path.insert(0, ROOT)

CORPORA = ("iam", "rimes", "bentham", "saintgall", "washington")
NAMES = dict(iam="IAM", rimes="RIMES", bentham="Bentham", saintgall="Saint Gall", washington="Washington")
# archive actually decoded by the selected Student of each corpus (Saint Gall: marker-free labels;
# the test labels are identical to the official ones, only training labels lose the "pt" letter)
ARCHIVE = dict(iam="iam", rimes="rimes", bentham="bentham", saintgall="saintgall_nopt", washington="washington")

# Validation-selected seed-1 Student artifacts (docs/FINAL_RESULTS_T5.md, results/final_selection.json)
SEED1 = {
    "iam": dict(raw="ft_iam_c128_student_kd_s1_iam", lm="ens_iam_ft_iam_c128_student_kd_s1_wiki"),
    "rimes": dict(raw="ft_rimes_c128_student_kd_s1_rimes", lm="ens_rimes_ft_rimes_c128_student_kd_s1_wiki"),
    "bentham": dict(raw="ft_bentham_c128_student_kd_s1_bentham", lm="ens_bentham_ft_bentham_c128_student_kd_s1_wiki"),
    "saintgall": dict(raw="nopt_sg_student_s1_saintgall_nopt", lm="ens_saintgall_nopt_student_wiki"),
    "washington": dict(raw="ft_washington_c128_student_s1_washington", lm="ens_washington_ft_washington_c128_student_s1_wiki"),
}
SEED1_RUN = dict(iam="ft_iam_c128_student_kd_s1", rimes="ft_rimes_c128_student_kd_s1",
                 bentham="ft_bentham_c128_student_kd_s1", saintgall="nopt_sg_student_s1",
                 washington="ft_washington_c128_student_s1")
# HTR-JAND base Student (68x864): the tuned five-corpus Student (preset paper_student, KD from the five-corpus
# Teacher, tau 2), greedy CTC, decoded on each official test set. Saint Gall: trained with the original labels
# (raw "pt" letter); its test references are the official ones.
BASE = {ds: f"combined_final_student_kd_s1_{ds}" for ds in CORPORA}
BASE_SEEDS = {s: {ds: f"combined_final_student_kd_s{s}_{ds}" for ds in CORPORA} for s in ("1", "2")}
BASE_LABEL = "HTR-JAND base (68×864)"

STAGES = ("raw CTC", "T5 on raw CTC", "char LM", "T5 on char LM")


def J(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def read_tsv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def test_gt(ds):
    with h5py.File(os.path.join(DATA, f"{ARCHIVE[ds]}_128x1632.hdf5")) as f:
        return [g.decode() if isinstance(g, bytes) else g for g in f["test"]["gt"][:]]


def train_gt(ds):
    with h5py.File(os.path.join(DATA, f"{ARCHIVE[ds]}_128x1632.hdf5")) as f:
        return [g.decode() if isinstance(g, bytes) else g for g in f["train"]["gt"][:]]


def test_image(ds, idx, size="128x1632"):
    with h5py.File(os.path.join(DATA, f"{ARCHIVE[ds]}_{size}.hdf5")) as f:
        return f["test"]["dt"][idx]


def selected_ids(ds):
    """Seed-1 validation-selected ids per stage from results/final_selection.json."""
    sel = J("final_selection.json")[ds]
    return {st: sel[f"Student | {st}"] for st in STAGES if f"Student | {st}" in sel}


def pick_path(stage_valid):
    """Validation-selected decoding path: lowest validation CER; an exact tie goes to the simpler
    path (order raw CTC < char LM < T5 on raw CTC < T5 on char LM)."""
    order = ["raw CTC", "char LM", "T5 on raw CTC", "T5 on char LM"]
    return min((s for s in order if s in stage_valid), key=lambda s: (round(stage_valid[s], 12), order.index(s)))


# ----------------------------------------------------------------------------- alignment
def align(ref, hyp):
    """Levenshtein alignment. Returns ops [(op, ref_char, hyp_char)] with op in
    '=' (match), 'S' (substitution), 'D' (ref char deleted), 'I' (hyp char inserted).
    Unit costs; ties in the backtrace prefer match/substitution, then deletion, then insertion."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]), d[i - 1, j] + 1, d[i, j - 1] + 1)
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("=" if ref[i - 1] == hyp[j - 1] else "S", ref[i - 1], hyp[j - 1])); i -= 1; j -= 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            ops.append(("D", ref[i - 1], "")); i -= 1
        else:
            ops.append(("I", "", hyp[j - 1])); j -= 1
    return ops[::-1]


def is_space_or_punct(c):
    return c == " " or unicodedata.category(c).startswith("P") or unicodedata.category(c).startswith("S")


def figkit():
    sys.path.insert(0, FIGKIT_ROOT)
    import figkit as fk          # noqa: E402
    from figkit import style     # noqa: E402
    return fk, style

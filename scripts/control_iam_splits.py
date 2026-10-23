#!/usr/bin/env python3
"""CONTROL C5 (review M1, M9): IAM split facts, by line ID.

Reads the official FKI large-writer-independent lists, the Aachen (RWTH / Laia) lists and
ascii/lines.txt from the raw corpus, and checks that each archive split is exactly the
list(s) in order (transcription by transcription, as prepare_data._iam builds them).
Reports the size of every list, the overlaps between FKI and Aachen partitions by line ID,
and the segmentation-status mix.  Writes results/control_iam_splits.json.
"""
import json
import os
import sys

import h5py

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.environ.get("HTRJAND_RAW_DIR", os.path.join(ROOT, "raw")) + "/iam"
DATA = os.environ.get("HTRJAND_DATA_DIR", os.path.join(ROOT, "data"))


def rd(p):
    return open(p).read().split()


def main():
    fki = os.path.join(RAW, "largeWriterIndependentTextLineRecognitionTask")
    L = {"fki_train": rd(f"{fki}/trainset.txt"), "fki_valid1": rd(f"{fki}/validationset1.txt"),
         "fki_valid2": rd(f"{fki}/validationset2.txt"), "fki_test": rd(f"{fki}/testset.txt"),
         "aachen_train": rd(f"{RAW}/aachen/trainset.txt"), "aachen_valid": rd(f"{RAW}/aachen/validationset.txt"),
         "aachen_test": rd(f"{RAW}/aachen/testset.txt")}
    gt, status = {}, {}
    for line in open(os.path.join(RAW, "ascii", "lines.txt")).read().splitlines():
        if line and line[0] != "#":
            s = line.split()
            gt[s[0]] = " ".join(s[8:]).replace("|", " ")
            status[s[0]] = s[1]
    S = {k: set(v) for k, v in L.items()}
    S["fki_valid"] = S["fki_valid1"] | S["fki_valid2"]
    allfki = S["fki_train"] | S["fki_valid"] | S["fki_test"]
    allaa = S["aachen_train"] | S["aachen_valid"] | S["aachen_test"]
    res = dict(control="C5 IAM split facts (review M1/M9)", lists={k: dict(n=len(v), unique=len(set(v)),
               in_lines_txt=sum(x in gt for x in v), seg_err=sum(status.get(x) == "err" for x in v),
               forms=len({"-".join(x.split("-")[:2]) for x in v}),
               writers_note="writer IDs need forms.txt") for k, v in L.items()},
               lines_txt_total=len(gt), lines_txt_err=sum(v == "err" for v in status.values()),
               fki_union=len(allfki), aachen_union=len(allaa), union_both=len(allfki | allaa),
               lines_in_neither=len(set(gt) - allfki - allaa))
    ov = {}
    for a in ("fki_train", "fki_valid1", "fki_valid2", "fki_test"):
        for b in ("aachen_train", "aachen_valid", "aachen_test"):
            ov[f"{a} & {b}"] = len(S[a] & S[b])
    ov["fki_test not in any aachen list"] = len(S["fki_test"] - allaa)
    ov["aachen_test not in any fki list"] = len(S["aachen_test"] - allfki)
    ov["fki_train == aachen_train (as sets)"] = S["fki_train"] == S["aachen_train"]
    ov["fki_train == aachen_train (same order)"] = L["fki_train"] == L["aachen_train"]
    res["overlaps_by_line_id"] = ov
    # form-level (page) overlap: a line of a form in the other partition's training split
    form = lambda ids: {"-".join(x.split("-")[:2]) for x in ids}
    res["form_overlaps"] = {"fki_test forms & aachen_train forms": len(form(S["fki_test"]) & form(S["aachen_train"])),
                            "aachen_test forms & fki_train forms": len(form(S["aachen_test"]) & form(S["fki_train"]))}
    # writers from forms.txt if present
    ft = os.path.join(RAW, "ascii", "forms.txt")
    if os.path.exists(ft):
        w = {}
        for line in open(ft).read().splitlines():
            if line and line[0] != "#":
                s = line.split(); w[s[0]] = s[1]
        wr = lambda ids: {w.get("-".join(x.split("-")[:2])) for x in ids}
        res["writer_overlaps"] = {"fki_test writers & aachen_train writers": len(wr(S["fki_test"]) & wr(S["aachen_train"])),
                                  "aachen_test writers & fki_train writers": len(wr(S["aachen_test"]) & wr(S["fki_train"])),
                                  "fki_test writers & fki_train writers": len(wr(S["fki_test"]) & wr(S["fki_train"])),
                                  "aachen_test writers & aachen_train writers": len(wr(S["aachen_test"]) & wr(S["aachen_train"]))}
        for k in L:
            res["lists"][k]["writers"] = len(wr(L[k]))
            res["lists"][k].pop("writers_note")
    # archives: split == list(s) in order
    arch = {}
    for f, part in (("iam.hdf5", "fki"), ("iam_128x1632.hdf5", "fki"), ("iam_aachen.hdf5", "aachen")):
        with h5py.File(os.path.join(DATA, f), "r") as h:
            d = {}
            for sp in ("train", "valid", "test"):
                ids = (L[f"fki_{sp}"] if sp != "valid" else L["fki_valid1"] + L["fki_valid2"]) if part == "fki" else L[f"aachen_{sp}"]
                want = [gt[x] for x in ids if x in gt and gt[x].strip()]
                got = [g.decode("utf-8") for g in h[sp]["gt"][:]]
                d[sp] = dict(n=len(got), equals_list_transcriptions_in_order=got == want, n_list=len(ids))
                if sp == "valid" and part == "fki":
                    d[sp]["first_900_equal_validationset1"] = got[:900] == [gt[x] for x in L["fki_valid1"]]
            arch[f] = d
    res["archives"] = arch
    res["iam_split_used"] = "6,161 / 1,840 (validationset1 + validationset2) / 1,861"
    json.dump(res, open(os.path.join(ROOT, "results", "control_iam_splits.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    sys.exit(main())

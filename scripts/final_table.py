#!/usr/bin/env python3
"""Validation-selected final results per corpus.

    python scripts/final_table.py      # writes docs/FINAL_RESULTS_T5.md and results/final_selection.json
    python scripts/final_table.py --out-dir /tmp/check   # same tables, written elsewhere

For every corpus, system (Student = one compact model; Teacher = Teacher ensemble or best single
Teacher) and stage (raw CTC, Wikipedia char LM, T5 post-processing on any base), every candidate
result found in results/ is listed and the one with the lowest VALIDATION CER (WER breaks ties) is
selected. Test numbers are read only for the selected rows (and listed for all, so nothing is hidden).
"""
import argparse
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
CORPORA = ["iam", "rimes", "bentham", "saintgall", "washington"]
STUDENTS = ["c128_student_s1", "c128_student_kd_s1"] + [f"ft_{c}_c128_student_kd_s1" for c in CORPORA] \
    + [f"ft_{c}_c128_student_s1" for c in CORPORA] + ["nopt_sg_student_s1"]
TEACHERS = ["nopt_sg_c128_teacher_s1", "nopt_sg_c128_teacher_s2", "nopt_sg_combined_final_teacher_s1",
            "nopt_sg_combined_final_teacher_s2", "nopt_sg_combined_syn_teacher_s1", "nopt_sg_combined_syn_teacher_s2",
            "c128_syn_teacher_s1", "c128_syn_teacher_s2"] + [f"ft_{c}_c128_syn_teacher_s{s}" for c in CORPORA for s in (1, 2)]


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def candidates(ds):
    rows = []
    for system, models in (("Student", STUDENTS), ("Teacher", TEACHERS)):
        for m, dsx in [(m, d) for m in models for d in (ds, ds + "_nopt")]:
            v, t = load(f"{R}/{m}_{dsx}_valid.json"), load(f"{R}/{m}_{dsx}_test.json")
            if v and t:
                rows.append(dict(system=system, stage="raw CTC", id=m, v=(v["cer"], v["wer"]),
                                 t=(t["cer"], t["wer"], t.get("ser"))))
    for f in sorted(glob.glob(f"{R}/ens_{ds}_*.json")):
        r = load(f)
        if not r or "test" not in r or r.get("dataset") not in (ds, ds + "_nopt"):
            continue
        members = [m.split("/")[-2] for m in r["members"]]
        if len(members) == 1 and "student" in members[0] and members[0].endswith("_s2"):
            continue   # seed-2 Students are reported by seed2_student_chain.py / seed2_saintgall_nopt.py
        if len(members) == 1:
            system = "Student" if "student" in members[0] else "Teacher"
            stage = "char LM"
        else:
            system, stage = "Teacher", ("ensemble (no LM)" if r["chosen"]["alpha"] == 0.0 and "nolm" in f else "ensemble + char LM")
        if r["chosen"]["members"] != list(range(len(members))):
            continue
        rows.append(dict(system=system, stage=stage, id=os.path.basename(f)[:-5], v=(r["chosen"]["cer"], r["chosen"]["wer"]),
                         t=(r["test"]["cer"], r["test"]["wer"], r["test"].get("ser"))))
    for f in sorted(glob.glob(f"{R}/t5_{ds}_*.json")):
        r = load(f)
        if not r or "test_t5" not in r:
            continue
        base = r["base"]
        if "s2student" in base:
            continue   # seed-2 Student T5 paths: see results/seed2_student.json
        system = "Teacher" if "teacher" in base else "Student"
        base_stage = ("ensemble (no LM)" if "ensnolm" in base else "ensemble + char LM") if "teacher_ens" in base \
            else ("char LM" if "_lm" in base else "raw CTC")
        rows.append(dict(system=system, stage=f"T5 on {base_stage}", id=os.path.basename(f)[:-5],
                         corrector=os.path.basename(r["corrector"]), v=(r["selected"]["cer"], r["selected"]["wer"]),
                         t=(r["test_t5"]["cer"], r["test_t5"]["wer"], r["test_t5"].get("ser")),
                         base_valid=r["valid_raw"]["cer"], base_test=r["test_raw"]["cer"], gate=r["selected"]))
    return rows


def pct(x):
    return "—" if x is None else f"{100 * x:.2f}"


def main(argv=None):
    global R
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=R)
    ap.add_argument("--out-dir", default=None, help="write FINAL_RESULTS_T5.md and final_selection.json here "
                                                   "(default: docs/ and results/ of the repository)")
    a = ap.parse_args(argv)
    R = a.results_dir
    out_md = os.path.join(a.out_dir, "FINAL_RESULTS_T5.md") if a.out_dir else os.path.join(ROOT, "docs", "FINAL_RESULTS_T5.md")
    out_js = os.path.join(a.out_dir, "final_selection.json") if a.out_dir else os.path.join(R, "final_selection.json")
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
    sel, lines = {}, ["# Final validation-selected results (T5 stage included)", "",
                      "Saint Gall candidates include models fine-tuned on the training labels without the raw 'pt' special "
                      "letter (annotated on the official training pages only; validation and test labels are unchanged).", "",
                      "Generated by `scripts/final_table.py`. Within each corpus x system x stage, the candidate with the "
                      "lowest validation CER is selected; test CER is shown for every candidate but never used to choose. "
                      "All numbers are CER % on the official test split unless marked valid.", ""]
    COLS = [("Student", "raw CTC"), ("Student", "T5 on raw CTC"), ("Student", "char LM"), ("Student", "T5 on char LM"),
            ("Teacher", "ensemble (no LM)"), ("Teacher", "T5 on ensemble (no LM)"),
            ("Teacher", "ensemble + char LM"), ("Teacher", "T5 on ensemble + char LM")]
    summary = ["| corpus | " + " | ".join(f"{a} {b}" for a, b in COLS) + " | best valid path: test CER |",
               "|---|" + "---:|" * len(COLS) + "---|"]
    for ds in CORPORA:
        rows = candidates(ds)
        pick = {}
        for r in rows:
            key = (r["system"], r["stage"])
            # lowest validation (CER, WER); exact ties broken by id so the table is deterministic
            if key not in pick or (r["v"], r["id"]) < (pick[key]["v"], pick[key]["id"]):
                pick[key] = r
        sel[ds] = {f"{k[0]} | {k[1]}": v for k, v in pick.items()}
        best = min(pick.values(), key=lambda r: r["v"])
        g = lambda s, st: pick.get((s, st))
        def cell(r):
            return f"{pct(r['t'][0])} (v {pct(r['v'][0])})" if r else "—"
        summary.append(f"| {ds} | " + " | ".join(cell(g(*c)) for c in COLS)
                       + f" | {best['system']} {best['stage']} `{best['id']}`: {pct(best['t'][0])} |")
        lines += [f"## {ds}", "", "| system | stage | id | corrector | valid CER / WER | test CER / WER / SER | selected |",
                  "|---|---|---|---|---:|---:|:-:|"]
        for r in sorted(rows, key=lambda r: (r["system"], r["stage"], r["v"])):
            chosen = any(r is p for p in pick.values())
            lines.append(f"| {r['system']} | {r['stage']} | `{r['id']}` | {r.get('corrector', '')} | "
                         f"{pct(r['v'][0])} / {pct(r['v'][1])} | {pct(r['t'][0])} / {pct(r['t'][1])} / {pct(r['t'][2])} | "
                         f"{'**yes**' if chosen else ''} |")
        lines.append("")
    out = lines[:5] + ["## Summary", "", "Cells: test CER (validation CER of the selected candidate).", ""] + summary + [""] + lines[5:]
    open(out_md, "w").write("\n".join(out) + "\n")
    json.dump(sel, open(out_js, "w"), indent=1, default=str)
    print("\n".join(summary))


if __name__ == "__main__":
    main()

"""Add the marker-free Saint Gall seed-2 Student to results/seed2_student.json.

seed2_student_chain.py ran before the Saint Gall marker-free labels existed, so its
"saintgall" entry is the original-label model. This script pools that entry with the
marker-free model (nopt_sg_student_s2, its char-LM path and both T5 paths) and selects
the path with the lowest VALIDATION CER, exactly like the chain. The original-label
entry is kept under "saintgall_original_labels".
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def j(p):
    return json.load(open(os.path.join(ROOT, "results", p)))


def main():
    path = os.path.join(ROOT, "results", "seed2_student.json")
    out = j("seed2_student.json")
    old = out.get("saintgall_original_labels", out["saintgall"])
    m = "nopt_sg_student_s2"
    e = j("ens_saintgall_nopt_student_s2_wiki.json")
    paths = {
        "raw CTC": (j(f"{m}_saintgall_nopt_valid.json")["cer"], j(f"{m}_saintgall_nopt_test.json")["cer"]),
        "char LM": (e["chosen"]["cer"], e["test"]["cer"]),
    }
    for name, k in (("T5 on raw CTC", "ctc"), ("T5 on char LM", "lm")):
        x = j(f"t5_saintgall_nopt_s2student_{k}_t5_ext_ctx1.json")
        paths[name] = (x["selected"]["cer"], x["test_t5"]["cer"])
    pool = {f"{k} (marker-free labels)": v for k, v in paths.items()}
    pool.update({f"{k} (original labels)": (v["valid"], v["test"]) for k, v in old["paths"].items()})
    best = min(pool, key=lambda k: pool[k][0])   # first minimum: dict order breaks ties
    out["saintgall_original_labels"] = old
    out["saintgall"] = dict(model=m if "marker-free" in best else old["model"], init=old.get("parent"),
                            dataset="saintgall_nopt", paths={k: dict(valid=a, test=b) for k, (a, b) in pool.items()},
                            selected_path=best, selected_test=pool[best][1],
                            note="pooled original-label and marker-free candidates; selection on validation CER")
    json.dump(out, open(path, "w"), indent=1)
    print(best, pool[best])


if __name__ == "__main__":
    main()

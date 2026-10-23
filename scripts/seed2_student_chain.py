#!/usr/bin/env python3
"""Seed-2 replication of the Student protocol, every choice on validation:

 1. wait for c128_student_s2 / c128_student_kd_s2 and their per-corpus valid + test evaluations;
 2. per corpus, parent = the one with lower greedy VALIDATION CER;
 3. per-corpus fine-tuning of that parent (seed 2, same recipe as configs/queues/runs_queue_FT.txt);
 4. adopt the fine-tuned model if its validation CER is lower than the parent's;
 5. Wikipedia char LM decoding (same LM per corpus as seed 1);
 6. T5 post-processing with the external-text corrector on the raw CTC and LM paths;
 7. write results/seed2_student.json with every decision and number.
"""
import json
import os
import subprocess
import sys
import time


if __name__ == "__main__" and any(a in ("-h", "--help") for a in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.environ.get("HTRJAND_DATA_DIR") or sys.exit("set HTRJAND_DATA_DIR to the directory holding the archives")
ENV = dict(os.environ, HTRJAND_DATA_DIR=D)
CORPORA = ["iam", "rimes", "bentham", "saintgall", "washington"]
LM = dict(iam="--lm-text synth/text_en.txt --lm-max-lines 400000 --order 10",
          rimes="--lm-text synth/text_fr.txt --lm-max-lines 200000 --order 9",
          bentham="--lm-text synth/text_en.txt --lm-max-lines 200000 --order 9",
          saintgall="--lm-text synth/text_la.txt --lm-max-lines 200000 --order 9",
          washington="--lm-text synth/text_en.txt --lm-max-lines 200000 --order 9")
FT = ("--gpu 0 --workers 6 --dataset {D}/{ds}_128x1632.hdf5 --preset paper_student --set pools=[[2,2],[2,2],[4,2]] "
      "--dropout 0.05 --dropout-cnn 0.0 --dropout-rnn 0.0 --weight-decay 1e-4 --sched cosine --lr 3e-4 --epochs 30 "
      "--warmup-epochs 1 --patience 0 --aug-set strong --ema 0.999 --batch-size 16 --seed 2 --init-ckpt runs/{m}/best.pth "
      "--keep-init-charset --tag {t}")


def sh(cmd):
    print(f"[{time.strftime('%H:%M:%S')}] {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, cwd=ROOT, env=ENV, capture_output=True, text=True)


def j(p):
    return json.load(open(os.path.join(ROOT, p)))


def done(m):
    return all(os.path.exists(os.path.join(ROOT, f"results/{m}_{ds}_{sp}.json")) for ds in CORPORA for sp in ("valid", "test"))


def main():
    while not (done("c128_student_s2") and done("c128_student_kd_s2")):
        time.sleep(300)
    out = {}
    for ds in CORPORA:
        v = {m: j(f"results/{m}_{ds}_valid.json")["cer"] for m in ("c128_student_s2", "c128_student_kd_s2")}
        parent = min(v, key=v.get)
        t = f"ft_{ds}_{parent}"
        if not os.path.exists(os.path.join(ROOT, f"results/{t}_{ds}_valid.json")):
            sh("python -m htrjand.train " + FT.format(D=D, ds=ds, m=parent, t=t))
            sh(f"python -m htrjand.evaluate --gpu 0 --split valid --ckpt runs/{t}/best.pth --dataset {ds} --tag {t}_{ds}")
            sh(f"python -m htrjand.evaluate --gpu 0 --ckpt runs/{t}/best.pth --dataset {ds} --tag {t}_{ds}")
        fv = j(f"results/{t}_{ds}_valid.json")["cer"]
        model = t if fv < v[parent] else parent
        r = sh(f"python -m htrjand.ensemble_rescore --dataset {ds} --tag {ds}_{model}_wiki --members runs/{model}/best.pth "
               f"{LM[ds]} --betas 0.0,0.5,1.0,1.5,2.0,3.0,4.0,5.0 --workers 12")
        paths = {"raw CTC": (f"results/{model}_{ds}_valid.json", f"results/{model}_{ds}_test.json")}
        e = j(f"results/ens_{ds}_{model}_wiki.json")
        paths["char LM"] = (e["chosen"]["cer"], e["test"]["cer"])
        for name, vp, tp in (("T5 on raw CTC", f"results/{model}_{ds}_valid_preds.tsv", f"results/{model}_{ds}_test_preds.tsv"),
                             ("T5 on char LM", f"results/ens_{ds}_{model}_wiki_valid_preds.tsv", f"results/ens_{ds}_{model}_wiki_test_preds.tsv")):
            o = f"results/t5_{ds}_s2student_{'ctc' if 'CTC' in name else 'lm'}_t5_ext_ctx1.json"
            sh(f"python -m htrjand.t5post apply --dataset {ds} --corrector postproc/t5_ext_ctx1 --tag {os.path.basename(o)[:-5]} "
               f"--val-preds {vp} --test-preds {tp} --out {o}")
            x = j(o)
            paths[name] = (x["selected"]["cer"], x["test_t5"]["cer"])
        rv, rt = paths["raw CTC"]
        paths["raw CTC"] = (j(rv)["cer"], j(rt)["cer"])
        best = min(paths, key=lambda k: paths[k][0])
        out[ds] = dict(parent_valid=v, parent=parent, finetuned=t, finetuned_valid=fv, model=model,
                       paths={k: dict(valid=a, test=b) for k, (a, b) in paths.items()}, selected_path=best,
                       selected_test=paths[best][1])
        json.dump(out, open(os.path.join(ROOT, "results/seed2_student.json"), "w"), indent=1)
        print(ds, json.dumps(out[ds]), flush=True)
    print("SEED2 CHAIN DONE", flush=True)


if __name__ == "__main__":
    main()

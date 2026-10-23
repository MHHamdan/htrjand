#!/usr/bin/env python3
"""Convert training checkpoints (runs/<id>/best.pth) into the release format.

    python scripts/export_hf.py --runs-dir runs --results-dir results --out-root ../htrjand-release-hf
    python scripts/export_hf.py ... --only htrjand-student/iam

For every model of the release plan (RELEASE below) it writes
    <out-root>/<repo>/<variant>/model.safetensors   the state dict, nothing else
    <out-root>/<repo>/<variant>/config.json         architecture, preprocessing, tokenizer,
                                                   decoder settings, provenance, metrics, sha256
The checkpoints hold no optimiser / scheduler / scaler / EMA state (verified and recorded in
config.json "export.checkpoint_keys"); their `args` carry machine paths and are written to the
config sanitised. Metrics are copied from results/*.json (raw CTC, char LM, T5) only when the
result file's own provenance names this run as its base; decoder settings (released LM weights,
T5 gates) come from results/release_eval/decoder_settings.json when it exists
(scripts/eval_release_decoders.py).
"""
import argparse
import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from htrjand import release as REL  # noqa: E402

CORPORA = ["iam", "rimes", "bentham", "saintgall", "washington"]
PAPER_VERSION = REL.RELEASE_VERSION
SOURCE_COMMIT = REL.SOURCE_COMMIT
STUDENT, STUDENT2, TEACHER = REL.STUDENT, REL.STUDENT_SEED2, REL.TEACHER

# repo, variant, run id, role, seed, corpus specialisation, how it was chosen
RELEASE = [
    (STUDENT, "general", "c128_student_kd_s1", "student", 1, None,
     "five-corpus 128 x 1632 Student with KD (seed 1)"),
    (STUDENT, "iam", "ft_iam_c128_student_kd_s1", "student", 1, "iam", "validation-selected per-corpus Student (final_selection.json)"),
    (STUDENT, "rimes", "ft_rimes_c128_student_kd_s1", "student", 1, "rimes", "validation-selected per-corpus Student (final_selection.json)"),
    (STUDENT, "bentham", "ft_bentham_c128_student_kd_s1", "student", 1, "bentham", "validation-selected per-corpus Student (final_selection.json)"),
    (STUDENT, "saintgall", "nopt_sg_student_s1", "student", 1, "saintgall",
     "validation-selected per-corpus Student, fine-tuned on marker-free Saint Gall training labels"),
    (STUDENT, "washington", "ft_washington_c128_student_s1", "student", 1, "washington", "validation-selected per-corpus Student (final_selection.json)"),
    (STUDENT2, "general", "c128_student_kd_s2", "student", 2, None,
     "seed-2 five-corpus Student: c128_student_kd_s2 (combined validation CER 4.95 %) preferred over c128_student_s2 (5.08 %)"),
    (STUDENT2, "iam", "ft_iam_c128_student_s2", "student", 2, "iam", "seed-2 per-corpus Student (seed2_student.json)"),
    (STUDENT2, "rimes", "ft_rimes_c128_student_kd_s2", "student", 2, "rimes", "seed-2 per-corpus Student (seed2_student.json)"),
    (STUDENT2, "bentham", "ft_bentham_c128_student_kd_s2", "student", 2, "bentham", "seed-2 per-corpus Student (seed2_student.json)"),
    (STUDENT2, "saintgall", "nopt_sg_student_s2", "student", 2, "saintgall",
     "seed-2 Saint Gall Student fine-tuned on marker-free labels (docs/FINAL_REPORT_T5.md 14b)"),
    (STUDENT2, "washington", "ft_washington_c128_student_s2", "student", 2, "washington", "seed-2 per-corpus Student (seed2_student.json)"),
    (TEACHER, "general", "c128_syn_teacher_s2", "teacher", 2, None,
     "five-corpus 128 x 1632 Teacher: c128_syn_teacher_s2 (combined validation CER 3.131 %) preferred over c128_syn_teacher_s1 (3.141 %)"),
]

RELEASED_CORRECTOR = "t5_ext_ctx1"
LANG = dict(iam="en", rimes="fr", bentham="en", saintgall="la", washington="en")
LANGUAGE_NAME = dict(en="English", fr="French", la="Latin")

PATH_RULES = [
    (re.compile(r"/home/[^\s'\"]*?/htr-hand-refined/"), ""),
    (re.compile(r"/home/[^\s'\"]*?/handwritten-text-recognition/data/"), "$HTRJAND_DATA_DIR/"),
    (re.compile(r"/home/[^\s'\"]*"), "<local path>"),
]


def sanitise(v):
    if isinstance(v, str):
        for rx, rep in PATH_RULES:
            v = rx.sub(rep, v)
        v = v.replace("htrjand_paper_v1", "htrjand")
        return v
    if isinstance(v, list):
        return [sanitise(x) for x in v]
    if isinstance(v, dict):
        return {k: sanitise(x) for k, x in v.items()}
    return v


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def pct(x):
    return None if x is None else round(100.0 * x, 4)


def metrics_for(run, corpus_list, R):
    """Every result whose provenance names `run` as its base, per corpus."""
    out = {}
    for ds in corpus_list:
        dss = [ds, ds + "_nopt"] if ds == "saintgall" else [ds]
        m = {}
        for d in dss:
            v, t = load(f"{R}/{run}_{d}_valid.json"), load(f"{R}/{run}_{d}_test.json")
            if v and t and "raw_ctc" not in m:
                m["raw_ctc"] = dict(valid_cer=pct(v["cer"]), valid_wer=pct(v["wer"]), test_cer=pct(t["cer"]),
                                    test_wer=pct(t["wer"]), test_ser=pct(t.get("ser")), labels=d,
                                    source=[f"results/{run}_{d}_valid.json", f"results/{run}_{d}_test.json"])
        for f in sorted(os.listdir(R)):
            if not (f.startswith("ens_") and f.endswith(".json")):
                continue
            r = load(os.path.join(R, f))
            if not r or "test" not in r or r.get("dataset") not in dss:
                continue
            if [x.split("/")[-2] for x in r["members"]] != [run]:
                continue
            m["char_lm_corpus_plus_wikipedia"] = dict(
                valid_cer=pct(r["chosen"]["cer"]), valid_wer=pct(r["chosen"]["wer"]), test_cer=pct(r["test"]["cer"]),
                test_wer=pct(r["test"]["wer"]), test_ser=pct(r["test"].get("ser")), labels=r["dataset"],
                lm=dict(order=r["lm"]["order"], text=["<corpus training transcriptions>"]
                        + [os.path.basename(x) for x in r["lm"].get("extra", [])], alpha=r["chosen"]["alpha"],
                        beta=r["chosen"]["beta"]),
                note="LM fitted on the corpus's TRAINING transcriptions + Wikipedia text: NOT the released LM",
                source=f"results/{f}")
        cand = {"t5_on_raw_ctc": [], "t5_on_char_lm": []}
        for f in sorted(os.listdir(R)):
            if not (f.startswith("t5_") and f.endswith(".json")) or f.endswith("_t5cache.json"):
                continue
            r = load(os.path.join(R, f))
            if not r or "test_t5" not in r or r.get("dataset") not in dss:
                continue
            vp = os.path.basename(r.get("val_preds", ""))
            if vp.startswith(f"{run}_"):
                key = "t5_on_raw_ctc"
            elif any(vp.startswith(f"ens_{x}_{run}_wiki") for x in dss) or \
                    (run.startswith("nopt_sg_student") and
                     vp == f"ens_saintgall_nopt_student{'_s2' if run.endswith('s2') else ''}_wiki_valid_preds.tsv"):
                key = "t5_on_char_lm"
            else:
                continue
            s = r["selected"]
            corr = os.path.basename(r.get("corrector", ""))
            cand[key].append(dict(
                id=f[:-5], valid_cer=pct(s["cer"]), valid_wer=pct(s["wer"]), test_cer=pct(r["test_t5"]["cer"]),
                test_wer=pct(r["test_t5"]["wer"]), test_ser=pct(r["test_t5"].get("ser")),
                base_valid_cer=pct(r["valid_raw"]["cer"]), base_test_cer=pct(r["test_raw"]["cer"]),
                gate=dict(detect=s["detect"], apply=s["apply"], eps_thr=s["eps_thr"], theta=s["theta"]),
                corrector=corr, corrector_released=corr == RELEASED_CORRECTOR, labels=r["dataset"],
                source=f"results/{f}", _v=(s["cer"], s["wer"])))
        fsel = (load(f"{R}/final_selection.json") or {}).get(ds, {})
        for key, stage in (("t5_on_raw_ctc", "Student | T5 on raw CTC"), ("t5_on_char_lm", "Student | T5 on char LM")):
            c = cand[key]
            if not c:
                continue
            sel_id = fsel.get(stage, {}).get("id")
            pick = next((x for x in c if x["id"] == sel_id), None)
            if pick is not None:
                pick["selected_by"] = f"results/final_selection.json '{stage}' (validation-selected over the three correctors)"
            else:
                # not the run of the seed-1 selection: same rule as final_table.py over this run's correctors
                # (seed 2 ran the released corrector only, as results/seed2_student.json records)
                pick = min(c, key=lambda x: (x["_v"], x["id"]))
                pick["selected_by"] = ("lowest validation CER over the correctors run for this model "
                                       f"({', '.join(sorted(x['corrector'] for x in c))})")
            m[key] = {k: v for k, v in pick.items() if k != "_v"}
            if not pick["corrector_released"]:
                m[key]["note"] = (f"selected corrector {pick['corrector']} is NOT released; the released corrector "
                                  f"({RELEASED_CORRECTOR}) is reported under '{key}_released_corrector'")
                rel = next((x for x in c if x["corrector_released"]), None)
                if rel is not None:
                    m[key + "_released_corrector"] = {k: v for k, v in rel.items() if k != "_v"}
        if m:
            out[ds] = m
    return out


def selected_paths(corpus, R, run):
    """The validation-selected Student path of final_selection.json / seed2_student.json for this corpus."""
    fs = load(f"{R}/final_selection.json") or {}
    best = None
    for k, v in fs.get(corpus, {}).items():
        if v["system"] == "Student" and (best is None or v["v"][0] < best["v"][0]):
            best = v
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--results-dir", default=os.path.join(ROOT, "results"))
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--decoder-settings", default=os.path.join(ROOT, "results", "release_eval", "decoder_settings.json"))
    ap.add_argument("--only", nargs="*", default=None, help="<repo>/<variant> entries to export")
    ap.add_argument("--t5-repo", default=REL.DEFAULT_T5)
    ap.add_argument("--lm-repo", default=REL.DEFAULT_CHARLM)
    a = ap.parse_args(argv)
    import torch
    from safetensors.torch import save_file
    from htrjand import models as M
    from htrjand.data import Tokenizer

    if not os.path.isdir(a.runs_dir):
        sys.exit(f"{a.runs_dir}: no runs directory (the training checkpoints are not part of the repository)")
    settings = load(a.decoder_settings) or {}
    R = a.results_dir
    report = []
    for repo, variant, run, role, seed, corpus, why in RELEASE:
        if a.only and f"{repo}/{variant}" not in a.only:
            continue
        ck_path = os.path.join(a.runs_dir, run, "best.pth")
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        keys = sorted(ck.keys())
        forbidden = [k for k in keys if any(x in k.lower() for x in ("optim", "sched", "scaler", "ema"))]
        if forbidden:
            sys.exit(f"{ck_path}: unexpected training state {forbidden}")
        sd = {k: v.contiguous() for k, v in ck["model_state_dict"].items()}
        net = M.HTRJAND(**ck["model_cfg"])
        net.load_state_dict(sd, strict=True)
        tok = Tokenizer(ck["charset"])
        assert tok.vocab_size == ck["model_cfg"]["vocab_size"]
        log = load(os.path.join(a.runs_dir, run, "train_log.json")) or {}
        out = os.path.join(a.out_root, repo, variant)
        os.makedirs(out, exist_ok=True)
        wpath = os.path.join(out, "model.safetensors")
        save_file(sd, wpath, metadata={"format": "pt"})   # one key: byte-reproducible (safetensors orders metadata by hash)
        n_total = sum(p.numel() for p in net.parameters())
        n_aux = sum(p.numel() for p in net.aux.parameters()) if net.aux is not None else 0
        corpora = [corpus] if corpus else CORPORA
        default = corpus or "iam"
        st = settings.get(f"{repo}/{variant}", {})
        args = sanitise(ck.get("args", {}))
        env = log.get("environment", {})
        cfg = dict(
            format="htrjand/1",
            model_type="htrjand",
            release=PAPER_VERSION, work=REL.WORK_DESCRIPTION, paper=REL.PAPER_TITLE,
            repo=REL.repo_id(repo), variant=variant, role=role, licence="CC BY-NC 4.0",
            description=why,
            architecture=ck["model_cfg"],
            parameters=dict(total=n_total, deployed=n_total - n_aux, aux_head=n_aux,
                            note="the auxiliary CTC head (training only) is kept in the weights for a strict load; "
                                 "inference does not use it; 'deployed' excludes it"),
            n_frames=net.n_frames,
            preprocessing=dict(
                input_size=list(ck["input_size"]), channels=1, normalisation=ck.get("normalisation", "paper"),
                recipe=["grayscale", "aspect-preserving scale s = min(W / w, H / h), cv2.INTER_AREA, int() truncation",
                        "centre pad with white (255) to H x W", "invert if mean > 128 (ink bright on a dark ground)",
                        "normalisation 'paper': per-image min-max to [-1, 1] (per-image min-max normalisation); blank image -> -1"],
                user_images="htrjand.inference.prepare_line: polarity check, ink crop (margin 8 % of ink height), "
                            "optional small-angle deskew, line check, then the recipe above"),
            tokenizer=dict(charset=tok.charset, chars=tok.chars, pad_index=tok.PAD, unk_index=tok.UNK,
                           blank_index=tok.BLANK, vocab_size=tok.vocab_size,
                           built_from="training transcriptions only (train split of the five corpora)"),
            corpus=dict(specialisation=corpus or "general (IAM, RIMES, Bentham, Saint Gall, Washington + synthetic lines)",
                        default=default, supported=corpora,
                        languages={c: LANGUAGE_NAME[LANG[c]] for c in corpora}),
            decoders=dict(
                default="greedy",
                greedy=dict(description="best-path CTC: arg-max per frame, collapse repeats, drop blanks"),
                lm=dict(repo=a.lm_repo, licence="CC BY-SA 4.0 (Wikipedia-derived text only)",
                        search=dict(beam=16, nbest=8, prune=-9.0, gen_alpha=0.2, gen_beta=0.5),
                        per_corpus=st.get("lm", {}),
                        description="CTC prefix beam search with the released Wikipedia character LM, n-best "
                                    "rescored by CTC log-likelihood + alpha * LM + beta * length (alpha, beta on validation)")),
            postprocessing=dict(t5=dict(corrector=a.t5_repo, gates=st.get("t5", {}),
                                        description="gated T5-small corrector (T5 post-processing); gate chosen on validation")),
            provenance=dict(
                training_run_id=run, seed=seed,
                source_release=f"{REL.SOURCE_COMMIT} of https://github.com/{REL.GITHUB_REPO}",
                private_source_commit=REL.PRIVATE_SOURCE_COMMIT,
                training_git_commit=(env.get("git_commit") or "")[:7] + ("+dirty" if env.get("git_dirty") else ""),
                trained_on=sanitise(ck.get("dataset")), init_ckpt=sanitise(args.get("init_ckpt")),
                teacher_ckpt=sanitise(ck.get("teacher_ckpt")), kd=bool(args.get("kd")),
                best_epoch=ck.get("epoch"), epochs_run=log.get("epochs_run"), finished=log.get("finished"),
                val_cer_at_selection=ck.get("val_cer"),
                selection="best validation CER over epochs (EMA or raw weights, whichever lower)",
                training_args=args,
                training_command=sanitise(env.get("command")),
                environment=dict(python=env.get("python"), torch=env.get("torch"), cuda=env.get("cuda"), gpu=env.get("gpu"))),
            metrics=dict(
                note="CER / WER / SER in %. Copied from results/*.json of the source repository; test decoded once "
                     "per checkpoint or path after validation selection.",
                per_corpus=metrics_for(run, corpora, R)),
            export=dict(checkpoint_keys=keys, stripped=[k for k in keys if k != "model_state_dict"],
                        note="no optimiser / scheduler / AMP scaler / EMA state was present in the checkpoint; "
                             "only model_state_dict is exported (metadata goes to this config)"),
            weights=dict(file="model.safetensors", sha256=sha256(wpath), bytes=os.path.getsize(wpath),
                         includes_aux_head=n_aux > 0),
        )
        if st.get("release_lm_validation"):
            cfg["metrics"]["released_lm_validation"] = st["release_lm_validation"]
        json.dump(cfg, open(os.path.join(out, "config.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        report.append((f"{repo}/{variant}", run, n_total - n_aux, cfg["weights"]["sha256"][:12]))
        print(f"{repo}/{variant:<11} {run:<32} deployed {n_total - n_aux:>9,}  total {n_total:>9,}  "
              f"sha256 {cfg['weights']['sha256'][:16]}", flush=True)
    return report


if __name__ == "__main__":
    main()

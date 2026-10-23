#!/usr/bin/env python3
"""Validation-only evaluation and tuning of the released decoders.

    HTRJAND_DATA_DIR=... python scripts/eval_release_decoders.py lm --hf-root ../htrjand-release-hf \
        --lm-dir release_lm --work-dir /tmp/htrjand_eval
    HTRJAND_DATA_DIR=... python scripts/eval_release_decoders.py t5 --hf-root ../htrjand-release-hf \
        --corrector ../htrjand-release-hf/htrjand-t5-corrector --work-dir /tmp/htrjand_eval

Nothing here reads a TEST split. For every released recogniser and every corpus it supports:

  lm   greedy CTC, then the character-LM decoder of `htrjand.lmdecode` (the single-member case of
       ensemble_rescore.py: n-best from the LM prefix beam search with gen_alpha 0.2 / gen_beta 0.5,
       rescored by CTC + alpha LM + beta length) with
         * each released Wikipedia-only LM candidate (--lm-specs), alpha x beta swept on the full
           validation split exactly as ensemble_rescore.py does (lowest CER, WER breaks ties);
         * for the seed-1 per-corpus Students, the ORIGINAL corpus+Wikipedia LM of the experiments
           (training transcriptions + Wikipedia lines, same order / line cap) at the alpha / beta that
           ensemble_rescore.py chose: this reproduces results/ens_<corpus>_<run>_wiki.json "chosen".
  t5   the gated T5 corrector on the greedy and on the released-LM outputs: the detect x apply x
       eps-threshold x theta grid of t5post.cmd_apply, lowest validation CER selected, with the
       Wikipedia lexicon shipped with the corrector (and, for comparison on the seed-1 per-corpus
       raw-CTC path, the corpus TRAINING lexicon of the experiments).

Writes metrics (no text) to results/release_eval/*.json and the selected settings to
results/release_eval/decoder_settings.json, which scripts/export_hf.py copies into config.json.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import get_context

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402

CORPORA = ["iam", "rimes", "bentham", "saintgall", "washington"]
ARCHIVE = dict(iam="iam", rimes="rimes", bentham="bentham", saintgall="saintgall_nopt", washington="washington")
LANG = dict(iam="en", rimes="fr", bentham="en", saintgall="la", washington="en")
# the experiments' LM per corpus (scripts/ft_student_paths.sh, saintgall_nopt.sh)
ORIG_LM = dict(iam=("text_en.txt", 400000, 10), rimes=("text_fr.txt", 200000, 9), bentham=("text_en.txt", 200000, 9),
               saintgall=("text_la.txt", 200000, 9), washington=("text_en.txt", 200000, 9))
ORIG_ENS = dict(iam="ens_iam_ft_iam_c128_student_kd_s1_wiki", rimes="ens_rimes_ft_rimes_c128_student_kd_s1_wiki",
                bentham="ens_bentham_ft_bentham_c128_student_kd_s1_wiki", saintgall="ens_saintgall_nopt_student_wiki",
                washington="ens_washington_ft_washington_c128_student_s1_wiki")
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
BETAS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
THETAS = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.01]

_W = {}


def targets(release, only=None):
    out = []
    for repo, variant, run, role, seed, corpus, _ in release:
        for ds in ([corpus] if corpus else CORPORA):
            key = f"{repo}/{variant}"
            if only and key not in only:
                continue
            out.append(dict(repo=repo, variant=variant, run=run, seed=seed, corpus=ds, key=key,
                            per_corpus=corpus is not None))
    return out


def read_split(ds, split, size_suffix="_128x1632"):
    import h5py
    p = os.path.join(os.environ["HTRJAND_DATA_DIR"], f"{ARCHIVE[ds]}{size_suffix}.hdf5")
    with h5py.File(p, "r") as h:
        imgs = h[split]["dt"][:] if size_suffix else None
        labels = [g.decode("utf-8") if isinstance(g, bytes) else g for g in h[split]["gt"][:]]
    return imgs, labels


def valid_logprobs(t, hf_root, work, device):
    path = os.path.join(work, "lp", f"{t['key'].replace('/', '__')}__{t['corpus']}.npy")
    if os.path.exists(path):
        return np.load(path), json.load(open(path[:-4] + "_labels.json"))
    from htrjand import HTRJANDModel
    from htrjand.inference import normalise_canvas
    m = HTRJANDModel.from_pretrained(os.path.join(hf_root, t["repo"]), variant=t["variant"], device=device)
    imgs, labels = read_split(t["corpus"], "valid")
    out = []
    for i in range(0, len(imgs), 64):
        x = np.stack([normalise_canvas(im, m.normalisation)[None] for im in imgs[i:i + 64]])
        out.append(m.log_probs(x).astype(np.float32))
    lp = np.concatenate(out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, lp)
    json.dump(labels, open(path[:-4] + "_labels.json", "w"))
    return lp, labels


def labels_for(t, work):
    p = os.path.join(work, "lp", f"{t['key'].replace('/', '__')}__{t['corpus']}_labels.json")
    return json.load(open(p))


def _cand(i):
    from htrjand.lmdecode import candidates, score_candidates
    lp = _W["lps"][i]
    c = candidates(lp, _W["tok"], _W["lm"])
    s, l, n = score_candidates(lp, _W["tok"], c, _W["lm"])
    return c, s.tolist(), l.tolist(), n.tolist()


def nbest_all(lps, tok, lm, workers):
    import torch
    torch.set_num_threads(1)
    _W.update(lps=lps, tok=tok, lm=lm)
    with get_context("fork").Pool(workers) as p:
        return p.map(_cand, range(len(lps)), chunksize=4)


def sweep(refs, rows, alphas=ALPHAS, betas=BETAS):
    from htrjand import metrics as Me
    grid = []
    for al in alphas:
        for be in betas:
            h = [c[int(np.argmax(np.array(s) + al * np.array(l) + be * np.array(n)))] for c, s, l, n in rows]
            grid.append(dict(alpha=al, beta=be, cer=Me.cer(refs, h), wer=Me.wer(refs, h)))
    best = min(grid, key=lambda r: (r["cer"], r["wer"]))
    h = [c[int(np.argmax(np.array(s) + best["alpha"] * np.array(l) + best["beta"] * np.array(n)))] for c, s, l, n in rows]
    return best, h, grid


def corpus_lm(ds, synth_dir):
    from htrjand.decode import CompactCharLM
    _, train = read_split(ds, "train", "")
    f, cap, order = ORIG_LM[ds]
    extra = [l.rstrip("\n") for l in open(os.path.join(synth_dir, f), encoding="utf-8") if l.strip()][:cap]
    return CompactCharLM(order).fit(train + extra), order


def cmd_lm(a):
    import torch
    from export_hf import RELEASE
    from htrjand import metrics as Me
    from htrjand.data import Tokenizer
    from htrjand.decode import CompactCharLM
    device = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    specs = [s.split(":") for s in a.lm_specs]                  # name:order:min_count
    out_dir = os.path.join(ROOT, "results", "release_eval")
    os.makedirs(out_dir, exist_ok=True)
    res_path = os.path.join(out_dir, "lm_validation.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    lm_cache = {}

    def get_lm(lang, name):
        k = (lang, name)
        if k not in lm_cache:
            lm_cache.clear()
            t0 = time.time()
            lm_cache[k] = CompactCharLM.load(os.path.join(a.lm_dir, f"{name}_{lang}.gz"))
            print(f"  loaded {name}_{lang} ({lm_cache[k].n_entries():,} entries, {time.time() - t0:.0f}s)", flush=True)
        return lm_cache[k]

    tl = targets(RELEASE, a.only)
    for t in tl:
        lp, refs = valid_logprobs(t, a.hf_root, a.work_dir, device)
        cfg = json.load(open(os.path.join(a.hf_root, t["repo"], t["variant"], "config.json")))
        tok = Tokenizer(cfg["tokenizer"]["charset"])
        g = tok.ctc_greedy_decode(torch.from_numpy(lp))
        k = f"{t['key']}|{t['corpus']}"
        r = res.get(k, dict(model=t["key"], run=t["run"], corpus=t["corpus"], n_valid=len(refs)))
        r["greedy"] = dict(cer=Me.cer(refs, g), wer=Me.wer(refs, g))
        print(f"[{k}] greedy valid CER {100 * r['greedy']['cer']:.2f}", flush=True)
        lps = list(lp)
        names = [s[0] for s in specs] if (t["per_corpus"] and t["seed"] == 1) or a.all_specs else [a.release_spec]
        for name in names:
            if name in r.get("wiki_lm", {}) and not a.redo:
                continue
            lm = get_lm(LANG[t["corpus"]], name)
            t0 = time.time()
            rows = nbest_all(lps, tok, lm, a.workers)
            best, _, _ = sweep(refs, rows)
            r.setdefault("wiki_lm", {})[name] = dict(best, lm_entries=lm.n_entries(), order=lm.order,
                                                    min_count=lm.meta.get("min_count"), seconds=round(time.time() - t0, 1),
                                                    lines_per_second=round(len(lps) / (time.time() - t0), 1))
            print(f"[{k}] wiki LM {name}: valid CER {100 * best['cer']:.2f} (alpha {best['alpha']}, beta {best['beta']}) "
                  f"{time.time() - t0:.0f}s", flush=True)
            json.dump(res | {k: r}, open(res_path, "w"), indent=1)
        if t["per_corpus"] and t["seed"] == 1 and "corpus_plus_wiki_lm" not in r:
            ens = json.load(open(os.path.join(ROOT, "results", ORIG_ENS[t["corpus"]] + ".json")))
            lm_cache.clear()
            t0 = time.time()
            lm, order = corpus_lm(t["corpus"], a.synth_dir)
            assert order == ens["lm"]["order"]
            rows = nbest_all(lps, tok, lm, a.workers)
            al, be = ens["chosen"]["alpha"], ens["chosen"]["beta"]
            best, _, _ = sweep(refs, rows, [al], [be])
            r["corpus_plus_wiki_lm"] = dict(best, order=order, reported_valid_cer=ens["chosen"]["cer"],
                                            reported_valid_wer=ens["chosen"]["wer"], source=f"results/{ORIG_ENS[t['corpus']]}.json",
                                            reproduces_reported=abs(best["cer"] - ens["chosen"]["cer"]) < 1e-12,
                                            seconds=round(time.time() - t0, 1))
            print(f"[{k}] corpus+wiki LM (order {order}, alpha {al}, beta {be}): valid CER {100 * best['cer']:.3f} "
                  f"(reported {100 * ens['chosen']['cer']:.3f})", flush=True)
            del lm
        res[k] = r
        json.dump(res, open(res_path, "w"), indent=1)
    # decoder settings: the released LM, alpha / beta chosen on validation per (model, corpus)
    st_path = os.path.join(out_dir, "decoder_settings.json")
    st = json.load(open(st_path)) if os.path.exists(st_path) else {}
    for k, r in res.items():
        key, ds = k.split("|")
        w = r.get("wiki_lm", {}).get(a.release_spec)
        if not w:
            continue
        st.setdefault(key, {}).setdefault("lm", {})[ds] = dict(
            lm_file=f"charlm_{LANG[ds]}.gz", language=LANG[ds], alpha=w["alpha"], beta=w["beta"],
            valid_cer=round(100 * w["cer"], 4), valid_wer=round(100 * w["wer"], 4),
            greedy_valid_cer=round(100 * r["greedy"]["cer"], 4),
            selected_on=f"{ds} validation split ({r['n_valid']} lines), grid alpha {ALPHAS} x beta {BETAS}")
    json.dump(st, open(st_path, "w"), indent=1)
    print("LM DONE", flush=True)


# --------------------------------------------------------------------------------------------- T5
def t5_sweep(refs, hyps, eps, cands, confs, corpus, corr, lex):
    from htrjand import metrics as Me
    from htrjand import t5post as P
    fmt = corr.formats[corpus]
    eps = np.array(eps, dtype=float)
    has_eps = not np.isnan(eps).all()
    modes = ["all", "oov"] + (["eps", "oov_or_eps"] if has_eps else [])
    eps_thrs = [float(np.nanquantile(eps, q)) for q in (0.5, 0.75, 0.9)] if has_eps else [0.0]

    def run(theta, mode, how, et):
        out, n = [], 0
        for h, e, c, s in zip(hyps, eps, cands, confs):
            flag, oov = P.detect(P.normalise(h), lex, e, mode, et)
            if flag and s > theta:
                c2 = fmt.restore(c)
                new = c2 if how == "line" else P.apply_segments(P.normalise(h), c2, oov)
                new = fmt.restore(new)
                n += new != P.normalise(h)
                out.append(new)
            else:
                out.append(h)
        return out, n
    grid = []
    for mode in modes:
        for how in ("line", "segment"):
            for et in (eps_thrs if "eps" in mode else [0.0]):
                for th in THETAS:
                    hyp, n = run(th, mode, how, et)
                    grid.append(dict(detect=mode, apply=how, eps_thr=et, theta=th, n_changed=n,
                                     cer=Me.cer(refs, hyp), wer=Me.wer(refs, hyp)))
    return min(grid, key=lambda r: (r["cer"], r["wer"])), dict(cer=Me.cer(refs, hyps), wer=Me.wer(refs, hyps))


def cmd_t5(a):
    import torch
    from export_hf import RELEASE
    from htrjand.data import Tokenizer
    from htrjand.lmdecode import pick
    from htrjand.t5correct import T5Corrector, load_lexicon
    from htrjand import t5post as P
    device = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    corr = T5Corrector.from_pretrained(a.corrector, device=device, resources=a.charlm_dir)
    out_dir = os.path.join(ROOT, "results", "release_eval")
    res_path = os.path.join(out_dir, "t5_validation.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    st_path = os.path.join(out_dir, "decoder_settings.json")
    st = json.load(open(st_path))
    lm_res = json.load(open(os.path.join(out_dir, "lm_validation.json")))
    from htrjand.decode import CompactCharLM
    lm_cache = {}
    for t in targets(RELEASE, a.only):
        k = f"{t['key']}|{t['corpus']}"
        if k in res and not a.redo:
            continue
        lp, refs = valid_logprobs(t, a.hf_root, a.work_dir, device)
        cfg = json.load(open(os.path.join(a.hf_root, t["repo"], t["variant"], "config.json")))
        tok = Tokenizer(cfg["tokenizer"]["charset"])
        g = tok.ctc_greedy_decode(torch.from_numpy(lp))
        eps = (1.0 - np.exp(lp).max(-1).mean(1)).tolist()
        lang = LANG[t["corpus"]]
        if lang not in lm_cache:
            lm_cache.clear()
            lm_cache[lang] = CompactCharLM.load(os.path.join(a.lm_dir, f"{a.release_spec}_{lang}.gz"))
        s = st[t["key"]]["lm"][t["corpus"]]
        rows = nbest_all(list(lp), tok, lm_cache[lang], a.workers)
        lmh = [pick(c, np.array(sc), np.array(l), np.array(n), s["alpha"], s["beta"]) for c, sc, l, n in rows]
        r = dict(model=t["key"], run=t["run"], corpus=t["corpus"], n_valid=len(refs))
        lex = corr.lexicon(t["corpus"])
        for base, hyps, e in (("ctc", g, eps), ("lm", lmh, [float("nan")] * len(lmh))):
            t0 = time.time()
            cands, confs = corr.generate(corr.sources(t["corpus"], hyps, context=True))
            best, raw = t5_sweep(refs, hyps, e, cands, confs, t["corpus"], corr, lex)
            r[base] = dict(base=raw, selected=best, lexicon=corr.meta["lexicons"][lang], seconds=round(time.time() - t0, 1))
            if base == "ctc" and t["per_corpus"] and t["seed"] == 1:
                _, train = read_split(t["corpus"], "train", "")
                tl = P.lexicon([P.normalise(x) for x in train])
                b2, _ = t5_sweep(refs, hyps, e, cands, confs, t["corpus"], corr, tl)
                r[base]["with_training_lexicon"] = b2
                alt = os.path.join(corr.resources, "lexicon", f"wiki_{lang}_all.txt.gz")
                if os.path.exists(alt):
                    b3, _ = t5_sweep(refs, hyps, e, cands, confs, t["corpus"], corr, load_lexicon(alt))
                    r[base]["with_wiki_all_words_lexicon"] = b3
            print(f"[{k}] T5 on {base}: valid CER {100 * raw['cer']:.2f} -> {100 * best['cer']:.2f} "
                  f"({best['detect']}/{best['apply']}/theta {best['theta']}, {best['n_changed']} changed)", flush=True)
        res[k] = r
        json.dump(res, open(res_path, "w"), indent=1)
    for k, r in res.items():
        key, ds = k.split("|")
        for base in ("ctc", "lm"):
            b = r[base]["selected"]
            st.setdefault(key, {}).setdefault("t5", {}).setdefault(ds, {})[base] = dict(
                detect=b["detect"], apply=b["apply"], eps_thr=b["eps_thr"], theta=b["theta"],
                valid_cer_before=round(100 * r[base]["base"]["cer"], 4), valid_cer_after=round(100 * b["cer"], 4),
                n_valid_changed=b["n_changed"], lexicon=r[base]["lexicon"],
                selected_on=f"{ds} validation split ({r['n_valid']} lines), t5post.cmd_apply grid")
    json.dump(st, open(st_path, "w"), indent=1)
    print("T5 DONE", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("lm", "t5"):
        p = sub.add_parser(name)
        p.add_argument("--hf-root", required=True)
        p.add_argument("--lm-dir", required=True, help="directory with <spec>_<lang>.gz LM files")
        p.add_argument("--work-dir", required=True)
        p.add_argument("--release-spec", default="charlm", help="LM spec whose settings go to the configs")
        p.add_argument("--only", nargs="*", default=None)
        p.add_argument("--workers", type=int, default=16)
        p.add_argument("--cpu", action="store_true")
        p.add_argument("--redo", action="store_true")
    sub.choices["lm"].add_argument("--lm-specs", nargs="*", default=[], help="LM file stems to compare, e.g. o10m1")
    sub.choices["lm"].add_argument("--all-specs", action="store_true")
    sub.choices["lm"].add_argument("--synth-dir", default=os.path.join(ROOT, "synth"))
    sub.choices["t5"].add_argument("--corrector", required=True)
    sub.choices["t5"].add_argument("--charlm-dir", required=True, help="local LM repo holding lexicon/")
    a = ap.parse_args(argv)
    if not os.environ.get("HTRJAND_DATA_DIR"):
        sys.exit("set HTRJAND_DATA_DIR to the directory holding the archives")
    dict(lm=cmd_lm, t5=cmd_t5)[a.cmd](a)


if __name__ == "__main__":
    main()

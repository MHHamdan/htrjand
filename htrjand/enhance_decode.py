"""Teacher ensembling and language-model decoding of the
recogniser, with every free weight chosen on the validation split.

    python -m htrjand.enhance_decode --dataset iam \
        --ckpts runs/combined_final_teacher_s1/best.pth runs/combined_final_teacher_s2/best.pth \
        --tag ens2_iam --gpu 0

Steps, per corpus:
  1. every checkpoint decodes valid and test once; frame log-probabilities are
     cached (results/enhance_cache/<ckpt-id>_<dataset>_<split>.npy);
  2. the ensemble is the log of the mean of the members' frame probabilities
     (all members share the 108-frame geometry and the combined charset);
  3. a character n-gram (htrjand/decode.py CharLM) is fitted on TRAINING
     transcriptions only -- the corpus's own, or --lm-text files that are checked
     for test-line overlap before use -- and the CTC prefix beam search weights
     (order, alpha, beta) are swept on validation;
  4. the chosen setting decodes test once.
No test transcription reaches any fitted quantity. Results: results/enh_<tag>.json
and the usual _test_preds.tsv / _test_counts.npz for paired tests.
"""
import argparse
import csv
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import data as D
from . import metrics as Me
from . import models as M

from ._compat import REPO_ROOT  # noqa: E402  (source checkout, else the working directory)
from . import decode as Dec  # noqa: E402  (CharLM, prefix_beam_search: pure Python)


@torch.no_grad()
def logprobs(ckpt, dataset, split, device, cache_dir, batch_size=64, tta=False):
    key = hashlib.md5(os.path.abspath(ckpt).encode()).hexdigest()[:10]
    path = os.path.join(cache_dir, f"{key}_{dataset}_{split}{'_tta' if tta else ''}.npy")
    model, ck = M.from_checkpoint(ckpt)
    tok = D.Tokenizer(ck["charset"])
    ds = D.LineDataset(D.resolve_for(dataset, ck["model_cfg"]["input_size"]), split, tok,
                       normalisation=ck.get("normalisation", "paper"))
    if os.path.exists(path):
        return np.load(path), ds.labels, tok
    model.to(device)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=D.collate)
    out = []
    for img, _, _ in dl:
        img = img.to(device)
        lp = torch.log_softmax(model(img).float(), -1)
        if tta:   # mean of probabilities over small horizontal/vertical shifts
            ps = [lp.exp()]
            for dy, dx in ((0, 2), (0, -2), (1, 0), (-1, 0)):
                ps.append(torch.softmax(model(torch.roll(img, (dy, dx), (2, 3))).float(), -1))
            lp = torch.stack(ps).mean(0).clamp_min(1e-12).log()
        out.append(lp.cpu().numpy().astype(np.float32))
    arr = np.concatenate(out)
    os.makedirs(cache_dir, exist_ok=True)
    np.save(path, arr)
    return arr, ds.labels, tok


def ensemble(arrs):
    if len(arrs) == 1:
        return arrs[0]
    p = np.mean([np.exp(a) for a in arrs], axis=0)
    return np.log(np.clip(p, 1e-12, None)).astype(np.float32)


def _beam(args):
    return Dec.prefix_beam_search(*args)


def beam_all(lps, tok, lm, alpha, beta, beam, workers):
    jobs = [(lp, lp.shape[0], tok.chars, tok.BLANK, beam, lm, alpha, beta) for lp in lps]
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(_beam, jobs, chunksize=8))


def greedy(lps, tok):
    return tok.ctc_greedy_decode(torch.from_numpy(lps))


def overlap_check(lm_lines, test_lines, min_len=25):
    """Fraction of test lines (>= min_len chars) that occur verbatim in the LM text."""
    big = "\n".join(lm_lines)
    t = [l for l in test_lines if len(l) >= min_len]
    hit = sum(1 for l in t if l in big)
    return hit / max(len(t), 1), hit, len(t)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tta", action="store_true", help="shift test-time augmentation per member")
    ap.add_argument("--lm-source", default="corpus", choices=("corpus", "combined", "none"),
                    help="training transcriptions the char LM is fitted on")
    ap.add_argument("--lm-text", nargs="*", default=[], help="extra LM text files (one line per line)")
    ap.add_argument("--orders", default="6,8,10")
    ap.add_argument("--alphas", default="0.0,0.2,0.4,0.6,0.8")
    ap.add_argument("--betas", default="0.0,0.5,1.0,1.5")
    ap.add_argument("--beam", type=int, default=16)
    ap.add_argument("--subset", type=int, default=600, help="validation lines used for the sweep")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "results"))
    a = ap.parse_args(argv)

    device = torch.device("cpu" if a.gpu < 0 or not torch.cuda.is_available() else f"cuda:{a.gpu}")
    cache = os.path.join(a.out_dir, "enhance_cache")
    split_lp = {}
    for split in ("valid", "test"):
        arrs = []
        for c in a.ckpts:
            lp, refs, tok = logprobs(c, a.dataset, split, device, cache, tta=a.tta)
            arrs.append(lp)
        split_lp[split] = (ensemble(arrs), refs)
    torch.cuda.empty_cache()
    v_lp, v_refs = split_lp["valid"]
    t_lp, t_refs = split_lp["test"]
    res = dict(tag=a.tag, dataset=a.dataset, ckpts=[os.path.abspath(c) for c in a.ckpts],
               tta=a.tta, n_valid=len(v_refs), n_test=len(t_refs), result_class="ENHANCED_DECODING")
    gv, gt = greedy(v_lp, tok), greedy(t_lp, tok)
    res["greedy_valid"] = dict(cer=Me.cer(v_refs, gv), wer=Me.wer(v_refs, gv))
    res["greedy_test"] = dict(cer=Me.cer(t_refs, gt), wer=Me.wer(t_refs, gt), ser=Me.ser(t_refs, gt))
    print(f"[{a.tag}] ensemble of {len(a.ckpts)} greedy: valid CER {100*res['greedy_valid']['cer']:.2f}"
          f"  test CER {100*res['greedy_test']['cer']:.2f} WER {100*res['greedy_test']['wer']:.2f}", flush=True)

    final_hyps, chosen = gt, None
    if a.lm_source != "none":
        import h5py
        with h5py.File(D.archive_path(a.dataset if a.lm_source == "corpus" else "combined"), "r") as h:
            lm_lines = [g.decode("utf-8") for g in h["train"]["gt"][:]]
        for f in a.lm_text:
            lm_lines += [l.rstrip("\n") for l in open(f, encoding="utf-8") if l.strip()]
        frac, hit, n = overlap_check(lm_lines, t_refs)
        res["lm"] = dict(source=a.lm_source, extra=a.lm_text, n_lines=len(lm_lines),
                         test_line_overlap=frac, test_lines_found=hit, test_lines_checked=n)
        print(f"[{a.tag}] LM text {len(lm_lines)} lines; verbatim test-line overlap {hit}/{n}", flush=True)
        if a.lm_text and frac > 0.01:
            raise SystemExit("LM text contains test lines; refusing to decode")
        lms = {o: Dec.CharLM(o).fit(lm_lines) for o in (int(x) for x in a.orders.split(","))}
        rng = np.random.default_rng(0)
        idx = np.arange(len(v_refs))
        if a.subset and a.subset < len(idx):
            idx = np.sort(rng.choice(idx, a.subset, replace=False))
        sub_lp, sub_refs = [v_lp[i] for i in idx], [v_refs[i] for i in idx]
        rows = []
        for o, lm in lms.items():
            for al in (float(x) for x in a.alphas.split(",")):
                for be in (float(x) for x in a.betas.split(",")):
                    if al == 0.0 and be != 0.0:
                        continue
                    t0 = time.time()
                    h = beam_all(sub_lp, tok, lm, al, be, a.beam, a.workers)
                    rows.append(dict(order=o, alpha=al, beta=be, cer=Me.cer(sub_refs, h), wer=Me.wer(sub_refs, h)))
                    print(f"  order {o} alpha {al:.1f} beta {be:.1f}: val CER {100*rows[-1]['cer']:.2f}"
                          f" WER {100*rows[-1]['wer']:.2f} ({time.time()-t0:.0f}s)", flush=True)
        chosen = min(rows, key=lambda r: (r["cer"], r["wer"]))
        res["sweep"], res["chosen"] = rows, chosen
        lm = lms[chosen["order"]]
        hv = beam_all(list(v_lp), tok, lm, chosen["alpha"], chosen["beta"], a.beam, a.workers)
        res["beam_lm_valid"] = dict(cer=Me.cer(v_refs, hv), wer=Me.wer(v_refs, hv))
        final_hyps = beam_all(list(t_lp), tok, lm, chosen["alpha"], chosen["beta"], a.beam, a.workers)
        res["beam_lm_test"] = dict(cer=Me.cer(t_refs, final_hyps), wer=Me.wer(t_refs, final_hyps),
                                   ser=Me.ser(t_refs, final_hyps))
        print(f"[{a.tag}] chosen {chosen['order']}/{chosen['alpha']}/{chosen['beta']}: valid CER "
              f"{100*res['beam_lm_valid']['cer']:.2f}  test CER {100*res['beam_lm_test']['cer']:.2f}"
              f" WER {100*res['beam_lm_test']['wer']:.2f}", flush=True)

    ce, cl, we, wl = Me.per_line_counts(t_refs, final_hyps)
    res["cer_ci95"] = list(Me.bootstrap_ci(ce, cl))
    os.makedirs(a.out_dir, exist_ok=True)
    stem = os.path.join(a.out_dir, f"enh_{a.tag}")
    json.dump(res, open(stem + ".json", "w"), indent=1)
    np.savez_compressed(stem + "_test_counts.npz", char_err=ce, char_len=cl, word_err=we, word_len=wl)
    with open(stem + "_test_preds.tsv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["idx", "ref", "hyp"])
        for i, (r, h) in enumerate(zip(t_refs, final_hyps)):
            w.writerow([i, r, h])
    return res


if __name__ == "__main__":
    main()

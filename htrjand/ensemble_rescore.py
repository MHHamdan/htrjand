"""Sequence-level Teacher ensemble with character-LM rescoring.

    python -m htrjand.ensemble_rescore --dataset iam --tag iam_ens5 \
        --members runs/combined_final_teacher_s1/best.pth runs/combined_final_teacher_s2/best.pth \
                  runs/iam_final_teacher_s1/best.pth ...

Averaging frame posteriors of independently trained CTC models splits their
spikes (they fire on different frames), so members are combined at the level of
whole transcriptions instead:
  1. each member runs a CTC prefix beam search with the character LM and returns
     its n best transcriptions; the candidate set of a line is their union;
  2. every candidate is scored by every member's exact CTC log-likelihood
     (torch ctc_loss on the member's own frame posteriors; a member whose charset
     cannot spell the candidate gives it -inf);
  3. score(c) = mean_m log p_m(c) + alpha * log P_LM(c) + beta * |c|; alpha, beta
     and the member subset are chosen on validation, then test is decoded once.
The LM is fitted on training transcriptions only (and optional --lm-text files,
which are refused if they contain a test line verbatim).
"""
import argparse
import csv
import itertools
import json
import os
import time
from multiprocessing import get_context

import numpy as np
import torch
import torch.nn.functional as F

from . import data as D
from . import metrics as Me
from .enhance_decode import Dec, logprobs, overlap_check

from ._compat import REPO_ROOT  # noqa: E402  (source checkout, else the working directory)
_W = {}   # fork-shared state for the beam-search workers (the LM is not pickled per job)


def _nbest(i):
    lp, tok = _W["lps"][i], _W["tok"]
    return Dec.prefix_beam_search(lp, lp.shape[0], tok.chars, tok.BLANK, _W["beam"], _W["lm"],
                                  _W["alpha"], _W["beta"], -9.0, 1, _W["n"])


def member_nbest(lps, tok, lm, beam, alpha, beta, n, workers):
    _W.update(lps=lps, tok=tok, lm=lm, beam=beam, alpha=alpha, beta=beta, n=n)
    with get_context("fork").Pool(workers) as p:
        out = p.map(_nbest, range(len(lps)), chunksize=8)
    return [[h["text"] for h in hs] for hs in out]


def _lm_score(texts):
    lm = _W["lm"]
    out = []
    for t in texts:
        s, ctx = 0.0, ""
        for ch in t:
            s += lm.logp(ctx, ch)
            ctx = (ctx + ch)[-(lm.order - 1):]
        out.append(s)
    return out


@torch.no_grad()
def ctc_scores(lp, tok, cands, device):
    """log p(c | x) under one member for every candidate of one line."""
    T = lp.shape[0]
    enc, ok = [], []
    for c in cands:
        e = tok.encode(c)
        good = len(c) > 0 and tok.UNK not in e
        enc.append(e if good else [2])
        ok.append(good)
    tgt = torch.tensor(list(itertools.chain.from_iterable(enc)), dtype=torch.long)
    tl = torch.tensor([len(e) for e in enc], dtype=torch.long)
    x = torch.from_numpy(lp).to(device)[:, None, :].expand(T, len(cands), lp.shape[1]).contiguous()
    nll = F.ctc_loss(x, tgt.to(device), torch.full((len(cands),), T, dtype=torch.long),
                     tl, blank=tok.BLANK, reduction="none", zero_infinity=False)
    s = (-nll).cpu().numpy()
    s[~np.array(ok)] = -np.inf
    s[~np.isfinite(s)] = -1e9
    return s


def build(dataset, split, members, device, cache, lm, beam, a0, b0, nbest, workers):
    per_member, refs = [], None
    for ck in members:
        lp, r, tok = logprobs(ck, dataset, split, device, cache)
        if refs is None:
            refs = list(r)
        elif list(r) != refs:
            # an archive built at another canvas may store the same lines in another
            # order (Saint Gall); align this member to the first member's order
            from collections import defaultdict
            pos = defaultdict(list)
            for j, t in enumerate(r):
                pos[t].append(j)
            if sorted(r) != sorted(refs):
                raise SystemExit(f"{ck}: {split} transcriptions differ from the first member's")
            lp = lp[[pos[t].pop(0) for t in refs]]
        per_member.append((lp, tok))
    t0 = time.time()
    cands = [set() for _ in refs]
    for lp, tok in per_member:
        for i, hs in enumerate(member_nbest(list(lp), tok, lm, beam, a0, b0, nbest, workers)):
            cands[i].update(h for h in hs if h)
    cands = [sorted(c) or [""] for c in cands]
    print(f"  {split}: {sum(map(len, cands)) / len(cands):.1f} candidates per line "
          f"({time.time() - t0:.0f}s)", flush=True)
    S = [np.stack([ctc_scores(lp[i], tok, cands[i], device) for lp, tok in per_member])
         for i in range(len(refs))]                                   # (members, n_cand)
    _W["lm"] = lm
    with get_context("fork").Pool(workers) as p:
        LMS = p.map(_lm_score, cands, chunksize=16)
    return refs, cands, S, [np.array(x) for x in LMS]


def pick(cands, S, LMS, members_idx, alpha, beta):
    out = []
    for c, s, l in zip(cands, S, LMS):
        tot = s[members_idx].mean(0) + alpha * l + beta * np.array([len(x) for x in c])
        out.append(c[int(np.argmax(tot))])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--lm-source", default="corpus", choices=("corpus", "combined"))
    ap.add_argument("--lm-text", nargs="*", default=[])
    ap.add_argument("--lm-no-train-text", action="store_true",
                    help="CONTROL ONLY: fit the LM on the --lm-text files alone, without the training transcriptions")
    ap.add_argument("--lm-max-lines", type=int, default=200000)
    ap.add_argument("--order", type=int, default=8)
    ap.add_argument("--beam", type=int, default=16)
    ap.add_argument("--nbest", type=int, default=8)
    ap.add_argument("--gen-alpha", type=float, default=0.2, help="LM weight inside each member's search")
    ap.add_argument("--gen-beta", type=float, default=0.5)
    ap.add_argument("--alphas", default="0.0,0.1,0.2,0.3,0.4,0.6,0.8,1.0")
    ap.add_argument("--betas", default="0.0,0.25,0.5,1.0,1.5,2.0,3.0")
    ap.add_argument("--subsets", action="store_true", help="also search member subsets on validation")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "results"))
    a = ap.parse_args(argv)

    import h5py
    device = torch.device("cpu" if a.gpu < 0 or not torch.cuda.is_available() else f"cuda:{a.gpu}")
    cache = os.path.join(a.out_dir, "enhance_cache")
    with h5py.File(D.archive_path(a.dataset if a.lm_source == "corpus" else "combined"), "r") as h:
        lm_lines = [g.decode("utf-8") for g in h["train"]["gt"][:]]
    with h5py.File(D.archive_path(a.dataset), "r") as h:
        test_refs = [g.decode("utf-8") for g in h["test"]["gt"][:]]
    extra = []
    for f in a.lm_text:
        extra += [l.rstrip("\n") for l in open(f, encoding="utf-8") if l.strip()][: a.lm_max_lines]
    # the official training split may repeat stock test lines (RIMES letter closings);
    # that is part of the benchmark. Only EXTERNAL text is gated.
    frac, hit, n = overlap_check(extra, test_refs) if extra else (0.0, 0, 0)
    tfrac, thit, _ = overlap_check(lm_lines, test_refs)
    if a.lm_no_train_text:
        if not extra:
            raise SystemExit("--lm-no-train-text needs --lm-text")
        lm_lines = []
    lm_lines += extra
    print(f"[{a.tag}] LM text {len(lm_lines)} lines; external-text verbatim test-line overlap {hit}/{n}; "
          f"official train/test line repeats {thit}", flush=True)
    if frac > 0.0:
        raise SystemExit("external LM text contains test lines; refusing to decode")
    lm = Dec.CharLM(a.order).fit(lm_lines)

    data = {}
    for split in ("valid", "test"):
        data[split] = build(a.dataset, split, a.members, device, cache, lm, a.beam,
                            a.gen_alpha, a.gen_beta, a.nbest, a.workers)
    refs, cands, S, LMS = data["valid"]
    alphas = [float(x) for x in a.alphas.split(",")]
    betas = [float(x) for x in a.betas.split(",")]
    M = len(a.members)
    subsets = [tuple(range(M))]
    if a.subsets:
        subsets = [s for k in range(1, M + 1) for s in itertools.combinations(range(M), k)]
    rows = []
    for sub in subsets:
        for al in alphas:
            for be in betas:
                h = pick(cands, S, LMS, list(sub), al, be)
                rows.append(dict(members=list(sub), alpha=al, beta=be, cer=Me.cer(refs, h), wer=Me.wer(refs, h)))
    best = min(rows, key=lambda r: (r["cer"], r["wer"]))
    single = {i: min((r for r in rows if r["members"] == [i]), key=lambda r: r["cer"]) for i in range(M)} \
        if a.subsets else {}
    oracle = Me.cer(refs, [min(c, key=lambda x: Me.cer([r], [x])) for r, c in zip(refs, cands)])
    v_hyps = pick(cands, S, LMS, best["members"], best["alpha"], best["beta"])
    t_refs, t_c, t_S, t_L = data["test"]
    hyps = pick(t_c, t_S, t_L, best["members"], best["alpha"], best["beta"])
    res = dict(tag=a.tag, dataset=a.dataset, members=[os.path.abspath(m) for m in a.members],
               result_class="ENHANCED_DECODING", lm=dict(source=a.lm_source, extra=a.lm_text, order=a.order,
               n_lines=len(lm_lines), external_test_line_overlap=frac, train_test_line_repeats=tfrac), chosen=best, valid_oracle_cer=oracle,
               valid_best_single={str(k): v for k, v in single.items()},
               test=dict(cer=Me.cer(t_refs, hyps), wer=Me.wer(t_refs, hyps), ser=Me.ser(t_refs, hyps)),
               grid_size=len(rows))
    if a.lm_no_train_text:
        res["lm"]["train_text"] = False
    ce, cl, we, wl = Me.per_line_counts(t_refs, hyps)
    res["cer_ci95"] = list(Me.bootstrap_ci(ce, cl))
    stem = os.path.join(a.out_dir, f"ens_{a.tag}")
    json.dump(res, open(stem + ".json", "w"), indent=1)
    np.savez_compressed(stem + "_test_counts.npz", char_err=ce, char_len=cl, word_err=we, word_len=wl)
    with open(stem + "_test_preds.tsv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["idx", "ref", "hyp"])
        for i, (r, h) in enumerate(zip(t_refs, hyps)):
            w.writerow([i, r, h])
    with open(stem + "_valid_preds.tsv", "w", newline="") as fh:   # input of the T5 stage's threshold sweep
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["idx", "ref", "hyp"])
        for i, (r, h) in enumerate(zip(refs, v_hyps)):
            w.writerow([i, r, h])
    print(f"[{a.tag}] chosen members {best['members']} alpha {best['alpha']} beta {best['beta']}: "
          f"valid CER {100*best['cer']:.2f} (oracle {100*oracle:.2f}); test CER {100*res['test']['cer']:.2f} "
          f"WER {100*res['test']['wer']:.2f}", flush=True)
    return res


if __name__ == "__main__":
    main()

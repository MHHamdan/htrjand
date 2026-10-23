"""Decode one split once and write the result files behind every reported number.

    python -m htrjand.evaluate --ckpt runs/iam_paper_teacher_s1/best.pth --split test --gpu 0

Writes, under --out-dir (default <repo>/results):
    <tag>_<split>.json        CER / WER / SER, 95 % bootstrap CI, parameter counts, latency
    <tag>_<split>_preds.tsv   idx, reference, hypothesis
    <tag>_<split>_counts.npz  per-line error counts for paired tests
Decoding is greedy best-path CTC (the paper names no other decoder). No lexicon,
no language model, no post-processing: every number is a raw recognition result.
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import data as D
from . import metrics as Me
from . import models as M

from ._compat import REPO_ROOT  # noqa: E402  (source checkout, else the working directory)


@torch.no_grad()
def decode_split(model, dataset, tok, device, batch_size=32, workers=4):
    """Greedy transcriptions and a per-line error estimate
    eps_hat = 1 - mean_t max_c p(c | x_t), the recogniser's own confidence, used
    by the post-processing gate (T5 post-processing)."""
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                    collate_fn=D.collate, pin_memory=device.type == "cuda")
    hyps, eps = [], []
    for img, _, _ in dl:
        logits = model(img.to(device, non_blocking=True))
        hyps += tok.ctc_greedy_decode(logits)
        eps += (1.0 - torch.softmax(logits, -1).max(-1).values.mean(1)).cpu().tolist()
    return list(dataset.labels), hyps, eps


@torch.no_grad()
def latency_ms(model, device, input_size=D.INPUT_SIZE, n=100, warmup=20):
    """Batch-1 forward time on one 68 x 864 line, after warm-up."""
    x = torch.randn(1, 1, *input_size, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(n):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - t0) / n * 1000.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=("valid", "test"))
    ap.add_argument("--dataset", default=None, help="override the checkpoint's dataset (cross-corpus decoding)")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--gpu", type=int, default=0, help="-1 = CPU")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "results"))
    ap.add_argument("--tag", default=None, help="default: name of the checkpoint's directory")
    ap.add_argument("--latency", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)

    device = torch.device("cpu" if a.gpu < 0 or not torch.cuda.is_available() else f"cuda:{a.gpu}")
    model, ck = M.from_checkpoint(a.ckpt)
    model.to(device)
    tok = D.Tokenizer(ck["charset"])
    dataset = a.dataset or ck["dataset"]
    ds = D.LineDataset(D.resolve_for(dataset, ck["model_cfg"]["input_size"]), a.split, tok, normalisation=ck.get("normalisation", "paper"),
                       root=a.data_dir, limit=a.limit)
    refs, hyps, eps = decode_split(model, ds, tok, device, a.batch_size, a.workers)
    ce, cl, we, wl = Me.per_line_counts(refs, hyps)
    lo, hi = Me.bootstrap_ci(ce, cl)
    tag = a.tag or os.path.basename(os.path.dirname(os.path.abspath(a.ckpt)))
    res = dict(tag=tag, result_class=ck.get("result_class", "HTRJAND_RUN"),
               dataset=dataset, trained_on=ck["dataset"], split=a.split, preset=ck["preset"],
               seed=ck.get("seed"), epoch=ck.get("epoch"), best_val_cer=ck.get("val_cer"),
               n_lines=len(refs), decoder="greedy", post_processing="none",
               cer=float(ce.sum() / max(cl.sum(), 1)), wer=float(we.sum() / max(wl.sum(), 1)),
               ser=Me.ser(refs, hyps), cer_ci95=[lo, hi],
               params_total=M.count_params(model), params_deployed=model.deployed_parameters(),
               ckpt=os.path.abspath(a.ckpt), model_cfg=ck["model_cfg"])
    if a.latency:
        res["latency_bs1_ms"] = latency_ms(model, device, tuple(ck["model_cfg"]["input_size"]))
        res["latency_device"] = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    os.makedirs(a.out_dir, exist_ok=True)
    stem = os.path.join(a.out_dir, f"{tag}_{a.split}")
    json.dump(res, open(stem + ".json", "w"), indent=1)
    np.savez_compressed(stem + "_counts.npz", char_err=ce, char_len=cl, word_err=we, word_len=wl)
    with open(stem + "_preds.tsv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["idx", "eps_hat", "ref", "hyp"])
        for i, (r, h, e) in enumerate(zip(refs, hyps, eps)):
            w.writerow([i, f"{e:.6f}", r, h])
    print(json.dumps({k: v for k, v in res.items() if k not in ("ckpt", "model_cfg")}, indent=1))
    return res


if __name__ == "__main__":
    main()

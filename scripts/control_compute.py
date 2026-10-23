#!/usr/bin/env python3
"""CONTROL C4 (review M6): MACs per line, T5 corrector MACs, LM memory, and forward latency.

MACs: an analytic counter (forward hooks on Conv2d / Linear / LSTM / MultiheadAttention, plus the
two Proxima matmuls) that is unit-tested against torch.utils.flop_counter.FlopCounterMode on every
module type FlopCounterMode understands (it does not count nn.LSTM, which is why the analytic counter
exists).  Element-wise work (sigmoid gates, BN, LN, softmax, SE pooling) is not counted - it is
< 1 % of the MACs.  Deployed path only (the auxiliary CTC head is training-only).

T5: FlopCounterMode around the real t5post generate() (beam 4, context window of the corrector)
on the first N test lines of a corpus, divided by N.

Latency: median wall time of >= 200 forward passes after 20 warm-up passes, batch 1 and 16,
CPU (torch.set_num_threads(8)) and GPU 0 (CUDA_VISIBLE_DEVICES=0, synchronised), eval mode,
no_grad, random input of the declared canvas.  Machine load (uptime, nvidia-smi) is recorded
before and after each device block.

    python scripts/control_compute.py --test          # unit test of the counter
    python scripts/control_compute.py                 # writes results/control_compute.json
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from htrjand import models as M  # noqa: E402

MODELS = {  # role: checkpoint (validation-selected / deployed models; 68x864 = combined-archive recipe)
    "Student 68x864": "runs/combined_final_student_kd_s1/best.pth",
    "Student 128x1632": "runs/ft_iam_c128_student_kd_s1/best.pth",
    "Teacher 68x864": "runs/combined_final_teacher_s1/best.pth",
    "Teacher 128x1632": "runs/ft_iam_c128_syn_teacher_s2/best.pth",
}


def count_macs(model, x, lstm_only=False):
    """Multiply-accumulates of one forward pass, by module type."""
    acc = {"conv": 0, "linear": 0, "lstm": 0, "attention_matmul": 0}
    hooks = []

    def conv(m, i, o):
        acc["conv"] += o.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]

    def lin(m, i, o):
        acc["linear"] += (o.numel() // m.out_features) * m.in_features * m.out_features

    def lstm(m, i, o):
        B, T = i[0].shape[0], i[0].shape[1]
        dirs = 2 if m.bidirectional else 1
        n = 0
        for layer in range(m.num_layers):
            inp = m.input_size if layer == 0 else m.hidden_size * dirs
            n += dirs * T * 4 * m.hidden_size * (inp + m.hidden_size)
        acc["lstm"] += B * n

    def mha(m, i, o):
        q = i[0]
        B, T, d = q.shape
        S = i[1].shape[1]
        acc["linear"] += B * (T * d * d + 2 * S * d * d + T * d * d)   # in-proj Q,K,V + out-proj
        acc["attention_matmul"] += B * 2 * T * S * d                  # QK^T and AV

    def prox(m, i, o):
        B, T, d = i[0].shape
        acc["attention_matmul"] += B * 2 * T * d                      # q.K^T and w.V (single query)

    for mod in model.modules():
        if isinstance(mod, nn.Conv2d):
            hooks.append(mod.register_forward_hook(conv))
        elif isinstance(mod, nn.Linear) and not isinstance(mod, nn.modules.linear.NonDynamicallyQuantizableLinear):
            hooks.append(mod.register_forward_hook(lin))
        elif isinstance(mod, nn.LSTM):
            hooks.append(mod.register_forward_hook(lstm))
        elif isinstance(mod, nn.MultiheadAttention):
            hooks.append(mod.register_forward_hook(mha))
        elif isinstance(mod, M.ProximaAttention):
            hooks.append(mod.register_forward_hook(prox))
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    acc["total"] = sum(acc.values())
    return acc


def flopcounter_macs(model, x):
    # train mode: nn.MultiheadAttention's eval-mode fast path (aten._native_multi_head_attention)
    # is invisible to FlopCounterMode; the arithmetic is identical in both modes
    was = model.training
    model.train()
    with torch.no_grad(), FlopCounterMode(display=False) as f:
        model(x)
    model.train(was)
    return f.get_total_flops() // 2


def test():
    """The analytic counter equals FlopCounterMode on everything FlopCounterMode counts, and the
    LSTM formula equals an explicit unrolled cell."""
    torch.manual_seed(0)
    for preset in ("paper_student", "paper_teacher", "abl_no_attention", "abl_mha_only", "abl_proxima_only"):
        m = M.build(preset, 80, input_size=(68, 864)).eval()
        x = torch.zeros(2, 1, 68, 864)
        a = count_macs(m, x)
        f = flopcounter_macs(m, x)
        ok = abs((a["total"] - a["lstm"]) - f) <= 0.001 * f
        print(f"{preset:18s} analytic(no LSTM) {a['total'] - a['lstm']:>14,d}  FlopCounterMode {f:>14,d}  {'OK' if ok else 'MISMATCH'}")
        assert ok
    # LSTM: unrolled cell with explicit matmuls, counted by FlopCounterMode
    l = nn.LSTM(10, 20, 2, batch_first=True, bidirectional=True)
    x = torch.zeros(3, 7, 10)
    a = count_macs(nn.Sequential(l), x)["lstm"]
    with torch.no_grad(), FlopCounterMode(display=False) as f:
        for layer in range(2):
            inp = 10 if layer == 0 else 40
            for _ in range(2):
                h = torch.zeros(3, 20)
                Wi, Wh = torch.zeros(80, inp), torch.zeros(80, 20)
                for t in range(7):
                    _ = torch.zeros(3, inp) @ Wi.T + h @ Wh.T
    assert a == f.get_total_flops() // 2, (a, f.get_total_flops() // 2)
    print(f"LSTM analytic {a:,d} == unrolled {f.get_total_flops() // 2:,d}  OK")


def load():
    out = uptime()
    return dict(uptime=out, nvidia_smi=subprocess.run(
        "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader",
        shell=True, capture_output=True, text=True).stdout.strip().splitlines())


def uptime():
    return subprocess.run("uptime", shell=True, capture_output=True, text=True).stdout.strip()


def latency(model, shape, device, batch, runs, warm=20):
    model = model.to(device).eval()
    x = torch.randn(batch, 1, *shape, device=device)
    ts = []
    with torch.no_grad():
        for k in range(warm + runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if k >= warm:
                ts.append(time.perf_counter() - t0)
    ts = np.array(ts) * 1000
    return dict(batch=batch, runs=runs, median_ms_per_batch=float(np.median(ts)),
                median_ms_per_line=float(np.median(ts) / batch), p10_ms=float(np.percentile(ts, 10)),
                p90_ms=float(np.percentile(ts, 90)))


def t5_macs(corrector, dataset, n, device):
    from htrjand import t5post as T
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(corrector)
    model = T5ForConditionalGeneration.from_pretrained(corrector).to(device).eval()
    ctx = int(json.load(open(os.path.join(corrector, "train_meta.json"))).get("context", True))
    import csv
    with open(os.path.join(ROOT, f"results/ens_iam_ft_iam_c128_student_kd_s1_wiki_test_preds.tsv")) as fh:
        hyps = [r["hyp"] for r in csv.DictReader(fh, delimiter="\t")][:n]
    src = T.with_context(dataset, [T.normalise(h) for h in hyps], ctx)
    enc_len = float(np.mean([len(tok(s).input_ids) for s in src]))
    with torch.no_grad(), FlopCounterMode(display=False) as f:
        outs, _ = T.generate(model, tok, src, device, batch=1, beams=4)
    dec_len = float(np.mean([len(tok(o).input_ids) for o in outs]))
    t0 = time.perf_counter()
    T.generate(model, tok, src, device, batch=1, beams=4)
    if device.type == "cuda":
        torch.cuda.synchronize()
    lat = (time.perf_counter() - t0) / n * 1000
    return dict(corrector=corrector, dataset=dataset, n_lines=n, beams=4, context_window=ctx,
                mean_source_tokens=enc_len, mean_output_tokens=dec_len,
                macs_per_line=f.get_total_flops() / 2 / n, params=sum(p.numel() for p in model.parameters()),
                latency_ms_per_line_batch1=lat, device=str(device),
                note="FlopCounterMode over transformers generate (beam 4, batch 1, KV cache on); MACs = FLOPs/2")


def lm_memory():
    """Resident-memory increase of fitting the largest LM (IAM, 10-gram, train + 400k Wikipedia lines)."""
    import gc
    import h5py
    import psutil
    from htrjand import data as D
    from htrjand.enhance_decode import Dec
    with h5py.File(D.archive_path("iam"), "r") as h:
        train = [g.decode("utf-8") for g in h["train"]["gt"][:]]
    ext = [l.rstrip("\n") for l in open(os.path.join(ROOT, "synth/text_en.txt"), encoding="utf-8") if l.strip()][:400000]
    lines = train + ext
    gc.collect()
    p = psutil.Process()
    r0 = p.memory_info().rss
    lm = Dec.CharLM(10).fit(lines)
    gc.collect()
    r1 = p.memory_info().rss
    n = sum(len(c) for c in lm.counts)
    return dict(lm="IAM combined (train + 400k Wikipedia lines), order 10", rss_increase_bytes=r1 - r0, ngram_entries=n,
                note="Python dict-of-tuples CharLM as used by ensemble_rescore; a compact trie/ARPA would be far smaller")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--skip-cpu", action="store_true")
    a = ap.parse_args()
    if a.test:
        return test()
    torch.set_num_threads(a.threads)
    res = dict(control="C4 compute (review M6)", label="CONTROL", torch=torch.__version__,
               cpu=subprocess.run("lscpu | grep 'Model name'", shell=True, capture_output=True, text=True).stdout.strip(),
               threads=a.threads, models={})
    for role, ck in MODELS.items():
        model, c = M.from_checkpoint(os.path.join(ROOT, ck))
        shape = tuple(c["input_size"])
        x = torch.zeros(1, 1, *shape)
        macs = count_macs(model, x)
        res["models"][role] = dict(ckpt=ck, input_size=list(shape), n_frames=model.n_frames,
                                   params_deployed=model.deployed_parameters(), macs_per_line=macs,
                                   gmacs_per_line=macs["total"] / 1e9, flopcounter_non_lstm_macs=flopcounter_macs(model, x))
        print(role, shape, f"{macs['total']/1e9:.3f} GMAC/line", macs, flush=True)
    res["load_before_gpu"] = load()
    dev = torch.device("cuda")
    for role, ck in MODELS.items():
        model, c = M.from_checkpoint(os.path.join(ROOT, ck))
        res["models"][role]["latency_gpu0"] = [latency(model, tuple(c["input_size"]), dev, b, a.runs) for b in (1, 16)]
        print(role, "GPU", res["models"][role]["latency_gpu0"], flush=True)
    res["load_after_gpu"] = load()
    res["t5"] = [t5_macs(os.path.join(ROOT, "postproc/t5_ext_ctx1"), "iam", 50, dev)]
    print(res["t5"], flush=True)
    torch.cuda.empty_cache()
    if not a.skip_cpu:
        res["load_before_cpu"] = load()
        for role, ck in MODELS.items():
            model, c = M.from_checkpoint(os.path.join(ROOT, ck))
            res["models"][role]["latency_cpu"] = [latency(model, tuple(c["input_size"]), torch.device("cpu"), b, a.runs)
                                                  for b in (1, 16)]
            print(role, "CPU", res["models"][role]["latency_cpu"], flush=True)
        res["load_after_cpu"] = load()
    res["lm_memory"] = lm_memory()
    print(res["lm_memory"], flush=True)
    json.dump(res, open(os.path.join(ROOT, "results", "control_compute.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

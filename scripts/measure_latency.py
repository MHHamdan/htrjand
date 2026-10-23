#!/usr/bin/env python3
"""CPU latency of the released models, batch 1, one line image, preprocessing included.

    CUDA_VISIBLE_DEVICES= python scripts/measure_latency.py --hf-root ../htrjand-release-hf \
        --image examples/ood_2_gettysburg_a.png --threads 4 --out results/release_eval/latency.json

For every (repo, variant) of the release and every decoder (greedy, lm, greedy + T5, lm + T5) the
median and 90th percentile of --n timed calls of `HTRJANDModel.transcribe` after --warmup calls.
One-off load times (model, LM file, T5 corrector) are reported separately.
"""
import argparse
import json
import os
import platform
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def cpu_name():
    try:
        for l in open("/proc/cpuinfo"):
            if l.startswith("model name"):
                return l.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "CPU"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-root", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--models", nargs="*", default=None, help="<repo>/<variant> (default: all)")
    ap.add_argument("--no-t5", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    import torch
    torch.set_num_threads(a.threads)
    from htrjand import HTRJANDModel
    from htrjand import release as REL
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from export_hf import RELEASE
    targets = [(r, v) for r, v, *_ in RELEASE if not a.models or f"{r}/{v}" in a.models]
    rows, loads, corr = [], {}, None
    for repo, variant in targets:
        t0 = time.perf_counter()
        m = HTRJANDModel.from_pretrained(os.path.join(a.hf_root, repo), variant=variant, device="cpu")
        loads[f"{repo}/{variant}"] = round(time.perf_counter() - t0, 2)
        lk = "lm_file_" + m.lm_settings()["language"]
        if lk not in loads:                      # first load of that language (later ones hit the cache)
            t0 = time.perf_counter()
            m.load_lm()
            loads[lk] = round(time.perf_counter() - t0, 2)
        decs = [("greedy", False), ("lm", False)]
        if not a.no_t5:
            if corr is None:
                t0 = time.perf_counter()
                corr = m.load_corrector()
                loads["t5_corrector"] = round(time.perf_counter() - t0, 2)
            m._corrector = corr
            decs += [("greedy", True), ("lm", True)]
        for dec, t5 in decs:
            for _ in range(a.warmup):
                m.transcribe(a.image, decoder=dec, t5=t5)
            ts = []
            for _ in range(a.n):
                t0 = time.perf_counter()
                m.transcribe(a.image, decoder=dec, t5=t5)
                ts.append(1000 * (time.perf_counter() - t0))
            ts.sort()
            rows.append(dict(repo=repo, variant=variant, decoder=dec + (" + T5" if t5 else ""),
                             median_ms=statistics.median(ts), p90_ms=ts[int(0.9 * (len(ts) - 1))]))
            print(f"{repo}/{variant:<11} {rows[-1]['decoder']:<12} median {rows[-1]['median_ms']:7.1f} ms  "
                  f"p90 {rows[-1]['p90_ms']:7.1f} ms", flush=True)
    res = dict(cpu=cpu_name(), threads=a.threads, torch=torch.__version__, image=os.path.basename(a.image),
               n=a.n, warmup=a.warmup, load_seconds=loads, rows=rows,
               note="shared machine; numbers are indicative. Preprocessing (decode, polarity, crop, resize) included.")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(loads))


if __name__ == "__main__":
    main()

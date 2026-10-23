#!/usr/bin/env python3
"""Check that every exported model reproduces its training checkpoint exactly.

    python scripts/verify_export.py --hf-root ../htrjand-release-hf --runs-dir runs \
        --old-code /path/to/source-checkout --n 64 [--raw-dir raw]

For each model of scripts/export_hf.py RELEASE:
  * the ORIGINAL checkpoint is loaded with the ORIGINAL code (`htrjand_paper_v1.models.from_checkpoint`
    of the source checkout given by --old-code) and decodes the first --n VALIDATION lines of the
    128 x 1632 archive of its corpus through `htrjand_paper_v1.data.LineDataset` (the evaluation path);
  * the EXPORTED model is loaded with `htrjand.HTRJANDModel.from_pretrained` (config.json only, strict
    safetensors load, SHA-256 checked) and decodes the same stored canvases through
    `htrjand.inference.normalise_canvas`;
  * logits must be bit-identical (torch.equal) and the greedy strings equal.
With --raw-dir, `htrjand.inference.prepare_line(crop=False)` on raw IAM line images is also compared
with the original `prepare_data.preprocess` (the archive builder).
Needs the licensed archives ($HTRJAND_DATA_DIR); exits with a clear message when they are absent.
"""
import argparse
import importlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ARCHIVE = dict(iam="iam", rimes="rimes", bentham="bentham", saintgall="saintgall_nopt", washington="washington")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-root", required=True)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--old-code", required=True, help="checkout containing htrjand_paper_v1/ (commit ebbfbcd)")
    ap.add_argument("--data-dir", default=os.environ.get("HTRJAND_DATA_DIR"))
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--out", default=None, help="write the report as JSON")
    a = ap.parse_args(argv)
    if not a.data_dir or not os.path.isdir(a.data_dir):
        sys.exit("set --data-dir or $HTRJAND_DATA_DIR to the directory holding the *_128x1632.hdf5 archives")
    import numpy as np
    import torch
    sys.path.insert(1, os.path.abspath(a.old_code))
    oldD = importlib.import_module("htrjand_paper_v1.data")
    oldM = importlib.import_module("htrjand_paper_v1.models")
    from htrjand import HTRJANDModel
    from htrjand.inference import normalise_canvas
    from export_hf import RELEASE, CORPORA
    torch.set_num_threads(8)
    rows, ok_all = [], True
    for repo, variant, run, role, seed, corpus, _ in RELEASE:
        old, ck = oldM.from_checkpoint(os.path.join(a.runs_dir, run, "best.pth"))
        new = HTRJANDModel.from_pretrained(os.path.join(a.hf_root, repo), variant=variant, device="cpu")
        assert new.tokenizer.chars == oldD.Tokenizer(ck["charset"]).chars
        for ds in ([corpus] if corpus else CORPORA):
            arch = os.path.join(a.data_dir, f"{ARCHIVE[ds]}_128x1632.hdf5")
            if not os.path.exists(arch):
                sys.exit(f"{arch} not found")
            otok = oldD.Tokenizer(ck["charset"])
            dset = oldD.LineDataset(arch, "valid", otok, normalisation=ck.get("normalisation", "paper"), limit=a.n)
            x_old = torch.stack([dset[i][0] for i in range(len(dset))])
            x_new = torch.from_numpy(np.stack([normalise_canvas(dset.images[i], new.normalisation)[None]
                                               for i in range(len(dset))]))
            with torch.no_grad():
                lo = old(x_old)
                ln = new.logits(x_new)
            same_in = torch.equal(x_old, x_new)
            same_logits = torch.equal(lo, ln)
            go, gn = otok.ctc_greedy_decode(lo), new.greedy(torch.log_softmax(ln, -1).numpy())
            same_text = go == gn
            ok = same_in and same_logits and same_text
            ok_all &= ok
            rows.append(dict(model=f"{repo}/{variant}", run=run, corpus=ds, archive=os.path.basename(arch),
                             lines=len(dset), inputs_identical=same_in, logits_identical=same_logits,
                             max_abs_logit_diff=float((lo - ln).abs().max()), greedy_identical=same_text,
                             greedy_cer_vs_reference=None))
            from htrjand import metrics as Me
            rows[-1]["greedy_cer_vs_reference"] = round(100 * Me.cer(dset.labels, gn), 3)
            print(f"{'OK ' if ok else 'FAIL'} {repo}/{variant:<11} {ds:<11} {len(dset)} lines  inputs={same_in} "
                  f"logits={same_logits} (max diff {rows[-1]['max_abs_logit_diff']:.1e}) greedy={same_text}", flush=True)
    if a.raw_dir:
        import glob
        oldP = importlib.import_module("htrjand_paper_v1.prepare_data")
        from htrjand.inference import prepare_line
        paths = sorted(glob.glob(os.path.join(a.raw_dir, "iam", "lines", "*", "*", "*.png")))[: a.n]
        same, flipped, diff = 0, [], []
        for p in paths:
            r = prepare_line(p, (128, 1632), crop=False, check=False)
            if r.inverted_input:          # light-on-dark input: inverted first by design, not comparable
                flipped.append(os.path.basename(p)); continue
            if np.array_equal(oldP.preprocess(p, (128, 1632)), r.canvas):
                same += 1
            else:
                diff.append(os.path.basename(p))
        rows.append(dict(check="prepare_line(crop=False) == prepare_data.preprocess on raw IAM lines",
                         lines=len(paths), identical=same, differing=diff, light_on_dark_inputs_skipped=flipped))
        ok_all &= not diff
        print(f"{'OK ' if not diff else 'FAIL'} raw-image preprocessing identical on {same}/{len(paths) - len(flipped)} "
              f"IAM lines ({len(flipped)} light-on-dark input(s) skipped: {flipped})")
    if a.out:
        json.dump(dict(ok=ok_all, rows=rows), open(a.out, "w"), indent=1)
    print("VERIFY:", "PASS" if ok_all else "FAIL")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()

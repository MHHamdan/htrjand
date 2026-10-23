#!/usr/bin/env python3
"""CONTROL C1/C4 (review M6, M7): size of every character LM used by the Student LM paths.

For each corpus, fits the same CharLM (same order, same text, same max-lines) as
ensemble_rescore for the three LM sources (training transcriptions only, Wikipedia only,
combined), and reports the number of distinct (context, char) n-gram entries per order,
the number of distinct contexts per order, the external-text size, and the size of the
model serialised with CharLM.save (JSON, written to a temporary file and measured; gzip
size also reported).  Writes results/control_lm_sizes.json.
"""
import gzip
import json
import os
import shutil
import sys
import tempfile
import time

import h5py

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from htrjand import data as D            # noqa: E402
from htrjand.enhance_decode import Dec   # noqa: E402

LM = dict(iam=("iam", "synth/text_en.txt", 400000, 10), rimes=("rimes", "synth/text_fr.txt", 200000, 9),
          bentham=("bentham", "synth/text_en.txt", 200000, 9), saintgall=("saintgall_nopt", "synth/text_la.txt", 200000, 9),
          washington=("washington", "synth/text_en.txt", 200000, 9))


def size(lm, tmp):
    p = os.path.join(tmp, "lm.json")
    lm.save(p)
    raw = os.path.getsize(p)
    with open(p, "rb") as fi, gzip.open(p + ".gz", "wb", compresslevel=6) as fo:
        shutil.copyfileobj(fi, fo)
    gz = os.path.getsize(p + ".gz")
    os.remove(p); os.remove(p + ".gz")
    return raw, gz


def main():
    out = {}
    tmp = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
    for corpus, (ds, txt, nmax, order) in LM.items():
        with h5py.File(D.archive_path(ds), "r") as h:
            train = [g.decode("utf-8") for g in h["train"]["gt"][:]]
        ext = [l.rstrip("\n") for l in open(os.path.join(ROOT, txt), encoding="utf-8") if l.strip()][:nmax]
        out[corpus] = dict(dataset=ds, order=order, external_text=txt, external_lines=len(ext),
                           external_chars=sum(map(len, ext)), train_lines=len(train),
                           train_chars=sum(map(len, train)), sources={})
        for src, lines in (("train_only", train), ("wiki_only", ext), ("combined", train + ext)):
            t0 = time.time()
            lm = Dec.CharLM(order).fit(lines)
            raw, gz = size(lm, tmp)
            out[corpus]["sources"][src] = dict(
                n_lines=len(lines), ngram_entries_per_order={str(n): len(lm.counts[n]) for n in range(1, order + 1)},
                contexts_per_order={str(n): len(lm.totals[n]) for n in range(1, order + 1)},
                ngram_entries_total=sum(len(lm.counts[n]) for n in range(1, order + 1)),
                json_bytes=raw, json_gzip_bytes=gz, vocab=len(lm.vocab))
            print(corpus, src, out[corpus]["sources"][src]["ngram_entries_total"], raw, f"{time.time()-t0:.0f}s", flush=True)
            del lm
        json.dump(out, open(os.path.join(ROOT, "results", "control_lm_sizes.json"), "w"), indent=1)
    shutil.rmtree(tmp)


if __name__ == "__main__":
    main()

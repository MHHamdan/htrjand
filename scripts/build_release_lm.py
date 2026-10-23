#!/usr/bin/env python3
"""Build the character n-gram LM files of the public release from Wikipedia-derived text.

    python scripts/build_release_lm.py --lang en --text synth/text_en.txt --order 8 --min-count 2 \
        --out release_lm/charlm_en.gz

The released LMs are fitted on the Wikipedia-derived lines in synth/text_{en,fr,la}.txt ONLY
(CC BY-SA; see docs/RELEASE_LM_EVAL.md). An n-gram fitted on a benchmark's training
transcriptions effectively redistributes them, so the corpus LM of the paper's decoding
experiments is not released; `scripts/build_corpus_lm.py` rebuilds it for users who hold the
archives.

`--grid` fits once at the largest order and reports, for every (order, min_count) pair, the
number of stored entries and the compressed file size, without keeping the files.
"""
import argparse
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from htrjand.decode import CompactCharLM  # noqa: E402


def read_lines(path, max_lines=None):
    out = [l.rstrip("\n") for l in open(path, encoding="utf-8") if l.strip()]
    return out[:max_lines] if max_lines else out


def derive(full, order, min_count, prune_from):
    """Order-k model from an order-K >= k fit: the first k tables, pruned per order."""
    m = CompactCharLM(order, full.alpha)
    m.vocab, m.V = full.vocab, full.V
    for n in range(1, order + 1):
        keep = min_count if (min_count > 1 and n >= prune_from) else 1
        m.ngrams[n] = {k: c for k, c in full.ngrams[n].items() if c >= keep} if keep > 1 else full.ngrams[n]
        ctxs = {k[:-1] for k in m.ngrams[n]}
        m.totals[n] = {k: c for k, c in full.totals[n].items() if k in ctxs} if keep > 1 else full.totals[n]
    m.meta = dict(min_count=min_count, prune_from=prune_from)
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--text", nargs="+", required=True, help="one sentence fragment per line")
    ap.add_argument("--max-lines", type=int, default=None)
    ap.add_argument("--order", type=int, default=8)
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--prune-from", type=int, default=3, help="lowest order that is pruned")
    ap.add_argument("--out", default=None)
    ap.add_argument("--grid", default=None, metavar="ORDERS:MINCOUNTS", help="e.g. 6,7,8,9,10:1,2,3,5")
    ap.add_argument("--stats-out", default=None)
    ap.add_argument("--save-dir", default=None, help="with --grid: keep every file as o<order>m<min_count>_<lang>.gz")
    a = ap.parse_args(argv)
    lines = []
    for p in a.text:
        lines += read_lines(p, a.max_lines)
    orders = [a.order]
    mins = [a.min_count]
    if a.grid:
        o, m = a.grid.split(":")
        orders, mins = [int(x) for x in o.split(",")], [int(x) for x in m.split(",")]
    t0 = time.time()
    full = CompactCharLM(max(orders)).fit(lines)
    print(f"[{a.lang}] fitted order {max(orders)} on {len(lines)} lines "
          f"({sum(map(len, lines))} chars) in {time.time() - t0:.0f}s", flush=True)
    rows = []
    for o in orders:
        for mc in mins:
            m = derive(full, o, mc, a.prune_from)
            if a.grid and a.save_dir:
                os.makedirs(a.save_dir, exist_ok=True)
                path = os.path.join(a.save_dir, f"o{o}m{mc}_{a.lang}.gz")
            elif a.grid:
                with tempfile.NamedTemporaryFile(suffix=".gz", delete=False) as tf:
                    path = tf.name
            else:
                path = a.out
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            m.save(path, lang=a.lang, source=[os.path.basename(p) for p in a.text], n_lines=len(lines),
                   licence="CC BY-SA (Wikipedia-derived text); see docs/RELEASE_LM_EVAL.md")
            size = os.path.getsize(path)
            rows.append(dict(lang=a.lang, order=o, min_count=mc, prune_from=a.prune_from,
                             entries=m.n_entries(), bytes=size))
            print(json.dumps(rows[-1]), flush=True)
            if a.grid and not a.save_dir:
                os.remove(path)
    if a.stats_out:
        json.dump(rows, open(a.stats_out, "w"), indent=1)


if __name__ == "__main__":
    main()

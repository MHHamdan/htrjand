#!/usr/bin/env python3
"""Rebuild the paper's corpus + Wikipedia character LM (and the corpus training lexicon) locally.

    HTRJAND_DATA_DIR=/path/to/archives python scripts/build_corpus_lm.py --corpus iam \
        --wiki synth/text_en.txt --out my_lm/iam_corpus_wiki.gz [--lexicon-out my_lm/iam_lexicon.txt.gz]

The released LMs contain Wikipedia text only. The decoding experiments of the paper fitted the LM on
the corpus's TRAINING transcriptions plus Wikipedia lines (scripts/ft_student_paths.sh: IAM order 10
with 400,000 English lines; RIMES / Bentham / Saint Gall / Washington order 9 with 200,000 lines of
French / English / Latin / English). This script rebuilds exactly that model (CompactCharLM, the same
probabilities as decode.CharLM for every query the decoder makes) for users who hold the archives.

DO NOT REDISTRIBUTE the output: an n-gram of this order fitted on a few thousand lines is close to a
copy of the benchmark transcriptions, which are licensed for non-commercial research only.

Use it with a released model:  model.load_lm(path="my_lm/iam_corpus_wiki.gz") and pass the returned
object as `lm=` to `model.decode_lm`, or set alpha / beta from results/ens_<corpus>_<run>_wiki.json.
"""
import argparse
import gzip
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DEFAULTS = dict(iam=(10, 400000), rimes=(9, 200000), bentham=(9, 200000), saintgall=(9, 200000), washington=(9, 200000))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, choices=sorted(DEFAULTS))
    ap.add_argument("--wiki", required=True, help="Wikipedia-derived text, one line fragment per line")
    ap.add_argument("--order", type=int, default=None, help="default: the paper's order for the corpus")
    ap.add_argument("--max-lines", type=int, default=None, help="Wikipedia lines used (default: the paper's cap)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lexicon-out", default=None, help="also write the corpus TRAINING lexicon (T5 gate)")
    ap.add_argument("--data-dir", default=None)
    a = ap.parse_args(argv)
    if a.data_dir:
        os.environ["HTRJAND_DATA_DIR"] = a.data_dir
    from htrjand import t5post as P
    from htrjand.decode import CompactCharLM
    ds = "saintgall_nopt" if a.corpus == "saintgall" else a.corpus
    try:
        train = P.read_labels(ds, "train")
    except FileNotFoundError as e:
        sys.exit(f"needs the {ds} archive: {e}")
    order, cap = DEFAULTS[a.corpus]
    order, cap = a.order or order, a.max_lines or cap
    wiki = [l.rstrip("\n") for l in open(a.wiki, encoding="utf-8") if l.strip()][:cap]
    lm = CompactCharLM(order).fit(train + wiki)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    lm.save(a.out, corpus=a.corpus, n_train_lines=len(train), n_wiki_lines=len(wiki),
            licence="CONTAINS BENCHMARK TRANSCRIPTIONS: local use only, do not redistribute")
    print(f"{a.out}: order {order}, {len(train)} training + {len(wiki)} Wikipedia lines, {lm.n_entries():,} entries")
    if a.lexicon_out:
        lex = sorted(P.lexicon([P.normalise(t) for t in train]))
        with gzip.open(a.lexicon_out, "wt", encoding="utf-8") as f:
            f.write("\n".join(lex) + "\n")
        print(f"{a.lexicon_out}: {len(lex)} word types")


if __name__ == "__main__":
    main()

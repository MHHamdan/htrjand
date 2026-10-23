#!/usr/bin/env python3
"""Deterministic qualitative-example selection on the official TEST splits of the five corpora.

    python scripts/build_qualitative_examples.py        # -> results/qualitative_examples.json

Systems (seed 1, validation-selected Student of results/final_selection.json; no test-set choice):
  raw CTC        greedy CTC of the selected Student (ft_<ds>_c128_student_kd_s1, ft_washington_c128_student_s1,
                 nopt_sg_student_s1)
  char LM        CTC prefix beam search + character n-gram LM (ens_<ds>_..._wiki)
  T5 on char LM  gated T5-small corrector applied to the char-LM output (the selected T5 id)
  T5 on raw CTC  gated T5-small corrector applied to the raw CTC output (the selected T5 id)
  final          the decoding path with the lowest VALIDATION CER (exact ties -> the simpler path)
  base           HTR-JAND base (68x864) Student: combined_final_student_kd_s1 (five-corpus training, greedy CTC)
                 decoded on the corpus's test set; Saint Gall model trained with the original labels.
Line alignment: the selected-Student TSVs are in archive order (verified: ref == archive gt at every
index). The base TSVs come from the 68x864 archives, whose Saint Gall order differs, so they
are mapped to archive indices through the reference text (k-th occurrence -> k-th occurrence); the
mapping is required to be a bijection with identical references.

Per-line CER = edit distance / reference length (no normalisation). All rules are evaluated per corpus
unless the rule names a corpus. Ties are broken by the lowest archive index. Rules:
  low_cer        raw-CTC CER closest to the corpus's 10th percentile of per-line raw-CTC CER
                 (percentile over all test lines, numpy linear); candidates: references >= 30 chars
  median_cer     same, 50th percentile
  high_cer       same, 90th percentile
  long_line      IAM only: the longest reference (characters)
  historical     Saint Gall and Washington: the median_cer line of that corpus (same rule, flagged)
  spacing_punct  among lines with >= 3 raw-CTC errors: the largest share of edit operations that involve
                 a space, punctuation or symbol character (on either side of the alignment)
  rare_vocab     lines whose reference contains a word (maximal run of letters, case-sensitive) absent
                 from the corpus's TRAINING-split transcriptions; among them the raw-CTC CER closest to
                 their median
  over_base      IAM only: the largest per-line CER reduction from the HTR-JAND base (68x864) to the
                 Student's greedy CTC (128x1632), references >= 20 chars
  lm_correction  the largest per-line CER reduction raw CTC -> char LM, references >= 20 chars
  t5_correction  the largest per-line CER reduction char LM -> T5 on char LM among lines T5 changed;
                 absent (reported) if T5 changed no test line of the corpus or improved none
  t5_failure     the largest per-line CER increase char LM -> T5 on char LM among lines T5 changed;
                 absent (reported) if no changed line got worse
  model_failure  the highest CER of the final (validation-selected) path, references >= 20 chars
Sample IDs are recovered from the raw corpus lists with htrjand.prepare_data's readers (the
archives were built from them in this order) and checked twice: the reader's label must equal the
archive label at that index, and the raw image re-preprocessed to 128x1632 must equal the archive image
pixel for pixel. A sample whose check fails gets id=null with the reason.
"""
import json
import os
import re
import sys
from collections import defaultdict

import editdistance
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refined_figs_common as C  # noqa: E402
from htrjand import prepare_data as P  # noqa: E402


if __name__ == "__main__" and any(a in ("-h", "--help") for a in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)


RULE_NAMES = ("low_cer", "median_cer", "high_cer", "long_line", "historical", "spacing_punct", "rare_vocab",
              "over_base", "lm_correction", "t5_correction", "t5_failure", "model_failure")
RULES = {k: " ".join(v.split()) for k, v in re.findall(r"^  (\w+)\s{2,}(.+?(?:\n {17}.+?)*)$", __doc__, re.M)
         if k in RULE_NAMES}


def cer(r, h):
    return editdistance.eval(r, h) / max(len(r), 1)


def map_by_ref(gt, refs, hyps):
    """archive index -> hypothesis, matching references occurrence by occurrence."""
    pos = defaultdict(list)
    for i, r in enumerate(refs):
        pos[r].append(i)
    out, used = [None] * len(gt), defaultdict(int)
    for i, g in enumerate(gt):
        k = used[g]; used[g] += 1
        out[i] = hyps[pos[g][k]]
    if len(refs) != len(gt) or sorted(refs) != sorted(gt):
        raise SystemExit("base references are not a permutation of the archive references")
    return out


def load_corpus(ds):
    gt = C.test_gt(ds)
    sel = C.selected_ids(ds)
    sys_ = {}
    raw = C.read_tsv(os.path.join(C.RES, f"{C.SEED1[ds]['raw']}_test_preds.tsv"))
    lm = C.read_tsv(os.path.join(C.RES, f"{C.SEED1[ds]['lm']}_test_preds.tsv"))
    assert sel["raw CTC"]["id"] + f"_{C.ARCHIVE[ds]}" == C.SEED1[ds]["raw"], (sel["raw CTC"]["id"], ds)
    assert sel["char LM"]["id"] == C.SEED1[ds]["lm"], ds
    for name, rows, col in (("raw CTC", raw, "hyp"), ("char LM", lm, "hyp")):
        assert [r["ref"] for r in rows] == gt, f"{ds} {name}: TSV order != archive order"
        sys_[name] = [r[col] for r in rows]
    t5src = {}
    for st, base in (("T5 on char LM", "char LM"), ("T5 on raw CTC", "raw CTC")):
        tid = sel[st]["id"]
        rows = C.read_tsv(os.path.join(C.RES, f"{tid}_test_preds.tsv"))
        assert [r["ref"] for r in rows] == gt, f"{ds} {st}: order"
        assert [r["raw"] for r in rows] == sys_[base], f"{ds} {st}: T5 input != {base} output"
        sys_[st] = [r["final"] for r in rows]
        t5src[st] = tid
    valid = {st: sel[st]["v"][0] for st in C.STAGES if st in sel}
    final_path = C.pick_path(valid)
    sys_["final"] = sys_[final_path]
    rec = C.read_tsv(os.path.join(C.RES, f"{C.BASE[ds]}_test_preds.tsv"))
    sys_["base"] = map_by_ref(gt, [r["ref"] for r in rec], [r["hyp"] for r in rec])
    ids = dict(raw=C.SEED1[ds]["raw"], lm=C.SEED1[ds]["lm"], **{st: t5src[st] for st in t5src},
               base=C.BASE[ds], final_path=final_path, valid_cer=valid)
    return gt, sys_, ids


def nearest(values, target, cand):
    cand = [i for i in cand]
    return min(cand, key=lambda i: (abs(values[i] - target), i))


def argbest(scores, cand, sign=+1):
    """max (sign=+1) of scores over cand; ties -> lowest index."""
    return min(cand, key=lambda i: (-sign * scores[i], i))


def letter_words(s):
    return re.findall(r"[^\W\d_]+", s)


def sample_ids(ds):
    parts = P.READERS[ds](os.path.join(C.RAW, ds))
    return [(im, g) for im, g in parts["test"] if g]


def describe_id(ds, item, all_items, idx):
    if ds == "rimes":
        (path, box) = item
        page = os.path.basename(path)
        k = sum(1 for j in range(idx) if os.path.basename(all_items[j][0][0]) == page)
        return f"{page}#line{k}", dict(page=os.path.relpath(path, C.RAW), box_top_bottom_left_right=list(box))
    return os.path.splitext(os.path.basename(item))[0], dict(file=os.path.relpath(item, C.RAW))


def main():
    out = dict(description=__doc__.strip(), rules=RULES, corpora={}, samples=[])
    for ds in C.CORPORA:
        gt, S, ids = load_corpus(ds)
        n = len(gt)
        L = np.array([len(g) for g in gt])
        c = {k: np.array([cer(g, h) for g, h in zip(gt, v)]) for k, v in S.items()}
        changed = {st: [i for i in range(n) if S[st][i] != S["char LM" if "char" in st else "raw CTC"][i]]
                   for st in ("T5 on char LM", "T5 on raw CTC")}
        info = dict(n_test_lines=n, ids=ids, final_path=ids["final_path"],
                    corpus_cer={k: float(sum(editdistance.eval(g, h) for g, h in zip(gt, v)) / L.sum())
                                for k, v in S.items()},
                    t5_changed_lines={k: len(v) for k, v in changed.items()}, notes=[])
        picks = {}
        ge30, ge20 = [i for i in range(n) if L[i] >= 30], [i for i in range(n) if L[i] >= 20]
        for name, q in (("low_cer", 10), ("median_cer", 50), ("high_cer", 90)):
            t = float(np.percentile(c["raw CTC"], q))
            picks[name] = (nearest(c["raw CTC"], t, ge30), dict(percentile=q, target_cer=t))
        if ds == "iam":
            picks["long_line"] = (argbest(L, range(n)), dict(ref_len=int(L.max())))
            d = c["base"] - c["raw CTC"]
            picks["over_base"] = (argbest(d, ge20), dict(cer_reduction=float(d[argbest(d, ge20)])))
        if ds in ("saintgall", "washington"):
            picks["historical"] = (picks["median_cer"][0], dict(same_line_as="median_cer"))
        # spacing / punctuation
        share = {}
        for i in range(n):
            ops = [o for o in C.align(gt[i], S["raw CTC"][i]) if o[0] != "="]
            if len(ops) >= 3:
                share[i] = sum(1 for o, a, b in ops if any(C.is_space_or_punct(x) for x in a + b)) / len(ops)
        if share:
            i = argbest(share, share.keys())
            picks["spacing_punct"] = (i, dict(error_share_space_punct=share[i], n_candidates=len(share)))
        # rare vocabulary
        lex = {w for g in C.train_gt(ds) for w in letter_words(g)}
        oov = {i: [w for w in letter_words(gt[i]) if w not in lex] for i in range(n)}
        cand = [i for i in range(n) if oov[i]]
        if cand:
            med = float(np.median(c["raw CTC"][cand]))
            i = nearest(c["raw CTC"], med, cand)
            picks["rare_vocab"] = (i, dict(oov_words=oov[i], n_candidates=len(cand), median_cer_of_candidates=med))
        d = c["raw CTC"] - c["char LM"]
        i = argbest(d, ge20)
        picks["lm_correction"] = (i, dict(cer_reduction=float(d[i])))
        ch = changed["T5 on char LM"]
        if not ch:
            info["notes"].append("T5 on char LM changed no test line of this corpus: no t5_correction / t5_failure sample")
        else:
            d = c["char LM"] - c["T5 on char LM"]
            ib, iw = argbest(d, ch), argbest(-d, ch)
            if d[ib] > 0:
                picks["t5_correction"] = (ib, dict(cer_reduction=float(d[ib]), n_changed=len(ch)))
            else:
                info["notes"].append(f"T5 on char LM changed {len(ch)} test line(s) but improved none")
            if d[iw] < 0:
                picks["t5_failure"] = (iw, dict(cer_increase=float(-d[iw]), n_changed=len(ch)))
            else:
                info["notes"].append(f"T5 on char LM changed {len(ch)} test line(s) and made none worse")
            info["t5_on_char_lm_changed"] = dict(n=len(ch), better=int((d[ch] > 0).sum()),
                                                 worse=int((d[ch] < 0).sum()), equal=int((d[ch] == 0).sum()))
        i = argbest(c["final"], ge20)
        picks["model_failure"] = (i, dict(final_cer=float(c["final"][i])))
        info["rule_picks"] = {k: int(v[0]) for k, v in picks.items()}
        out["corpora"][ds] = info

        items = sample_ids(ds)
        by_idx = defaultdict(list)
        for rule, (i, extra) in picks.items():
            by_idx[i].append((rule, extra))
        for i in sorted(by_idx):
            sid, src, why = None, None, None
            item = items[i] if i < len(items) else None
            if item is None or item[1] != gt[i]:
                why = "raw reader label differs from archive label at this index"
            else:
                img = P.preprocess(item[0], size=(128, 1632))
                if np.array_equal(img, C.test_image(ds, i)):
                    sid, src = describe_id(ds, item[0], [x[0] for x in items] if ds != "rimes" else items, i)
                else:
                    why = "re-preprocessed raw image differs from the archive image"
            preds = {}
            for k in S:
                preds[k] = dict(text=S[k][i], cer=float(c[k][i]), edits=int(editdistance.eval(gt[i], S[k][i])))
            out["samples"].append(dict(
                dataset=ds, split="test", archive=f"data/{C.ARCHIVE[ds]}_128x1632.hdf5", archive_index=int(i),
                sample_id=sid, sample_source=src, sample_id_unrecoverable_reason=why,
                gt=gt[i], ref_len=len(gt[i]), rules=[r for r, _ in by_idx[i]],
                rule_details={r: e for r, e in by_idx[i]}, final_path=ids["final_path"], predictions=preds))
    path = os.path.join(C.RES, "qualitative_examples.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    for s in out["samples"]:
        print(f"{s['dataset']:10s} {s['archive_index']:5d} {str(s['sample_id']):28s} {','.join(s['rules']):40s} "
              f"raw {100*s['predictions']['raw CTC']['cer']:5.1f} lm {100*s['predictions']['char LM']['cer']:5.1f} "
              f"final {100*s['predictions']['final']['cer']:5.1f} base {100*s['predictions']['base']['cer']:5.1f}")
    for ds, v in out["corpora"].items():
        print(ds, v["final_path"], v["t5_changed_lines"], v.get("t5_on_char_lm_changed"), v["notes"])
    print("wrote", path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Dataset-overlap audit: official-split archives vs the historical re-split `*ss` archives.

    python scripts/overlap_audit.py --data-dir $HTRJAND_DATA_DIR --raw-dir raw [--workers 24]

For every corpus (IAM, RIMES, Bentham, Saint Gall, Washington) and every archive
(official 68x864, official 128x1632 used by the refined work, historical *ss 68x864)
it measures, for each pair of partitions, within one archive and across archives:

  A  identical sample IDs. The archives store no IDs. IDs are recovered by replaying
     the official partition readers of htrjand/prepare_data.py on --raw-dir,
     re-rendering every raw line at the archive's canvas with the same recipe, and
     mapping each archive row to the raw line(s) with the same pixel MD5. The map is
     only used where it is unambiguous; its coverage is reported.
  B  identical image arrays: MD5 over the raw uint8 bytes plus the shape.
  C  near-duplicate images: the non-background bounding box of the line is cropped,
     background-subtracted (local box mean), resized to 32x256 (INTER_AREA),
     mean-centred and L2-normalised (see feature()); two rows are
     near-duplicates when the cosine (= Pearson correlation) of these vectors is
     >= TAU and the log aspect ratios of the crops differ by <= ASPECT_TOL.
     Valid across canvases. The threshold is validated in the output on (i) the
     same raw line rendered at 68x864 and 128x1632 (positives) and (ii) official
     test vs official train of the same corpus (negatives: disjoint by construction).
  D  identical normalised transcriptions (NFC, whitespace collapsed), case-sensitive
     and case-insensitive (casefold).
  E  text matches whose images are not duplicates (string repeated, image not
     shared): D minus the lines whose matching-text partner is also an image match.

Deterministic: no sampling, no randomness; output results/overlap_audit.json.
"""
import argparse
import functools
import hashlib
import json
import os
import sys
import unicodedata
from collections import Counter, defaultdict
from multiprocessing import Pool

import cv2
import h5py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SPLITS = ("train", "valid", "test")
FEAT_HW = (32, 256)
TAU = 0.88
ASPECT_TOL = 0.15
CHUNK = 256

CORPORA = {
    # corpus: {archive_key: filename}
    "iam": {"official68": "iam.hdf5", "official128": "iam_128x1632.hdf5", "ss": "iamss.hdf5"},
    "rimes": {"official68": "rimes.hdf5", "official128": "rimes_128x1632.hdf5", "ss": "rimesss.hdf5"},
    "bentham": {"official68": "bentham.hdf5", "official128": "bentham_128x1632.hdf5", "ss": "benthamss.hdf5"},
    "saintgall": {"official68": "saintgall.hdf5", "official128": "saintgall_nopt_128x1632.hdf5",
                  "official68_nopt": "saintgall_nopt.hdf5", "ss": "saintgallss.hdf5"},
    "washington": {"official68": "washington.hdf5", "official128": "washington_128x1632.hdf5", "ss": "washingtonss.hdf5"},
}


# ----------------------------------------------------------------------------- text
def norm_text(s):
    return " ".join(unicodedata.normalize("NFC", s).split())


# ----------------------------------------------------------------------------- image features
def feature(img):
    """Content vector robust to the canvas.

    1. crop to the rendered line: rows/columns that are not entirely the pad value
       (the builders centre-pad with a constant, found in the corner pixel);
    2. remove the paper background with a local box-mean (kernel = half the crop height)
       and keep the positive residual (ink strokes); without this step, page texture and
       illumination dominate the vector on Bentham;
    3. resize to FEAT_HW with INTER_AREA, mean-centre, L2-normalise.
    Returns (vector, log aspect ratio of the crop)."""
    pad = img[0, 0]
    m = img != pad
    rows, cols = np.where(m.any(1))[0], np.where(m.any(0))[0]
    crop = img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1] if len(rows) else img
    f = crop.astype(np.float32)
    k = max(3, (crop.shape[0] // 2) | 1)
    hp = np.clip(f - cv2.blur(f, (k, k)), 0, None)
    v = cv2.resize(hp, (FEAT_HW[1], FEAT_HW[0]), interpolation=cv2.INTER_AREA).ravel()
    v -= v.mean()
    n = np.linalg.norm(v)
    return (v / n if n > 0 else v), float(np.log(crop.shape[1] / crop.shape[0]))


def md5(img):
    return hashlib.md5(str(img.shape).encode() + img.tobytes()).hexdigest()


def read_archive(path):
    """Per split: md5 list, feature matrix, log-aspect vector, normalised labels, shape."""
    out = {}
    with h5py.File(path, "r") as h:
        for sp in SPLITS:
            dt, gt = h[sp]["dt"], h[sp]["gt"]
            n = dt.shape[0]
            m, F, A = [], np.zeros((n, FEAT_HW[0] * FEAT_HW[1]), np.float32), np.zeros(n, np.float32)
            for s in range(0, n, CHUNK):
                block = dt[s:s + CHUNK]
                for j, img in enumerate(block):
                    m.append(md5(img))
                    F[s + j], A[s + j] = feature(img)
            labels = [norm_text(g.decode("utf-8") if isinstance(g, bytes) else g) for g in gt[:]]
            out[sp] = dict(md5=m, F=F, A=A, text=labels, shape=list(dt.shape))
    return out


# ----------------------------------------------------------------------------- raw replay (sample IDs)
_imread = cv2.imread


@functools.lru_cache(maxsize=4)
def _cached_read(path, flag):
    return _imread(path, flag)


def _render(args):
    item, size = args
    from htrjand import prepare_data as P
    cv2.imread = lambda p, f=cv2.IMREAD_GRAYSCALE: _cached_read(p, f)   # RIMES: one page, many lines
    try:
        return md5(P.preprocess(item, size=tuple(size)))
    except Exception as e:  # noqa: BLE001
        return "ERR:" + type(e).__name__


def raw_ids(corpus, raw_dir):
    """Official partition lists in builder order: [(id, item, label)] per split, empty labels dropped."""
    from htrjand import prepare_data as P
    parts = P.READERS[corpus](os.path.join(raw_dir, P._raw_subdir(corpus)))
    out = {}
    for sp in SPLITS:
        rows = []
        for item, g in parts[sp]:
            if not g:
                continue
            if isinstance(item, tuple):          # RIMES: (page, (top, bottom, left, right))
                page, box = item
                lid = f"{os.path.basename(page)}#{'_'.join(map(str, box))}"
            else:
                lid = os.path.splitext(os.path.basename(item))[0]
            rows.append((lid, item, norm_text(g)))
        out[sp] = rows
    return out


def render_hashes(ids, size, workers):
    jobs, keys = [], []
    for sp in SPLITS:
        for lid, item, _ in ids[sp]:
            jobs.append((item, size)); keys.append((sp, lid))
    order = sorted(range(len(jobs)), key=lambda i: str(jobs[i][0]))   # page locality for RIMES
    with Pool(workers) as pool:
        res = pool.map(_render, [jobs[i] for i in order], chunksize=16)
    h = {}
    for i, r in zip(order, res):
        h[keys[i]] = r
    return h


# ----------------------------------------------------------------------------- comparisons
def best_match(FQ, AQ, FR, AR):
    """For each query row: max correlation over reference rows with compatible aspect, and its index."""
    best = np.full(len(FQ), -1.0, np.float32)
    arg = np.full(len(FQ), -1, np.int64)
    if len(FR) == 0 or len(FQ) == 0:
        return best, arg
    for s in range(0, len(FQ), 512):
        C = FQ[s:s + 512] @ FR.T
        C[np.abs(AQ[s:s + 512, None] - AR[None, :]) > ASPECT_TOL] = -1.0
        arg[s:s + 512] = C.argmax(1)
        best[s:s + 512] = C[np.arange(len(C)), arg[s:s + 512]]
    return best, arg


def pair_corr(FQ, AQ, i, FR, AR, j):
    if abs(AQ[i] - AR[j]) > ASPECT_TOL:
        return -1.0
    return float(FQ[i] @ FR[j])


def compare(Q, R, same_canvas, idQ=None, idR=None, examples=5):
    nq = len(Q["md5"])
    res = {"n_query": nq, "n_reference": len(R["md5"])}
    # A: IDs
    if idQ is not None and idR is not None:
        refids = {x for x in idR if x}
        hit = [i for i in range(nq) if idQ[i] and idQ[i] in refids]
        res["A_identical_id"] = len(hit)
        res["A_query_rows_with_id"] = sum(1 for x in idQ if x)
        res["A_reference_rows_with_id"] = len(refids)
    else:
        res["A_identical_id"] = None
    # B: bytes
    rset = set(R["md5"])
    bhit = np.array([m in rset for m in Q["md5"]]) if same_canvas else np.zeros(nq, bool)
    res["B_byte_identical"] = int(bhit.sum()) if same_canvas else None
    # C: near-duplicates
    best, arg = best_match(Q["F"], Q["A"], R["F"], R["A"])
    near = best >= TAU
    res["C_near_duplicate"] = int(near.sum())
    res["C_near_duplicate_not_byte_identical"] = int((near & ~bhit).sum()) if same_canvas else None
    res["C_best_corr_quantiles"] = {q: round(float(np.quantile(best, q)), 4) for q in (0.0, 0.05, 0.5, 0.95, 1.0)} if nq else {}
    res["C_corr_band_0.50_TAU"] = int(((best >= 0.50) & (best < TAU)).sum())
    # D/E: text
    rtext = defaultdict(list)
    rtext_ci = set()
    for j, t in enumerate(R["text"]):
        rtext[t].append(j); rtext_ci.add(t.casefold())
    d_cs = [i for i, t in enumerate(Q["text"]) if t in rtext]
    d_ci = [i for i, t in enumerate(Q["text"]) if t.casefold() in rtext_ci]
    shared, only = [], []
    for i in d_cs:
        js = rtext[Q["text"][i]]
        img_shared = any((same_canvas and Q["md5"][i] == R["md5"][j]) or
                         pair_corr(Q["F"], Q["A"], i, R["F"], R["A"], j) >= TAU for j in js)
        (shared if img_shared else only).append(i)
    res["D_text_identical_case_sensitive"] = len(d_cs)
    res["D_text_identical_case_insensitive"] = len(d_ci)
    res["D_text_and_image_shared"] = len(shared)
    res["E_text_only_repeat"] = len(only)
    res["E_examples"] = [t for t, _ in Counter(Q["text"][i] for i in only).most_common(examples)]
    # image shared but text differs (same line image carrying a different label)
    first = {}
    for j, m in enumerate(R["md5"]):
        first.setdefault(m, j)
    partner = [first[Q["md5"][i]] if bhit[i] else (int(arg[i]) if near[i] else None) for i in range(nq)]
    res["image_shared_text_differs"] = sum(1 for i, j in enumerate(partner)
                                           if j is not None and Q["text"][i] != R["text"][j])
    res["_best"] = best  # stripped before writing
    return res


def strip(d):
    return {k: v for k, v in d.items() if not k.startswith("_")}


def pct(a, b):
    return None if (a is None or not b) else round(100.0 * a / b, 2)


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.environ.get("HTRJAND_DATA_DIR") or os.path.join(ROOT, "data"))
    ap.add_argument("--raw-dir", default=os.environ.get("HTRJAND_RAW_DIR") or os.path.join(ROOT, "raw"),
                    help="raw corpora (for sample IDs); category A is skipped if absent")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "overlap_audit.json"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--corpora", nargs="*", default=list(CORPORA))
    a = ap.parse_args(argv)

    report = {"method": {"feature": f"pad-crop, local-mean background removal, {FEAT_HW[0]}x{FEAT_HW[1]} INTER_AREA, mean-centred, L2",
                         "tau": TAU, "aspect_tol_log": ASPECT_TOL, "md5": "md5(str(shape) + raw uint8 bytes)",
                         "text_norm": "NFC + whitespace collapse; case-insensitive = casefold",
                         "data_dir": os.path.abspath(a.data_dir), "raw_dir": os.path.abspath(a.raw_dir)},
              "corpora": {}, "threshold_validation": {}}
    for corpus in a.corpora:
        print(f"== {corpus}", file=sys.stderr)
        C = {"archives": {}, "within": {}, "cross": {}, "ids": {}}
        arch = {}
        for key, fn in CORPORA[corpus].items():
            p = os.path.join(a.data_dir, fn)
            if not os.path.exists(p):
                C["archives"][key] = {"file": fn, "missing": True}
                continue
            arch[key] = read_archive(p)
            C["archives"][key] = {"file": fn, "shape": {sp: arch[key][sp]["shape"] for sp in SPLITS},
                                  "within_split_byte_duplicates": {sp: len(arch[key][sp]["md5"]) - len(set(arch[key][sp]["md5"])) for sp in SPLITS}}
            print(f"   read {fn}", file=sys.stderr)

        # --- A: recover IDs
        ids = {k: None for k in arch}
        raw_ok = os.path.isdir(os.path.join(a.raw_dir, corpus))
        C["ids"]["raw_available"] = raw_ok
        if raw_ok:
            rid = raw_ids(corpus, a.raw_dir)
            C["ids"]["raw_lines"] = {sp: len(rid[sp]) for sp in SPLITS}
            raw_split = {lid: sp for sp in SPLITS for lid, _, _ in rid[sp]}
            raw_text = {lid: t for sp in SPLITS for lid, _, t in rid[sp]}
            for canvas, keys in (((68, 864), [k for k in arch if k.startswith("official68") or k == "ss"]),
                                 ((128, 1632), [k for k in arch if k == "official128"])):
                if not keys:
                    continue
                hh = render_hashes(rid, canvas, a.workers)
                by_md5 = defaultdict(set)
                for (sp, lid), m in hh.items():
                    by_md5[m].add(lid)
                C["ids"][f"render_errors_{canvas[0]}x{canvas[1]}"] = sum(1 for v in hh.values() if v.startswith("ERR"))
                C["ids"][f"raw_lines_sharing_md5_{canvas[0]}x{canvas[1]}"] = sum(len(v) for v in by_md5.values() if len(v) > 1)
                for k in keys:
                    ids[k] = {}
                    stat = {}
                    for sp in SPLITS:
                        rows = []
                        uniq = amb = none = 0
                        same_split = text_eq = idx_eq = 0
                        for r, m in enumerate(arch[k][sp]["md5"]):
                            cand = by_md5.get(m, set())
                            if len(cand) == 1:
                                lid = next(iter(cand)); uniq += 1
                                same_split += raw_split[lid] == sp
                                text_eq += raw_text[lid] == arch[k][sp]["text"][r]
                            elif cand:
                                # ambiguous md5: disambiguate by label, else leave unresolved
                                c2 = [x for x in cand if raw_text[x] == arch[k][sp]["text"][r]]
                                lid = c2[0] if len(c2) == 1 else None
                                amb += 1
                            else:
                                lid = None; none += 1
                            if r < len(rid[sp]) and hh[(sp, rid[sp][r][0])] == m:
                                idx_eq += 1
                            rows.append(lid)
                        ids[k][sp] = rows
                        stat[sp] = dict(rows=len(rows), unique_md5_to_raw=uniq, ambiguous=amb, unmapped=none,
                                        mapped_from_same_official_split=same_split, mapped_label_equal=text_eq,
                                        index_order_equal_to_builder=idx_eq)
                    C["ids"][k] = stat
            # for the ss archive: from which official split does each ss row come?
            if ids.get("ss"):
                C["ids"]["ss_rows_by_official_origin"] = {
                    sp: dict(Counter(raw_split[x] if x else "unmapped" for x in ids["ss"][sp])) for sp in SPLITS}

        # --- within-archive split pairs
        for k in arch:
            W = {}
            for q, r in (("valid", "train"), ("test", "train"), ("test", "valid")):
                res = compare(arch[k][q], arch[k][r], True,
                              ids[k][q] if ids.get(k) else None, ids[k][r] if ids.get(k) else None)
                res["B_pct_of_query"] = pct(res["B_byte_identical"], res["n_query"])
                res["C_pct_of_query"] = pct(res["C_near_duplicate"], res["n_query"])
                W[f"{r}∩{q}"] = strip(res)
                if k == "official68" and (q, r) == ("test", "train"):
                    report["threshold_validation"].setdefault(corpus, {})["negatives_official68_test_vs_train_best_corr"] = res["C_best_corr_quantiles"]
                    report["threshold_validation"][corpus]["negatives_at_or_above_tau"] = res["C_near_duplicate"]
            C["within"][k] = W

        # --- cross-archive pairs: official vs ss
        if "ss" in arch:
            for ok in [k for k in arch if k.startswith("official")]:
                same = arch[ok]["train"]["shape"][1:] == arch["ss"]["train"]["shape"][1:]
                for q, r in (("test", "train"), ("test", "valid"), ("test", "test"),
                             ("valid", "train"), ("train", "test"), ("train", "valid")):
                    res = compare(arch[ok][q], arch["ss"][r], same,
                                  ids[ok][q] if ids.get(ok) else None, ids["ss"][r] if ids.get("ss") else None)
                    for f in ("A_identical_id", "B_byte_identical", "C_near_duplicate"):
                        res[f.split("_")[0] + "_pct_of_query"] = pct(res[f], res["n_query"])
                    C["cross"][f"{ok}.{q}∩ss.{r}"] = strip(res)

        # --- threshold validation: same raw line at two canvases (positives)
        if "official68" in arch and "official128" in arch and ids.get("official68") and ids.get("official128"):
            pos = []
            for sp in SPLITS:
                i68 = {x: i for i, x in enumerate(ids["official68"][sp]) if x}
                for j, x in enumerate(ids["official128"][sp]):
                    if x in i68:
                        i = i68[x]
                        pos.append(float(arch["official68"][sp]["F"][i] @ arch["official128"][sp]["F"][j])
                                   if abs(arch["official68"][sp]["A"][i] - arch["official128"][sp]["A"][j]) <= ASPECT_TOL else -1.0)
            pos = np.array(pos)
            tv = report["threshold_validation"].setdefault(corpus, {})
            tv["positives_same_id_68_vs_128"] = len(pos)
            tv["positives_corr_quantiles"] = {q: round(float(np.quantile(pos, q)), 4) for q in (0.0, 0.001, 0.01, 0.05, 0.5)} if len(pos) else {}
            tv["positives_below_tau"] = int((pos < TAU).sum())
        report["corpora"][corpus] = C

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(report, fh, indent=1, ensure_ascii=False, sort_keys=False)
    print(f"wrote {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

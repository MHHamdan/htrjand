#!/usr/bin/env python3
"""Saint Gall "pt" annotation-convention audit (CPU, deterministic, read-only).

    python scripts/saintgall_audit.py [--raw-dir RAW] [--data-dir DATA]   # -> results/saintgall_audit.json

Checks, from the raw IAM-HistDB Saint Gall ground truth and the HDF5 archives:
  a. the README definition of the special letters "pt" / "et" (quoted with line numbers) and raw examples;
  b. the code path that joins letters into words (prepare_data._saintgall, build_saintgall_nopt.py);
  c. "pt" / "et" counts by split and page, in the raw ground truth and in every Saint Gall archive;
     Washington's own punctuation code ("s_pt") and archives, to confirm the fix was not applied there;
  d. lexical checks: does any evaluation reference contain "pt" as letters; old -> new training-label diff,
     every removed substring traced back to a raw "pt" special-letter token;
  e. per-row SHA-256 of images and labels: validation / test identical between saintgall* and saintgall_nopt*;
  f. old vs marker-free results (validation and test) from results/*.json, the validation-based selection,
     and the share of the old models' error that is "pt" insertion (CER after deleting emitted "pt").
Nothing is trained or decoded; no GPU.
"""
import argparse
import glob
import hashlib
import inspect
import json
import os
import re
import sys
from collections import Counter, defaultdict

import editdistance
import h5py

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
R = os.path.join(ROOT, "results")
SPLITS = ("train", "valid", "test")
PT_FINAL = re.compile(r"(?<=\S)pt(?=\s|$)")        # "pt" closing a word (how the joined special letter looks)
PT_ALONE = re.compile(r"(?:(?<=\s)|^)pt(?=\s|$)")    # "pt" as a whole word


def rel(p):
    return os.path.relpath(p, ROOT) if os.path.abspath(p).startswith(ROOT) else p


def dec(x):
    return x.decode("utf-8") if isinstance(x, bytes) else x


def cer(refs, hyps):
    return sum(editdistance.eval(r, h) for r, h in zip(refs, hyps)) / max(sum(len(r) for r in refs), 1)


GENUINE_PT_WORDS = set()    # words that end in the LETTERS p,t in the marker-free training labels ("ppt" = propter); set in main


def strip_pt(h):
    """Delete one emitted word-final "pt" (the trained-in special letter) per word, and stand-alone "pt" words.
    Words that genuinely end in the letters p,t in the corrected TRAINING labels are left alone."""
    out = []
    for w in h.split(" "):
        if w == "pt":
            continue
        if w.endswith("pt") and w not in GENUINE_PT_WORDS:
            w = w[:-2]
        out.append(w)
    return " ".join(out)


def count_pt(s):
    return dict(substring=s.count("pt"), word_final=len(PT_FINAL.findall(s)), whole_word=len(PT_ALONE.findall(s)),
                et_substring=s.count("et"))


def add(a, b):
    for k, v in b.items():
        a[k] = a.get(k, 0) + v
    return a


# ----------------------------------------------------------------------------- a. README + raw
def readme(raw):
    p = os.path.join(raw, "saintgall", "README.txt")
    lines = open(p, encoding="utf-8", errors="replace").read().splitlines()
    keep = [i for i, l in enumerate(lines, 1) if "transcription.txt" in l or "special letters" in l
            or l.startswith("[lineID]") or l.startswith("[character_1]") or "Spellings are given" in l]
    return dict(path=p, lines={i: lines[i - 1] for i in range(min(keep), max(keep) + 1) if lines[i - 1].strip()})


def raw_saintgall(raw):
    src = os.path.join(raw, "saintgall")
    sets = {s: open(os.path.join(src, "sets", f"{s}.txt")).read().split() for s in SPLITS}
    page2split = {p: s for s, ps in sets.items() for p in ps}
    rows = []
    path = os.path.join(src, "ground_truth", "transcription.txt")
    for n, line in enumerate(open(path, encoding="utf-8"), 1):
        lid, spell, lab = line.split()
        page = lid.rsplit("-", 1)[0]
        words = [w.split("-") for w in spell.split("|")]
        rows.append(dict(lineno=n, lid=lid, page=page, split=page2split[page], spelling=spell, edition_labels=lab,
                         words=words,
                         old=" ".join("".join(w) for w in words),                     # == prepare_data._saintgall
                         new=" ".join(x for x in ("".join(c for c in w if c != "pt") for w in words) if x)))
    return path, sets, rows


def raw_stats(sets, rows):
    out = {}
    for s in SPLITS:
        rs = [r for r in rows if r["split"] == s]
        per_page = Counter({p: 0 for p in sets[s]})
        et_page = Counter({p: 0 for p in sets[s]})
        pos = Counter()
        pt_pair_words, pt_pair_tokens = [], 0          # genuine letters p,t adjacent (e.g. s-c-r-i-p-t-u-m)
        n_words = 0
        for r in rs:
            for w in r["words"]:
                n_words += 1
                for i, c in enumerate(w):
                    if c == "pt":
                        per_page[r["page"]] += 1
                        pos["alone" if len(w) == 1 else "final" if i == len(w) - 1 else "initial" if i == 0 else "medial"] += 1
                    if c == "et":
                        et_page[r["page"]] += 1
                for a, b in zip(w, w[1:]):
                    if a == "p" and b == "t":
                        pt_pair_tokens += 1
                        pt_pair_words.append("-".join(w))
        out[s] = dict(pages=len(sets[s]), lines=len(rs), words=n_words,
                      pt_tokens=sum(per_page.values()), pages_with_pt=sum(v > 0 for v in per_page.values()),
                      lines_with_pt=sum(any(c == "pt" for w in r["words"] for c in w) for r in rs),
                      pt_position=dict(pos), pt_per_page=dict(sorted(per_page.items())),
                      et_tokens=sum(et_page.values()), pages_with_et=sum(v > 0 for v in et_page.values()),
                      et_per_page=dict(sorted(et_page.items())),
                      letter_pair_p_t=dict(tokens=pt_pair_tokens, distinct_words=sorted(set(pt_pair_words))))
    return out


def raw_examples(rows, k=3):
    ex = []
    for r in rows:
        if r["split"] == "train" and any(c == "pt" for w in r["words"] for c in w):
            ex.append(dict(line_id=r["lid"], transcription_txt_line=r["lineno"], raw_spelling=r["spelling"],
                           edition_labels=r["edition_labels"], old_archive_label=r["old"], nopt_archive_label=r["new"]))
            if len(ex) == k:
                break
    return ex


# ----------------------------------------------------------------------------- b. code path
def code_path():
    import htrjand.prepare_data as P
    src, start = inspect.getsourcelines(P._saintgall)
    join = [start + i for i, l in enumerate(src) if 'replace("-", "")' in l]
    out = dict(prepare_data=dict(file=rel(inspect.getsourcefile(P)), function="_saintgall",
                                 lines=[start, start + len(src) - 1], join_line=join,
                                 join_code=[src[j - start].strip() for j in join]))
    b = os.path.join(ROOT, "scripts", "build_saintgall_nopt.py")
    L = open(b).read().splitlines()
    find = lambda pat: [i for i, l in enumerate(L, 1) if pat in l]
    out["build_saintgall_nopt"] = dict(file=rel(b), old_label_line=find('old = " ".join'), new_label_line=find("new = "),
                                       code=[L[i - 1].strip() for i in find('old = " ".join') + find("new = ")],
                                       copies_images_line=find('create_dataset(f"{sp}/dt"'),
                                       maps_labels_line=find("new = [old2new[g]"),
                                       mentions_washington=any("washington" in l.lower() for l in L))
    D = __import__("htrjand.data", fromlist=["OFFICIAL"])
    out["dataset_registry"] = dict(file="htrjand/data.py",
                                   nopt_entries=sorted(k for k in D.OFFICIAL if "nopt" in k))
    return out


# ----------------------------------------------------------------------------- c/e. archives
def read_archive(path):
    with h5py.File(path, "r") as h:
        out = {}
        for s in SPLITS:
            dt, gt = h[s]["dt"], h[s]["gt"]
            out[s] = dict(gt=[dec(g) for g in gt[:]],
                          gt_bytes=[g if isinstance(g, bytes) else g.encode("utf-8") for g in gt[:]],
                          img_sha=[hashlib.sha256(dt[i].tobytes()).hexdigest() for i in range(dt.shape[0])],
                          shape=list(dt.shape), dtype=str(dt.dtype),
                          gt_h5_encoding=str(h5py.check_string_dtype(gt.dtype).encoding) if h5py.check_string_dtype(gt.dtype) else None)
        return out


def archive_counts(arc, label_to_lids=None, lid_page=None):
    out = {}
    for s in SPLITS:
        tot = {}
        page = Counter()
        lines_pt = 0
        unmapped = 0
        for g in arc[s]["gt"]:
            c = count_pt(g)
            add(tot, c)
            lines_pt += c["word_final"] > 0
            if label_to_lids is not None:
                lids = label_to_lids[s].get(g, [])
                pages = {lid_page[x] for x in lids}
                if len(pages) == 1:
                    page[pages.pop()] += c["word_final"]
                else:
                    unmapped += 1
        out[s] = dict(lines=len(arc[s]["gt"]), lines_with_word_final_pt=lines_pt, **tot)
        if label_to_lids is not None:
            out[s]["word_final_pt_per_page"] = dict(sorted(page.items()))
            out[s]["pages_with_word_final_pt"] = sum(v > 0 for v in page.values())
            out[s]["rows_not_mapped_to_one_page"] = unmapped
    return out


def identity(a, b):
    out = {}
    for s in SPLITS:
        n = min(len(a[s]["gt"]), len(b[s]["gt"]))
        same_img = sum(x == y for x, y in zip(a[s]["img_sha"], b[s]["img_sha"]))
        same_gt = sum(x == y for x, y in zip(a[s]["gt_bytes"], b[s]["gt_bytes"]))
        digest = lambda arc: hashlib.sha256("".join(h + hashlib.sha256(g).hexdigest()
                                                    for h, g in zip(arc[s]["img_sha"], arc[s]["gt_bytes"])).encode()).hexdigest()
        out[s] = dict(rows=[len(a[s]["gt"]), len(b[s]["gt"])], shape_equal=a[s]["shape"] == b[s]["shape"],
                      images_identical=same_img, labels_identical=same_gt,
                      all_images_identical=same_img == n == len(a[s]["gt"]) == len(b[s]["gt"]),
                      all_labels_identical=same_gt == n == len(a[s]["gt"]) == len(b[s]["gt"]),
                      split_digest=[digest(a), digest(b)],
                      gt_h5_encoding=[a[s]["gt_h5_encoding"], b[s]["gt_h5_encoding"]])
    return out


def train_diff(old, new, raw_rows):
    """Row-wise old -> new training labels; each removed piece must be the raw "pt" token of that word."""
    by_old = defaultdict(list)
    for r in raw_rows:
        if r["split"] == "train":
            by_old[r["old"]].append(r)
    changed = removed_chars = removed_pt = 0
    bad, survived_letter_pt_old, survived_letter_pt_new = [], 0, 0
    word_final_letter_pt = []                          # word-final "pt" in old labels that is NOT the special letter
    for i, (o, n) in enumerate(zip(old["train"]["gt"], new["train"]["gt"])):
        cands = by_old.get(o, [])
        exp = {r["new"] for r in cands}
        if len(exp) != 1 or n not in exp:
            bad.append(dict(row=i, reason="no unique raw line", old=o, new=n))
            continue
        r = cands[0]
        if o != n:
            changed += 1
        removed_chars += len(o) - len(n)
        ow, nw = o.split(" "), n.split(" ")
        if len(ow) != len(nw):
            bad.append(dict(row=i, reason="word count changed", old=o, new=n))
            continue
        for w_raw, a, b in zip(r["words"], ow, nw):
            k = sum(c == "pt" for c in w_raw)
            if a != b:
                if not (k == 1 and w_raw[-1] == "pt" and a == b + "pt"):
                    bad.append(dict(row=i, reason="removal not a raw pt token", old_word=a, new_word=b, raw="-".join(w_raw)))
                removed_pt += 1
            elif k:
                bad.append(dict(row=i, reason="raw pt token kept", word=a))
            letters = "".join(c for c in w_raw if c != "pt")
            survived_letter_pt_old += sum(1 for x, y in zip(w_raw, w_raw[1:]) if x == "p" and y == "t")
            survived_letter_pt_new += b.count("pt")
            if a.endswith("pt") and not (w_raw[-1] == "pt"):
                word_final_letter_pt.append(dict(row=i, word=a, raw="-".join(w_raw), line_id=r["lid"]))
            if letters != b:
                bad.append(dict(row=i, reason="letters differ", raw="-".join(w_raw), new_word=b))
    return dict(rows=len(old["train"]["gt"]), labels_changed=changed, chars_removed=removed_chars,
                words_shortened=removed_pt, removed_substrings_all_raw_pt_tokens=not bad, violations=bad[:20],
                n_violations=len(bad),
                genuine_letter_pair_p_t_in_raw=survived_letter_pt_old, substring_pt_in_new_labels=survived_letter_pt_new,
                word_final_pt_from_letters_not_special=word_final_letter_pt)


# ----------------------------------------------------------------------------- d. lexical
def lexical(raw_rows, arcs, la_text):
    out = {}
    for s in SPLITS:
        words = [w for g in arcs["saintgall"][s]["gt"] for w in g.split()]
        out[s] = dict(words=len(words),
                      words_ending_pt=sorted(Counter(w for w in words if w.endswith("pt")).items()),
                      words_containing_pt=sorted(Counter(w for w in words if "pt" in w).items())[:40],
                      n_words_containing_pt=sum("pt" in w for w in words))
    # edition labels (3rd column) of words that carry the pt letter: does the lexical label end in "pt"?
    lab_end_pt = tot = 0
    for r in raw_rows:
        labs = r["edition_labels"].split("|")
        if len(labs) != len(r["words"]):
            continue
        for w, l in zip(r["words"], labs):
            if w and w[-1] == "pt":
                tot += 1
                lab_end_pt += l.endswith("pt")
    out["edition_label_of_pt_words"] = dict(aligned_pt_words=tot, label_ends_in_pt=lab_end_pt)
    if os.path.exists(la_text):
        c = Counter()
        n = 0
        for line in open(la_text, encoding="utf-8"):
            for w in re.findall(r"[A-Za-z]+", line.lower()):
                n += 1
                if w.endswith("pt"):
                    c[w] += 1
        out["latin_lm_text"] = dict(file=rel(la_text), words=n, words_ending_pt=sum(c.values()),
                                    rate_per_10k=round(1e4 * sum(c.values()) / max(n, 1), 3), top=c.most_common(10))
    return out


# ----------------------------------------------------------------------------- Washington
def washington(raw, data_dir):
    import htrjand.prepare_data as P
    src = os.path.join(raw, "washington")
    gt_raw = {}
    for line in open(os.path.join(src, "ground_truth", "transcription.txt"), encoding="utf-8"):
        s = line.split()
        gt_raw[s[0]] = s[1]
    sets = {s: open(os.path.join(src, "sets", "cv1", f"{s}.txt")).read().split() for s in SPLITS}
    parts = P._washington(src)
    out = dict(raw={}, archives={}, nopt_archives_present=sorted(os.path.basename(p) for p in glob.glob(os.path.join(data_dir, "washington*nopt*"))))
    for s in SPLITS:
        toks = [c for l in sets[s] for w in gt_raw[l].split("|") for c in w.split("-")]
        out["raw"][s] = dict(lines=len(sets[s]), s_pt_tokens=toks.count("s_pt"), letter_pair_p_t=sum(
            1 for l in sets[s] for w in gt_raw[l].split("|") for a, b in zip(w.split("-"), w.split("-")[1:]) if a == "p" and b == "t"))
    for f in ("washington.hdf5", "washington_128x1632.hdf5"):
        p = os.path.join(data_dir, f)
        if not os.path.exists(p):
            continue
        with h5py.File(p, "r") as h:
            d = {}
            for s in SPLITS:
                gt = [dec(g) for g in h[s]["gt"][:]]
                words = [w for g in gt for w in g.split()]
                d[s] = dict(lines=len(gt), **{k: sum(count_pt(g)[k] for g in gt) for k in ("substring", "word_final", "whole_word")},
                            words_containing_pt=sorted(Counter(w for w in words if "pt" in w).items()),
                            labels_equal_raw_builder=Counter(gt) == Counter(g for _, g in parts[s]))
            out["archives"][f] = d
    return out


# ----------------------------------------------------------------------------- f. results
def J(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else None


def raw_res(tag, ds):
    v, t = J(f"{tag}_{ds}_valid.json"), J(f"{tag}_{ds}_test.json")
    return None if not (v and t) else dict(id=tag, dataset=ds, valid=v["cer"], test=t["cer"],
                                           valid_preds=f"{tag}_{ds}_valid_preds.tsv", test_preds=f"{tag}_{ds}_test_preds.tsv")


def ens_res(tag):
    r = J(f"ens_saintgall_{tag}.json")
    return None if not r else dict(id=f"ens_saintgall_{tag}", dataset=r["dataset"], valid=r["chosen"]["cer"], test=r["test"]["cer"],
                                   alpha=r["chosen"]["alpha"], beta=r["chosen"]["beta"],
                                   members=[m.split("/")[-2] for m in r["members"]], lm=r.get("lm"),
                                   valid_preds=f"ens_saintgall_{tag}_valid_preds.tsv", test_preds=f"ens_saintgall_{tag}_test_preds.tsv")


def best(cands):
    cands = [c for c in cands if c]
    return min(cands, key=lambda c: c["valid"]) if cands else None


def read_tsv(name):
    p = os.path.join(R, name)
    if not os.path.exists(p):
        return None
    L = open(p, encoding="utf-8").read().split("\n")
    head = L[0].split("\t")
    ri, hi = head.index("ref"), head.index("hyp")
    refs, hyps = [], []
    for l in L[1:]:
        if not l:
            continue
        f = l.split("\t")
        f += [""] * (len(head) - len(f))
        refs.append(f[ri])
        hyps.append(f[hi])
    return refs, hyps


def pt_effect(name, orders):
    """orders: {archive name: its reference list for this split}; the TSV must follow one of them row for row."""
    t = read_tsv(name)
    if t is None:
        return None
    refs, hyps = t
    stripped = [strip_pt(h) for h in hyps]
    ins = sum(1 for h in hyps for w in h.split(" ") if w.endswith("pt") and w not in GENUINE_PT_WORDS)
    e_all = sum(editdistance.eval(r, h) for r, h in zip(refs, hyps))
    e_str = sum(editdistance.eval(r, h) for r, h in zip(refs, stripped))
    return dict(file=name, lines=len(refs), refs_row_order=[k for k, v in orders.items() if refs == v],
                refs_equal_archive_multiset=all(Counter(refs) == Counter(v) for v in orders.values()),
                cer=cer(refs, hyps), cer_pt_deleted=cer(refs, stripped), emitted_pt=ins,
                lines_with_emitted_pt=sum(1 for h, x in zip(hyps, stripped) if h != x),
                char_errors=e_all, char_errors_pt_deleted=e_str,
                share_of_errors_removed_by_pt_deletion=(e_all - e_str) / max(e_all, 1))


def results(arcs):
    old_teachers = ["ft_saintgall_c128_syn_teacher_s1", "ft_saintgall_c128_syn_teacher_s2", "c128_syn_teacher_s1", "c128_syn_teacher_s2"]
    new_teachers = ["nopt_sg_c128_teacher_s1", "nopt_sg_c128_teacher_s2", "nopt_sg_combined_final_teacher_s1",
                    "nopt_sg_combined_final_teacher_s2", "nopt_sg_combined_syn_teacher_s1", "nopt_sg_combined_syn_teacher_s2"]
    old_students_s1 = ["ft_saintgall_c128_student_s1", "ft_saintgall_c128_student_kd_s1", "c128_student_s1", "c128_student_kd_s1"]
    rows = {
        "Teacher raw CTC (best single, by valid)": (best(raw_res(m, "saintgall") for m in old_teachers),
                                                    best(raw_res(m, "saintgall_nopt") for m in new_teachers)),
        "Teacher ensemble (no LM)": (ens_res("ensft_nolm"), ens_res("nopt_ens_nolm")),
        "Teacher ensemble + LM": (best(ens_res(t) for t in ("ensft_wiki", "ens3_wiki", "ens4_wiki", "ens5_wiki", "ens6_wiki", "ens7_wiki")),
                                  ens_res("nopt_ens_wiki")),
        "Student raw CTC, seed 1": (best(raw_res(m, "saintgall") for m in old_students_s1), raw_res("nopt_sg_student_s1", "saintgall_nopt")),
        "Student + LM, seed 1": (best(ens_res(t) for t in ("ft_saintgall_c128_student_s1_wiki", "c128_student_s1_wiki", "c128_student_kd_s1_wiki")),
                                 ens_res("nopt_student_wiki")),
        "Student raw CTC, seed 2": (raw_res("ft_saintgall_c128_student_kd_s2", "saintgall"), raw_res("nopt_sg_student_s2", "saintgall_nopt")),
        "Student + LM, seed 2": (ens_res("ft_saintgall_c128_student_kd_s2_wiki"), ens_res("nopt_student_s2_wiki")),
    }
    s2 = J("seed2_student.json") or {}
    sg2 = s2.get("saintgall", {})
    if not rows["Student raw CTC, seed 2"][0] and sg2:
        rows["Student raw CTC, seed 2"] = (dict(id=sg2.get("model"), valid=sg2["paths"]["raw CTC"]["valid"],
                                                test=sg2["paths"]["raw CTC"]["test"]), rows["Student raw CTC, seed 2"][1])
    table = {}
    for k, (o, n) in rows.items():
        ent = dict(old=o, new=n, selected_on_valid=("new" if (n and o and n["valid"] < o["valid"]) else "old" if o else "new"))
        for tag, r in (("old", o), ("new", n)):
            if r and "valid_preds" in r:
                for sp in ("valid", "test"):
                    r[f"pt_effect_{sp}"] = pt_effect(r[f"{sp}_preds"], {n: arcs[n][sp]["gt"] for n in arcs})
        table[k] = ent
    sel = J("final_selection.json") or {}
    selection = {k: dict(id=v["id"], valid=v["v"][0], test=v["t"][0]) for k, v in sel.get("saintgall", {}).items()}
    seed2 = dict(file="results/seed2_student.json", saintgall_model=sg2.get("model"), saintgall_selected_path=sg2.get("selected_path"),
                 saintgall_selected_test=sg2.get("selected_test"),
                 note="seed2_student.json still records the pt-trained seed-2 Student for Saint Gall; the marker-free "
                      "seed-2 numbers come from nopt_sg_student_s2_* and ens_saintgall_nopt_student_s2_wiki.json")
    return dict(old_vs_new=table, final_selection_saintgall=selection, seed2_student_json=seed2)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default=os.environ.get("HTRJAND_RAW_DIR",
                                                         os.path.join(ROOT, "raw")))
    ap.add_argument("--data-dir", default=os.environ.get("HTRJAND_DATA_DIR", os.path.join(ROOT, "data")))
    ap.add_argument("--out", default=os.path.join(R, "saintgall_audit.json"))
    a = ap.parse_args()

    tpath, sets, rows = raw_saintgall(a.raw_dir)
    rstats = raw_stats(sets, rows)
    lid_page = {r["lid"]: r["page"] for r in rows}
    old_map = {s: defaultdict(list) for s in SPLITS}
    new_map = {s: defaultdict(list) for s in SPLITS}
    for r in rows:
        old_map[r["split"]][r["old"]].append(r["lid"])
        new_map[r["split"]][r["new"]].append(r["lid"])

    GENUINE_PT_WORDS.update(w for r in rows if r["split"] == "train" for w in r["new"].split() if w.endswith("pt"))
    names = ["saintgall", "saintgall_128x1632", "saintgall_nopt", "saintgall_nopt_128x1632"]
    arcs = {n: read_archive(os.path.join(a.data_dir, f"{n}.hdf5")) for n in names}
    counts = {}
    for n in names:
        m = new_map if "nopt" in n else old_map
        counts[n] = archive_counts(arcs[n], m, lid_page)
        for s in SPLITS:                                # archive labels == raw-derived labels (as multisets)
            exp = Counter(r["new" if "nopt" in n else "old"] for r in rows if r["split"] == s)
            counts[n][s]["labels_equal_raw_builder"] = Counter(arcs[n][s]["gt"]) == exp
            counts[n][s]["non_ascii_label_chars"] = sorted({c for g in arcs[n][s]["gt"] for c in g if ord(c) > 127})

    out = dict(
        generated_by="scripts/saintgall_audit.py", raw_dir=a.raw_dir, data_dir=os.path.realpath(a.data_dir),
        a_readme=readme(a.raw_dir), a_transcription_file=tpath, a_raw_examples=raw_examples(rows),
        b_code_path=code_path(),
        c_raw_counts=rstats,
        c_archive_counts=counts,
        c_washington=washington(a.raw_dir, a.data_dir),
        d_lexical=lexical(rows, arcs, os.path.join(ROOT, "synth", "text_la.txt")),
        d_train_diff={"saintgall -> saintgall_nopt": train_diff(arcs["saintgall"], arcs["saintgall_nopt"], rows),
                      "saintgall_128x1632 -> saintgall_nopt_128x1632": train_diff(arcs["saintgall_128x1632"], arcs["saintgall_nopt_128x1632"], rows)},
        e_identity={"saintgall vs saintgall_nopt": identity(arcs["saintgall"], arcs["saintgall_nopt"]),
                    "saintgall_128x1632 vs saintgall_nopt_128x1632": identity(arcs["saintgall_128x1632"], arcs["saintgall_nopt_128x1632"])},
        f_results=results(arcs),
    )
    ids = out["e_identity"]
    out["verdict"] = dict(
        pt_only_on_train_pages=rstats["train"]["pt_tokens"] > 0 and rstats["valid"]["pt_tokens"] == rstats["test"]["pt_tokens"] == 0,
        eval_splits_byte_identical=all(ids[k][s]["all_images_identical"] and ids[k][s]["all_labels_identical"]
                                       for k in ids for s in ("valid", "test")),
        train_images_identical=all(ids[k]["train"]["all_images_identical"] for k in ids),
        train_diff_only_raw_pt=all(v["removed_substrings_all_raw_pt_tokens"] for v in out["d_train_diff"].values()),
        eval_reference_words_ending_in_pt=out["d_lexical"]["valid"]["words_ending_pt"] + out["d_lexical"]["test"]["words_ending_pt"],
        genuine_word_final_pt_in_training=sorted(GENUINE_PT_WORDS),
        washington_untouched=not out["c_washington"]["nopt_archives_present"] and not out["b_code_path"]["build_saintgall_nopt"]["mentions_washington"],
    )
    json.dump(out, open(a.out, "w"), indent=1, sort_keys=False, default=str)
    print(json.dumps(out["verdict"], indent=1))
    print("wrote", rel(a.out))


if __name__ == "__main__":
    main()

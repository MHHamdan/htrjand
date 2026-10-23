"""Section V of the paper: T5 post-processing (Algorithm 5, "T5P"), leakage-free.

Pipeline, with the paper's line numbers of Algorithm 5:

  normalise      Unicode NFC, typographic quotes/dashes -> the corpora's ASCII forms,
                 whitespace collapse (Sec. V.2). Spelling is never modernised: historical
                 content is preserved, only encodings are unified.
  SentencePiece  (line 2) T5-small's pretrained SentencePiece model, extended with the
                 corpus characters it cannot represent (computed from the TRAINING
                 transcriptions). Retraining SentencePiece from scratch would discard the
                 pretrained embeddings T5-small is chosen for; this is the declared reading.
  Segment        (line 4) one segment = one recognised text line.
  BuildContext   (line 5) the previous and next recognised lines of the same split, in
                 archive order (archive order follows the page for every corpus built by
                 prepare_data.py); the window (0 or 1 neighbour each side) is chosen on
                 validation.
  DetectErrors   (line 7) words absent from D = the corpus's TRAINING-split lexicon, and/or
                 the recogniser's own per-line error estimate eps_hat; a line without a
                 detected error is passed through unchanged.
  T5 candidate   (lines 9-10) the fine-tuned T5-small (60 M) generates the corrected line
                 (beam 4) conditioned on corpus tag and context.
  Confidence     (line 11) exp(mean token log-probability) of the candidate under T5
                 (transformers' length-normalised beam score).
  theta          (line 12) the candidate is applied only when confidence > theta.
  ApplyCorr.     (line 13) either the whole line, or only the aligned word segments that
                 contain a detected error ("segment" mode); chosen on validation.
  Format         (line 16) the corpus's punctuation-spacing convention, learned from its
                 TRAINING transcriptions, is restored on the output.

Training data (Sec. V.3), per corpus, never touching validation or test labels:
  * real pairs: out-of-fold recognitions of the TRAINING split (a fold model that did not
    train on a line transcribes it), paired with its ground truth;
  * systematic-error pairs: TRAINING lines corrupted with the substitution / insertion /
    deletion profile (spaces, case, punctuation and hyphens included, as they occur)
    measured on those real pairs, with the number of errors per line drawn from the
    observed per-line error distribution;
  * identity pairs are kept, so the corrector learns to leave correct text alone.
Validation is used for T5 early stopping (loss on out-of-fold VALIDATION recognitions),
and for theta / mode / context selection. Test labels are read only by the final metric.

Subcommands:  pairs | train | apply
"""
import argparse
import csv
import difflib
import json
import os
import random
import re
import time
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import torch

from ._compat import lazy

h5py = lazy("h5py", "pip install h5py")

from . import data as D
from . import metrics as Me
from . import models as M

from ._compat import REPO_ROOT  # noqa: E402  (source checkout, else the working directory)
CORPORA = ("iam", "rimes", "bentham", "saintgall", "washington")
LANG = dict(iam="English", rimes="French", bentham="English", saintgall="Latin", washington="English",
            saintgall_nopt="Latin")
_TYPO = {"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-",
         " ": " ", " ": " ", " ": " "}


# --------------------------------------------------------------------------- text conventions
def normalise(t):
    t = unicodedata.normalize("NFC", t)
    t = "".join(_TYPO.get(c, c) for c in t)
    return re.sub(r"\s+", " ", t).strip()


class Format:
    """Per-corpus punctuation spacing learned from TRAINING transcriptions: for each
    punctuation mark, whether the corpus writes a space before / after it (IAM and
    Bentham write "word , word"; RIMES writes French spacing). Restoration rewrites the
    spacing of the T5 output to the corpus's majority convention."""

    PUNCT = ",.;:!?)(\"'-"

    def __init__(self, train_lines):
        before, after = defaultdict(Counter), defaultdict(Counter)
        for t in train_lines:
            for i, c in enumerate(t):
                if c in self.PUNCT and i > 0 and t[i - 1].isalnum() or (c in self.PUNCT and i > 1 and t[i - 1] == " " and t[i - 2].isalnum()):
                    before[c][t[i - 1] == " "] += 1
                if c in self.PUNCT and i < len(t) - 1:
                    after[c][t[i + 1] == " "] += 1
        self.rule = {}
        for c in self.PUNCT:
            n = sum(before[c].values())
            if n >= 50:
                self.rule[c] = (before[c][True] / n, after[c][True] / max(sum(after[c].values()), 1))

    def restore(self, t):
        t = normalise(t)
        out = list(t)
        for i in range(len(out)):
            c = out[i]
            if c not in self.rule or i == 0:
                continue
            pb, _ = self.rule[c]
            if pb >= 0.995 and out[i - 1] not in (" ",) and out[i - 1].isalnum():
                out[i] = " " + c
            elif pb <= 0.005 and out[i - 1] == " " and i >= 2 and out[i - 2].isalnum():
                out[i - 1] = ""
        return normalise("".join(out))


def lexicon(train_lines):
    return {w.strip(".,;:!?\"'()-") for t in train_lines for w in t.split()} - {""}


def read_labels(dataset, split, root=None):
    with h5py.File(D.archive_path(dataset, root), "r") as h:
        return [g.decode("utf-8") if isinstance(g, bytes) else g for g in h[split]["gt"][:]]


# --------------------------------------------------------------------------- error profile
def align_ops(clean, noisy):
    sm = difflib.SequenceMatcher(a=clean, b=noisy, autojunk=False)
    return sm.get_opcodes()


def error_profile(pairs):
    """Character edit statistics of (noisy, clean) pairs and the per-line error-count
    distribution, plus a categorised summary for the audit trail."""
    subs, dels, ins = defaultdict(Counter), Counter(), Counter()
    per_line, cats = [], Counter()
    for noisy, clean in pairs:
        n = 0
        for op, i1, i2, j1, j2 in align_ops(clean, noisy):
            if op == "equal":
                continue
            a, b = clean[i1:i2], noisy[j1:j2]
            n += max(len(a), len(b))
            if op == "replace":
                for x, y in zip(a, b):
                    subs[x][y] += 1
                    cats["case" if x.lower() == y.lower() else "space" if " " in (x, y)
                         else "punctuation" if not (x.isalnum() and y.isalnum()) else "substitution"] += 1
                for x in a[len(b):]:
                    dels[x] += 1
                for y in b[len(a):]:
                    ins[y] += 1
            elif op == "delete":
                for x in a:
                    dels[x] += 1
                    cats["space_deletion" if x == " " else "hyphen" if x == "-" else "deletion"] += 1
            else:
                for y in b:
                    ins[y] += 1
                    cats["space_insertion" if y == " " else "hyphen" if y == "-" else "insertion"] += 1
        per_line.append(n)
    tot = sum(sum(c.values()) for c in subs.values()) + sum(dels.values()) + sum(ins.values())
    return dict(subs=subs, dels=dels, ins=ins, per_line=per_line, categories=dict(cats),
                p_sub=sum(sum(c.values()) for c in subs.values()) / max(tot, 1),
                p_del=sum(dels.values()) / max(tot, 1))


def corrupt(text, prof, rng):
    """Inject k errors, k drawn from the observed per-line error counts (zeros
    excluded: identity pairs come from the real data); each error is a substitution
    (from the observed confusions of that character), deletion (weighted by the
    observed deletion counts) or insertion (observed inserted characters)."""
    k = rng.choice([x for x in prof["per_line"] if x > 0] or [1])
    chars = list(text)
    ins_pool = list(prof["ins"].elements()) or [" "]
    for _ in range(k):
        if not chars:
            break
        r = rng.random()
        i = rng.randrange(len(chars))
        c = chars[i]
        if r < prof["p_sub"] and prof["subs"].get(c):
            chars[i] = rng.choice(list(prof["subs"][c].elements()))
        elif r < prof["p_sub"] + prof["p_del"] and (prof["dels"].get(c) or rng.random() < 0.2):
            del chars[i]
        else:
            chars.insert(i, rng.choice(ins_pool))
    return "".join(chars)


# --------------------------------------------------------------------------- input format
def source(corpus, cur, prev=None, nxt=None):
    s = f"correct {LANG[corpus]} {corpus.replace('_nopt', '')}: {cur}"
    if prev is not None or nxt is not None:
        s += f" | before: {prev or ''} | after: {nxt or ''}"
    return s


def with_context(corpus, hyps, window):
    out = []
    for i, h in enumerate(hyps):
        if window:
            out.append(source(corpus, h, hyps[i - 1] if i > 0 else "", hyps[i + 1] if i + 1 < len(hyps) else ""))
        else:
            out.append(source(corpus, h))
    return out


# --------------------------------------------------------------------------- pairs
@torch.no_grad()
def decode_holdout(ckpt, combined, split, fold, K, device):
    """Greedy transcriptions + eps_hat of the combined archive's `split`; for the
    training split, only the lines of `fold` (index % K == fold), which the fold model
    did not train on."""
    from torch.utils.data import DataLoader
    model, ck = M.from_checkpoint(ckpt)
    model.to(device)
    tok = D.Tokenizer(ck["charset"])
    ds = D.LineDataset(combined, split, tok, normalisation=ck.get("normalisation", "paper"),
                       fold=("holdout", fold, K) if split == "train" else None)
    dl = DataLoader(ds, batch_size=48, shuffle=False, num_workers=6, collate_fn=D.collate)
    hyps, eps = [], []
    for img, _, _ in dl:
        lg = model(img.to(device))
        hyps += tok.ctc_greedy_decode(lg)
        eps += (1.0 - torch.softmax(lg.float(), -1).max(-1).values.mean(1)).cpu().tolist()
    idx = np.arange(len(read_labels(combined, split)))
    if split == "train":
        idx = idx[idx % K == fold]
    return idx, list(ds.labels), hyps, eps


def corpus_ranges(combined, split, root=None):
    """The combined archive concatenates the five corpora in CORPORA order."""
    out, start = {}, 0
    for c in CORPORA:
        n = len(read_labels(c, split, root))
        out[c] = (start, start + n)
        start += n
    assert start == len(read_labels(combined, split, root)), "combined archive is not the concatenation"
    return out


def cmd_pairs(a):
    device = torch.device(f"cuda:{a.gpu}" if a.gpu >= 0 and torch.cuda.is_available() else "cpu")
    K = len(a.fold_ckpts)
    rec = {"train": {}, "valid": {}}
    for k, ck in enumerate(a.fold_ckpts):
        idx, refs, hyps, eps = decode_holdout(ck, a.combined, "train", k, K, device)
        for i, r, h, e in zip(idx, refs, hyps, eps):
            rec["train"][int(i)] = (r, h, e)
        print(f"fold {k}: {len(idx)} held-out training lines, CER {100*Me.cer(refs, hyps):.2f}", flush=True)
    # validation recognitions for T5 early stopping: fold 0 model (never saw validation)
    idx, refs, hyps, eps = decode_holdout(a.fold_ckpts[0], a.combined, "valid", 0, K, device)
    for i, r, h, e in zip(idx, refs, hyps, eps):
        rec["valid"][int(i)] = (r, h, e)
    rng = random.Random(a.seed)
    meta = dict(fold_ckpts=[os.path.abspath(c) for c in a.fold_ckpts], K=K, corpora={})
    os.makedirs(a.out_dir, exist_ok=True)
    for split in ("train", "valid"):
        rng_c = corpus_ranges(a.combined, split)
        rows = []
        for c in CORPORA:
            s, e = rng_c[c]
            lines = [rec[split][i] for i in range(s, e)]
            refs = [normalise(r) for r, _, _ in lines]
            hyps = [normalise(h) for _, h, _ in lines]
            src = with_context(c, hyps, 1), with_context(c, hyps, 0)
            for j in range(len(lines)):
                rows.append((c, "real", src[0][j], refs[j]))
                rows.append((c, "real", src[1][j], refs[j]))
            if split == "train":
                prof = error_profile(list(zip(hyps, refs)))
                n_syn = 0
                for j, r in enumerate(refs):
                    for _ in range(a.synthetic_per_line):
                        noisy = corrupt(r, prof, rng)
                        prev = corrupt(refs[j - 1], prof, rng) if j > 0 else ""
                        nxt = corrupt(refs[j + 1], prof, rng) if j + 1 < len(refs) else ""
                        rows.append((c, "systematic", source(c, noisy, prev, nxt), r))
                        rows.append((c, "systematic", source(c, noisy), r))
                        n_syn += 1
                ext = dict(x.split("=", 1) for x in a.extra_text).get(c)
                n_ext = 0
                if ext:
                    path, n_max = ext.rsplit(":", 1)
                    allowed = set("".join(refs)) | {" "}
                    ext_lines = [normalise(l) for l in open(path, encoding="utf-8")]
                    ext_lines = [l for l in ext_lines if l and set(l) <= allowed][: int(n_max)]
                    test_refs = [normalise(t) for t in read_labels(c, "test")]
                    big = "\n".join(ext_lines)
                    leak = sum(1 for t in test_refs if len(t) >= 25 and t in big)
                    if leak:
                        raise SystemExit(f"{path}: {leak} test lines of {c} occur verbatim; refusing")
                    for j, r in enumerate(ext_lines):
                        noisy = corrupt(r, prof, rng)
                        prev = corrupt(ext_lines[j - 1], prof, rng) if j > 0 else ""
                        nxt = corrupt(ext_lines[j + 1], prof, rng) if j + 1 < len(ext_lines) else ""
                        rows.append((c, "systematic_external", source(c, noisy, prev, nxt), r))
                        rows.append((c, "systematic_external", source(c, noisy), r))
                        n_ext += 1
                meta["corpora"][c] = dict(n_real=len(refs), heldout_cer=Me.cer(refs, hyps), n_external=n_ext,
                                          identity_lines=int(sum(h == r for h, r in zip(hyps, refs))),
                                          n_systematic=n_syn, error_categories=prof["categories"],
                                          top_substitutions={x: dict(cnt.most_common(4)) for x, cnt in sorted(
                                              prof["subs"].items(), key=lambda kv: -sum(kv[1].values()))[:12]})
            else:
                meta["corpora"].setdefault(c, {})["valid_cer_fold0"] = Me.cer(refs, hyps)
        path = os.path.join(a.out_dir, f"pairs_{split}.tsv")
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow(["corpus", "kind", "source", "target"])
            w.writerows(rows)
        print(f"{split}: {len(rows)} pairs -> {path}", flush=True)
    json.dump(meta, open(os.path.join(a.out_dir, "pairs_meta.json"), "w"), indent=1)


# --------------------------------------------------------------------------- T5
def load_t5(path_or_name, device, charset=None):
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(path_or_name)
    model = T5ForConditionalGeneration.from_pretrained(path_or_name)
    if charset is not None:
        missing = [ch for ch in charset if ch != " " and tok.decode(
            tok(ch, add_special_tokens=False)["input_ids"], clean_up_tokenization_spaces=False).strip() != ch]
        if missing:
            tok.add_tokens(missing)
            model.resize_token_embeddings(len(tok))
        print(f"SentencePiece vocabulary extended with {len(missing)} corpus characters: {missing}", flush=True)
    return tok, model.to(device)


def read_pairs(path, kinds=None):
    with open(path) as fh:
        r = csv.reader(fh, delimiter="\t")
        next(r)
        return [(c, s, t) for c, k, s, t in r if kinds is None or k in kinds]


def cmd_train(a):
    from torch.utils.data import DataLoader
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device(f"cuda:{a.gpu}" if a.gpu >= 0 and torch.cuda.is_available() else "cpu")
    charset = "".join(sorted(set("".join(read_labels("combined", "train")))))
    tok, model = load_t5(a.model, device, charset)
    tr = read_pairs(os.path.join(a.pairs_dir, "pairs_train.tsv"))
    if a.max_pairs:
        random.shuffle(tr); tr = tr[: a.max_pairs]
    va = read_pairs(os.path.join(a.pairs_dir, "pairs_valid.tsv"), kinds={"real"})
    va = [p for p in va if " | before:" in p[1]] if a.context else [p for p in va if " | before:" not in p[1]]
    if not a.context:
        tr = [p for p in tr if " | before:" not in p[1]]

    def collate(batch):
        X = tok([s for _, s, _ in batch], max_length=a.max_len, truncation=True, padding=True, return_tensors="pt")
        Y = tok([t for _, _, t in batch], max_length=a.max_len, truncation=True, padding=True, return_tensors="pt")
        lab = Y["input_ids"].masked_fill(Y["attention_mask"] == 0, -100)
        return X["input_ids"], X["attention_mask"], lab

    dl = DataLoader(tr, batch_size=a.batch_size, shuffle=True, collate_fn=collate, num_workers=2)
    vdl = DataLoader(va, batch_size=64, shuffle=False, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * len(dl)
    warm = max(1, int(0.03 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 1 - s / steps))
    best, bad, log = float("inf"), 0, []
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        tot = 0.0
        for X, A, Y in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = model(input_ids=X.to(device), attention_mask=A.to(device), labels=Y.to(device)).loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item()
        model.eval()
        vl, n = 0.0, 0
        with torch.no_grad():
            for X, A, Y in vdl:
                vl += model(input_ids=X.to(device), attention_mask=A.to(device), labels=Y.to(device)).loss.item() * len(X)
                n += len(X)
        vl /= max(n, 1)
        log.append(dict(epoch=ep, train_loss=tot / len(dl), valid_loss=vl, elapsed_s=time.time() - t0))
        print(f"[t5] epoch {ep} train {tot/len(dl):.4f} valid {vl:.4f} ({(time.time()-t0)/60:.0f} min)", flush=True)
        if vl < best - 1e-4:
            best, bad = vl, 0
            model.save_pretrained(a.out); tok.save_pretrained(a.out)
        else:
            bad += 1
            if bad >= a.patience:
                break
    json.dump(dict(model=a.model, params=sum(p.numel() for p in model.parameters()), pairs_dir=os.path.abspath(a.pairs_dir),
                   n_train=len(tr), n_valid=len(va), context=a.context, epochs_run=len(log), best_valid_loss=best,
                   lr=a.lr, batch_size=a.batch_size, seed=a.seed, log=log,
                   selection="lowest validation loss on out-of-fold VALIDATION recognitions"),
              open(os.path.join(a.out, "train_meta.json"), "w"), indent=1)


@torch.no_grad()
def generate(model, tok, sources, device, batch=48, max_len=192, beams=4):
    outs, confs = [], []
    for i in range(0, len(sources), batch):
        enc = tok(sources[i:i + batch], return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
        g = model.generate(**enc, max_length=max_len, num_beams=beams, early_stopping=True,
                           output_scores=True, return_dict_in_generate=True)
        outs += tok.batch_decode(g.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        confs += torch.exp(g.sequences_scores).float().cpu().tolist()
    return outs, confs


# --------------------------------------------------------------------------- apply (Algorithm 5)
def read_preds(path):
    refs, hyps, eps = [], [], []
    with open(path) as fh:
        r = csv.reader(fh, delimiter="\t")
        hd = next(r)
        for row in r:
            d = dict(zip(hd, row))
            refs.append(d["ref"]); hyps.append(d["hyp"])
            eps.append(float(d["eps_hat"]) if "eps_hat" in d else float("nan"))
    return refs, hyps, np.array(eps)


def detect(hyp, lex, eps, mode, eps_thr):
    oov = [w for w in hyp.split() if w.strip(".,;:!?\"'()-") and w.strip(".,;:!?\"'()-") not in lex
           and not w.strip(".,;:!?\"'()-").isdigit()]
    flag_oov = bool(oov)
    flag_eps = (not np.isnan(eps)) and eps > eps_thr
    flag = dict(all=True, oov=flag_oov, eps=flag_eps, oov_or_eps=flag_oov or flag_eps)[mode]
    return flag, set(oov)


def apply_segments(hyp, cand, oov):
    """ApplyCorrection restricted to the aligned word segments that contain a detected
    (out-of-lexicon) word; every other recognised word is kept."""
    hw, cw = hyp.split(), cand.split()
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=hw, b=cw, autojunk=False).get_opcodes():
        seg = hw[i1:i2]
        if op != "equal" and any(w in oov for w in seg):
            out += cw[j1:j2]          # a segment holding a detected error takes the candidate
        else:
            out += seg                # everything else keeps the recognised words
    return " ".join(out)


def cmd_apply(a):
    device = torch.device(f"cuda:{a.gpu}" if a.gpu >= 0 and torch.cuda.is_available() else "cpu")
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(a.corrector)
    model = T5ForConditionalGeneration.from_pretrained(a.corrector).to(device).eval()
    train_lines = [normalise(t) for t in read_labels(a.dataset, "train")]
    lex, fmt = lexicon(train_lines), Format(train_lines)
    ctx_window = json.load(open(os.path.join(a.corrector, "train_meta.json"))).get("context", True)

    cache_path = a.out.replace(".json", "_t5cache.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

    def t5(split, hyps):
        key = f"{split}:{ctx_window}:{len(hyps)}"
        if key not in cache:
            cache[key] = generate(model, tok, with_context(a.dataset, [normalise(h) for h in hyps], int(ctx_window)),
                                  device, beams=a.beams)
            json.dump(cache, open(cache_path, "w"))
        return cache[key]

    def run(hyps, eps, cands, confs, theta, mode, how, eps_thr):
        out, n = [], 0
        for h, e, c, s in zip(hyps, eps, cands, confs):
            flag, oov = detect(normalise(h), lex, e, mode, eps_thr)
            if flag and s > theta:
                c = fmt.restore(c)
                new = c if how == "line" else apply_segments(normalise(h), c, oov)
                new = fmt.restore(new)
                n += new != normalise(h)
                out.append(new)
            else:
                out.append(h)
        return out, n

    v_refs, v_hyps, v_eps = read_preds(a.val_preds)
    v_c, v_s = t5("valid", v_hyps)
    modes = ["all", "oov"] + (["eps", "oov_or_eps"] if not np.isnan(v_eps).all() else [])
    eps_thrs = [float(np.nanquantile(v_eps, q)) for q in (0.5, 0.75, 0.9)] if not np.isnan(v_eps).all() else [0.0]
    thetas = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.01]
    sweep = []
    for mode in modes:
        for how in ("line", "segment"):
            for et in (eps_thrs if "eps" in mode else [0.0]):
                for th in thetas:
                    hyp, n = run(v_hyps, v_eps, v_c, v_s, th, mode, how, et)
                    sweep.append(dict(detect=mode, apply=how, eps_thr=et, theta=th, n_changed=n,
                                      cer=Me.cer(v_refs, hyp), wer=Me.wer(v_refs, hyp)))
    raw_v = dict(cer=Me.cer(v_refs, v_hyps), wer=Me.wer(v_refs, v_hyps))
    best = min(sweep, key=lambda r: (r["cer"], r["wer"]))
    t_refs, t_hyps, t_eps = read_preds(a.test_preds)
    t_c, t_s = t5("test", t_hyps)
    t_out, t_n = run(t_hyps, t_eps, t_c, t_s, best["theta"], best["detect"], best["apply"], best["eps_thr"])
    t_all, _ = run(t_hyps, t_eps, t_c, t_s, -1.0, "all", "line", 0.0)
    res = dict(dataset=a.dataset, base=a.tag, corrector=os.path.abspath(a.corrector), context_window=int(ctx_window),
               val_preds=os.path.abspath(a.val_preds), test_preds=os.path.abspath(a.test_preds),
               selected=best, valid_raw=raw_v, n_test_changed=t_n,
               test_raw=dict(cer=Me.cer(t_refs, t_hyps), wer=Me.wer(t_refs, t_hyps), ser=Me.ser(t_refs, t_hyps)),
               test_t5=dict(cer=Me.cer(t_refs, t_out), wer=Me.wer(t_refs, t_out), ser=Me.ser(t_refs, t_out)),
               test_t5_ungated=dict(cer=Me.cer(t_refs, t_all), wer=Me.wer(t_refs, t_all)),
               sweep=sweep, selection="(detect, apply, eps_thr, theta) minimising VALIDATION CER; test decoded once")
    json.dump(res, open(a.out, "w"), indent=1)
    with open(a.out.replace(".json", "_test_preds.tsv"), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["idx", "ref", "raw", "t5_candidate", "t5_confidence", "final"])
        for i in range(len(t_refs)):
            w.writerow([i, t_refs[i], t_hyps[i], t_c[i], f"{t_s[i]:.4f}", t_out[i]])
    ce, cl, we, wl = Me.per_line_counts(t_refs, t_out)
    np.savez_compressed(a.out.replace(".json", "_test_counts.npz"), char_err=ce, char_len=cl, word_err=we, word_len=wl)
    print(f"[{a.tag}] valid raw {100*raw_v['cer']:.2f} -> {100*best['cer']:.2f} ({best['detect']}/{best['apply']}/"
          f"theta {best['theta']}); test raw {100*res['test_raw']['cer']:.2f} -> T5 {100*res['test_t5']['cer']:.2f} "
          f"(ungated {100*res['test_t5_ungated']['cer']:.2f}), WER {100*res['test_raw']['wer']:.2f} -> "
          f"{100*res['test_t5']['wer']:.2f}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pairs")
    p.add_argument("--combined", required=True, help="combined archive the fold models were trained on")
    p.add_argument("--fold-ckpts", nargs="+", required=True)
    p.add_argument("--synthetic-per-line", type=int, default=2)
    p.add_argument("--extra-text", nargs="*", default=[], metavar="CORPUS=PATH:N",
                   help="external clean text corrupted with the corpus's observed error profile (gated: no test line)")
    p.add_argument("--seed", type=int, default=1); p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    p = sub.add_parser("train")
    p.add_argument("--pairs-dir", required=True); p.add_argument("--out", required=True)
    p.add_argument("--model", default="t5-small"); p.add_argument("--context", type=int, default=1)
    p.add_argument("--epochs", type=int, default=8); p.add_argument("--patience", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64); p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--max-pairs", type=int, default=0, help="smoke tests only")
    p.add_argument("--seed", type=int, default=1); p.add_argument("--gpu", type=int, default=0)
    p = sub.add_parser("apply")
    p.add_argument("--dataset", required=True, choices=CORPORA + ("saintgall_nopt",))
    p.add_argument("--val-preds", required=True); p.add_argument("--test-preds", required=True)
    p.add_argument("--corrector", required=True); p.add_argument("--tag", required=True)
    p.add_argument("--out", required=True); p.add_argument("--beams", type=int, default=4)
    p.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args(argv)
    dict(pairs=cmd_pairs, train=cmd_train, apply=cmd_apply)[a.cmd](a)


if __name__ == "__main__":
    main()

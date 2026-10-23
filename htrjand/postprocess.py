"""Earlier post-processing module: T5 corrector and a lexicon-based correction,
implemented without any leakage:

* correction pairs come from K-fold CROSS-DECODING of the TRAINING split
  (a model trained on the complement of fold k decodes fold k), so their error
  profile resembles test time; the paper's "systematic errors introduced from
  observed patterns"  are added by corrupting clean training lines with
  the confusions observed in those pairs;
* the lexicon is the TRAINING-split word list of the corpus;
* the T5-small corrector is fine-tuned on those pairs only;
* the gate threshold theta (T5 post-processing) and the lexicon on/off choice are
  chosen on VALIDATION predictions; the frozen setting is applied ONCE to the
  test predictions, and every condition is reported beside the raw CER.

Subcommands:
    pairs     cross-decode the training split with fold models -> pairs TSV
    train-t5  fine-tune t5-small on the pairs
    apply     sweep theta on validation, apply once to test, write the four conditions
"""
import argparse
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

from . import data as D
from . import metrics as Me
from . import models as M

from ._compat import REPO_ROOT  # noqa: E402  (source checkout, else the working directory)
PREFIX = "correct: "


# --------------------------------------------------------------------------- pairs
def read_preds(path):
    idx, eps, refs, hyps = [], [], [], []
    with open(path) as fh:
        r = csv.reader(fh, delimiter="\t")
        header = next(r)
        for row in r:
            idx.append(int(row[0]))
            if "eps_hat" in header:
                eps.append(float(row[1])); refs.append(row[2]); hyps.append(row[3])
            else:
                eps.append(float("nan")); refs.append(row[1]); hyps.append(row[2])
    return np.array(eps), refs, hyps


def confusions(pairs):
    """Character-level edit statistics of (noisy, clean) pairs: substitution
    counts per clean character, deletion and insertion counts, and the per-line
    error rate, from an alignment by difflib."""
    import difflib
    subs, dels, ins = defaultdict(Counter), Counter(), Counter()
    n_err = n_char = 0
    for noisy, clean in pairs:
        sm = difflib.SequenceMatcher(a=clean, b=noisy, autojunk=False)
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == "replace":
                for c, n in zip(clean[i1:i2], noisy[j1:j2]):
                    subs[c][n] += 1
                n_err += max(i2 - i1, j2 - j1)
            elif op == "delete":
                for c in clean[i1:i2]:
                    dels[c] += 1
                n_err += i2 - i1
            elif op == "insert":
                for c in noisy[j1:j2]:
                    ins[c] += 1
                n_err += j2 - j1
        n_char += len(clean)
    return dict(subs=subs, dels=dels, ins=ins, rate=n_err / max(n_char, 1))


def corrupt(text, stats, rng, rate=None):
    """Introduce errors into a clean line following the observed confusions
    ('systematic errors ... based on these observed patterns')."""
    rate = stats["rate"] if rate is None else rate
    ins_chars = list(stats["ins"].elements()) or [" "]
    out = []
    for c in text:
        r = rng.random()
        if r < rate:
            choice = rng.random()
            if c in stats["subs"] and choice < 0.7:
                cands = list(stats["subs"][c].elements())
                out.append(rng.choice(cands))
            elif choice < 0.85:
                continue                                    # deletion
            else:
                out.append(c); out.append(rng.choice(ins_chars))   # insertion
        else:
            out.append(c)
    return "".join(out)


@torch.no_grad()
def cmd_pairs(a):
    device = torch.device("cpu" if a.gpu < 0 or not torch.cuda.is_available() else f"cuda:{a.gpu}")
    K = len(a.fold_ckpts)
    rows, stats = [], []
    from torch.utils.data import DataLoader
    for k, ck_path in enumerate(a.fold_ckpts):
        model, ck = M.from_checkpoint(ck_path)
        model.to(device)
        tok = D.Tokenizer(ck["charset"])
        ds = D.LineDataset(a.dataset, "train", tok, normalisation=ck.get("normalisation", "paper"),
                           root=a.data_dir, fold=("holdout", k, K))
        dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, collate_fn=D.collate)
        hyps = []
        for img, _, _ in dl:
            hyps += tok.ctc_greedy_decode(model(img.to(device)))
        refs = list(ds.labels)
        cer = Me.cer(refs, hyps)
        stats.append(dict(fold=k, n=len(refs), cer=cer, ckpt=os.path.abspath(ck_path)))
        print(f"fold {k}: {len(refs)} held-out lines, cross-decoded CER {100*cer:.2f} %", flush=True)
        rows += list(zip(hyps, refs))
    real = list(dict.fromkeys(rows))
    st = confusions(real)
    rng = random.Random(a.seed)
    _, train_labels = D.read_split(a.dataset, "train", a.data_dir)
    synth = [(corrupt(t, st, rng), t) for t in train_labels for _ in range(a.synthetic_per_line)]
    synth = [(n, c) for n, c in synth if n != c]
    all_rows = real + synth
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["noisy", "clean"])
        for n, c in all_rows:
            w.writerow([n, c])
    ident = sum(1 for n, c in real if n == c)
    meta = dict(dataset=a.dataset, K=K, folds=stats, n_real=len(real), n_identity=ident,
                n_synthetic=len(synth), observed_char_error_rate=st["rate"],
                top_substitutions={c: dict(cnt.most_common(3)) for c, cnt in
                                   sorted(st["subs"].items(), key=lambda kv: -sum(kv[1].values()))[:15]})
    json.dump(meta, open(a.out.replace(".tsv", "_meta.json"), "w"), indent=1)
    print(f"wrote {len(all_rows)} pairs ({len(real)} cross-decoded, {ident} identity, "
          f"{len(synth)} synthetic) -> {a.out}")


# --------------------------------------------------------------------------- T5
def cmd_train_t5(a):
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device("cpu" if a.gpu < 0 or not torch.cuda.is_available() else f"cuda:{a.gpu}")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = T5ForConditionalGeneration.from_pretrained(a.model).to(device)
    pairs = []
    with open(a.pairs) as fh:
        r = csv.reader(fh, delimiter="\t"); next(r)
        pairs = [row for row in r if len(row) == 2]

    class PairSet(Dataset):
        def __len__(self):
            return len(pairs)

        def __getitem__(self, i):
            n, c = pairs[i]
            return (tok(PREFIX + n, max_length=a.max_len, truncation=True)["input_ids"],
                    tok(c, max_length=a.max_len, truncation=True)["input_ids"])

    def collate(batch):
        xs, ys = zip(*batch)
        mx, my = max(map(len, xs)), max(map(len, ys))
        X = torch.full((len(xs), mx), tok.pad_token_id, dtype=torch.long)
        A = torch.zeros((len(xs), mx), dtype=torch.long)
        Y = torch.full((len(ys), my), -100, dtype=torch.long)
        for i, (x, y) in enumerate(zip(xs, ys)):
            X[i, :len(x)] = torch.tensor(x); A[i, :len(x)] = 1; Y[i, :len(y)] = torch.tensor(y)
        return X, A, Y

    dl = DataLoader(PairSet(), batch_size=a.batch_size, shuffle=True, collate_fn=collate, num_workers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / max(steps, 1)))
    model.train()
    for ep in range(a.epochs):
        tot = 0.0
        for X, A, Y in dl:
            X, A, Y = X.to(device), A.to(device), Y.to(device)
            loss = model(input_ids=X, attention_mask=A, labels=Y).loss
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); tot += loss.item()
        print(f"[t5] epoch {ep} loss {tot/len(dl):.4f}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out); tok.save_pretrained(a.out)
    json.dump(dict(model=a.model, pairs=os.path.abspath(a.pairs), n_pairs=len(pairs), epochs=a.epochs,
                   lr=a.lr, batch_size=a.batch_size, seed=a.seed), open(os.path.join(a.out, "train_meta.json"), "w"), indent=1)
    print("saved", a.out)


# --------------------------------------------------------------------------- lexicon-based correction
class Lexicon:
    """Word list from the TRAINING transcriptions only. A hypothesis word that is
    not in the list is replaced by the most frequent training word within
    Levenshtein distance 1 of it, if one exists (the historical rule, minus the
    masked-LM re-ranking and minus its test-set lexicon)."""

    def __init__(self, train_labels):
        self.freq = Counter(w for t in train_labels for w in t.split())
        self.words = set(self.freq)
        self.by_len = defaultdict(list)
        for w in self.words:
            self.by_len[len(w)].append(w)

    def correct_word(self, w):
        import editdistance
        core = w.strip(".,;:!?\"'()")
        if not core or core in self.words or core.isdigit() or len(core) <= 1:
            return w
        best, best_f = None, 0
        for L in (len(core) - 1, len(core), len(core) + 1):
            for cand in self.by_len.get(L, ()):
                if editdistance.eval(core, cand) <= 1 and self.freq[cand] > best_f:
                    best, best_f = cand, self.freq[cand]
        return w.replace(core, best) if best else w

    def correct(self, text):
        return " ".join(self.correct_word(w) for w in text.split())


# --------------------------------------------------------------------------- apply (T5 post-processing)
@torch.no_grad()
def t5_correct(model, tok, texts, device, batch=32, max_len=192, beams=4):
    out = []
    for i in range(0, len(texts), batch):
        enc = tok([PREFIX + t for t in texts[i:i + batch]], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len).to(device)
        gen = model.generate(**enc, max_length=max_len, num_beams=beams, early_stopping=True,
                             no_repeat_ngram_size=3, repetition_penalty=1.2)
        # IAM references keep a space before punctuation and clitics ("there 's", "trouble ,");
        # the tokenizer's default clean-up would delete those spaces and be charged as errors
        out += tok.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return out


def cmd_apply(a):
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    device = torch.device("cpu" if a.gpu < 0 or not torch.cuda.is_available() else f"cuda:{a.gpu}")
    tok = AutoTokenizer.from_pretrained(a.corrector)
    model = T5ForConditionalGeneration.from_pretrained(a.corrector).to(device).eval()
    _, train_labels = D.read_split(a.dataset, "train", a.data_dir)
    lex = Lexicon(train_labels)

    import editdistance

    def accept(e, h, c, theta, delta):
        """T5 post-processing: apply a candidate only when the recogniser's error
        estimate exceeds theta and the candidate stays within a relative edit
        distance delta of the recognised line (a candidate that rewrites the
        line is not a correction)."""
        return e > theta and editdistance.eval(h, c) <= delta * max(len(h), 1)

    import difflib

    def word_merge(h, c, wdelta):
        """T5 post-processing at word level: align the recognised line with the
        candidate and take the candidate's words only where every recognised word
        in the aligned segment is absent from the training lexicon (DetectErrors
        with dictionary D), the segment keeps its length within one word, and the
        candidate segment stays within a relative edit distance wdelta of the
        recognised one (a replacement by an unrelated word is not a correction)."""
        hw, cw = h.split(), c.split()
        out = []
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=hw, b=cw, autojunk=False).get_opcodes():
            seg = hw[i1:i2]
            if op == "replace" and seg and all(w.strip(".,;:!?\"'()") not in lex.words for w in seg) \
                    and abs((i2 - i1) - (j2 - j1)) <= 1:
                a_txt, b_txt = " ".join(seg), " ".join(cw[j1:j2])
                if editdistance.eval(a_txt, b_txt) <= wdelta * max(len(a_txt), 1):
                    out += cw[j1:j2]
                    continue
            out += seg
        return " ".join(out)

    lex_cache = {}

    def lex_lines(hyps):
        key = id(hyps)
        if key not in lex_cache:
            lex_cache[key] = [lex.correct(h) for h in hyps]
        return lex_cache[key]

    def conditions(eps, hyps, t5, theta, delta=1.0):
        sel = [c if accept(e, h, c, theta, delta) else h for e, h, c in zip(eps, hyps, t5)]
        wsel = [word_merge(h, c, delta if delta < 1.0 else 0.34) if e > theta else h
                for e, h, c in zip(eps, hyps, t5)]
        return {"raw": hyps, "lexicon": lex_lines(hyps), "t5_all": t5,
                "t5_selective": sel, "t5_word_selective": wsel,
                "t5_selective_lexicon": [lex.correct(s) for s in sel]}

    # 1. validation: T5 outputs once, theta sweep, condition choice
    cache_path = a.out.replace(".json", "_t5cache.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

    def cached_t5(key, hyps):
        if key not in cache or len(cache[key]) != len(hyps):
            cache[key] = t5_correct(model, tok, hyps, device)
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            json.dump(cache, open(cache_path, "w"))
        return cache[key]

    v_eps, v_refs, v_hyps = read_preds(a.val_preds)
    v_t5 = cached_t5("valid", v_hyps)
    thetas = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3, 1.01]
    deltas = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    sweep = []
    for th in thetas:
        for de in deltas:
            conds = conditions(v_eps, v_hyps, v_t5, th, de)
            n_acc = sum(accept(e, h, c, th, de) for e, h, c in zip(v_eps, v_hyps, v_t5))
            sweep.append(dict(theta=th, delta=de, n_corrected=int(n_acc),
                              val_cer={k: Me.cer(v_refs, v) for k, v in conds.items()}))
    # choose (theta, delta) by the selective T5 condition; report every condition there
    best = min(sweep, key=lambda d: min(d["val_cer"]["t5_selective"], d["val_cer"]["t5_word_selective"]))
    theta, delta = best["theta"], best["delta"]
    val_best_cond = min(best["val_cer"], key=best["val_cer"].get)
    print(f"theta = {theta}, delta = {delta}, {best['n_corrected']} validation lines corrected (validation: "
          + ", ".join(f"{k} {100*v:.2f}" for k, v in best["val_cer"].items()) + ")")

    # 2. test: once
    t_eps, t_refs, t_hyps = read_preds(a.test_preds)
    t_t5 = cached_t5("test", t_hyps)
    conds = conditions(t_eps, t_hyps, t_t5, theta, delta)
    res = dict(dataset=a.dataset, corrector=os.path.abspath(a.corrector), theta=theta, delta=delta,
               n_test=len(t_refs),
               n_corrected=int(sum(accept(e, h, c, theta, delta) for e, h, c in zip(t_eps, t_hyps, t_t5))),
               validation_best_condition=val_best_cond, val_sweep=sweep,
               test={k: dict(cer=Me.cer(t_refs, v), wer=Me.wer(t_refs, v), ser=Me.ser(t_refs, v))
                     for k, v in conds.items()},
               note="post-processed results; the raw row is the recogniser alone")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    with open(a.out.replace(".json", "_preds.tsv"), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["idx", "eps_hat", "ref"] + list(conds))
        for i in range(len(t_refs)):
            w.writerow([i, f"{t_eps[i]:.6f}", t_refs[i]] + [conds[k][i] for k in conds])
    print(json.dumps({k: {m: round(100 * x, 2) for m, x in v.items()} for k, v in res["test"].items()}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pairs"); p.add_argument("--dataset", required=True)
    p.add_argument("--fold-ckpts", nargs="+", required=True, help="fold-k checkpoints in order k = 0..K-1")
    p.add_argument("--synthetic-per-line", type=int, default=1, help="corrupted copies of each training line ")
    p.add_argument("--seed", type=int, default=1); p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--data-dir", default=None); p.add_argument("--out", required=True)
    p = sub.add_parser("train-t5"); p.add_argument("--pairs", required=True); p.add_argument("--out", required=True)
    p.add_argument("--model", default="t5-small", help="the paper names T5-small (60 M)")
    p.add_argument("--epochs", type=int, default=5); p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--max-len", type=int, default=192)
    p.add_argument("--seed", type=int, default=1); p.add_argument("--gpu", type=int, default=0)
    p = sub.add_parser("apply"); p.add_argument("--dataset", required=True)
    p.add_argument("--val-preds", required=True); p.add_argument("--test-preds", required=True)
    p.add_argument("--corrector", required=True); p.add_argument("--out", required=True)
    p.add_argument("--data-dir", default=None); p.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args(argv)
    {"pairs": cmd_pairs, "train-t5": cmd_train_t5, "apply": cmd_apply}[a.cmd](a)


if __name__ == "__main__":
    main()

"""CTC decoding.

Three decoders, in increasing cost:

  greedy          argmax per frame, collapse repeats, drop blanks.
  beam            CTC prefix beam search, no language model.
  beam + char LM  the same search with a character n-gram estimated from the
                  TRAINING transcriptions of the corpus being decoded.

The language model deserves a word on provenance. It is fitted on the same
6,161 IAM (or 10,192 RIMES) training lines the recogniser is fitted on and on
nothing else: no Brown corpus, no LOB, no web text, no external lexicon. That
keeps it inside the no-external-data setting, and it is a weaker model than the
word-level LMs most published IAM numbers use. Results are reported with and
without it in every table so the two settings are never conflated.

Release note: this module was vendored unchanged from the original working
repository (it was imported from there through sys.path by enhance_decode.py
and ensemble_rescore.py). `CompactCharLM` below was added for the public release:
the same stupid-backoff model, stored compactly, with optional count pruning.
"""
import gzip
import json
import math
from collections import Counter
import numpy as np
from collections import defaultdict

NEG_INF = -1e30


def _logsumexp(a, b):
    if a <= NEG_INF:
        return b
    if b <= NEG_INF:
        return a
    m = a if a > b else b
    return m + math.log(math.exp(a - m) + math.exp(b - m))


class CharLM:
    """Katz-style character n-gram with stupid backoff.

    Stupid backoff rather than Kneser-Ney because the search calls the model
    once per (prefix, character) pair per frame and needs a constant-time
    lookup; with a backoff factor of 0.4 the scores are not normalised
    probabilities, which is harmless inside a beam search that only ranks.
    """

    def __init__(self, order=8, alpha=0.4):
        self.order, self.alpha = order, alpha
        self.counts = [defaultdict(int) for _ in range(order + 1)]
        self.totals = [defaultdict(int) for _ in range(order + 1)]
        self.vocab = set()

    def fit(self, texts, bos="\x02"):
        for t in texts:
            self.vocab.update(t)
            s = bos * (self.order - 1) + t
            for n in range(1, self.order + 1):
                for i in range(self.order - 1, len(s)):
                    ctx, ch = s[i - n + 1:i], s[i]
                    self.counts[n][(ctx, ch)] += 1
                    self.totals[n][ctx] += 1
        self.V = max(len(self.vocab), 1)
        return self

    def logp(self, ctx, ch):
        """log P(ch | ctx) with stupid backoff; ctx is the recent history."""
        w = 0.0
        for n in range(self.order, 0, -1):
            c = ctx[-(n - 1):] if n > 1 else ""
            if len(ctx) < n - 1:
                continue
            tot = self.totals[n].get(c, 0)
            if tot:
                cnt = self.counts[n].get((c, ch), 0)
                if cnt:
                    return w + math.log(cnt / tot)
            w += math.log(self.alpha)
        return w + math.log(1.0 / self.V)

    def save(self, path):
        json.dump(dict(order=self.order, alpha=self.alpha, vocab=sorted(self.vocab),
                       counts=[{f"{k[0]}\x00{k[1]}": v for k, v in c.items()} for c in self.counts],
                       totals=[dict(t) for t in self.totals]), open(path, "w"))

    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        m = cls(d["order"], d["alpha"])
        m.vocab = set(d["vocab"]); m.V = max(len(m.vocab), 1)
        m.counts = [defaultdict(int, {tuple(k.split("\x00", 1)): v for k, v in c.items()})
                    for c in d["counts"]]
        m.totals = [defaultdict(int, t) for t in d["totals"]]
        return m


class CompactCharLM:
    """`CharLM` with string keys, no begin-of-line padding n-grams, optional pruning,
    and a compact on-disk format (release addition; `CharLM` itself is unchanged).

    Why it is the same model: `prefix_beam_search` queries `logp(prefix, ch)` with
    the recognised prefix only, never with the "\x02" padding `CharLM.fit` adds, and
    `CharLM.logp` skips every order longer than the prefix. The padded n-grams are
    therefore never read, and dropping them changes no score. With `min_count=1`
    `CompactCharLM.logp(ctx, ch) == CharLM.logp(ctx, ch)` for every context without
    "\x02" (tests/test_release.py checks it).

    Pruning (`min_count > 1`, orders >= `prune_from`) removes n-grams seen fewer than
    `min_count` times. Context totals keep their UNPRUNED value, so a retained
    n-gram keeps its unpruned probability and a pruned one backs off exactly as an
    unseen one does (stupid backoff, factor `alpha`).
    """

    MAGIC = "HTRJAND-CHARLM-1"

    def __init__(self, order=8, alpha=0.4):
        self.order, self.alpha = order, alpha
        self.ngrams = [dict() for _ in range(order + 1)]   # n -> {ctx + ch: count}
        self.totals = [dict() for _ in range(order + 1)]   # n -> {ctx: count of ctx}
        self.vocab = set()
        self.V = 1
        self.meta = {}

    def fit(self, texts, min_count=1, prune_from=3):
        grams = [Counter() for _ in range(self.order + 1)]
        for t in texts:
            self.vocab.update(t)
            L = len(t)
            for n in range(1, self.order + 1):
                g = grams[n]
                for j in range(n - 1, L):
                    g[t[j - n + 1:j + 1]] += 1
        self.V = max(len(self.vocab), 1)
        for n in range(1, self.order + 1):
            tot = Counter()
            for k, c in grams[n].items():
                tot[k[:-1]] += c
            keep = min_count if (min_count > 1 and n >= prune_from) else 1
            self.ngrams[n] = {k: c for k, c in grams[n].items() if c >= keep}
            ctxs = {k[:-1] for k in self.ngrams[n]}
            self.totals[n] = {k: c for k, c in tot.items() if k in ctxs}
            grams[n] = None
        self.meta.update(min_count=int(min_count), prune_from=int(prune_from))
        return self

    def logp(self, ctx, ch):
        """log P(ch | ctx) with stupid backoff; identical to CharLM.logp."""
        w = 0.0
        for n in range(self.order, 0, -1):
            if len(ctx) < n - 1:
                continue
            c = ctx[-(n - 1):] if n > 1 else ""
            tot = self.totals[n].get(c, 0)
            if tot:
                cnt = self.ngrams[n].get(c + ch, 0)
                if cnt:
                    return w + math.log(cnt / tot)
            w += math.log(self.alpha)
        return w + math.log(1.0 / self.V)

    def n_entries(self):
        return sum(len(x) for x in self.ngrams) + sum(len(x) for x in self.totals)

    def save(self, path, **meta):
        """gzip text: a JSON header line, then per order one 'N n k' line followed by
        k lines 'gram<TAB>count' (sorted), then 'T n k' and k lines 'ctx<TAB>total'.
        Characters are written verbatim (UTF-8); tab / newline never occur in a line."""
        self.meta.update(meta)
        head = dict(magic=self.MAGIC, order=self.order, alpha=self.alpha, V=self.V,
                    vocab="".join(sorted(self.vocab)), meta=self.meta)
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as f:
            f.write(json.dumps(head, ensure_ascii=False) + "\n")
            for n in range(1, self.order + 1):
                for tag, table in (("N", self.ngrams[n]), ("T", self.totals[n])):
                    f.write(f"{tag} {n} {len(table)}\n")
                    for k in sorted(table):
                        f.write(f"{k}\t{table[k]}\n")
        return path

    @classmethod
    def load(cls, path):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            head = json.loads(f.readline())
            if head.get("magic") != cls.MAGIC:
                raise ValueError(f"{path}: not an HTR-JAND character LM")
            m = cls(head["order"], head["alpha"])
            m.V, m.vocab, m.meta = head["V"], set(head["vocab"]), head.get("meta", {})
            for _ in range(2 * m.order):
                tag, n, k = f.readline().split()
                n, k = int(n), int(k)
                table = {}
                for _ in range(k):
                    line = f.readline()[:-1]
                    key, _, cnt = line.rpartition("\t")
                    table[key] = int(cnt)
                (m.ngrams if tag == "N" else m.totals)[n] = table
        return m


def load_lm(path):
    """A release LM file (CompactCharLM, .gz) or a CharLM JSON written by CharLM.save."""
    return CompactCharLM.load(path) if str(path).endswith(".gz") else CharLM.load(path)


def greedy_decode(logp, lengths, chars, blank, unk=1):
    ids = logp.argmax(-1)
    out = []
    for b in range(ids.shape[0]):
        prev, buf = -1, []
        for t in range(int(lengths[b])):
            k = int(ids[b, t])
            if k != prev and k != blank and k > unk:
                buf.append(chars[k])
            prev = k
        out.append("".join(buf))
    return out


def prefix_beam_search(logp, length, chars, blank, beam=16, lm=None, alpha=0.5,
                       beta=0.6, prune=-9.0, unk=1, nbest=0):
    """Standard CTC prefix beam search (Hannun et al., 2014).

    Each beam entry keeps two scores: p_b, the probability that the prefix has
    been emitted and the last frame was blank, and p_nb, that it was the prefix's
    own last character. Keeping them apart is what lets the search decide
    whether a repeated character collapses or doubles. `alpha` weights the
    language model and `beta` rewards length, which offsets the LM's systematic
    preference for short strings.
    """
    T = int(length)
    beams = {(): (0.0, NEG_INF, 0.0)}          # prefix -> (p_b, p_nb, lm_score)
    for t in range(T):
        row = logp[t]
        cand = np.where(row > prune)[0]
        if blank not in cand:
            cand = np.append(cand, blank)
        nxt = {}

        def put(pref, pb, pnb, lms):
            o = nxt.get(pref)
            if o is None:
                nxt[pref] = (pb, pnb, lms)
            else:
                nxt[pref] = (_logsumexp(o[0], pb), _logsumexp(o[1], pnb), lms)

        for pref, (pb, pnb, lms) in beams.items():
            ptot = _logsumexp(pb, pnb)
            for k in cand:
                p = float(row[k])
                if k == blank or k <= unk:
                    put(pref, ptot + p, NEG_INF, lms)
                    continue
                if pref and pref[-1] == k:
                    # extend without growing the prefix (repeat collapses)
                    put(pref, NEG_INF, pnb + p, lms)
                    # grow only from a blank-terminated path
                    new = pref + (k,)
                    put(new, NEG_INF, pb + p,
                        lms + (lm.logp("".join(chars[i] for i in pref), chars[k]) if lm else 0.0))
                else:
                    new = pref + (k,)
                    put(new, NEG_INF, ptot + p,
                        lms + (lm.logp("".join(chars[i] for i in pref), chars[k]) if lm else 0.0))
        scored = sorted(nxt.items(),
                        key=lambda kv: -(_logsumexp(kv[1][0], kv[1][1])
                                         + alpha * kv[1][2] + beta * len(kv[0])))
        beams = dict(scored[:beam])
    def total(kv):
        return _logsumexp(kv[1][0], kv[1][1]) + alpha * kv[1][2] + beta * len(kv[0])

    ranked = sorted(beams.items(), key=total, reverse=True)
    if nbest:
        # The n-best list the attention decoder rescores. Each entry carries the
        # search's own score split into its acoustic and language parts, so the
        # rescoring weight can be retuned without re-running the search.
        return [dict(ids=list(pref),
                     text="".join(chars[i] for i in pref),
                     ctc=_logsumexp(v[0], v[1]), lm=v[2], n=len(pref))
                for pref, v in ranked[:nbest]]
    return "".join(chars[i] for i in ranked[0][0])


def beam_decode(logp, lengths, chars, blank, beam=16, lm=None, alpha=0.5, beta=0.6,
                prune=-9.0, unk=1, workers=0):
    args = [(logp[b], lengths[b]) for b in range(logp.shape[0])]
    if workers and workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        import functools
        f = functools.partial(_one, chars=chars, blank=blank, beam=beam, lm=lm,
                              alpha=alpha, beta=beta, prune=prune, unk=unk)
        with ProcessPoolExecutor(workers) as ex:
            return list(ex.map(f, args, chunksize=8))
    return [prefix_beam_search(a[0], a[1], chars, blank, beam, lm, alpha, beta, prune, unk)
            for a in args]


def _one(a, **kw):
    return prefix_beam_search(a[0], a[1], **kw)

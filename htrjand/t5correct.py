"""Gated T5 refinement of recognised lines with the released corrector (release addition).

Inference-only form of T5 post-processing as implemented in `t5post.py` (`cmd_apply`), with the
gate chosen on validation and stored in the recogniser's config.json:

  input       t5post.normalise(hyp); `correct <Language> <corpus>: <line>` and, for a corrector
              trained with context, ` | before: <prev> | after: <next>` (empty when the line
              has no neighbours, as for the first and last line of a validation split)
  candidate   T5 beam search (beam 4, max length 192), confidence = exp(length-normalised
              beam log-probability)  -- t5post.generate
  detect      "all" (every line), "oov" (a word absent from the lexicon), "eps" (the
              recogniser's error estimate above eps_thr), or "oov_or_eps"  -- t5post.detect
  apply       only when detected AND confidence > theta: the whole line ("line") or only the
              aligned word segments holding an out-of-lexicon word ("segment")
  format      the corpus's punctuation-spacing rules (t5post.Format, learned from training
              transcriptions; only the per-mark fractions are shipped)

Lexicon: word lists built from the Wikipedia-derived text (`lexicon/wiki_<lang>.txt.gz` of the
CC BY-SA character-LM repo, release.CHARLM); the experiments used each corpus's training-split
lexicon, which is not redistributed (docs/RELEASE_LM_EVAL.md reports the validation effect).
Requires `pip install transformers` (and sentencepiece is NOT needed: tokenizer.json is shipped).
"""
import gzip
import json
import os

import numpy as np

from . import t5post as P

LEX_LANG = dict(iam="en", rimes="fr", bentham="en", saintgall="la", washington="en", saintgall_nopt="la")


def _resolve(name_or_path, revision=None, cache_dir=None, token=None):
    if os.path.isdir(name_or_path):
        return os.path.abspath(name_or_path)
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=name_or_path, revision=revision, cache_dir=cache_dir, token=token)


def load_lexicon(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return {l.rstrip("\n") for l in f if l.strip()}


def make_format(rules):
    f = P.Format.__new__(P.Format)
    f.rule = {c: tuple(v) for c, v in rules.items()}
    return f


class T5Corrector:
    def __init__(self, root, device="cpu", resources=None, cache_dir=None, token=None):
        import torch
        try:
            from transformers import AutoTokenizer, T5ForConditionalGeneration
            from transformers.utils import logging as hf_logging
            hf_logging.disable_progress_bar()
        except ImportError as e:
            raise ImportError("T5 refinement needs transformers: pip install transformers "
                              "(or pip install 'htrjand[t5]')") from e
        self.root = root
        self.device = torch.device(device)
        self.tok = AutoTokenizer.from_pretrained(root)
        self.model = T5ForConditionalGeneration.from_pretrained(root).to(self.device).eval()
        meta = json.load(open(os.path.join(root, "htrjand_corrector.json"), encoding="utf-8"))
        self.meta = meta
        self.context_window = int(meta["context_window"])
        self.formats = {c: make_format(r) for c, r in meta["format_rules"].items()}
        self.resources = resources or meta.get("lexicon_repo")      # local dir or Hub repo id
        self.cache_dir, self.token = cache_dir, token
        self._lex = {}

    @classmethod
    def from_pretrained(cls, name_or_path, revision=None, device="cpu", cache_dir=None, token=None, resources=None):
        return cls(_resolve(name_or_path, revision, cache_dir, token), device, resources, cache_dir, token)

    def lexicon_path(self, lang):
        rel = self.meta["lexicons"][lang]
        if self.resources and os.path.isdir(self.resources):
            return os.path.join(self.resources, rel)
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=self.resources, filename=rel, cache_dir=self.cache_dir, token=self.token)

    def lexicon(self, corpus):
        lang = LEX_LANG[corpus]
        if lang not in self._lex:
            self._lex[lang] = load_lexicon(self.lexicon_path(lang))
        return self._lex[lang]

    def generate(self, sources, beams=4, max_len=192, batch=48):
        return P.generate(self.model, self.tok, sources, self.device, batch=batch, max_len=max_len, beams=beams)

    def sources(self, corpus, hyps, context=True):
        hn = [P.normalise(h) for h in hyps]
        if self.context_window and not context:
            # a corrector trained with context always saw the separators; isolated lines
            # get empty neighbours, exactly like the first/last line of a split
            return [P.source(corpus, h, "", "") for h in hn]
        return P.with_context(corpus, hn, self.context_window)

    def correct(self, hyps, corpus, gate, eps=None, context=True, beams=4):
        """hyps: recognised lines (in reading order if context=True). Returns a list of
        (final text, T5 confidence, applied?) -- the gate of t5post.cmd_apply.run."""
        eps = [float("nan")] * len(hyps) if eps is None else eps
        cands, confs = self.generate(self.sources(corpus, hyps, context), beams=beams)
        return self.apply_gate(hyps, eps, cands, confs, corpus, gate)

    def apply_gate(self, hyps, eps, cands, confs, corpus, gate):
        lex, fmt = self.lexicon(corpus), self.formats[corpus]
        out = []
        for h, e, c, s in zip(hyps, eps, cands, confs):
            flag, oov = P.detect(P.normalise(h), lex, e, gate["detect"], gate["eps_thr"])
            if flag and s > gate["theta"]:
                c2 = fmt.restore(c)
                new = c2 if gate["apply"] == "line" else P.apply_segments(P.normalise(h), c2, oov)
                new = fmt.restore(new)
                out.append((new, float(s), new != P.normalise(h)))
            else:
                out.append((h, float(s), False))
        return out

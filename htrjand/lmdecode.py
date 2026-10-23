"""Character-LM decoding of one recogniser (release addition).

This is the single-member case of `ensemble_rescore.py`, the decoder behind every
"char LM" number of the HTR-JAND results, reorganised to work line by line:

  1. CTC prefix beam search (`decode.prefix_beam_search`, beam 16, prune -9) with the LM
     inside the search at weight gen_alpha and length bonus gen_beta returns the n-best
     transcriptions (n = 8); empty strings are dropped and the set is sorted;
  2. every candidate c is scored log p_CTC(c | x) (exact, `ensemble_rescore.ctc_scores`)
     + alpha * log P_LM(c) + beta * |c|;
  3. the arg-max is returned (first in sorted order on ties).
gen_alpha / gen_beta are the `ensemble_rescore.py` defaults (0.2 / 0.5); alpha and beta are
chosen on validation (`scripts/eval_release_decoders.py`) and stored in config.json.
"""
import numpy as np
import torch

from . import decode as Dec
from .ensemble_rescore import ctc_scores


def lm_score(lm, text):
    """Sum of log P_LM over the characters of `text` (same loop as ensemble_rescore._lm_score)."""
    s, ctx = 0.0, ""
    for ch in text:
        s += lm.logp(ctx, ch)
        ctx = (ctx + ch)[-(lm.order - 1):]
    return s


def candidates(lp, tok, lm, beam=16, nbest=8, gen_alpha=0.2, gen_beta=0.5, prune=-9.0):
    hs = Dec.prefix_beam_search(lp, lp.shape[0], tok.chars, tok.BLANK, beam, lm, gen_alpha, gen_beta,
                                prune, 1, nbest)
    return sorted({h["text"] for h in hs if h["text"]}) or [""]


def score_candidates(lp, tok, cands, lm, device="cpu"):
    """(ctc log-likelihoods, LM log-probabilities, lengths) of every candidate."""
    s = ctc_scores(np.ascontiguousarray(lp, dtype=np.float32), tok, cands, torch.device(device))
    return s, np.array([lm_score(lm, c) for c in cands]), np.array([len(c) for c in cands])


def pick(cands, s, l, n, alpha, beta):
    return cands[int(np.argmax(s + alpha * l + beta * n))]


def nbest_rescore(lp, tok, lm, beam=16, nbest=8, gen_alpha=0.2, gen_beta=0.5, alpha=0.2, beta=1.0,
                  prune=-9.0, device="cpu"):
    """lp: (T, V) log-softmax of one line -> best transcription."""
    c = candidates(lp, tok, lm, beam, nbest, gen_alpha, gen_beta, prune)
    s, l, n = score_candidates(lp, tok, c, lm, device)
    return pick(c, s, l, n, alpha, beta)

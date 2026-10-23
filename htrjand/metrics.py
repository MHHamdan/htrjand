"""CER / WER / SER, corpus-aggregated, plus the per-line counts used by the
paired bootstrap. No text normalisation is applied: hypothesis and reference
are compared as the archive stores them (case, punctuation and inner spaces
all count), which is also what the paper-era script measured with jiwer.
"""
import numpy as np

from ._compat import lazy

editdistance = lazy("editdistance", "pip install editdistance")


def per_line_counts(refs, hyps):
    ce = np.array([editdistance.eval(r, h) for r, h in zip(refs, hyps)], dtype=float)
    cl = np.array([len(r) for r in refs], dtype=float)
    we = np.array([editdistance.eval(r.split(), h.split()) for r, h in zip(refs, hyps)], dtype=float)
    wl = np.array([len(r.split()) for r in refs], dtype=float)
    return ce, cl, we, wl


def cer(refs, hyps):
    ce, cl, _, _ = per_line_counts(refs, hyps)
    return float(ce.sum() / max(cl.sum(), 1))


def wer(refs, hyps):
    _, _, we, wl = per_line_counts(refs, hyps)
    return float(we.sum() / max(wl.sum(), 1))


def ser(refs, hyps):
    return float(sum(r != h for r, h in zip(refs, hyps)) / max(len(refs), 1))


def bootstrap_ci(err, length, n_boot=2000, seed=0, alpha=0.05):
    """Percentile bootstrap over lines for a ratio-of-sums rate."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(err), size=(n_boot, len(err)))
    vals = err[idx].sum(axis=1) / np.maximum(length[idx].sum(axis=1), 1)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_bootstrap(err_a, err_b, length, n_boot=2000, seed=0):
    """Paired bootstrap for rate_a - rate_b over the same lines.
    Returns (observed difference, lo, hi, two-sided p)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(length), size=(n_boot, len(length)))
    a = err_a[idx].sum(axis=1) / np.maximum(length[idx].sum(axis=1), 1)
    b = err_b[idx].sum(axis=1) / np.maximum(length[idx].sum(axis=1), 1)
    d = a - b
    obs = err_a.sum() / max(length.sum(), 1) - err_b.sum() / max(length.sum(), 1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(obs), float(lo), float(hi), float(min(1.0, p))

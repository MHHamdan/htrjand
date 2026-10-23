"""OPTIONAL external helper: naive line splitting of a multi-line image.

This is NOT part of HTR-JAND and was never evaluated: the released recognisers are
line-level models trained on lines segmented by each benchmark's own ground truth.
This helper only cuts a clean, roughly horizontal, single-column region of
handwriting into text bands at the minima of its horizontal projection profile, so
that each band can be passed to the recogniser. It fails on skewed or curved lines,
touching ascenders/descenders, marginalia, multi-column pages, tables and drawings.
For real documents use a dedicated layout-analysis / line-segmentation tool.

Used only when the user asks for it (`predict.py --segment-lines`, or
`HTRJANDModel.transcribe(..., segment_lines=True)`).
"""
import numpy as np

from .inference import crop_to_ink, ink_mask, load_gray, normalise_polarity


def split_lines(image, min_gap_frac=0.2, min_band_frac=0.3, smooth_frac=0.01, min_component=4):
    """Return a list of dark-on-light uint8 crops, one per detected text band, top to bottom.

    A band is a run of rows whose (smoothed) ink count exceeds 10 % of the profile's
    peak; bands shorter than `min_band_frac` x the median band height are merged into
    the nearest band; a cut is placed at the profile minimum between two bands."""
    gray = load_gray(image)
    g, _ = normalise_polarity(gray)
    g, _ = crop_to_ink(g, min_component=min_component)
    m = ink_mask(g, min_component)
    if not m.any():
        return []
    prof = m.sum(axis=1).astype(np.float64)
    k = max(1, int(round(smooth_frac * len(prof))))
    if k > 1:
        prof = np.convolve(prof, np.ones(k) / k, mode="same")
    on = prof > 0.10 * prof.max()
    bands, start = [], None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        elif not v and start is not None:
            bands.append([start, i]); start = None
    if start is not None:
        bands.append([start, len(on)])
    if not bands:
        return [g]
    med = float(np.median([b - a for a, b in bands]))
    merged = []
    for b in bands:                                       # merge slivers (dots, accents)
        if merged and ((b[1] - b[0]) < min_band_frac * med or (b[0] - merged[-1][1]) < min_gap_frac * med * 0.25):
            merged[-1][1] = b[1]
        else:
            merged.append(b)
    if len(merged) > 1 and (merged[0][1] - merged[0][0]) < min_band_frac * med:
        merged[1][0] = merged[0][0]; merged.pop(0)
    cuts = [0]
    for (a0, b0), (a1, b1) in zip(merged, merged[1:]):
        seg = prof[b0:a1] if a1 > b0 else prof[a1:b0 + 1]
        off = b0 if a1 > b0 else a1
        cuts.append(int(off + int(np.argmin(seg))) if len(seg) else b0)
    cuts.append(len(prof))
    out = []
    for t, b in zip(cuts, cuts[1:]):
        crop, _ = crop_to_ink(g[t:b], min_component=min_component)
        if ink_mask(crop, min_component).any():
            out.append(crop)
    return out

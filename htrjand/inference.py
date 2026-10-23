"""Preprocessing of user line images for the released models (release addition).

The models were trained on archives built by `prepare_data.preprocess`: a grayscale
line image, scaled by ONE factor s = min(W / w, H / h) (cv2.INTER_AREA, aspect ratio
kept), centre-padded with white to H x W (128 x 1632 for the released models),
inverted when the mean exceeds 128 so the ink is bright on a dark ground, then
normalised per image to [-1, 1] (per-image min-max normalisation, "paper") or to [0, 1] ("unit").

`prepare_line` applies exactly that geometry and normalisation, after four steps that
make arbitrary user images look like a segmented line of the archives:

  1. decoding     PNG / JPEG / ... , 8 or 16 bit, gray / RGB / RGBA (transparent pixels
                  are composited on white), or a PIL image / numpy array;
  2. polarity     a light-on-dark image (median above the Otsu threshold is dark) is
                  inverted to dark-on-light first, so the white padding of the recipe
                  is background in both cases;
  3. crop         excess background is cropped to the ink bounding box plus a margin
                  (connected components smaller than `min_component` px are ignored as
                  speckle); on by default;
  4. deskew       optional (off by default), small angles only: the rotation in
                  [-max_angle, +max_angle] maximising the variance of the horizontal
                  projection profile, applied only if it exceeds 0.3 degrees.

`line_check` flags inputs that are probably not a single text line (two or more
well-separated text bands in the horizontal projection profile, or an image taller
than it is wide); `HTRJANDModel.transcribe` refuses them unless `force=True`.
"""
from dataclasses import dataclass, field
import os

import numpy as np

from ._compat import lazy

cv2 = lazy("cv2", "pip install opencv-python-headless")

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


MIN_HEIGHT, MIN_WIDTH = 16, 32        # smallest input accepted (px)
LOW_RES_INK = 20                       # ink height below which a low-resolution warning is given (px)


class NotALineError(ValueError):
    """The input looks like a page or a multi-line region, not one text line."""


# --------------------------------------------------------------------------- decoding
def load_gray(image):
    """Path / bytes / PIL.Image / numpy array -> uint8 (H, W), dark ink on light paper
    is NOT assumed here (see `normalise_polarity`)."""
    if isinstance(image, (str, os.PathLike)):
        data = np.fromfile(str(image), dtype=np.uint8)
        arr = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise ValueError(f"cannot decode image {image!r} (supported: {', '.join(IMAGE_EXTENSIONS)})")
        bgr_order = True
    elif isinstance(image, (bytes, bytearray)):
        arr = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise ValueError("cannot decode image bytes")
        bgr_order = True
    elif hasattr(image, "convert") and hasattr(image, "mode"):          # PIL.Image
        mode = image.mode
        if mode in ("I;16", "I;16B", "I"):
            arr = np.asarray(image).astype(np.float64)
        elif mode in ("RGBA", "LA", "P", "PA"):
            arr = np.asarray(image.convert("RGBA"))
        elif mode in ("L", "1"):
            arr = np.asarray(image.convert("L"))
        else:
            arr = np.asarray(image.convert("RGB"))
        bgr_order = False
    else:
        arr = np.asarray(image)
        bgr_order = False
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:                                   # 16-bit / float input
        a = arr.astype(np.float64)
        if a.max() <= 1.0 and arr.dtype.kind == "f":
            a = a * 255.0
        elif a.max() > 255:
            a = a * (255.0 / (65535.0 if a.max() > 4095 else a.max()))
        arr = np.clip(np.rint(a), 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3:
        raise ValueError(f"unsupported image shape {arr.shape}")
    c = arr.shape[2]
    if c == 1:
        return arr[:, :, 0]
    if c == 2:                                                   # gray + alpha
        g, al = arr[:, :, 0].astype(np.float32), arr[:, :, 1].astype(np.float32) / 255.0
        return np.clip(np.rint(g * al + 255.0 * (1.0 - al)), 0, 255).astype(np.uint8)
    if c == 4:                                                   # composite on white
        rgb, al = arr[:, :, :3].astype(np.float32), arr[:, :, 3:4].astype(np.float32) / 255.0
        comp = np.clip(np.rint(rgb * al + 255.0 * (1.0 - al)), 0, 255).astype(np.uint8)
        if int(comp.max()) - int(comp.min()) < 16 and int(arr[:, :, 3].max()) - int(arr[:, :, 3].min()) >= 16:
            # light ink drawn as opacity on a transparent page (e.g. white strokes): the alpha channel IS the ink
            return (255 - arr[:, :, 3]).astype(np.uint8)
        arr = comp
    else:
        arr = arr[:, :, :3]
    code = cv2.COLOR_BGR2GRAY if bgr_order else cv2.COLOR_RGB2GRAY
    return cv2.cvtColor(np.ascontiguousarray(arr), code)


# --------------------------------------------------------------------------- polarity / ink
def otsu_threshold(gray):
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(t)


def is_dark_background(gray):
    """True when the background (the majority class of an Otsu split) is the dark one."""
    t = otsu_threshold(gray)
    return float(np.mean(gray > t)) < 0.5


def normalise_polarity(gray):
    """Return (dark-ink-on-light uint8 image, inverted?)."""
    if is_dark_background(gray):
        return (255 - gray).astype(np.uint8), True
    return gray, False


def ink_mask(gray_dark_on_light, min_component=4):
    """Binary ink mask (uint8 0/1) of a dark-on-light image, speckle removed."""
    t = otsu_threshold(gray_dark_on_light)
    # a blank or nearly blank image has no meaningful Otsu split
    if float(gray_dark_on_light.max()) - float(gray_dark_on_light.min()) < 16:
        return np.zeros_like(gray_dark_on_light, dtype=np.uint8)
    m = (gray_dark_on_light <= t).astype(np.uint8)
    if min_component > 1 and m.any():
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        small = stats[:, cv2.CC_STAT_AREA] < min_component
        small[0] = False
        if small.any():
            m[small[lab]] = 0
    return m


def crop_to_ink(gray, margin_frac=0.08, min_margin=2, min_component=4):
    """Crop a dark-on-light image to its ink bounding box plus a margin of
    `margin_frac` x ink height (at least `min_margin` px). Returns (image, box) where
    box = (top, bottom, left, right) in the input, or the input unchanged when no ink
    is found."""
    m = ink_mask(gray, min_component)
    if not m.any():
        return gray, (0, gray.shape[0], 0, gray.shape[1])
    ys, xs = np.where(m)
    t, b, l, r = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    mg = max(min_margin, int(round(margin_frac * (b - t))))
    t, b = max(0, t - mg), min(gray.shape[0], b + mg)
    l, r = max(0, l - mg), min(gray.shape[1], r + mg)
    return gray[t:b, l:r], (t, b, l, r)


# --------------------------------------------------------------------------- deskew
def estimate_skew(gray, max_angle=5.0, step=0.25, min_component=4):
    """Small-angle skew (degrees, counter-clockwise positive as in cv2) of a
    dark-on-light line: the rotation that maximises the variance of the row profile."""
    m = ink_mask(gray, min_component).astype(np.float32)
    if m.sum() < 20:
        return 0.0
    h, w = m.shape
    best, best_a = -1.0, 0.0
    for a in np.arange(-max_angle, max_angle + 1e-9, step):
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(a), 1.0)
        rot = cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        v = float(np.var(rot.sum(axis=1)))
        if v > best + 1e-9:
            best, best_a = v, float(a)
    return best_a


def rotate(gray, angle):
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(angle), 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * s + w * c + 0.5), int(h * c + w * s + 0.5)
    M[0, 2] += nw / 2.0 - w / 2.0
    M[1, 2] += nh / 2.0 - h / 2.0
    return cv2.warpAffine(gray, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=255)


# --------------------------------------------------------------------------- line check
def text_bands(gray, min_component=4, smooth_frac=0.02):
    """Text bands of a dark-on-light image from its horizontal projection profile:
    list of (top, bottom) row ranges whose ink density exceeds 15 % of the peak."""
    m = ink_mask(gray, min_component)
    if not m.any():
        return []
    prof = m.sum(axis=1).astype(np.float64)
    k = max(1, int(round(smooth_frac * len(prof))))
    if k > 1:
        prof = np.convolve(prof, np.ones(k) / k, mode="same")
    on = prof > 0.15 * prof.max()
    bands, start = [], None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        elif not v and start is not None:
            bands.append((start, i)); start = None
    if start is not None:
        bands.append((start, len(on)))
    return bands


def line_check(gray, max_height_ratio=1.0, min_component=4):
    """(ok, reason) for a dark-on-light, cropped image. Not ok when the profile has two
    or more substantial text bands separated by a gap of at least 25 % of the median
    band height, or when the image is taller than `max_height_ratio` x its width."""
    h, w = gray.shape
    bands = text_bands(gray, min_component)
    if len(bands) >= 2:
        hs = [b - a for a, b in bands]
        tall = max(hs)
        big = [(a, b) for a, b in bands if (b - a) >= 0.35 * tall]
        if len(big) >= 2:
            med = float(np.median([b - a for a, b in big]))
            gaps = [big[i + 1][0] - big[i][1] for i in range(len(big) - 1)]
            if max(gaps) >= 0.25 * med:
                return False, (f"{len(big)} separate text bands found in the horizontal projection profile; "
                               "this looks like a multi-line region or a page")
    if h > max_height_ratio * w:
        return False, f"the image is taller than it is wide ({h} x {w} px); a text line is wide and flat"
    return True, ""


# --------------------------------------------------------------------------- training geometry
def fit_canvas(gray_u8, size):
    """`prepare_data.preprocess` on an in-memory grayscale array (identical arithmetic):
    aspect-preserving scale into H x W with INTER_AREA, centre pad with white, invert if
    the mean exceeds 128 (ink bright on a dark ground)."""
    th, tw = size
    img = gray_u8
    h, w = img.shape
    s = min(tw / w, th / h)
    nw, nh = int(w * s), int(h * s)
    if nw == 0 or nh == 0:
        raise ValueError(f"degenerate line image: {img.shape}")
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    dh, dw = th - nh, tw - nw
    img = cv2.copyMakeBorder(img, dh // 2, dh - dh // 2, dw // 2, dw - dw // 2,
                             cv2.BORDER_CONSTANT, value=255)
    if img.mean() > 128:
        img = 255 - img
    return img.astype(np.uint8)


def normalise_canvas(canvas_u8, how="paper"):
    """The archive path of `data.LineDataset.__getitem__` (no augmentation):
    uint8 -> /255 -> back to uint8 -> `data.normalise`."""
    from .data import normalise
    img = canvas_u8.astype(np.float32) / 255.0
    return normalise((img * 255.0).astype(np.uint8), how) if how == "paper" else img


@dataclass
class PreparedLine:
    canvas: np.ndarray                 # uint8 (H, W), ink bright on dark: what the model sees
    tensor_input: np.ndarray           # float32 (1, H, W), normalised
    cropped: np.ndarray                # uint8, dark-on-light, after polarity / crop / deskew
    inverted_input: bool = False
    crop_box: tuple = None
    skew_deg: float = 0.0
    is_line: bool = True
    warnings: list = field(default_factory=list)


def prepare_line(image, size=(128, 1632), normalisation="paper", crop=True, deskew=False,
                 max_angle=5.0, margin_frac=0.08, min_component=4, check=True):
    gray = load_gray(image)
    if gray.ndim != 2 or gray.shape[0] < MIN_HEIGHT or gray.shape[1] < MIN_WIDTH:
        raise ValueError(f"image too small ({gray.shape[0]} x {gray.shape[1]} px); a text line image needs at least "
                         f"{MIN_HEIGHT} px height and {MIN_WIDTH} px width (scan at 200 dpi or more)")
    warnings = []
    g, inv = normalise_polarity(gray)
    if inv:
        warnings.append("light ink on a dark background detected; the image was inverted")
    box = (0, g.shape[0], 0, g.shape[1])
    if crop:
        g, box = crop_to_ink(g, margin_frac, min_component=min_component)
        if not ink_mask(g, min_component).any():
            warnings.append("no ink found; the image may be blank")
    angle = 0.0
    if deskew:
        angle = estimate_skew(g, max_angle, min_component=min_component)
        if abs(angle) >= 0.3:
            g = rotate(g, angle)
            if crop:
                g, _ = crop_to_ink(g, margin_frac, min_component=min_component)
        else:
            angle = 0.0
    m = ink_mask(g, min_component)
    if m.any():
        ys = np.where(m.any(axis=1))[0]
        if ys.max() - ys.min() + 1 < LOW_RES_INK:
            warnings.append(f"very low resolution (ink height {ys.max() - ys.min() + 1} px); the transcription is "
                            "unreliable; scan at 200 dpi or more")
    ok, reason = line_check(g, min_component=min_component) if check else (True, "")
    if not ok:
        warnings.append("not a single text line? " + reason)
    canvas = fit_canvas(g, tuple(size))
    x = normalise_canvas(canvas, normalisation)[None].astype(np.float32)
    return PreparedLine(canvas=canvas, tensor_input=x, cropped=g, inverted_input=inv, crop_box=box,
                        skew_deg=angle, is_line=ok, warnings=warnings)


# --------------------------------------------------------------------------- confidence
def frame_confidence(probs, blank):
    """Line confidence = mean over the frames whose arg-max is not blank of the
    arg-max probability (mean over all frames if every frame is blank).
    probs: (T, V) softmax of one line."""
    am = probs.argmax(-1)
    mx = probs.max(-1)
    sel = am != blank
    return float(mx[sel].mean() if sel.any() else mx.mean())


def char_confidences(probs, blank, pad=0):
    """Per emitted character of the greedy path: max probability over the frames of
    its run (consecutive frames with the same non-blank arg-max)."""
    am = probs.argmax(-1)
    mx = probs.max(-1)
    out, prev, cur = [], None, None
    for k, p in zip(am, mx):
        k = int(k)
        if k != prev:
            if cur is not None:
                out.append(cur)
            cur = float(p) if (k != blank and k != pad) else None
        elif cur is not None:
            cur = max(cur, float(p))
        prev = k
    if cur is not None:
        out.append(cur)
    return out

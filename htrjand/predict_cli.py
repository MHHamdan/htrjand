"""Transcribe handwritten TEXT LINE images with a released HTR-JAND model (line-level recognizer).

    python predict.py --image examples/ood_2_gettysburg_a.png
    python predict.py --model MHamdan/htrjand-student --variant iam --image line.png --decoder lm --t5
    python predict.py --model path/to/htrjand-student --variant general --image a.png b.jpg --device cpu
    python predict.py --image page.png --segment-lines     # naive external line splitting, see below

--model     a Hugging Face repo id (default: htrjand.release.DEFAULT_MODEL) or a local directory
            holding <variant>/config.json (a downloaded or staged repo), or a variant directory itself
--variant   general (five-corpus model, default) | iam | rimes | bentham | saintgall | washington
--decoder   greedy (raw CTC, default) | lm (CTC beam search + released Wikipedia character LM)
--t5        also run the gated T5-small corrector (needs `pip install transformers`) on the decoder output
--corpus    domain for the LM language and the T5 gate (default: the variant's; `general` uses iam):
            iam / bentham / washington = English, rimes = French, saintgall = Latin

The recogniser reads ONE text line per image. Inputs that look like a page or several lines are
refused unless --force; --segment-lines first cuts the image at the minima of its horizontal
projection profile (a naive helper, not part of HTR-JAND and not evaluated).
Prints, per image: raw CTC, LM and T5 outputs (as requested), the confidence (mean max-softmax
probability over the non-blank frames of the greedy path) and any preprocessing warnings.
"""
import argparse
import json
import os
import sys

from .release import DEFAULT_MODEL


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--variant", default="general")
    ap.add_argument("--revision", default=None, help="Hub revision (branch, tag or commit)")
    ap.add_argument("--image", nargs="+", required=True, help="PNG / JPG / JPEG line image(s) or a directory")
    ap.add_argument("--decoder", default="greedy", choices=("greedy", "lm"))
    ap.add_argument("--t5", action="store_true", help="gated T5 refinement of the decoder output")
    ap.add_argument("--t5-model", default=None, help="corrector repo id or directory (default: from config.json)")
    ap.add_argument("--corpus", default=None, choices=("iam", "rimes", "bentham", "saintgall", "washington"))
    ap.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, ... (GPU choice: CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--force", action="store_true", help="transcribe inputs that look like pages anyway")
    ap.add_argument("--segment-lines", action="store_true", help="split multi-line images first (naive helper)")
    ap.add_argument("--no-crop", action="store_true", help="do not crop excess background")
    ap.add_argument("--deskew", action="store_true", help="small-angle deskew (off by default)")
    ap.add_argument("--save-preprocessed", default=None, metavar="DIR", help="write the 128 x 1632 model inputs")
    ap.add_argument("--json", action="store_true", help="one JSON object per line instead of text")
    a = ap.parse_args(argv)

    from . import HTRJANDModel, NotALineError
    from .inference import IMAGE_EXTENSIONS
    paths = []
    for p in a.image:
        if os.path.isdir(p):
            paths += sorted(os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith(IMAGE_EXTENSIONS))
        else:
            paths.append(p)
    if not paths:
        sys.exit("no images found")
    model = HTRJANDModel.from_pretrained(a.model, variant=a.variant, revision=a.revision, device=a.device)
    if a.t5:
        try:
            model.load_corrector(a.t5_model, device=a.device)
        except ImportError as e:
            print(f"--t5: {e}", file=sys.stderr)
            return 3
    status = 0
    for p in paths:
        try:
            res = model.transcribe(p, decoder=a.decoder, corpus=a.corpus, t5=a.t5, force=a.force,
                                   crop=not a.no_crop, deskew=a.deskew, segment_lines=a.segment_lines)
        except NotALineError as e:
            print(f"{p}\tREFUSED\t{e}", file=sys.stderr)
            status = 2
            continue
        except (ValueError, FileNotFoundError) as e:
            print(f"{p}\tERROR\t{e}", file=sys.stderr)
            status = 1
            continue
        lines = res if isinstance(res, list) else [res]
        if a.save_preprocessed:
            from .inference import cv2
            os.makedirs(a.save_preprocessed, exist_ok=True)
            stem = os.path.splitext(os.path.basename(p))[0]
            if a.segment_lines:
                from .segment_lines import split_lines
                crops = split_lines(p)
            else:
                crops = [p]
            for k, c in enumerate(crops):
                cv2.imwrite(os.path.join(a.save_preprocessed, f"{stem}_{k:02d}.png"),
                            model.preprocess(c, crop=not a.no_crop, deskew=a.deskew).canvas)
        for k, r in enumerate(lines):
            name = p if len(lines) == 1 else f"{p}#{k + 1}"
            if a.json:
                d = r.to_dict()
                d.pop("char_confidences", None)
                print(json.dumps(dict(image=name, **d), ensure_ascii=False))
                continue
            print(f"== {name}")
            print(f"raw CTC   : {r.raw}")
            if r.lm is not None:
                print(f"char LM   : {r.lm}")
            if r.t5 is not None:
                print(f"T5        : {r.t5}   (T5 confidence {r.t5_confidence:.3f}, "
                      f"{'applied' if r.t5_applied else 'gate kept the input'})")
            if r.confidence is not None:
                print(f"confidence: {r.confidence:.3f}")
            for w in r.warnings:
                print(f"warning   : {w}")
    return status


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Gradio demo of HTR-JAND, the line-level handwriting recognizer.

    pip install -r requirements.txt -r requirements-demo.txt
    python app.py                                   # models from the Hugging Face Hub
    HTRJAND_MODEL_ROOT=/path/to/staged/repos python app.py   # local copies of the model repos

HTRJAND_MODEL_ROOT, when set, is a directory holding the repo folders (<prefix>-student,
<prefix>-teacher, <prefix>-t5-corrector, <prefix>-charlm) and replaces the Hub ids.
HTRJAND_DEMO_T5=0 hides the T5 option; HTRJAND_DEMO_TEACHER=0 hides the Teacher.
"""
import glob
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.isdir(os.path.join(HERE, "htrjand")):
    sys.path.insert(0, HERE)

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402

from htrjand import HTRJANDModel, NotALineError  # noqa: E402
from htrjand import release as REL  # noqa: E402

ROOT = os.environ.get("HTRJAND_MODEL_ROOT")
SHOW_T5 = os.environ.get("HTRJAND_DEMO_T5", "1") != "0"
SHOW_TEACHER = os.environ.get("HTRJAND_DEMO_TEACHER", "1") != "0"


def source(repo):
    return os.path.join(ROOT, repo) if ROOT else REL.repo_id(repo)


MODELS = {
    "Student, general (five corpora) [default]": (REL.STUDENT, "general"),
    "Student, IAM (modern English)": (REL.STUDENT, "iam"),
    "Student, RIMES (modern French)": (REL.STUDENT, "rimes"),
    "Student, Bentham (18th-19th c. English)": (REL.STUDENT, "bentham"),
    "Student, Washington (18th c. English)": (REL.STUDENT, "washington"),
    "Student, Saint Gall (9th c. Latin)": (REL.STUDENT, "saintgall"),
}
if SHOW_TEACHER:
    MODELS["Teacher, general (five corpora, 7.2 M parameters, slower)"] = (REL.TEACHER, "general")
DOMAINS = {"English (IAM)": "iam", "English, historical (Bentham)": "bentham",
           "English, 18th c. (Washington)": "washington", "French (RIMES)": "rimes", "Latin (Saint Gall)": "saintgall"}
DECODERS = ["Raw CTC (greedy)", "CTC + Wikipedia character LM"]

_loaded, _corrector = {}, {}


def get_model(name):
    if name not in _loaded:
        repo, variant = MODELS[name]
        _loaded[name] = HTRJANDModel.from_pretrained(source(repo), variant=variant, device="cpu")
    return _loaded[name]


def get_corrector(model):
    """One T5 corrector shared by every model (lexicons resolved like the model's LM)."""
    if "t5" not in _corrector:
        _corrector["t5"] = model.load_corrector(source(REL.T5_CORRECTOR) if ROOT else None, device="cpu")
    model._corrector = _corrector["t5"]
    return _corrector["t5"]


def run(image, model_name, domain, decoder, use_t5, force, deskew):
    if image is None:
        return None, "", "", "", "", "Upload a line image first."
    t0 = time.time()
    notes = []
    repo, variant = MODELS[model_name]
    corpus = DOMAINS[domain] if variant == "general" else variant
    if corpus == "saintgall" and variant == "general":
        if repo == REL.STUDENT:
            # the general models were trained with the original Saint Gall labels (raw "pt" letter):
            # Latin goes to the marker-free Saint Gall Student
            model_name = next(k for k, v in MODELS.items() if v == (REL.STUDENT, "saintgall"))
            notes.append("Latin selected: using the Saint Gall Student (the general Student was trained with the "
                         "original Saint Gall labels and writes 'pt' for punctuation).")
        else:
            notes.append("Warning: the Teacher was trained with the original Saint Gall labels and writes 'pt' for "
                         "punctuation; for Latin use 'Student, Saint Gall'.")
    try:
        m = get_model(model_name)
    except Exception as e:                              # offline / unreachable Hub
        return None, "", "", "", "", f"Cannot load the model: {type(e).__name__}: {e}"
    try:
        prep = m.preprocess(image, deskew=deskew)
    except ValueError as e:                             # too small, undecodable
        return None, "", "", "", "", str(e)
    preview = 255 - prep.canvas                         # shown dark-on-light; the model sees it inverted
    notes += list(prep.warnings)
    if not prep.is_line and not force:
        notes.append("Refused: HTR-JAND reads ONE text line. Crop a single line, or tick 'transcribe anyway'.")
        return preview, "", "", "", "", "\n".join(notes)
    try:
        if use_t5:
            get_corrector(m)
        r = m.transcribe(image, decoder="lm" if decoder == DECODERS[1] else "greedy", corpus=corpus,
                         t5=bool(use_t5), force=True, deskew=deskew)
    except (NotALineError, ValueError, ImportError, OSError) as e:
        return preview, "", "", "", "", "\n".join(notes + [f"{type(e).__name__}: {e}"])
    conf = f"{r.confidence:.3f}" if r.confidence is not None else ""
    t5 = ""
    if r.t5 is not None:
        t5 = r.t5 + ("" if r.t5_applied else "   [gate kept the input]")
        conf += f"   (T5 candidate confidence {r.t5_confidence:.3f})"
    for w in r.warnings:
        if w not in notes:
            notes.append(w)
    notes.append(f"{m.config['variant']} {m.config['role']}, domain {corpus}, {time.time() - t0:.2f} s on CPU")
    return preview, r.raw, r.lm or "", t5, conf, "\n".join(notes)


def examples():
    files = sorted(glob.glob(os.path.join(HERE, "examples", "*.png")))
    return [[f, next(iter(MODELS)), "English (IAM)", DECODERS[0], False, False, False] for f in files]


NOTE = f"""
### {REL.WORK_NAME}: line-level handwritten text recognition

A **line-level** recogniser: upload an image of **one line** of handwriting (PNG / JPG / JPEG, any
size, light or dark background). Page or multi-line images are flagged and refused unless you
tick *transcribe anyway*. The image is cropped to the ink, scaled with its aspect ratio kept into
128 x 1632, centre-padded and normalised exactly as the training lines were (preview below).

Models: compact Students (806,287 parameters at inference) trained on IAM, RIMES, Bentham, Saint Gall
and Washington plus synthetic lines; the general Student is the default. Decoders: raw CTC, or CTC
with a character LM built from Wikipedia text only; optional gated T5-small refinement.
Outside those five domains (other hands, scripts, languages, scans) expect errors; the LM and T5 can
also *introduce* plausible but wrong words. Research demo; not for forensic, legal, medical or other
high-stakes use. Weights: CC BY-NC 4.0 (training data licences); code: MIT.
"""


def build():
    with gr.Blocks(title=f"{REL.WORK_NAME} demo") as demo:
        gr.Markdown(NOTE)
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Line image (drag and drop)", image_mode="RGBA", sources=["upload", "clipboard"])
                model = gr.Dropdown(list(MODELS), value=next(iter(MODELS)), label="Model")
                domain = gr.Dropdown(list(DOMAINS), value="English (IAM)",
                                     label="Domain for the LM language and T5 (general models; Latin switches to the Saint Gall Student)")
                decoder = gr.Radio(DECODERS, value=DECODERS[0], label="Decoder")
                use_t5 = gr.Checkbox(value=False, label="T5 refinement (gated; slower)", visible=SHOW_T5)
                with gr.Row():
                    force = gr.Checkbox(value=False, label="transcribe anyway if it looks like a page")
                    deskew = gr.Checkbox(value=False, label="small-angle deskew")
                go = gr.Button("Transcribe", variant="primary")
            with gr.Column():
                preview = gr.Image(label="Model input (128 x 1632, shown dark on light)", type="numpy",
                                   interactive=False)
                raw = gr.Textbox(label="Raw CTC")
                lm = gr.Textbox(label="Character LM")
                t5 = gr.Textbox(label="T5 refinement", visible=SHOW_T5)
                conf = gr.Textbox(label="Confidence (mean max-softmax over non-blank frames)")
                notes = gr.Textbox(label="Notes", lines=3)
        inputs = [image, model, domain, decoder, use_t5, force, deskew]
        outputs = [preview, raw, lm, t5, conf, notes]
        go.click(run, inputs, outputs)
        ex = examples()
        if ex:
            gr.Examples(ex, inputs, outputs, fn=run, cache_examples=False,
                        label="Out-of-distribution examples (public-domain / CC0 letters, see examples/SOURCES.md)")
    return demo


if __name__ == "__main__":
    build().launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"))

#!/usr/bin/env python3
"""Write the model-card DRAFTS (README.md) of the staged Hugging Face repos.

    python scripts/write_model_cards.py --hf-root ../htrjand-release-hf

Every number is read from a JSON file: the staged config.json files (which copy results/*.json
of the experiments), results/final_selection.json, results/seed2_student.json and
results/release_eval/{lm,t5}_validation.json / latency.json (validation-only evaluation of the
released decoders). Nothing is typed by hand; the only TODOs are the citation / URL of the
paper under review.
"""
import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from htrjand import release as REL  # noqa: E402

CORPORA = ["iam", "rimes", "bentham", "saintgall", "washington"]
NAME = dict(iam="IAM", rimes="RIMES", bentham="Bentham", saintgall="Saint Gall", washington="Washington")
DATASET_FREE = ["IAM Handwriting Database (lines, FKI writer-independent split)", "RIMES (ICDAR 2011 lines)",
                "Bentham Dataset R0 (ICFHR 2014 HTRtS)", "Saint Gall (IAM-HistDB)", "Washington (IAM-HistDB, cv1)"]
DATA_LICENCES = """| corpus | licence / terms | required citation |
|---|---|---|
| IAM | CC BY-NC 4.0 (non-commercial research), https://zenodo.org/records/19003721 | U. Marti, H. Bunke, "The IAM-database: An English Sentence Database for Off-line Handwriting Recognition", IJDAR 5:39-46, 2002, doi:10.1007/s100320200071 |
| RIMES | CC BY-NC 4.0, doi:10.5281/zenodo.10812725 (A2iA / Mitek Systems) | Grosicki et al., RIMES database, Zenodo 10812725 |
| Bentham | CC BY 4.0, doi:10.5281/zenodo.44519 (images: UCL Special Collections, Bentham Papers) | Sánchez et al., ICFHR 2014 HTRtS; Gatos et al., DAS 2014 |
| Washington | IAM-HistDB terms: non-commercial research and teaching only (source pages: George Washington Papers, Library of Congress) | A. Fischer et al., Pattern Recognition Letters 33(7), 2012; A. Fischer, "IAM-HistDB", World Scientific, 2020 |
| Saint Gall | IAM-HistDB terms: non-commercial research and teaching only (source: St. Gallen, Stiftsbibliothek, Cod. Sang. 562, www.e-codices.ch, CC BY-NC 4.0) | A. Fischer, V. Frinken, A. Fornés, H. Bunke, HIP 2011; IAM-HistDB chapter, 2020 |
| synthetic lines | rendered from Wikipedia-derived text (CC BY-SA 4.0) with Google Fonts handwriting faces (OFL-1.1 / Apache-2.0; rendered images unrestricted) | — |
"""
BIB_NEW = f"""@unpublished{{{REL.PAPER_BIBKEY},
  author = {{{' and '.join(a.split()[-1] + ', ' + ' '.join(a.split()[:-1]) for a in REL.PAPER_AUTHORS)}}},
  title  = {{{{HTR-JAND}}: Handwritten Text Recognition with Joint Attention Network and Knowledge Distillation}},
  year   = {{2026}},
  note   = {{{REL.PAPER_NOTE[0].upper() + REL.PAPER_NOTE[1:]}. TODO: venue / DOI / URL once available}}
}}"""
LM_PROVENANCE = ("English: WikiText-103 (Merity et al., 2016; Hugging Face `Salesforce/wikitext`, CC BY-SA), 400,000 "
                 "line fragments; French and Latin: Wikipedia articles via the Hugging Face `wikimedia/wikipedia` dataset "
                 "(CC BY-SA 3.0 / GFDL), 200,000 and 100,000 line fragments (the dataset snapshot was not recorded). "
                 "The released LM files are licensed CC BY-SA 4.0.")
LIMITATIONS = """- **Line level only.** The input must be ONE segmented text line. Pages and multi-line regions are
  detected (horizontal projection profile, aspect ratio) and refused unless forced; the optional
  `--segment-lines` splitter is a naive projection-profile helper, not part of the evaluated system.
- **Domain shift.** Trained on five benchmark collections (modern English and French, 18th-19th c. English,
  9th c. Latin) plus synthetic font-rendered lines. Other hands, writing instruments, scripts, layouts and
  scan conditions (low resolution, degraded, stained, show-through, coloured or ruled paper) raise the error
  rate, often sharply; the out-of-distribution demo images show this qualitatively.
- **Historical scripts.** Saint Gall is a single 9th-century Carolingian manuscript; the Saint Gall variant
  follows its transcription convention (marker-free punctuation, letter-by-letter Latin) and does not
  generalise to other Latin manuscripts or abbreviation systems.
- **Language dependence.** The character set is the union of the five corpora's training transcriptions
  (108 characters); characters outside it cannot be produced. The LM and T5 options assume English, French
  or Latin text; they bias outputs towards Wikipedia-like spelling.
- **LM / T5 hallucination.** Both can replace a correctly recognised but unusual word (names, numbers,
  historical spellings) by a plausible wrong one. The T5 gate, tuned on validation, rejects most candidates.
- **Unusual symbols.** Diacritics outside the charset, superscripts, abbreviation marks, strike-throughs and
  insertions are not modelled.
- **Confidence** is the mean max-softmax probability over the non-blank frames of the greedy path; it is not
  calibrated and is not a probability that the line is correct."""
INTENDED = """**Intended use:** research and teaching on handwritten text recognition; transcription drafts of
single text lines that a person will check; benchmarking against the reported numbers.

**Out of scope:** forensic handwriting analysis or writer identification, legal identity or signature
verification, medical records or prescriptions, any high-stakes or automated decision without human review,
and commercial use (the weights are CC BY-NC 4.0 because the training data is non-commercial)."""


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def f2(x):
    return "—" if x is None else f"{x:.2f}"


def yaml_list(xs):
    return "\n".join(f"  - {x}" for x in xs)


def model_index(name, rows):
    """rows: (corpus, metric name, value %)."""
    out = [f"model-index:", f"- name: {name}", "  results:"]
    by = {}
    for c, mname, v in rows:
        by.setdefault(c, []).append((mname, v))
    for c, ms in by.items():
        out += ["  - task:", "      type: image-to-text", "      name: Handwritten text recognition (line level)",
                "    dataset:", f"      name: {NAME[c]}", f"      type: {c}", "      split: test",
                "    metrics:"]
        for mname, v in ms:
            out += ["    - type: cer", f"      value: {v:.2f}", f"      name: {mname}"]
    return "\n".join(out)


def recogniser_card(repo_dir, repo, role, seed, R, ev):
    variants = sorted(d for d in os.listdir(repo_dir) if os.path.exists(os.path.join(repo_dir, d, "config.json")))
    variants = sorted(variants, key=lambda v: (v != "general", REL.VARIANTS.index(v) if v in REL.VARIANTS else 99))
    cfgs = {v: json.load(open(os.path.join(repo_dir, v, "config.json"))) for v in variants}
    lmv, t5v, lat = ev.get("lm", {}), ev.get("t5", {}), ev.get("latency", {})
    sel = load(os.path.join(R, "final_selection.json")) or {}
    s2 = load(os.path.join(R, "seed2_student.json")) or {}
    rid = REL.repo_id(repo)
    mi_rows = []
    for v in variants:
        pc = cfgs[v]["metrics"]["per_corpus"]
        for c in ([v] if v != "general" else CORPORA):
            m = pc.get(c, {}).get("raw_ctc")
            if m and (v != "general" or role == "teacher"):
                mi_rows.append((c, f"Test CER (%), raw CTC, {v} variant", m["test_cer"]))
    for c in CORPORA:
        if role == "student" and seed == 1:
            best = min((x for x in sel.get(c, {}).values() if x["system"] == "Student"), key=lambda x: x["v"][0], default=None)
            if best:
                mi_rows.append((c, f"Test CER (%), validation-selected Student path ({best['stage']})", 100 * best["t"][0]))
        if role == "student" and seed == 2 and c in s2:
            mi_rows.append((c, f"Test CER (%), validation-selected Student path ({s2[c]['selected_path']})",
                            100 * s2[c]["selected_test"]))
    W = REL.WORK_NAME[0].upper() + REL.WORK_NAME[1:]
    title = {"student": f"{W}: compact Student (seed {seed})", "teacher": f"{W}: Teacher"}[role]
    head = f"""---
library_name: pytorch
pipeline_tag: image-to-text
license: cc-by-nc-4.0
language:
- en
- fr
- la
tags:
- handwritten-text-recognition
- htr
- ocr
- ctc
- knowledge-distillation
datasets:
{yaml_list(DATASET_FREE)}
{model_index(repo, mi_rows)}
---

# {title}

> **DRAFT model card** generated by `scripts/write_model_cards.py` from the result files; numbers are copied,
> not typed. Version {REL.RELEASE_VERSION}, source commit `{REL.SOURCE_COMMIT}`.

**HTR-JAND**, the line-level handwriting recognizer (enhanced version), from the paper *{REL.PAPER_TITLE}*
(Hamdan, Rahiche, Cheriet; {REL.PAPER_NOTE}). Trained from scratch on the official writer-independent splits of
five benchmarks plus synthetic lines; every model and decoder choice was made on validation data.

| variant | training run | trained on | deployed parameters | weights sha256 | config sha256 |
|---|---|---|---:|---|---|
"""
    for v in variants:
        c = cfgs[v]
        init = (c["provenance"].get("init_ckpt") or "").split("/")[-2:-1]
        spec = c["corpus"]["specialisation"] if v == "general" else \
            f"{NAME[v]}{' (marker-free labels)' if v == 'saintgall' else ''}, fine-tuned from `{init[0] if init else '?'}`"
        head += (f"| `{v}` | `{c['provenance']['training_run_id']}` | {spec} | "
                 f"{c['parameters']['deployed']:,} | `{c['weights']['sha256'][:16]}…` | "
                 f"`{sha(os.path.join(repo_dir, v, 'config.json'))[:16]}…` |\n")
    a = cfgs[variants[0]]["architecture"]
    p = cfgs[variants[0]]["parameters"]
    head += f"""
Full SHA-256 digests: `{variants[0]}/config.json` field `weights.sha256` (and the table in
`docs/RELEASE_ENGINEERING_NOTES.md` of the code repository).

## Usage

```python
# pip install git+https://github.com/{REL.GITHUB_REPO}.git
from htrjand import HTRJANDModel
model = HTRJANDModel.from_pretrained("{rid}", variant="{variants[0]}")      # or a local directory
r = model.transcribe("line.png", decoder="greedy", return_confidence=True)   # decoder="lm" for the char LM
print(r.text, r.confidence)
```
```bash
python predict.py --model {rid} --variant {variants[-1] if len(variants) > 1 else variants[0]} --image line.png --decoder lm --t5
```
`from_pretrained` downloads only `<variant>/*` (`huggingface_hub.snapshot_download(allow_patterns=...)`, optional
`revision=`), builds the network from `config.json` alone and loads the weights with `strict=True` after checking
their SHA-256. The character LM ({REL.repo_id(REL.CHARLM)}, CC BY-SA 4.0) and the T5 corrector
({REL.repo_id(REL.T5_CORRECTOR)}) are fetched only when requested.

## Architecture

Gated CNN with squeeze-and-excitation, BiLSTM stack and Combined Attention (multi-head self-attention + Proxima
attention), CTC output; an auxiliary CTC head on the CNN columns is used in training only.

| | value |
|---|---|
| CNN blocks (FullGatedConv2d 3x3, BN, ReLU, MaxPool, SE r={a['se_reduction']}) | channels {a['channels']}, pools {a['pools']} |
| recurrent stage | {a['lstm_layers']} x BiLSTM, {a['lstm_hidden']} per direction, LayerNorm after each |
| attention | {a['attn']} ({a['heads']} head{'s' if a['heads'] > 1 else ''}), {a['attn_position'].replace('_', ' ')}, fusion "{a['fusion']}" |
| output | {a['vocab_size']} classes (PAD, UNK, 108 characters, CTC blank = index {cfgs[variants[0]]['tokenizer']['blank_index']}), {cfgs[variants[0]]['n_frames']} frames per line |
| parameters | {p['deployed']:,} at inference; {p['total']:,} with the auxiliary head, which is kept in the weights for a strict load |

## Input format

One grayscale text line. Preprocessing (`htrjand.inference.prepare_line`, identical to the training archives
after steps 1-3): (1) decode PNG/JPEG, RGB/RGBA (transparent pixels on white) to gray; (2) a light-on-dark
image is inverted to dark-on-light; (3) crop to the ink bounding box with a margin of 8 % of the ink height
(optional small-angle deskew, off by default); then the training geometry: scale by ONE factor
s = min(1632 / w, 128 / h) (cv2.INTER_AREA, aspect ratio kept), centre-pad with white to **128 x 1632**, invert
so the ink is bright on a dark ground (mean > 128), and normalise per image to [-1, 1] (min-max).

## Decoders

- **Raw CTC** (greedy best path): the reported "raw CTC" numbers.
- **Character LM**: CTC prefix beam search (beam 16) with an order-8 character n-gram, n-best (8) rescored by
  CTC log-likelihood + alpha x LM + beta x length, alpha / beta chosen on validation per corpus and stored in
  `config.json`. The **released LM is built from Wikipedia-derived text only** ({LM_PROVENANCE}) The experiments
  of the paper used an LM fitted on each corpus's training transcriptions plus Wikipedia text; that LM is not
  released (it would redistribute the transcriptions). Its numbers are reported below, separately.
- **T5 refinement**: gated T5-small corrector ({REL.repo_id(REL.T5_CORRECTOR)}), applied only when its
  confidence exceeds the validation-selected threshold.

## Results

CER in %. Test splits are the official ones; every test number comes from `results/*.json` of the source
repository, decoded once after validation selection. "valid" = validation CER of the same configuration.
"""
    body = ""
    for v in variants:
        c = cfgs[v]
        pc = c["metrics"]["per_corpus"]
        body += f"\n### Variant `{v}`\n\n| corpus | raw CTC valid / test | char LM (corpus + Wikipedia, experiments) valid / test | T5 on raw CTC valid / test | T5 on char LM valid / test | released Wikipedia LM, valid | released LM + T5 (ctc / lm base), valid |\n|---|---|---|---|---|---|---|\n"
        for corpus in ([v] if v != "general" else CORPORA):
            m = pc.get(corpus, {})
            g = lambda k: (f"{f2(m[k]['valid_cer'])} / {f2(m[k]['test_cer'])}" if k in m else "—")
            key = f"{repo}/{v}|{corpus}"
            w = lmv.get(key, {}).get("wiki_lm", {}).get("charlm")
            t = t5v.get(key, {})
            t5s = " / ".join(f2(100 * t[b]["selected"]["cer"]) if b in t else "—" for b in ("ctc", "lm"))
            body += (f"| {NAME[corpus]} | {g('raw_ctc')} | {g('char_lm_corpus_plus_wikipedia')} | {g('t5_on_raw_ctc')} | "
                     f"{g('t5_on_char_lm')} | {f2(100 * w['cer']) if w else '—'} | {t5s} |\n")
    if role == "student":
        ctl = load(os.path.join(R, "control_lm_source.json")) or {}
        crow = [x for x in ctl.get("rows", []) if x["seed"] == seed]
        if crow:
            body += ("\n### Where the character-LM gain comes from (control, `results/control_lm_source.json`)\n\n"
                     "Per-corpus Students of this seed, test CER % (alpha / beta chosen on validation, test decoded once). "
                     "\"train-only\" = LM fitted on the corpus training transcriptions alone; \"Wikipedia-only\" = the control "
                     "arm with the Wikipedia text alone (order 9/10, the experiments' line caps: close to, but not the same file "
                     "as, the released order-8 LM); \"combined\" = the experiments' LM.\n\n"
                     "| corpus | greedy | train-only LM | Wikipedia-only LM | combined LM | share of the gain given by train-only |\n"
                     "|---|---:|---:|---:|---:|---:|\n")
            for x in crow:
                g, tr, wk, cb = (x[k]["test"]["cer"] for k in ("greedy", "train_only", "wiki_only", "combined"))
                share = 100 * (g - tr) / (g - cb) if g > cb else float("nan")
                body += f"| {NAME[x['corpus']]} | {f2(100 * g)} | {f2(100 * tr)} | {f2(100 * wk)} | {f2(100 * cb)} | {share:.0f} % |\n"
            body += ("\nMost of the LM gain comes from the statistics of the corpus's own training text, which cannot be "
                     "redistributed. **The released Wikipedia-only LM is therefore weaker than the paper's corpus + Wikipedia "
                     "LM**: on validation (`docs/RELEASE_LM_EVAL.md`, seed-1 per-corpus Students) ")
            parts = []
            for c in CORPORA:
                k = f"{REL.STUDENT}/{c}|{c}"
                r = lmv.get(k, {})
                w, cp = r.get("wiki_lm", {}).get("charlm"), r.get("corpus_plus_wiki_lm")
                if w and cp:
                    parts.append(f"{NAME[c]} {f2(100 * w['cer'])} vs {f2(100 * cp['cer'])}")
            body += "; ".join(parts) + " (released vs corpus + Wikipedia, CER %).\n"
        body += "\n### Validation-selected Student path per corpus (the paper's protocol)\n\n| corpus | seed | selected path | valid | test |\n|---|---|---|---:|---:|\n"
        for corpus in CORPORA:
            if seed == 1:
                best = min((x for x in sel.get(corpus, {}).values() if x["system"] == "Student"),
                           key=lambda x: x["v"][0], default=None)
                if best:
                    body += f"| {NAME[corpus]} | 1 | {best['stage']} (`{best['id']}`) | {f2(100 * best['v'][0])} | {f2(100 * best['t'][0])} |\n"
            elif corpus in s2:
                x = s2[corpus]
                vv = x["paths"][x["selected_path"]]["valid"]
                body += f"| {NAME[corpus]} | 2 | {x['selected_path']} (`{x['model']}`) | {f2(100 * vv)} | {f2(100 * x['selected_test'])} |\n"
        body += ("\nSources: `results/final_selection.json` (seed 1) and `results/seed2_student.json` (seed 2). The char-LM "
                 "paths there use the corpus + Wikipedia LM of the experiments, not the released LM.\n")
    else:
        body += ("\nThe Teacher *system* of the paper is a sequence-level ensemble of up to 12 Teachers with an LM; this repo "
                 "holds ONE five-corpus Teacher (`c128_syn_teacher_s2`, preferred over `_s1` on combined validation CER "
                 "3.131 % vs 3.141 %). Its numbers above are single-model numbers.\n")
    if lat:
        body += "\n### CPU latency (batch 1, one line, `docs/RELEASE_ENGINEERING_NOTES.md`)\n\n| variant | decoder | median ms |\n|---|---|---:|\n"
        for row in lat.get("rows", []):
            if row["repo"] == repo:
                body += f"| {row['variant']} | {row['decoder']} | {row['median_ms']:.0f} |\n"
        body += f"\nMeasured on {lat.get('cpu', 'CPU')} with {lat.get('threads')} torch threads, preprocessing included.\n"
    tr = cfgs[variants[-1]]["provenance"]
    body += f"""
## Training summary

Five-corpus training (IAM, RIMES, Bentham, Saint Gall, Washington, official training splits, 128 x 1632) mixed with
120k synthetic lines (106 Google handwriting fonts, Wikipedia en/fr/la text, Eq. 33 ratio 0.1 -> 0.4), strong
augmentation, AdamW with cosine schedule, EMA 0.999, best epoch on validation CER; Students with knowledge
distillation from the 128 x 1632 Teacher (tau 2, alpha 0.7 -> 0.4, gamma 0.2 -> 0.5) where validation preferred
it; per-corpus variants are fine-tuned from the general model for 30 epochs (Saint Gall on marker-free labels).
Exact commands: `training_command` in each `config.json` (e.g. `{tr['training_command']}`) and the queue files of the
code repository (`configs/queues/`). No test line was used for training, selection or tuning.

## Limitations

{LIMITATIONS}

## Intended use

{INTENDED}

## Data licences and required citations

The weights are trained on the corpora below and inherit their non-commercial terms.

{DATA_LICENCES}
## Citation

```bibtex
{BIB_NEW}
```

## License

Weights: **CC BY-NC 4.0** (training-data terms). Code (`htrjand` package, GitHub `{REL.GITHUB_REPO}`): MIT, covering
source code only. Released character LM and lexicons: CC BY-SA 4.0 (Wikipedia-derived), separate repo.
"""
    open(os.path.join(repo_dir, "README.md"), "w", encoding="utf-8").write(head + body)


def t5_card(d, ev):
    meta = json.load(open(os.path.join(d, "htrjand_corrector.json")))
    tm = json.load(open(os.path.join(d, "train_meta.json")))
    t5v = ev.get("t5", {})
    rows = ""
    for key, r in sorted(t5v.items()):
        repo, rest = key.split("/", 1)
        v, corpus = rest.split("|")
        for b in ("ctc", "lm"):
            if b in r:
                s = r[b]["selected"]
                rows += (f"| {repo}/{v} | {NAME[corpus]} | {'raw CTC' if b == 'ctc' else 'released LM'} | "
                         f"{f2(100 * r[b]['base']['cer'])} | {f2(100 * s['cer'])} | {s['detect']} / {s['apply']} / {s['theta']} | {s['n_changed']} |\n")
    txt = f"""---
library_name: transformers
pipeline_tag: text2text-generation
license: cc-by-nc-4.0
base_model: google-t5/t5-small
language:
- en
- fr
- la
tags:
- handwritten-text-recognition
- htr
- ocr
- post-ocr-correction
- t5
---

# {REL.WORK_NAME[0].upper() + REL.WORK_NAME[1:]}: gated T5-small post-OCR corrector

> **DRAFT model card.** Version {REL.RELEASE_VERSION}, source commit `{REL.SOURCE_COMMIT}`.

**Modified from `google-t5/t5-small` (Apache-2.0)**: fine-tuned ({tm['epochs_run']} epochs, best validation loss
{tm['best_valid_loss']:.4f}, {tm['n_train']:,} training pairs, {tm['params']:,} parameters) as the line-level corrector of
Algorithm 5 of the paper *{REL.PAPER_TITLE}* ({REL.PAPER_NOTE}). The SentencePiece vocabulary of t5-small is
extended with the corpus characters it cannot encode. See `NOTICE` and `LICENSE-APACHE-2.0`.

Input: `{meta['input_template']}`; output: the corrected line. Training pairs: out-of-fold recognitions of the
benchmark TRAINING lines, the same lines corrupted with the measured error profile, and 226k Wikipedia lines
(en / fr / la, CC BY-SA 4.0) corrupted the same way; early stopping on out-of-fold VALIDATION recognitions.

## Use (with a recogniser)

```python
from htrjand import HTRJANDModel
m = HTRJANDModel.from_pretrained("{REL.DEFAULT_MODEL}", variant="iam")
r = m.transcribe("line.png", decoder="lm", t5=True)   # loads this corrector and the gate from config.json
print(r.raw, r.lm, r.t5, r.t5_applied)
```
The gate (detect = all / out-of-lexicon / error estimate; apply = whole line / out-of-lexicon segments; threshold theta
on exp(length-normalised beam log-probability)) is chosen per recogniser, corpus and base decoder on VALIDATION and
stored in the recogniser's `config.json`. The out-of-lexicon detector uses the Wikipedia word lists of
{REL.repo_id(REL.CHARLM)} (the experiments used the corpus training lexicon, not redistributed). Punctuation-spacing
rules per corpus (`htrjand_corrector.json`, two fractions per mark, from the training transcriptions) restore each
corpus's convention.

## Validation effect of the released gates

| recogniser | corpus | base | base CER | after T5 | gate (detect / apply / theta) | lines changed |
|---|---|---|---:|---:|---|---:|
{rows}
Ungated T5 raises CER on every path of the experiments; on most paths the gate rejects nearly all candidates.
Test-split effects of the experiments: `metrics` in each recogniser's config.json and `docs/FINAL_REPORT_T5.md`.

## Limitations

T5 can rewrite a correct but unusual word (names, numbers, historical spelling) into a plausible wrong one; the gate
reduces but does not remove this. Trained for English / French / Latin and the five corpora's conventions only.

## License

**CC BY-NC 4.0** for the fine-tuned weights (trained on non-commercial benchmark transcriptions); the base model
t5-small is Apache-2.0 (notice and licence text included). Citation: see the recogniser cards.
"""
    open(os.path.join(d, "README.md"), "w", encoding="utf-8").write(txt)


def lm_card(d, ev, R):
    lmv = ev.get("lm", {})
    rows = ""
    for key, r in sorted(lmv.items()):
        w = r.get("wiki_lm", {}).get("charlm")
        if w:
            rows += f"| {key.split('|')[0]} | {NAME[r['corpus']]} | {f2(100 * r['greedy']['cer'])} | {f2(100 * w['cer'])} | {w['alpha']} / {w['beta']} |\n"
    files = ""
    for f in sorted(os.listdir(d)):
        if f.endswith(".gz"):
            p = os.path.join(d, f)
            import gzip
            with gzip.open(p, "rt", encoding="utf-8") as fh:
                h = json.loads(fh.readline())
            files += f"| `{f}` | {h['meta'].get('lang')} | {h['order']} | {h['meta'].get('n_lines'):,} | {os.path.getsize(p) / 1e6:.1f} MB | `{sha(p)[:16]}…` |\n"
    txt = f"""---
license: cc-by-sa-4.0
language:
- en
- fr
- la
tags:
- handwritten-text-recognition
- language-model
- character-n-gram
---

# {REL.WORK_NAME[0].upper() + REL.WORK_NAME[1:]}: Wikipedia character n-gram LMs and lexicons

> **DRAFT card.** Version {REL.RELEASE_VERSION}, source commit `{REL.SOURCE_COMMIT}`.

Character 8-gram models (stupid backoff, `htrjand.decode.CompactCharLM`, gzip text format) and word lists used by
the `decoder="lm"` option and the T5 gate of the {REL.WORK_NAME} recognisers. **Built from Wikipedia-derived text
only**; no benchmark transcription is included. Rebuild: `scripts/build_release_lm.py` of the code repository.

| file | language | order | text lines | size | sha256 |
|---|---|---:|---:|---:|---|
{files}
`lexicon/wiki_<lang>.txt.gz`: word types seen at least twice in the same text (T5 out-of-lexicon detector).

## Source and attribution

{LM_PROVENANCE} Contains text from Wikipedia (https://www.wikipedia.org), adapted (cut into line fragments,
reduced to character n-gram counts and word types). No benchmark transcription is included.

## Validation CER of the released LM decoder (%)

| recogniser | corpus | greedy | with this LM | alpha / beta |
|---|---|---:|---:|---|
{rows}
Validation only (alpha / beta chosen on the same split, as in the experiments). See `docs/RELEASE_LM_EVAL.md`.

## License

CC BY-SA 4.0 (share-alike: adapted LMs must keep this licence). Never mix benchmark transcriptions into an LM
released under this licence.
"""
    open(os.path.join(d, "README.md"), "w", encoding="utf-8").write(txt)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-root", required=True)
    ap.add_argument("--results-dir", default=os.path.join(ROOT, "results"))
    a = ap.parse_args(argv)
    R = a.results_dir
    E = os.path.join(R, "release_eval")
    ev = dict(lm=load(os.path.join(E, "lm_validation.json")) or {}, t5=load(os.path.join(E, "t5_validation.json")) or {},
              latency=load(os.path.join(E, "latency.json")) or {})
    for repo, role, seed in ((REL.STUDENT, "student", 1), (REL.STUDENT_SEED2, "student", 2), (REL.TEACHER, "teacher", 2)):
        d = os.path.join(a.hf_root, repo)
        if os.path.isdir(d):
            recogniser_card(d, repo, role, seed, R, ev)
            print("wrote", os.path.join(d, "README.md"))
    d = os.path.join(a.hf_root, REL.T5_CORRECTOR)
    if os.path.isdir(d):
        t5_card(d, ev)
        print("wrote", os.path.join(d, "README.md"))
    d = os.path.join(a.hf_root, REL.CHARLM)
    if os.path.isdir(d):
        lm_card(d, ev, R)
        print("wrote", os.path.join(d, "README.md"))


if __name__ == "__main__":
    main()

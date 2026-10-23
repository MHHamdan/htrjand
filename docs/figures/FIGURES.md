# HTR-JAND figures (`docs/figures/`)

> **In this repository.** Shipped: `architecture.{png,svg,pdf}`, `student_comparison.{png,svg}`, `decoding_ablation.{png,svg}`, `pipeline.png` and `attention.png` (IAM line images only; IAM Handwriting Database, Marti & Bunke 2002, CC BY-NC 4.0, non-commercial research illustration), `qualitative_examples_readme.png` (IAM, CC BY-NC 4.0, and Bentham, CC BY 4.0, with the attribution footer). Not shipped: `qualitative.*` and `qualitative_caption.txt` (Saint Gall / Washington images; IAM-HistDB terms forbid redistribution), the other PDF/SVG variants and the data JSONs; `results/qualitative_examples.json` is excluded (per-sample transcriptions). The figure scripts write to `docs/figures/`.

**Licensing** (`docs/audit_2026_10/LICENSING.md`):
* IAM and RIMES are CC BY-NC 4.0. They are acceptable in a non-commercial research README if
  attributed and limited to a few lines.
* Bentham is CC BY 4.0.
* Saint Gall and Washington (IAM-HistDB) are for non-commercial research and teaching only, and must
  not appear in the public README.

Image content per figure:
* `qualitative.*` has IAM, Saint Gall and Washington images, so it is for the paper only, with the
  attribution sentence in its caption.
* `qualitative_examples_readme.png` has IAM and Bentham only, with a footer, and is README-safe.
* `pipeline.*`, `attention.*` and `architecture.*` have IAM images only.
* `student_comparison` and `decoding_ablation` have no images.

**Environment.** Run from the repo root. The style comes from `figkit` in
`~/figures4papers`; set `FIGKIT_ROOT` if it lives elsewhere. Archives are read from `data/`, or from
`HTRJAND_DATA_DIR` if set. Raw corpora, used only to recover sample IDs and the original IAM image,
come from `HTRJAND_RAW_DIR` (default `<repo>/raw`).
The GPU is needed only for `attention_activations.py`; use `CUDA_VISIBLE_DEVICES=0`.

**Print size.** All figures are built at print size: 160 mm wide, except the architecture figure at
170 mm. Include them with `\includegraphics[width=\textwidth]{...}`. PNGs are 200 dpi; PDF and SVG
are vector with embedded TrueType text.

Regenerate everything:
```bash
python scripts/build_qualitative_examples.py          # results/qualitative_examples.json
python scripts/fig_qualitative.py                     # qualitative.*, qualitative_examples_readme.png, qualitative_caption.txt
python scripts/fig_student_comparison.py              # student_comparison.*, decoding_ablation.*, student_comparison_data.json
python scripts/fig_pipeline.py                        # pipeline.*
CUDA_VISIBLE_DEVICES=0 python scripts/attention_activations.py   # results/attention_student_iam_test.npz, attention_stats.json
python scripts/fig_attention.py                       # attention.*
python scripts/verify_architecture.py                 # architecture_shapes.json
python scripts/fig_architecture.py                    # architecture.*
```
Shared helpers: `scripts/refined_figs_common.py`. It holds the artifact ids, the Levenshtein
alignment, the validation-path rule and the style hook.

---

## 1. `qualitative.{pdf,png,svg}` (paper) and `qualitative_examples_readme.png` (public README)
* **Command:** `python scripts/build_qualitative_examples.py && python scripts/fig_qualitative.py`
* **Sources:**
  * `results/qualitative_examples.json`. Deterministic rules are documented in the script docstring
    and in the JSON's `rules`, with 45 samples over the five test sets.
  * Seed-1 validation-selected Student predictions: `results/ft_*_c128_student*_test_preds.tsv`,
    `nopt_sg_student_s1_*`, `ens_*_wiki_test_preds.tsv`, and `t5_*_test_preds.tsv` for the ids in
    `results/final_selection.json`.
  * HTR-JAND base (68×864): `results/combined_final_student_kd_s1_<corpus>_test_preds.tsv`. This is
    the tuned five-corpus Student at 68×864, decoded greedily. Its lines are mapped to archive indices
    through the reference text. The Saint Gall model was trained with the original labels.
  * Line images: `data/<corpus>_128x1632.hdf5` test split (Saint Gall: `saintgall_nopt_128x1632.hdf5`).
* **Row rule:** (a) IAM `lm_correction`, (b) IAM `t5_correction`, (c) Saint Gall `historical`,
  (d) Washington `historical`, (e) IAM `t5_failure`, (f) IAM `model_failure`.
* **README version (`qualitative_examples_readme.png`):** only IAM (CC BY-NC 4.0) and Bentham (CC BY 4.0)
  rows, because IAM-HistDB images (Saint Gall, Washington) must not appear in the public README
  (`docs/audit_2026_10/LICENSING.md`). Rows: IAM `lm_correction`, `t5_correction`, `t5_failure`,
  `model_failure`, and Bentham `median_cer`. Footer: "Line images: IAM Handwriting Database (Marti & Bunke
  2002, CC BY-NC 4.0) and Bentham (Sánchez et al. 2014, CC BY 4.0); test split; used for non-commercial
  research illustration." The earlier `qualitative_examples.png`, which contained a Saint Gall row,
  was deleted.
* **Proposed caption:**
  > Test-set examples selected by fixed rules: (a) IAM, largest per-line CER reduction from greedy CTC
  > to the character LM; (b) IAM, largest reduction from the character LM to the gated T5 corrector;
  > (c) Saint Gall and (d) Washington, the line at the median greedy-CTC CER; (e) IAM, largest CER
  > increase caused by T5; (f) IAM, highest CER of the validation-selected decoding among references of
  > at least 20 characters. Each block shows the 128×1632 network input, the reference, the
  > HTR-JAND base Student (68×864, greedy CTC), and the HTR-JAND Student (seed 1) with greedy CTC, with
  > the character LM, and with T5 when T5 changed the line. Red marks substituted or inserted
  > characters, and a red wedge marks a deleted reference character. Right: per-line CER (%).
  > Line images: IAM Handwriting Database (U. Marti and H. Bunke, IJDAR 5:39–46, 2002; CC BY-NC 4.0);
  > Saint Gall, IAM-HistDB (A. Fischer et al.), from St. Gallen, Stiftsbibliothek, Cod. Sang. 562, p. 30
  > (www.e-codices.ch; CC BY-NC 4.0); Washington, IAM-HistDB (A. Fischer et al.), George Washington
  > Papers, Library of Congress; official test splits, used for non-commercial research.
  > (The longer text is in `qualitative_caption.txt`.)

## 2. `student_comparison.{pdf,png,svg}`
* **Command:** `python scripts/fig_student_comparison.py`
* **Sources:**
  * Seed 1: `results/final_selection.json`.
  * Seed 2: `results/seed2_student.json`. Its `saintgall` entry is the marker-free Student
    `nopt_sg_student_s2` (the "(marker-free labels)" paths). That entry includes the T5 arms
    `results/t5_saintgall_nopt_s2student_{ctc,lm}_t5_ext_ctx1.json`, where the gate selected θ = 1.01,
    so T5 changed no line: test 2.30 raw and 1.57 LM.
  * HTR-JAND base (68×864): `results/combined_final_student_kd_s{1,2}_<corpus>_test.json`, the tuned
    five-corpus Student decoded greedily. † The Saint Gall base model was trained with the original
    labels (raw "pt" letter).
  * Plotted numbers: `student_comparison_data.json`.
* **Proposed caption:**
  > Test CER per corpus. Grey: HTR-JAND base Student (68×864 input), greedy CTC. Light blue: HTR-JAND
  > Student (128×1632 input), greedy CTC. Dark blue: HTR-JAND Student, decoding path selected on
  > validation (character LM, optionally followed by the gated T5 corrector; the path per seed is
  > printed under each corpus). Bars are the mean of two seeds, and markers show each seed. † The Saint
  > Gall base model was trained with the original labels, which contain the raw "pt" letter.

## 3. `decoding_ablation.{pdf,png,svg}`
* **Command:** same script as 2.
* **Sources:** same as 2, using the stages raw CTC, char LM and T5 on char LM.
* **Proposed caption:**
  > Test CER of the HTR-JAND Student after greedy CTC, after prefix beam search with the character LM,
  > and after the gated T5 corrector, for seeds 1 and 2. The dotted line is the HTR-JAND base Student
  > (68×864, greedy CTC, mean of two seeds). † The Saint Gall base model was trained with the original labels.

## 4. `pipeline.{pdf,png,svg}`
* **Command:** `python scripts/fig_pipeline.py`
* **Sources:**
  * The line is the IAM `t5_correction` sample (test index 1431, `p01-174-01`).
  * The original image is `raw/iam/lines/p01/p01-174/p01-174-01.png`; the network input is the archive image.
  * Strings come from `ft_iam_c128_student_kd_s1_iam_test_preds.tsv`,
    `ens_iam_ft_iam_c128_student_kd_s1_wiki_test_preds.tsv` and `t5_iam_ftstudent_lm_t5_ext_ctx1_test_preds.tsv`
    (T5 confidence 0.9974 > θ = 0.98).
  * α, β and θ come from the matching result JSONs.
* **Proposed caption:**
  > Recognition of IAM test line p01-174-01. The scanned line is scaled and padded into the 128×1632
  > input. The Student's greedy CTC output ("asthe dawn was brexteng , aund", CER 16.7 %) is
  > re-decoded with prefix beam search and the character LM ("asthe dawn was breaking , ourd", 13.3 %).
  > The gated T5 corrector then replaces the line (the reference line, 0.0 %; text not reproduced here) because it
  > contains an out-of-vocabulary word and the candidate's confidence (0.997) exceeds θ = 0.98. The
  > LM and T5 are external to the network.

## 5. `attention.{pdf,png,svg}` (+ `attention_notes.md`, `attention_stats.json`)
* **Command:** `CUDA_VISIBLE_DEVICES=0 python scripts/attention_activations.py && python scripts/fig_attention.py`
* **Sources:**
  * Forward hooks on `runs/ft_iam_c128_student_kd_s1/best.pth` over all 1861 IAM test lines. The
    greedy decode matches the published TSV on 1861/1861 lines.
  * The line shown is the IAM `lm_correction` sample (test 1849).
* **Proposed caption:**
  > Activations of the IAM Student on test line p06-096-09. (a) Input. (b) Greedy CTC emissions at
  > their frames (red: wrong). (c) MHSA weight received per key frame, averaged over queries. (d)
  > Proxima weights. (e) Per-frame L2 norms of the three summands of Eq. 18. (f) Per-frame maximum
  > posterior. (g) MHSA attention matrix. (h) Confidence at the emitting frame for correct and wrong
  > emissions over all IAM test lines. On the test set, MHSA is not local: mass within ±4 frames is
  > 4.3 %, against 4.7 % for uniform attention. It is largely query-independent and concentrates on
  > frames around emitted spaces (2.3× their share). Proxima spreads over about 122 of 193 frames and
  > favours character-emission frames (2.5×), and its output is the same for every frame. Low
  > confidence flags wrong emissions with AUROC 0.87.

## 6. `architecture.{pdf,png,svg}` (+ `architecture_notes.md`, `architecture_shapes.json`)
* **Command:** `python scripts/verify_architecture.py && python scripts/fig_architecture.py`
* **Sources:**
  * Shapes and parameter counts come from hooks on `runs/ft_iam_c128_student_kd_s1`,
    `runs/c128_student_kd_s1` and `runs/c128_syn_teacher_s1`.
  * Training settings come from the checkpoint `args`. Decoder and corrector settings come from
    `results/ens_*_wiki.json` and `results/t5_iam_ftstudent_lm_t5_ext_ctx1.json`.
  * The thumbnail is IAM test line 0.
* **Proposed caption:**
  > HTR-JAND Student. A 128×1632 line passes through three gated-convolution + SE blocks
  > (16/32/64 channels, pools 2×2, 2×2 and 4×2) to 204 columns of 512 features. These feed four
  > BiLSTM layers with LayerNorm (64 units per direction), then multi-head self-attention (one head) in
  > parallel with Proxima attention (one learned query, broadcast summary), fused as
  > LN(W_f[A;P]+b+X). A linear layer gives CTC posteriors over 111 symbols, for 806,287 parameters at
  > inference. Training only: a frozen Teacher (7,220,911 parameters) provides soft targets (τ = 2),
  > synthetic lines are mixed in, and an auxiliary CTC head on the CNN columns is trained with δ = 0.1.
  > Decoding uses greedy CTC or prefix beam search with an external character n-gram LM, optionally
  > followed by a gated T5-small corrector. Both run outside the network.

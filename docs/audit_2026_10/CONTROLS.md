# Reviewer-requested controls, 2026-10-17

> **Release-export note.** Written in the original working repository; kept verbatim except for mechanical edits: absolute machine paths replaced by repo-relative paths, `$HTRJAND_DATA_DIR/` or descriptive placeholders (`<historical repo>/...` is the earlier working repository, provenance only); `htrjand_paper_v1/` renamed `htrjand/`; the earlier repository's own `htrjand/` directory (a different model, not in this release) written `htrjand-jan/`; `docs/figures_refined/` renamed `docs/figures/`.


Every number here is an **ABLATION / CONTROL**. None of it is a newly selected system and none of it should replace a headline number. Each hyper-parameter (LM alpha/beta, T5 gate) is chosen on validation by the existing code. Test is decoded once per configuration. Oracle rows read the labels of the split they are scored on and are **diagnostic upper bounds only**.

Environment: `HTRJAND_DATA_DIR=$HTRJAND_DATA_DIR`, `CUDA_VISIBLE_DEVICES=0`, repo root `htr-hand-refined/`. Logs are in `logs/controls/`. The intermediate ensemble/T5 outputs are in `results/controls/`.

## C1. LM-source ablation (review M7): `results/control_lm_source.json`, `results/control_lm_sizes.json`

Code change: `htrjand/ensemble_rescore.py` gets a new flag, `--lm-no-train-text` (CONTROL ONLY). It fits the LM on the `--lm-text` files alone. Default behaviour and default JSON output are unchanged. Orders, max-lines, the alpha grid (default 0.0–1.0) and the beta grid `0.0,…,5.0` are those of `scripts/seed2_student_chain.py`. Single member; gen-alpha 0.2, beam 16, n-best 8 (defaults).

```
python scripts/control_lm_source.py --arm train     # (a) training transcriptions only: no --lm-text
python scripts/control_lm_source.py --arm wiki      # (b) --lm-text synth/text_<lang>.txt --lm-max-lines N --lm-no-train-text
python scripts/control_lm_source.py --collect
python scripts/control_lm_sizes.py
```
Each arm runs, for example: `python -m htrjand.ensemble_rescore --gpu 0 --dataset iam --tag ctl_lmwiki_iam_ft_iam_c128_student_kd_s1 --members runs/ft_iam_c128_student_kd_s1/best.pth --order 10 --betas 0.0,0.5,1.0,1.5,2.0,3.0,4.0,5.0 --workers 12 --out-dir results/controls --lm-text synth/text_en.txt --lm-max-lines 400000 --lm-no-train-text` (the full list is in the JSON under `commands`). "combined" is the existing `results/ens_<tag>.json`.

CER % (valid / test). Seed 1 = `final_selection.json` "Student | raw CTC". Seed 2 = `seed2_student.json`. Saint Gall uses `saintgall_nopt`.

| corpus | seed | model | greedy v/t | train-only LM v/t | Wikipedia-only LM v/t | combined LM v/t | test WER train / wiki / comb |
|---|---|---|---|---|---|---|---|
| iam | 1 | `ft_iam_c128_student_kd_s1` | 5.16 / 6.22 | 3.92 / 4.94 | 3.59 / 4.74 | 3.42 / 4.49 | 15.36 / 14.99 / 13.68 |
| iam | 2 | `ft_iam_c128_student_s2` | 5.46 / 6.62 | 4.06 / 5.17 | 3.74 / 4.86 | 3.46 / 4.56 | 15.51 / 15.26 / 13.77 |
| rimes | 1 | `ft_rimes_c128_student_kd_s1` | 3.58 / 4.80 | 2.29 / 3.23 | 2.54 / 3.53 | 2.21 / 3.13 | 10.00 / 11.53 / 9.61 |
| rimes | 2 | `ft_rimes_c128_student_kd_s2` | 3.57 / 4.75 | 2.31 / 3.18 | 2.51 / 3.59 | 2.18 / 3.11 | 9.93 / 11.54 / 9.40 |
| bentham | 1 | `ft_bentham_c128_student_kd_s1` | 3.74 / 5.64 | 2.91 / 4.80 | 3.24 / 4.92 | 2.90 / 4.81 | 30.22 / 30.20 / 30.22 |
| bentham | 2 | `ft_bentham_c128_student_kd_s2` | 3.63 / 5.55 | 2.80 / 4.53 | 3.10 / 4.75 | 2.75 / 4.58 | 29.25 / 29.76 / 29.48 |
| saintgall | 1 | `nopt_sg_student_s1` | 2.55 / 2.49 | 2.08 / 1.98 | 2.04 / 1.86 | 1.88 / 1.77 | 16.02 / 14.59 / 13.93 |
| saintgall | 2 | `nopt_sg_student_s2` | 2.14 / 2.30 | 1.78 / 1.80 | 1.76 / 1.65 | 1.66 / 1.57 | 15.44 / 13.44 / 12.73 |
| washington | 1 | `ft_washington_c128_student_s1` | 5.03 / 6.29 | 4.04 / 5.13 | 3.74 / 4.57 | 3.54 / 4.58 | 19.01 / 16.52 / 16.44 |
| washington | 2 | `ft_washington_c128_student_s2` | 4.50 / 5.45 | 3.67 / 4.52 | 3.22 / 4.21 | 3.07 / 4.07 | 16.44 / 14.81 / 14.55 |

On validation, the combined LM is best in 10 of 10 cells. Share of the greedy-to-combined test gain that the training-text-only LM already delivers: IAM 74 / 70 %, RIMES 94 / 96 %, Bentham 102 / 106 %, Saint Gall 70 / 67 %, Washington 68 / 67 % (seed 1 / seed 2). Increment of the external text over the training-only LM, on test: IAM +0.45 / +0.62 pp, RIMES +0.11 / +0.07, Bentham −0.02 / −0.06 (none), Saint Gall +0.21 / +0.24, Washington +0.54 / +0.45. On test, the training-text LM is slightly better than the combined LM on Bentham seed 2 (4.53 vs 4.58). The Wikipedia-only LM beats the training-only LM on IAM, Saint Gall and Washington, and is worse on RIMES and Bentham.

LM size (Python `CharLM`: one entry per distinct (context, char) pair over all orders 1..n). "JSON MB" is `CharLM.save`; gz is gzip -6. The Bentham and Washington Wikipedia-only LMs are the same LM (same 200k English lines). The Latin text file has only 100k lines.

| corpus | order | Wikipedia lines (chars) | train-only: n-gram entries / JSON MB / gz MB | Wikipedia-only | combined |
|---|---|---|---|---|---|
| iam | 10 | 400,000 (14,066,999) | 757,683 / 26.0 / 5.4 | 18,849,334 / 633.1 / 141.0 | 19,134,615 / 642.5 / 143.1 |
| rimes | 9 | 200,000 (7,588,741) | 509,492 / 17.2 / 3.9 | 7,802,412 / 260.7 / 59.6 | 8,023,226 / 267.9 / 61.4 |
| bentham | 9 | 200,000 (7,028,542) | 745,038 / 23.7 / 5.3 | 7,978,002 / 251.0 / 58.7 | 8,327,310 / 261.8 / 61.3 |
| saintgall | 9 | 100,000 (3,608,747) | 111,714 / 3.6 / 0.7 | 6,033,403 / 195.5 / 43.8 | 6,082,801 / 197.1 / 44.1 |
| washington | 9 | 200,000 (7,028,542) | 50,872 / 1.7 / 0.4 | 7,978,002 / 251.0 / 58.7 | 7,992,891 / 251.5 / 58.8 |

Entries per order, IAM combined 10-gram: 1,236 / 8,478 / 48,127 / 213,547 / 636,842 / 1,385,610 / 2,408,231 / 3,589,460 / 4,829,742 / 6,013,342. Resident memory of that LM in the Python decoder: 3.41 GiB (C4).

## C2. T5 correctors on a second recognizer family (review M2, M3, Q8.2): `results/control_t5_second_family.json`

Leakage check (read from `htrjand/t5post.py:241-313` and `postproc/t5data*/pairs_meta.json`):
- Pairs are out-of-fold greedy recognitions of the TRAINING split of `combined_128x1632` by `runs/c128_fold{0,1}of2`. Each fold model trains on the complement fold (`--fold k/2`, index % 2) and has no init checkpoint. The models are paper_teacher at 128×1632 (synthetic data from `synth_120k`).
- External-text pairs (t5_ext_ctx1 only) refuse any text containing a ≥25-char test line verbatim.
- Validation pairs come from fold 0's recognitions of the validation split. They are used only for the corrector's early stopping (`train_meta.json`: "lowest validation loss"). Test labels are read only by the final metric.

Applying the correctors to the valid/test outputs of any recognizer is therefore test-leak-free. One caveat for the validation-selected gate: the correctors' early stopping saw validation labels, so validation is mildly optimistic for T5 as a whole. This holds for the paper's own T5 paths too.

The 68×864 reconstruction models had no valid preds, so they were generated:
```
python -m htrjand.evaluate --gpu 0 --split valid --ckpt runs/<m>/best.pth --dataset <corpus> --tag <m>_<corpus> --out-dir results/controls
python scripts/control_t5_second_family.py --run      # t5post apply --val-preds results/controls/<tag>_valid_preds.tsv --test-preds results/<tag>_test_preds.tsv
python scripts/control_t5_second_family.py --collect
```
Recognizers: `combined_paper_student_kd_s1` (literal Student, 32 units/dir, KD), `combined_paper_teacher_s1` (literal Teacher, 64/dir), `combined_final_teacher_s1` (68×864 final-recipe Teacher, 128/dir), `iam_paper_student_{kd,nokd}_s1` and `iam_paper_teacher_s1` (IAM only), and the plain CNN-BiLSTM-CTC ablation `iam_tuned_abl_cnn_lstm_s1` (M2). All decoding is greedy CTC. Saint Gall uses the original labels (that is how the reconstructions were trained). CER %:

| recognizer | corpus | corrector | valid base -> gated | test base | test gated (rel) | test ungated (rel) | gate |
|---|---|---|---|---|---|---|---|
| `combined_paper_student_kd_s1` | iam | t5_ext_ctx1 | 10.93 -> 10.60 | 11.92 | 11.66 (-2.1%) | 15.21 (+27.6%) | all/line/θ=0.95 |
| `combined_paper_student_kd_s1` | iam | t5_ctx1 | 10.93 -> 10.86 | 11.92 | 11.87 (-0.4%) | 18.42 (+54.6%) | all/line/θ=0.95 |
| `combined_paper_student_kd_s1` | rimes | t5_ext_ctx1 | 8.79 -> 7.06 | 10.17 | 8.24 (-19.0%) | 9.74 (-4.2%) | oov_or_eps/line/θ=0.9 |
| `combined_paper_student_kd_s1` | rimes | t5_ctx1 | 8.79 -> 7.65 | 10.17 | 9.11 (-10.4%) | 12.48 (+22.7%) | oov_or_eps/line/θ=0.95 |
| `combined_paper_student_kd_s1` | bentham | t5_ext_ctx1 | 9.09 -> 8.79 | 10.43 | 10.15 (-2.7%) | 12.06 (+15.7%) | all/segment/θ=0.95 |
| `combined_paper_student_kd_s1` | bentham | t5_ctx1 | 9.09 -> 8.96 | 10.43 | 10.31 (-1.1%) | 13.32 (+27.7%) | all/line/θ=0.95 |
| `combined_paper_student_kd_s1` | saintgall | t5_ext_ctx1 | 11.92 -> 11.92 | 11.65 | 11.64 (-0.1%) | 11.87 (+1.9%) | all/line/θ=0.98 |
| `combined_paper_student_kd_s1` | saintgall | t5_ctx1 | 11.92 -> 11.88 | 11.65 | 11.69 (+0.3%) | 11.70 (+0.4%) | eps/line/θ=0.85 |
| `combined_paper_student_kd_s1` | washington | t5_ext_ctx1 | 13.34 -> 13.13 | 16.14 | 15.88 (-1.6%) | 19.43 (+20.4%) | all/line/θ=0.98 |
| `combined_paper_student_kd_s1` | washington | t5_ctx1 | 13.34 -> 13.19 | 16.14 | 16.14 (+0.0%) | 19.79 (+22.6%) | eps/line/θ=0.8 |
| `combined_paper_teacher_s1` | iam | t5_ext_ctx1 | 7.19 -> 6.93 | 8.14 | 7.86 (-3.5%) | 10.46 (+28.5%) | all/line/θ=0.95 |
| `combined_paper_teacher_s1` | iam | t5_ctx1 | 7.19 -> 7.11 | 8.14 | 8.09 (-0.7%) | 12.80 (+57.2%) | oov/line/θ=0.98 |
| `combined_paper_teacher_s1` | rimes | t5_ext_ctx1 | 4.82 -> 4.04 | 6.08 | 5.18 (-14.8%) | 6.21 (+2.2%) | all/segment/θ=0.95 |
| `combined_paper_teacher_s1` | rimes | t5_ctx1 | 4.82 -> 4.31 | 6.08 | 5.59 (-8.1%) | 7.65 (+25.8%) | all/segment/θ=0.95 |
| `combined_paper_teacher_s1` | bentham | t5_ext_ctx1 | 5.19 -> 4.99 | 7.22 | 7.06 (-2.2%) | 8.39 (+16.1%) | all/segment/θ=0.95 |
| `combined_paper_teacher_s1` | bentham | t5_ctx1 | 5.19 -> 5.09 | 7.22 | 7.16 (-0.8%) | 8.97 (+24.3%) | all/segment/θ=0.95 |
| `combined_paper_teacher_s1` | saintgall | t5_ext_ctx1 | 8.75 -> 8.72 | 8.74 | 8.75 (+0.2%) | 9.01 (+3.0%) | eps/segment/θ=0.0 |
| `combined_paper_teacher_s1` | saintgall | t5_ctx1 | 8.75 -> 8.73 | 8.74 | 8.74 (-0.0%) | 8.90 (+1.9%) | eps/line/θ=0.95 |
| `combined_paper_teacher_s1` | washington | t5_ext_ctx1 | 6.30 -> 5.81 | 7.74 | 7.57 (-2.2%) | 9.45 (+22.1%) | all/segment/θ=0.9 |
| `combined_paper_teacher_s1` | washington | t5_ctx1 | 6.30 -> 6.01 | 7.74 | 7.71 (-0.4%) | 10.77 (+39.2%) | all/line/θ=0.9 |
| `combined_final_teacher_s1` | iam | t5_ext_ctx1 | 4.30 -> 4.19 | 5.12 | 5.01 (-2.2%) | 7.32 (+43.0%) | all/line/θ=0.98 |
| `combined_final_teacher_s1` | iam | t5_ctx1 | 4.30 -> 4.25 | 5.12 | 5.10 (-0.3%) | 8.56 (+67.2%) | all/segment/θ=0.98 |
| `combined_final_teacher_s1` | rimes | t5_ext_ctx1 | 2.53 -> 2.45 | 3.60 | 3.49 (-3.2%) | 4.18 (+16.0%) | all/segment/θ=0.98 |
| `combined_final_teacher_s1` | rimes | t5_ctx1 | 2.53 -> 2.51 | 3.60 | 3.53 (-1.9%) | 5.22 (+45.0%) | all/segment/θ=0.98 |
| `combined_final_teacher_s1` | bentham | t5_ext_ctx1 | 3.38 -> 3.34 | 5.25 | 5.19 (-1.1%) | 6.02 (+14.7%) | oov/segment/θ=0.98 |
| `combined_final_teacher_s1` | bentham | t5_ctx1 | 3.38 -> 3.35 | 5.25 | 5.25 (-0.0%) | 6.39 (+21.8%) | oov/segment/θ=0.98 |
| `combined_final_teacher_s1` | saintgall | t5_ext_ctx1 | 7.50 -> 7.49 | 7.48 | 7.49 (+0.1%) | 7.79 (+4.1%) | eps/line/θ=0.98 |
| `combined_final_teacher_s1` | saintgall | t5_ctx1 | 7.50 -> 7.50 | 7.48 | 7.48 (+0.0%) | 7.59 (+1.5%) | all/line/θ=1.01 |
| `combined_final_teacher_s1` | washington | t5_ext_ctx1 | 3.06 -> 3.02 | 3.71 | 3.68 (-0.8%) | 5.53 (+49.0%) | all/segment/θ=0.95 |
| `combined_final_teacher_s1` | washington | t5_ctx1 | 3.06 -> 3.02 | 3.71 | 3.76 (+1.3%) | 5.33 (+43.5%) | eps/line/θ=0.95 |
| `iam_paper_student_kd_s1` | iam | t5_ext_ctx1 | 11.14 -> 10.77 | 12.00 | 11.79 (-1.8%) | 15.33 (+27.7%) | all/line/θ=0.95 |
| `iam_paper_student_kd_s1` | iam | t5_ctx1 | 11.14 -> 11.08 | 12.00 | 11.99 (-0.1%) | 18.74 (+56.1%) | all/line/θ=0.95 |
| `iam_paper_student_nokd_s1` | iam | t5_ext_ctx1 | 11.15 -> 10.72 | 11.93 | 11.70 (-1.9%) | 15.50 (+30.0%) | all/line/θ=0.9 |
| `iam_paper_student_nokd_s1` | iam | t5_ctx1 | 11.15 -> 11.04 | 11.93 | 11.87 (-0.5%) | 18.61 (+56.1%) | all/line/θ=0.95 |
| `iam_paper_teacher_s1` | iam | t5_ext_ctx1 | 8.99 -> 8.58 | 9.97 | 9.70 (-2.7%) | 12.56 (+25.9%) | all/line/θ=0.95 |
| `iam_paper_teacher_s1` | iam | t5_ctx1 | 8.99 -> 8.86 | 9.97 | 9.92 (-0.5%) | 15.71 (+57.5%) | all/line/θ=0.95 |
| `iam_tuned_abl_cnn_lstm_s1` | iam | t5_ext_ctx1 | 7.26 -> 7.00 | 7.98 | 7.80 (-2.2%) | 10.73 (+34.6%) | oov/line/θ=0.95 |
| `iam_tuned_abl_cnn_lstm_s1` | iam | t5_ctx1 | 7.26 -> 7.20 | 7.98 | 7.91 (-0.8%) | 12.81 (+60.6%) | oov/line/θ=0.98 |

Mean relative test-CER change per recognizer (gate always chosen on validation; "ungated" = every line replaced by the T5 line output):

| recognizer | corrector | corpora | mean rel. gated | mean rel. ungated |
|---|---|---|---|---|
| `combined_paper_student_kd_s1` | t5_ext_ctx1 | 5 | -5.11% | +12.27% |
| `combined_paper_student_kd_s1` | t5_ctx1 | 5 | -2.31% | +25.62% |
| `combined_paper_teacher_s1` | t5_ext_ctx1 | 5 | -4.51% | +14.39% |
| `combined_paper_teacher_s1` | t5_ctx1 | 5 | -1.99% | +29.67% |
| `combined_final_teacher_s1` | t5_ext_ctx1 | 5 | -1.43% | +25.35% |
| `combined_final_teacher_s1` | t5_ctx1 | 5 | -0.20% | +35.78% |
| `iam_paper_student_kd_s1` | t5_ext_ctx1 | 1 | -1.80% | +27.67% |
| `iam_paper_student_kd_s1` | t5_ctx1 | 1 | -0.12% | +56.13% |
| `iam_paper_student_nokd_s1` | t5_ext_ctx1 | 1 | -1.94% | +29.95% |
| `iam_paper_student_nokd_s1` | t5_ctx1 | 1 | -0.49% | +56.06% |
| `iam_paper_teacher_s1` | t5_ext_ctx1 | 1 | -2.74% | +25.91% |
| `iam_paper_teacher_s1` | t5_ctx1 | 1 | -0.53% | +57.51% |
| `iam_tuned_abl_cnn_lstm_s1` | t5_ext_ctx1 | 1 | -2.18% | +34.57% |
| `iam_tuned_abl_cnn_lstm_s1` | t5_ctx1 | 1 | -0.81% | +60.56% |

## C3. Oracle gate, changed lines, paired bootstrap (review M3, M4): `results/control_t5_oracle.json`

`python scripts/control_t5_oracle.py`. No decoding. It uses the stored T5 candidates (`results/<id>_t5cache.json`) and the prediction files. ORACLE = per-line min(edit distance of base, of T5 line-mode candidate), a DIAGNOSTIC UPPER BOUND using the evaluated split's labels. Δ = CER(T5 path) − CER(base) on test, from a paired bootstrap (10,000 line resamples, `numpy.default_rng(20261017)`, one stream in row order). The sign test is exact two-sided on improved vs worsened lines among the changed ones. "sel." marks the path that the existing selection chose on validation. CER %:

| corpus | seed | path | sel. | valid base / gated / ORACLE | test base / gated / ungated / ORACLE | changed raw / normalised (ws-only) | +improved / -worsened / =equal | Δ test CER pp [95% CI] | sign p |
|---|---|---|---|---|---|---|---|---|---|
| iam | 1 | T5 on raw CTC |  | 5.16 / 4.97 / 4.17 | 6.22 / 6.04 / 8.52 / 5.27 | 326 / 324 (2) | +217 / -79 / =30 | -0.174 [-0.250, -0.099] | 4.9e-16 |
| iam | 1 | T5 on char LM | yes | 3.42 / 3.35 / 2.99 | 4.49 / 4.44 / 6.15 / 4.06 | 101 / 99 (2) | +68 / -24 / =9 | -0.053 [-0.095, -0.011] | 4.9e-06 |
| rimes | 1 | T5 on raw CTC |  | 3.58 / 3.16 / 2.50 | 4.80 / 4.25 / 5.04 / 3.56 | 209 / 209 (0) | +162 / -38 / =9 | -0.557 [-0.720, -0.397] | 2e-19 |
| rimes | 1 | T5 on char LM | yes | 2.21 / 2.21 / 1.91 | 3.13 / 3.08 / 3.82 / 2.76 | 41 / 41 (0) | +28 / -12 / =1 | -0.045 [-0.118, +0.031] | 0.017 |
| bentham | 1 | T5 on raw CTC |  | 3.74 / 3.60 / 3.19 | 5.64 / 5.52 / 6.27 / 5.06 | 114 / 110 (4) | +73 / -32 / =9 | -0.120 [-0.219, -0.016] | 7.8e-05 |
| bentham | 1 | T5 on char LM | yes | 2.90 / 2.86 / 2.63 | 4.81 / 4.79 / 5.43 / 4.58 | 31 / 20 (11) | +19 / -5 / =7 | -0.026 [-0.066, +0.024] | 0.0066 |
| saintgall | 1 | T5 on raw CTC |  | 2.55 / 2.54 / 2.52 | 2.49 / 2.49 / 2.63 / 2.43 | 34 / 33 (1) | +16 / -13 / =5 | +0.000 [-0.030, +0.030] | 0.71 |
| saintgall | 1 | T5 on char LM |  | 1.88 / 1.88 / 1.87 | 1.77 / 1.77 / 1.92 / 1.75 | 0 / 0 (0) | +0 / -0 / =0 | +0.000 [+0.000, +0.000] | – |
| washington | 1 | T5 on raw CTC |  | 5.03 / 4.87 / 4.45 | 6.29 / 6.20 / 9.10 / 5.79 | 10 / 10 (0) | +7 / -2 / =1 | -0.093 [-0.206, +0.000] | 0.18 |
| washington | 1 | T5 on char LM | yes | 3.54 / 3.47 / 3.06 | 4.58 / 4.55 / 6.28 / 4.24 | 3 / 3 (0) | +2 / -1 / =0 | -0.031 [-0.112, +0.031] | 1 |
| iam | 2 | T5 on raw CTC |  | 5.46 / 5.17 / 4.41 | 6.62 / 6.41 / 8.73 / 5.59 | 342 / 340 (2) | +233 / -73 / =36 | -0.201 [-0.286, -0.113] | 1.2e-20 |
| iam | 2 | T5 on char LM | yes | 3.46 / 3.37 / 3.03 | 4.56 / 4.53 / 6.28 / 4.16 | 100 / 92 (8) | +66 / -30 / =4 | -0.023 [-0.064, +0.020] | 0.00031 |
| rimes | 2 | T5 on raw CTC |  | 3.57 / 3.14 / 2.52 | 4.75 / 4.36 / 5.30 / 3.73 | 199 / 198 (1) | +138 / -42 / =19 | -0.389 [-0.556, -0.224] | 3.8e-13 |
| rimes | 2 | T5 on char LM |  | 2.18 / 2.18 / 1.90 | 3.11 / 3.11 / 3.94 / 2.80 | 0 / 0 (0) | +0 / -0 / =0 | +0.000 [+0.000, +0.000] | – |
| bentham | 2 | T5 on raw CTC |  | 3.63 / 3.53 / 3.12 | 5.55 / 5.47 / 6.22 / 4.92 | 103 / 102 (1) | +73 / -23 / =7 | -0.081 [-0.175, +0.021] | 3.1e-07 |
| bentham | 2 | T5 on char LM | yes | 2.75 / 2.72 / 2.52 | 4.58 / 4.55 / 5.08 / 4.28 | 30 / 17 (13) | +20 / -4 / =6 | -0.031 [-0.070, +0.020] | 0.0015 |
| saintgall | 2 | T5 on raw CTC |  | 2.14 / 2.14 / 2.06 | 2.30 / 2.30 / 2.91 / 2.17 | 0 / 0 (0) | +0 / -0 / =0 | +0.000 [+0.000, +0.000] | – |
| saintgall | 2 | T5 on char LM |  | 1.66 / 1.66 / 1.59 | 1.57 / 1.57 / 2.26 / 1.53 | 0 / 0 (0) | +0 / -0 / =0 | +0.000 [+0.000, +0.000] | – |
| washington | 2 | T5 on raw CTC |  | 4.50 / 4.35 / 3.64 | 5.45 / 5.30 / 8.20 / 4.86 | 8 / 8 (0) | +8 / -0 / =0 | -0.155 [-0.279, -0.060] | 0.0078 |
| washington | 2 | T5 on char LM | yes | 3.07 / 3.00 / 2.90 | 4.07 / 4.12 / 5.92 / 3.85 | 4 / 4 (0) | +1 / -3 / =0 | +0.047 [-0.030, +0.139] | 0.62 |

**99 vs 101 is resolved.** `t5post` counts `new != normalise(raw)`, i.e. after NFC/typography/whitespace normalisation. The prediction-file diff compares raw strings. On IAM seed 1 LM, 2 of the 101 changed lines differ from the base only in whitespace/typography. Both numbers are correct; the paper should say which one it uses. Bentham shows the same effect more strongly (31 vs 20, seed 1).

Mean relative test change, LM path (5 corpora, gate-closed cells count as 0): seed 1 −0.77 % (−0.96 % over the 4 cells where the gate opened), seed 2 −0.01 % (−0.02 % over 3 cells). Raw-CTC path: −3.60 % / −3.11 %. Oracle line gate: LM path −6.9 % / −6.6 %, raw-CTC path −12.4 % / −12.9 %. So the validation-selected gate recovers only about 11 % (LM) and 24–29 % (CTC) of the oracle headroom, and correction quality itself is bounded: the ungated T5 output is worse than the base everywhere.

The bootstrap CI of the LM-path effect excludes 0 only for IAM seed 1 (−0.053 pp [−0.095, −0.011]). The sign test is significant (p<0.05) on IAM s1/s2, RIMES s1 and Bentham s1/s2. In those cells T5 fixes more lines than it breaks, but by amounts too small to move corpus CER beyond the bootstrap noise.

## C4. Compute (review M6): `results/control_compute.json`

`python scripts/control_compute.py --test` (analytic MAC counter equals FlopCounterMode on conv/linear/attention for 5 presets; LSTM formula equals an unrolled cell) and then `python scripts/control_compute.py`. MACs exclude element-wise ops and the training-only auxiliary head. Latency is the median of 200 runs after 20 warm-ups, random input, eval + no_grad, `torch.set_num_threads(8)`.

| model | canvas | frames | deployed params | GMAC/line (conv / LSTM / linear+attn) | GPU0 ms/line b1 / b16 | CPU(8 thr) ms/line b1 / b16 |
|---|---|---|---|---|---|---|
| Student `combined_final_student_kd_s1` | 68x864 | 108 | 806,287 | 0.370 (0.288 / 0.064 / 0.019) | 3.09 / 0.23 | 9.5 / 10.6 |
| Student `ft_iam_c128_student_kd_s1` | 128x1632 | 204 | 806,287 | 1.183 (1.023 / 0.120 / 0.040) | 3.69 / 0.81 | 21.6 / 40.2 |
| Teacher `combined_final_teacher_s1` | 68x864 | 108 | 6,172,335 | 2.966 (2.646 / 0.255 / 0.066) | 3.88 / 0.76 | 18.9 / 24.6 |
| Teacher `ft_iam_c128_syn_teacher_s2` | 128x1632 | 204 | 7,220,911 | 10.576 (9.746 / 0.695 / 0.134) | 6.76 / 2.13 | 88.4 / 90.0 |

T5 corrector (`postproc/t5_ext_ctx1`, 60,496,896 params, beam 4, context 1): 3.78 GMAC/line (mean 51.8 source / 12.9 output tokens, 50 IAM test lines), 118 ms/line on GPU 0 at batch 1.
LM memory: IAM combined 10-gram (19,134,615 entries) adds 3.41 GiB resident memory in the Python CharLM.

Machine load: before GPU block `21:13:07 up 47 days,  3:37, 10 users,  load average: 10.26, 10.78, 9.45`, nvidia-smi ['0, 100 %, 13443 MiB, 97887 MiB', '1, 48 %, 28944 MiB, 97887 MiB']; before CPU block `21:13:56 up 47 days,  3:38, 10 users,  load average: 8.87, 10.32, 9.36`; after CPU block `21:24:11 up 47 days,  3:48, 10 users,  load average: 13.42, 12.56, 11.09`, nvidia-smi ['0, 49 %, 15747 MiB, 97887 MiB', '1, 100 %, 9015 MiB, 97887 MiB']. CPU: Intel(R) Xeon(R) 6507P, torch 2.9.1+cu128.

**These are NOT idle-machine numbers.** During the GPU block, GPU 0 was at 100 % utilisation from another process plus this study's T5 runs. During the CPU block, the load average was 9–13 on 32 cores. The CPU batch-16 per-line times above batch-1 (Student 128×1632: 40 vs 22 ms) persist under these conditions, so the reviewer's contention reading cannot be ruled out. An idle-machine re-run with `python scripts/control_compute.py` is still needed before quoting latency in the paper. The MACs are exact and load-independent. Moving 68×864 → 128×1632 (3.56× pixels) multiplies Student MACs by 3.2× and Teacher MACs by 3.6×. One T5 call (3.8 GMAC, beam 4) costs about 3× a Student forward pass at 128×1632 and about 30× its GPU latency.

## C5. IAM split facts (review M1, M9): `results/control_iam_splits.json`

`python scripts/control_iam_splits.py`. Lists are in `raw/iam/largeWriterIndependentTextLineRecognitionTask/` (FKI) and `raw/iam/aachen/`.

| list | lines | forms | writers |
|---|---|---|---|
| FKI trainset | 6,161 | 747 | 283 |
| FKI validationset1 | 900 | 105 | 46 |
| FKI validationset2 | 940 | 115 | 43 |
| FKI testset | 1,861 | 232 | 128 |
| Aachen trainset | 6,161 | 747 | 283 |
| Aachen validationset | 966 | 115 | 56 |
| Aachen testset | 2,915 | 336 | 161 |

- Our splits are exactly train = FKI trainset (6,161), valid = validationset1 + validationset2 (900 + 940 = 1,840), test = FKI testset (1,861). The archive transcriptions equal the lists in order for `iam.hdf5` and `iam_128x1632.hdf5`, and the first 900 valid lines are validationset1. 6,161 + 1,840 + 1,861 = 9,862, all present in `lines.txt` (13,353 lines total).
- Aachen: the train set is the **same 6,161 lines** (same set, different order). Aachen valid 966 = 141 (FKI v1) + 221 (FKI v2) + 548 (FKI test) + 56 from neither list. Aachen test 2,915 = 1,313 (FKI test) + 759 (v1) + 709 (v2) + 134 from neither.
- None of our IAM test lines are in Aachen train, and none of Aachen test is in our train. There is no form or writer overlap either way. **But** 1,468 Aachen test lines (50.4 %) are in our validation set, which chose checkpoints, LM weights and T5 gates. Our existing models are therefore train-clean but selection-contaminated on the Aachen test set. An Aachen evaluation needs re-selection on Aachen valid; retraining is not needed for the train split, since it is identical. `iam_aachen.hdf5` exists and matches the Aachen lists (6,161 / 966 / 2,915).
- The 6,482 / 976 / 2,915 variant: no such list exists in `raw/`. Only the 6,161 / 966 / 2,915 Laia lists are present, so it could not be checked.


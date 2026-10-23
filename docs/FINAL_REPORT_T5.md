# HTR-JAND — enhanced configuration, decoding and T5 post-processing

Date: 2026-10-16. Repository `<repo root>`, branch `cleanup-for-submission`.
Every number below is read from `results/*.json` by `scripts/final_table.py` (full candidate lists in
`docs/FINAL_RESULTS_T5.md`); every run is in `docs/EXPERIMENT_REGISTRY.md` (`scripts/registry.py`).
CER / WER / SER in %, official test splits, test decoded once per checkpoint or path.
Two configurations are kept apart throughout: the **base configuration** (68 x 864, `configs/htrjand_base.yaml`,
results in `docs/RESULTS_RECONSTRUCTION.md`) and the **enhanced HTR-JAND configuration** (this report).

## 1. Final Student architecture (enhanced)

```
line image 128 x 1632, Eq. 1 normalisation
 -> 3 x [FullGatedConv2d 3x3 -> BN -> ReLU -> MaxPool -> SE(r=8)]  channels 16, 32, 64
    pools (2,2), (2,2), (4,2)            -> (B, 64, 8, 204)
 -> columns (B, 204, 512)  -- auxiliary CTC head (training only, delta = 0.1)
 -> 4 x [BiLSTM 64/direction -> LayerNorm]          -> (B, 204, 128)
 -> Combined Attention: LN(W_f [MHSA_1 head(X) ; Proxima(X)] + X)
 -> dropout -> Linear -> CTC (greedy or beam decoding)
```
Trained on the five corpora + synthetic lines (Eq. 33 mixing), with KD from the 128 x 1632 Teacher
(tau 2, alpha 0.7 -> 0.4, gamma 0.2 -> 0.5) where validation preferred it, then fine-tuned per corpus.

## 2. Final Teacher architecture (enhanced)

Same pipeline with 5 blocks (channels 32-512, pools (2,2)x3 + (2,1)x2 -> (B, 512, 4, 204)),
columns 2048-d, 4 x BiLSTM 128/direction, Combined Attention with 2 heads. The Teacher **system** is a
sequence-level ensemble (Table VI "+EL") of up to 12 such Teachers (two per-corpus fine-tuned
128 x 1632 Teachers, two 128 x 1632, four 68 x 864, single-corpus Teachers), see Section 12.

## 3. Exact parameter counts

| model | total | deployed (no aux head) |
|---|---:|---:|
| Student 128 x 1632 (final) | 863,230 | **806,287** |
| Teacher 128 x 1632 (final members) | 7,448,350 | **7,220,911** |
| Teacher 68 x 864 (base-configuration geometry) | 6,286,110 | 6,172,335 |
| T5-small corrector | 60,496,896 | 60,496,896 |

The 128 x 1632 Teacher is +17 % over the 68 x 864 Teacher, all in
the first BiLSTM's input weights; the Student keeps its count exactly (third pool (4,2)).

## 4. Base vs enhanced configuration

| # | component | base configuration | enhanced configuration | why / evidence |
|---|---|---|---|---|
| 1 | input | 68 x 864 | 128 x 1632, same recipe (aspect-preserving scale, centre pad, invert) | IAM Teacher 4.73 -> 4.39, every corpus better |
| 2 | Student pooling | (2,2) x 3 | (2,2),(2,2),(4,2) | keeps 806,287 parameters at 128 px |
| 3 | regularisation | dropout 0.01 | Teacher 0.2 / CNN 0.1 / RNN 0.2, wd 5e-4; Student 0.05 / 0 / 0, wd 1e-4 | Student underfit (train CTC 0.361 vs val 0.379) |
| 4 | optimiser | early stop 10 / val loss | AdamW, cosine lr 1e-3, warm-up, EMA 0.999, batch 16, 120-150 epochs, best val CER | D-005 |
| 5 | augmentation | affine + photometric jitter ("base") | elastic, projective, blots, stroke width ("strong") | screens |
| 6 | synthetic data (Alg. 2) | none | 200k / 120k lines, 106 Google handwriting fonts, en/fr/la Wikipedia text (0 test lines) | IAM 5.12 -> 4.73 |
| 7 | curriculum | Eq. 33 + Alg. 3 stages | Eq. 33 synthetic ratio 0.1 -> 0.4 exactly; Alg. 3 implemented as `--curriculum acp` (declared reading: five stages by line-length quantile 0.25/0.5/0.75/1/1, base then strong augmentation, synthetic from stage 3; advance when 1 - val CER > T, T0 0.70, Delta_T 0.05) but NOT adopted | ACP Student worse on validation on all five corpora (combined 5.11 vs 4.96; IAM 5.59 vs 5.51); stage 5 reached at epoch 11 (train_log.json) |
| 8 | KD | tau 4 | tau 2, alpha/gamma schedule 0.7 -> 0.4 / 0.2 -> 0.5; L_ce omitted (beta = 0 under these weights) | D-004 |
| 9 | multi-task | dynamic lambda_k | joint five-corpus training, then per-corpus fine-tuning chosen on validation | all 5 Students and 10 Teachers improved on validation |
| 10 | ensemble (+EL) | none | union of member n-best lists rescored by every member's CTC likelihood | prob. averaging hurt (spikes misalign) |
| 11 | decoding | greedy | greedy; extension: char n-gram LM (training text + Wikipedia) beam search | extension, reported separately |
| 12 | T5 (Sec. V) | — | Algorithm 5, Section 5 | — |

## 4b. Saint Gall label convention (found 2026-10-17)

The raw Saint Gall ground truth spells words letter by letter and uses two special letters, "pt" for a
punctuation dot and "et" for "&" (`raw/saintgall/README.txt`). The "pt" letter is annotated on the
official TRAINING pages only: 716 tokens on the 20 training pages, 0 on the 10 validation and 30 test
pages, although the dots are visible on every page. The archive builder joins the letters, so 717
training words end in "pt" and every model trained on them inserts "pt" that no validation or test
reference contains (317 insertions on the 235 validation lines of the ensemble). This explains the
whole Saint Gall gap of the earlier tables.

Found from validation errors and the README; the zero count on the test pages was also checked during
diagnosis, but the decision follows from the README and the validation labels alone. Fix: archives
`saintgall_nopt.hdf5` / `saintgall_nopt_128x1632.hdf5` with the "pt" letter removed (439 training labels
change; validation and test labels are byte-identical to before), the Saint Gall Student and six Teachers
fine-tuned on them (`scripts/saintgall_nopt.sh`, 30 epochs each), and the LM refitted on the corrected
training text. The old Saint Gall models stay in the candidate pool; validation selects the new ones.

## 5. T5 implementation (`htrjand/t5post.py`, Algorithm 5)

| Alg. 5 / Sec. V element | implementation |
|---|---|
| T5-small, 60 M | `t5-small`, 60,496,896 parameters |
| Train SentencePiece on D (line 2) | T5-small's pretrained SentencePiece model **extended** with the 9 corpus characters it cannot encode (`{ } ¤ § ² À ë œ ⊥`); retraining SentencePiece would discard the pretrained embeddings |
| Unicode / whitespace normalisation | NFC, typographic quotes and dashes to the corpora's ASCII forms, whitespace collapse; no spelling modernisation. Leaves all 4,792 validation and all training transcriptions unchanged |
| Segment (line 4) | one recognised line |
| BuildContext (line 5) | previous and next recognised lines; window 0 or 1 is a selectable corrector (`t5_ctx0`, `t5_ctx1`, `t5_ext_ctx1`) |
| DetectErrors(s, D) (line 7) | words absent from the corpus's TRAINING lexicon, the recogniser's eps_hat, both, or every line; chosen on validation |
| TokenizeSP + T5 (lines 9-10) | input `correct <language> <corpus>: <line> \| before: <prev> \| after: <next>`, beam 4 |
| Confidence (line 11) | exp(length-normalised beam log-probability of the candidate) |
| theta (line 12) | grid 0 ... 1.01 (1.01 = never correct), chosen on validation |
| ApplyCorrection (line 13) | whole line, or only aligned word segments containing a detected word; chosen on validation |
| Format (line 16) | per-corpus punctuation spacing learned from TRAINING text, enforced only where >= 99.5 % consistent (alters 1 of 4,792 clean validation lines) |

## 6. T5 training data

1. Two out-of-fold Teachers (128 x 1632 recipe, 60 epochs), each trained on the training lines with
   index % 2 != k of all five corpora plus synthetic lines; each transcribes the other half
   (held-out CER 3.72 / 3.71 %). 26,340 real (recognition, ground truth) pairs.
2. Per-corpus error profile from those pairs only (substitution / deletion / insertion tables, spaces,
   case, punctuation, hyphens, per-line error-count distribution; `postproc/t5data/pairs_meta.json`).
3. Systematic-error pairs: every training line corrupted twice with its corpus's profile.
4. Corrector `t5_ext_ctx1` adds 226k Wikipedia lines (en / fr / la by corpus), corrupted with the same
   profiles and gated for test-line overlap (none).
5. Each pair appears with and without context. T5 early stopping: loss on out-of-fold VALIDATION
   recognitions (fold-0 model). Pair counts: 158,040 (ctx1), 79,020 (ctx0), 610,464 (ext).

## 7. No test text or labels in training or tuning

`scripts/leakage_audit.py` -> `results/leakage_audit.json`, verdict **PASS**:
- no T5 training target equals a test line unless the same line is in the official TRAINING split
  (RIMES 88, Bentham 39, Washington 3 stock lines repeated by the benchmark; none unexplained);
- T5 early-stopping pairs are validation labels only;
- Wikipedia / synthetic text: 0 verbatim test lines (>= 25 characters) for every corpus;
- every checkpoint's charset is a subset of training characters;
- every T5 gate setting equals the validation optimum of its recorded sweep;
- fold models trained and decoded disjoint halves.
Model, decoder, corrector and gate are all selected on validation (`scripts/final_table.py`); the
4.47 non-distilled IAM Student is not used because its validation CER is higher.

## 8. Confidence threshold selection

For each base path, corpus and corrector, the grid detect x apply x eps-threshold x theta
(11 values) is evaluated on the validation predictions of that base path; the minimum validation CER
(WER breaks ties) is frozen and applied once to test. Selected thetas are 0.95-0.98 or 1.01 (no
correction); ungated T5 raises CER on every path (e.g. IAM Student raw CTC 6.22 -> 8.52-10.75).

## 9-13. Results (test CER %, validation CER in brackets; selection by validation)

| corpus | Student raw CTC | Student T5 (on raw CTC) | Student char LM | Student LM -> T5 | Teacher ens. (no LM) | Teacher ens. -> T5 | Teacher ens. + LM | Teacher ens. + LM -> T5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IAM | 6.22 (5.16) | 6.04 (4.97) | 4.49 (3.42) | **4.44** (3.35) | 3.57 (2.82) | 3.48 (2.74) | 2.88 (2.05) | **2.90** (2.05) |
| RIMES | 4.80 (3.58) | 4.25 (3.16) | 3.13 (2.21) | **3.08** (2.21) | 2.59 (1.70) | 2.56 (1.68) | **2.16** (1.34) | 2.16 (1.34) |
| Bentham | 5.64 (3.74) | 5.52 (3.60) | 4.81 (2.90) | **4.79** (2.86) | 4.22 (2.08) | 4.15 (2.05) | **3.85** (1.77) | 3.85 (1.77) |
| Saint Gall | 2.49 (2.55) | 2.49 (2.54) | **1.77** (1.88) | 1.77 (1.88) | 1.36 (1.37) | 1.42 (1.36) | **1.11** (1.20) | 1.11 (1.20) |
| Washington | 6.29 (5.03) | 6.20 (4.87) | 4.58 (3.54) | **4.55** (3.47) | 2.21 (1.80) | 2.22 (1.73) | 2.22 (1.64) | **2.22** (1.61) |

Bold = validation-selected best per system. Single-Teacher raw CTC (validation-selected per-corpus
fine-tuned Teacher): IAM 4.31, RIMES 2.99, Bentham 4.50, Saint Gall 1.74 (marker-free `nopt_sg_combined_final_teacher_s1`; 7.10 under the original labels), Washington 3.04.

**T5's effect** (mean relative test CER change over its base, five corpora): raw CTC Student -3.6 %
(RIMES -11.6 %), ensemble without LM -0.0 %, Student + LM -0.8 %, ensemble + LM +0.1 %. On validation the
gate rejects nearly every candidate once the recogniser (or the LM) has removed the errors a T5 word-level
rewrite can fix, and ungated T5 always hurts.

## 14b. Seed replication of the Student (seed 2)

The whole Student protocol was repeated with seed 2 (queues S3/S4, `scripts/seed2_student_chain.py`,
`results/seed2_student.json`): distilled and non-distilled 128 x 1632 Students, per-corpus parent chosen on
validation, per-corpus fine-tuning, Wikipedia char LM, T5 (`t5_ext_ctx1`), path chosen on validation;
Saint Gall fine-tuned on the marker-free labels (`nopt_sg_student_s2`). Test CER %:

| corpus | seed 1 | seed 2 (validation-selected path) | mean |
|---|---:|---:|---:|
| iam | 4.44 | 4.53 (T5 on char LM) | 4.49 |
| rimes | 3.08 | 3.11 (char LM) | 3.10 |
| bentham | 4.79 | 4.55 (T5 on char LM) | 4.67 |
| saintgall | 1.77 | 1.57 (char LM (marker-free labels)) | 1.67 |
| washington | 4.55 | 4.12 (T5 on char LM) | 4.33 |

Seed-1 T5 cells are the validation-best of three correctors; seed 2 ran `t5_ext_ctx1` only. The Saint Gall seed-2 entry is the marker-free model (`scripts/seed2_saintgall_nopt.py`; T5 arms run, gate rejected).

## 15. Best validation-selected configuration per corpus

`results/final_selection.json` lists the id of every selected path. Student: per-corpus fine-tuned
`ft_<corpus>_c128_student_kd_s1` (IAM, RIMES, Bentham) / `ft_washington_c128_student_s1` (Washington) /
`nopt_sg_student_s1` (Saint Gall, marker-free labels), Wikipedia char LM, corrector `t5_ext_ctx1`. Teacher: `ens_<corpus>_ensft_wiki`
(Washington: `ens_washington_ens7_wiki`; Saint Gall: `ens_saintgall_nopt_ens_wiki`), T5 gate as selected (IAM `t5_ctx1`, Washington `t5_ctx0`;
RIMES / Bentham / Saint Gall gates select theta = 1.01, i.e. no correction).

## 16. Reproduction

```bash
export HTRJAND_DATA_DIR=/path/to/archives
# archives at 128 x 1632, combined archive, synthetic lines (fonts + text in synth/)
scripts/build_c128.sh
# recognisers (queue files list every exact command)
python scripts/run_queue.py configs/queues/runs_queue_P5.txt logs/P5     # 128 x 1632 Teacher s1 (P6: s2)
python scripts/run_queue.py configs/queues/runs_queue_S1.txt logs/S1     # Student, no KD (S2: with KD)
python scripts/run_queue.py configs/queues/runs_queue_FT.txt logs/FT     # per-corpus Student fine-tuning
python scripts/run_queue.py configs/queues/runs_queue_FTT1.txt logs/FTT1 # per-corpus Teacher fine-tuning (FTT2)
python scripts/run_queue.py configs/queues/runs_queue_F0.txt logs/F0     # out-of-fold Teachers (F1)
# decoding paths and T5
scripts/lm_paths.sh; scripts/base_paths.sh; scripts/t5_pipeline.sh
python -m htrjand.t5post pairs --combined $HTRJAND_DATA_DIR/combined_128x1632.hdf5 \
   --fold-ckpts runs/c128_fold0of2/best.pth runs/c128_fold1of2/best.pth --synthetic-per-line 2 \
   --extra-text iam=synth/text_en.txt:60000 rimes=synth/text_fr.txt:60000 bentham=synth/text_en.txt:60000 \
   saintgall=synth/text_la.txt:40000 washington=synth/text_en.txt:20000 --out-dir postproc/t5data_ext
python -m htrjand.t5post train --pairs-dir postproc/t5data_ext --out postproc/t5_ext_ctx1 --context 1 --epochs 6
scripts/t5_apply_ext.sh; scripts/ft_student_paths.sh; scripts/ft_teacher_paths.sh; scripts/paper_path.sh
# tables, registry, audit
python scripts/final_table.py; python scripts/registry.py; python scripts/leakage_audit.py
```

## 17. Commits and checkpoints

Commits: e56c959 (pre-T5 state), 3193dbe (T5 module, registry), 335f0ee (T5 results, per-corpus
Student fine-tuning), and the commit adding this report. Checkpoints (not committed): `runs/<id>/best.pth`
(ids in `docs/EXPERIMENT_REGISTRY.md`), correctors `postproc/t5_ctx1`, `postproc/t5_ctx0`,
`postproc/t5_ext_ctx1`.

## 18. Summary

The enhanced HTR-JAND keeps every defining component (gated convolutions, SE, BiLSTM, MHSA + Proxima
Combined Attention, CTC, Teacher-Student KD, Eq. 33 curriculum, synthetic data, auxiliary loss, T5
post-processing); every result above is leakage-free and validation-selected. The T5 stage yields 0-12 %
relative CER reduction over its base. Open items: Teacher-side seed repeats; the Alg. 3 stage definitions
(the declared reading did not help and is kept only as an option).

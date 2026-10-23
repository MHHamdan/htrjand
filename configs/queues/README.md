# Queue files: the exact training commands

Each file is a list of shell commands run in order by `python scripts/run_queue.py <queue> <log_dir>` (lines starting
with `#` are comments). Every training run of `docs/EXPERIMENT_REGISTRY.md` is one line of a queue, **except** the
runs listed under "Runs made by scripts" below. Data paths are `$HTRJAND_DATA_DIR/...`; `--gpu 0` is the first device
of `CUDA_VISIBLE_DEVICES`.

Lines marked `# fixed for the current code` were recorded before the preset rename of decision D-001 (and before the
KD temperature default changed from 4 to 2, D-004); the command below such a comment reproduces the recorded run with
today's code (the run itself, its log and its results are unchanged). `runs_queue_FT.txt` keeps the original-label
Saint Gall fine-tune, marked `SUPERSEDED`: the released Saint Gall Student comes from `scripts/saintgall_nopt.sh`.

| queue | stage | what | training runs (`--tag`) | evaluate commands |
|---|---|---|---|---:|
| `runs_queue_A.txt` | base configuration, initial recipe | IAM Teacher / Student (KD and no KD), 3 seeds, 68 x 864 | `iam_paper_teacher_s1`, `iam_paper_student_nokd_s1`, `iam_paper_student_kd_s1`, `iam_paper_teacher_s2`, `iam_paper_teacher_s3`, `iam_paper_student_kd_s2`, `iam_paper_student_nokd_s2`, `iam_paper_student_kd_s3`, `iam_paper_student_nokd_s3` | 9 |
| `runs_queue_A2.txt` | base configuration, initial recipe | remainder of A after D-001, explicit 64 / 32 per-direction presets | `iam_paper_teacher_s3`, `iam_paper_student_kd_s2`, `iam_paper_student_nokd_s2`, `iam_paper_student_kd_s3`, `iam_paper_student_nokd_s3` | 5 |
| `runs_queue_B.txt` | base configuration, initial recipe | IAM component ablations; single-corpus Teachers of the other four corpora | `iam_abl_plain_conv_s1`, `iam_abl_no_attention_s1`, `iam_abl_mha_only_s1`, `iam_abl_proxima_only_s1`, `iam_abl_no_se_s1`, `iam_abl_cnn_lstm_s1`, `saintgall_paper_teacher_s1`, `washington_paper_teacher_s1`, `rimes_paper_teacher_s1`, `bentham_paper_teacher_s1` | 10 |
| `runs_queue_C.txt` | development (IAM) | one-lever tuning screens and declared alternatives (D-002, D-003, D-001) | `iam_tune_dropout_s1`, `iam_tune_escer_s1`, `iam_alt_d256_s1`, `iam_alt_fusion_hist_s1`, `iam_alt_attn_before_s1`, `iam_alt_unitnorm_s1`, `iam_tune_kd_tau1_s1`, `iam_tune_kd_tau2_s1`, `iam_historical_teacher_s1` | 9 |
| `runs_queue_D.txt` | development (IAM), tuned recipe | tuned Teacher / Students, 3 seeds; tuned single-corpus Teachers | `iam_tuned_teacher_s1`, `iam_tuned_student_kd_s1`, `iam_tuned_student_nokd_s1`, `iam_tuned_teacher_s2`, `iam_tuned_student_kd_s2`, `iam_tuned_student_nokd_s2`, `iam_tuned_teacher_s3`, `iam_tuned_student_kd_s3`, `iam_tuned_student_nokd_s3`, `rimes_tuned_teacher_s1`, `bentham_tuned_teacher_s1`, `saintgall_tuned_teacher_s1`, `washington_tuned_teacher_s1` | 13 |
| `runs_queue_E.txt` | development (IAM), tuned recipe | component ablations at the tuned operating point | `iam_tuned_abl_plain_conv_s1`, `iam_tuned_abl_no_se_s1`, `iam_tuned_abl_mha_only_s1`, `iam_tuned_abl_proxima_only_s1`, `iam_tuned_abl_no_attention_s1`, `iam_tuned_abl_cnn_lstm_s1` | 6 |
| `runs_queue_F.txt` | development (IAM) | tuned KD Students with tau 2, 3 seeds (D-004) | `iam_tuned_student_kd_tau2_s1`, `iam_tuned_student_kd_tau2_s2`, `iam_tuned_student_kd_tau2_s3` | 3 |
| `runs_queue_F0.txt` | T5 data 128 x 1632 | out-of-fold Teacher, fold 0 of 2 (T5 training pairs) | `c128_fold0of2` | 0 |
| `runs_queue_F1.txt` | T5 data 128 x 1632 | out-of-fold Teacher, fold 1 of 2 (T5 training pairs) | `c128_fold1of2` | 0 |
| `runs_queue_F2.txt` | development (IAM) | seed 2 of F | `iam_tuned_student_kd_tau2_s2` | 1 |
| `runs_queue_F3.txt` | development (IAM) | seed 3 of F | `iam_tuned_student_kd_tau2_s3` | 1 |
| `runs_queue_FT.txt` | **released Students** 128 x 1632 | per-corpus fine-tuning, seed 1 (IAM, RIMES, Bentham, Washington variants) | `ft_iam_c128_student_kd_s1`, `ft_rimes_c128_student_kd_s1`, `ft_bentham_c128_student_kd_s1`, `ft_saintgall_c128_student_s1`, `ft_washington_c128_student_s1` | 10 |
| `runs_queue_FTT1.txt` | Teacher side 128 x 1632 | per-corpus Teacher fine-tuning, seed 1 (ensemble members) | `ft_iam_c128_syn_teacher_s1`, `ft_rimes_c128_syn_teacher_s1`, `ft_bentham_c128_syn_teacher_s1`, `ft_saintgall_c128_syn_teacher_s1`, `ft_washington_c128_syn_teacher_s1` | 10 |
| `runs_queue_FTT2.txt` | Teacher side 128 x 1632 | per-corpus Teacher fine-tuning, seed 2 (ensemble members) | `ft_iam_c128_syn_teacher_s2`, `ft_rimes_c128_syn_teacher_s2`, `ft_bentham_c128_syn_teacher_s2`, `ft_saintgall_c128_syn_teacher_s2`, `ft_washington_c128_syn_teacher_s2` | 10 |
| `runs_queue_G.txt` | development | tuned single-corpus Teachers (RIMES, Bentham, Saint Gall, Washington) | `rimes_tuned_teacher_s1`, `bentham_tuned_teacher_s1`, `saintgall_tuned_teacher_s1`, `washington_tuned_teacher_s1` | 4 |
| `runs_queue_Ha.txt` | development (IAM) | screen H: one lever at a time (cosine, augmentation) | `iam_scr_cosine_s1`, `iam_scr_aug_egb_s1` | 2 |
| `runs_queue_Hb.txt` | development (IAM) | screen H (strong augmentation, recurrent dropout + wd) | `iam_scr_aug_strong_s1`, `iam_scr_rnndrop_wd_s1` | 2 |
| `runs_queue_Hc.txt` | development (IAM) | screen H (EMA, batch 16) | `iam_scr_ema_s1`, `iam_scr_bs16_s1` | 2 |
| `runs_queue_Hd.txt` | development (IAM) | screen H (column max-pool, SE r 8) | `iam_scr_maxpool_s1`, `iam_scr_se8_s1` | 2 |
| `runs_queue_Ia.txt` | development (IAM) | combination arms | `iam_cmb_cos_strong_s1` | 1 |
| `runs_queue_Ib.txt` | development (IAM) | combination arms | `iam_cmb_cos_strong_rnnwd_s1` | 1 |
| `runs_queue_Ic.txt` | development (IAM) | combination arms | `iam_cmb_all_s1` | 1 |
| `runs_queue_K.txt` | T5 data, 68 x 864 (earlier) | IAM out-of-fold models for the earlier post-processing module | `iam_fold0of2`, `iam_fold1of2` | 0 |
| `runs_queue_L1.txt` | final IAM recipe (D-005) | IAM Teacher / Student seed 2 | `iam_final_teacher_s2`, `iam_final_student_kd_s2` | 2 |
| `runs_queue_L2.txt` | final IAM recipe (D-005) | IAM Teacher / Student seed 3 | `iam_final_teacher_s3`, `iam_final_student_kd_s3` | 2 |
| `runs_queue_L3.txt` | final IAM recipe (D-005) | IAM Students without KD | `iam_final_student_nokd_s1`, `iam_final_student_nokd_s2` | 2 |
| `runs_queue_L4.txt` | final IAM recipe (D-005) | IAM KD / no-KD Students | `iam_final_student_kd_s1`, `iam_final_student_nokd_s3` | 2 |
| `runs_queue_M.txt` | evaluation | validation decodes of the final IAM models | — | 6 |
| `runs_queue_M1.txt` | evaluation | validation decodes (Teachers) | — | 3 |
| `runs_queue_M2.txt` | evaluation | validation decodes (Students) | — | 3 |
| `runs_queue_N1.txt` | **base configuration** (results base row) | five-corpus Teacher + KD Student, 68 x 864, seed 1 | `combined_final_teacher_s1`, `combined_final_student_kd_s1` | 12 |
| `runs_queue_N2.txt` | base configuration, initial recipe | five-corpus Teacher + KD Student with the package defaults | `combined_paper_teacher_s1`, `combined_paper_student_kd_s1` | 12 |
| `runs_queue_N3.txt` | **base configuration** (results base row) | five-corpus Teacher, seed 2 | `combined_final_teacher_s2` | 5 |
| `runs_queue_N4.txt` | **base configuration** (results base row) | five-corpus KD Student, seed 2 | `combined_final_student_kd_s2` | 6 |
| `runs_queue_P1.txt` | Teacher side, 68 x 864 + synthetic | five-corpus Teacher with synthetic mixing, seed 1 (ensemble member) | `combined_syn_teacher_s1` | 6 |
| `runs_queue_P2.txt` | Teacher side, 68 x 864 + synthetic | synthetic pretraining then real fine-tuning (ensemble member) | `combined_synpre_s1`, `combined_synft_teacher_s1` | 6 |
| `runs_queue_P3.txt` | Teacher side, 68 x 864 + synthetic | seed 2 of P1 (ensemble member) | `combined_syn_teacher_s2` | 6 |
| `runs_queue_P4.txt` | development | distilled Student from the P1 Teacher, 68 x 864 + synthetic | `combined_syn_student_kd_s1` | 6 |
| `runs_queue_P5.txt` | **released Teacher family** 128 x 1632 | five-corpus Teacher, seed 1 (`c128_syn_teacher_s1`) | `c128_syn_teacher_s1` | 6 |
| `runs_queue_P6.txt` | **released Teacher** 128 x 1632 | seed 2 (`c128_syn_teacher_s2` = htrjand-teacher) | `c128_syn_teacher_s2` | 6 |
| `runs_queue_S1.txt` | **released Students** 128 x 1632 | five-corpus Student, no KD, seed 1 | `c128_student_s1` | 6 |
| `runs_queue_S2.txt` | **released Students** 128 x 1632 | five-corpus KD Student, seed 1 (`c128_student_kd_s1` = htrjand-student general) | `c128_student_kd_s1` | 6 |
| `runs_queue_S3.txt` | **released Students** 128 x 1632 | seed-2 Student, no KD | `c128_student_s2` | 10 |
| `runs_queue_S4.txt` | **released Students** 128 x 1632 | seed-2 KD Student (`c128_student_kd_s2` = htrjand-student-seed2 general) | `c128_student_kd_s2` | 10 |
| `runs_queue_S5.txt` | development | adaptive curriculum (ACP) Student; not adopted | `c128_student_kd_acp_s1` | 10 |
| `runs_queue_SG2.txt` | **released Students** 128 x 1632 | seed-2 marker-free Saint Gall Student (htrjand-student-seed2 saintgall); recorded from its train_log.json | `nopt_sg_student_s2` | 2 |

## Runs made by scripts (not queues)

| runs | script | what |
|---|---|---|
| `nopt_sg_student_s1`, `nopt_sg_c128_teacher_s1/s2`, `nopt_sg_combined_{final,syn}_teacher_s1/s2` | `scripts/saintgall_nopt.sh` | Saint Gall fine-tunes on the marker-free labels (`saintgall_nopt*.hdf5`, `scripts/build_saintgall_nopt.py`); `nopt_sg_student_s1` is the released seed-1 `saintgall` variant |
| `ft_iam_c128_student_s2`, `ft_rimes_c128_student_kd_s2`, `ft_bentham_c128_student_kd_s2`, `ft_saintgall_c128_student_kd_s2`, `ft_washington_c128_student_s2` | `scripts/seed2_student_chain.py` | seed-2 per-corpus fine-tunes (parent chosen on validation); IAM / RIMES / Bentham / Washington are released seed-2 variants |
| (`nopt_sg_student_s2`) | launched by hand; its exact command is now `runs_queue_SG2.txt` (and `provenance.training_command` of its `config.json`) | `scripts/seed2_saintgall_nopt.py` pools it into `results/seed2_student.json` (no training) |

Decoding (character LM, ensembles) and T5 stages: `scripts/lm_paths.sh`, `scripts/base_paths.sh`, `scripts/ft_student_paths.sh`,
`scripts/ft_teacher_paths.sh`, `scripts/paper_path.sh`, `scripts/t5_pipeline.sh`, `scripts/t5_apply_ext.sh`; order in
`docs/FINAL_REPORT_T5.md` §16.

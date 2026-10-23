# Saint Gall "pt" annotation-convention audit (2026-10-17)

> **Release-export note.** Written in the original working repository; kept verbatim except for mechanical edits: absolute machine paths replaced by repo-relative paths, `$HTRJAND_DATA_DIR/` or descriptive placeholders (`<historical repo>/...` is the earlier working repository, provenance only); `htrjand_paper_v1/` renamed `htrjand/`; the earlier repository's own `htrjand/` directory (a different model, not in this release) written `htrjand-jan/`; `docs/figures_refined/` renamed `docs/figures/`.


Reproduce (CPU, about 30 s, deterministic: two runs give the same JSON md5):

```bash
python scripts/saintgall_audit.py      # -> results/saintgall_audit.json
```

Inputs: the raw IAM-HistDB corpus at `$HTRJAND_RAW_DIR/saintgall`
(`--raw-dir`), the archives in `data/` (a symlink to `handwritten-text-recognition/data`), and `results/*.json|tsv`.
The script does not train or decode. All numbers below come from `results/saintgall_audit.json`. CER is
recomputed from the prediction TSVs with the package metric (edit distance summed over lines / reference
characters), and it matches every stored `*.json` CER to two decimals.

**Verdict.**
1. The raw "pt" special letter appears only on the training pages.
2. The marker-free archives change only training labels, and each removed piece is exactly one raw "pt" token.
3. Validation and test images and labels are byte-identical, row for row.
4. Washington is untouched.
5. About 65-83 % of the old models' character errors are emitted "pt". After those are deleted, the old models
   are within -0.01 to +0.26 pp of the retrained ones. The Saint Gall gain is therefore almost entirely an
   annotation-convention effect and only marginally a model effect.

## a. The raw annotation convention

`raw/saintgall/README.txt` (full path `$HTRJAND_RAW_DIR/saintgall/README.txt`):

> l.41 `* ground_truth/transcription.txt`
> l.43 `[lineID] [spelling_1]|...|[spelling_n] [label_1]|...|[label_n]`
> l.45 `Text line transcription with accurate spellings that correspond with the manuscript images. Spellings are given by:`
> l.47 `[character_1]-...-[character_m]`
> l.49 `Besides lower case and upper case letters, we have also employed two special letters, i.e., "pt" for punctuation marks (typically dots between words) and "et" for the special character "&" that is frequently used throughout the data set. The word spellings are followed by the corresponding labels from the text edition. The labels are lower case only and without punctuation marks.`

Raw examples (`ground_truth/transcription.txt`, lines 1, 3, 4). In the raw files, `pt` is one *character* of
the word, separated by `-`:

| line id | raw spelling (excerpt) | edition label | old archive label (excerpt) | marker-free label |
|---|---|---|---|---|
| csg562-003-01 | `h-a-b-e-r-e-t-u-r-pt` | `haberetur` | `habereturpt et veluti` | `haberetur et veluti` |
| csg562-003-03 | `a-m-o-r-e-m-pt` | `amorem` | `ret amorempt sicuti` | `ret amorem sicuti` |
| csg562-003-04 | `i-n-d-i-c-a-t-pt` | `indicat` | `indicatpt inter ceteros` | `indicat inter ceteros` |

The Migne edition (`data/edition.txt`) has `haberetur,` at the first of these: a real punctuation mark.

## b. Archive-builder behaviour

- `htrjand/prepare_data.py:_saintgall` (lines 119-133). At line 125,
  `gt[s[0]] = s[1].replace("-", "").replace("|", " ")` deletes the character separator and turns word
  separators into spaces. The multi-letter tokens `pt` and `et` are written out as the two letters
  `p`,`t` / `e`,`t`, so `h-a-b-e-r-e-t-u-r-pt` becomes `habereturpt`. `et` gets the same treatment (`c-et-e-r-o-s` becomes
  `ceteros`), but it is annotated on every split, so the convention is the same for training and evaluation.
  Washington's reader (`_washington`, lines 136-150) is the contrast: it maps `s_pt` to `.` (line 148).
- `scripts/build_saintgall_nopt.py`:
  - line 24 rebuilds the old label exactly as above.
  - line 25 rebuilds the new label after dropping every character token equal to `pt`:
    `"".join(c for c in w.split("-") if c != "pt")`. Empty words are dropped, although none occur because `pt` is never a word on its own.
  - line 33 maps each archive row old -> new through that table, and stops if a label is ambiguous.
  - line 34 copies `dt` (the images) unchanged.

  The filter acts on whole tokens, so the letter sequence `p-t` inside a word (`s-c-r-i-p-t-u-m`) can never match.
  The script names only `saintgall*.hdf5`. `htrjand/data.py` registers `saintgall_nopt` as its only `*_nopt` dataset.

## c. Counts

Raw ground truth (special-letter tokens):

| split | pages | lines | `pt` tokens | pages with `pt` | lines with `pt` | `pt` position | `et` tokens (pages) |
|---|---:|---:|---:|---:|---:|---|---:|
| train | 20 | 468 | **716** | 20 / 20 (14-45 per page) | 439 | word-final: 716 | 254 (20/20) |
| valid | 10 | 235 | **0** | 0 / 10 | 0 | — | 115 (10/10) |
| test | 30 | 707 | **0** | 0 / 30 | 0 | — | 377 (30/30) |

The per-page counts are in `c_raw_counts.<split>.pt_per_page`. `et` is annotated on every page of every split. `pt` appears only on the training pages.

Archive strings (identical for the 68x864 and 128x1632 canvases):

| archive | split | word-final `pt` | lines with it | `pt` substring | `et` substring |
|---|---|---:|---:|---:|---:|
| saintgall[_128x1632] | train | 717 | 439 | 747 | 274 |
| | valid | 1 | 1 | 19 | 118 |
| | test | 0 | 0 | 57 | 384 |
| saintgall_nopt[_128x1632] | train | 1 | 1 | 31 | 274 |
| | valid | 1 | 1 | 19 | 118 |
| | test | 0 | 0 | 57 | 384 |

- 717 = 716 special letters + 1 genuine `ppt` (raw `p-p-t`, the abbreviation of *propter*, line csg562-013-05).
  `ppt` is kept in the marker-free labels. The single word-final `pt` on validation is also `ppt` (page csg562-059).
- Every archive's label multiset equals the raw-derived labels for every split (`labels_equal_raw_builder`).
- Older archives show the same pattern: `saintgallss.hdf5` has 718 word-final `pt` in train and 0 in test/valid, and `saintgall_h128.hdf5` has 717 / 1 / 0.

**Washington (control).** The raw Washington ground truth codes punctuation as `s_pt`, annotated on every split
(159 / 64 / 107 tokens). `_washington` maps it to `.`. The archives keep only lexical `pt`
(`Captain`, `kept`, `receipt`, `except`, ...; word-final 1 / 0 / 5). No `washington*nopt*` archive exists, and
the nopt builder does not mention Washington. The fix was not applied there, and it did not need to be.

## d. Why `pt` is punctuation, not lexical content

- The README defines it as a punctuation mark (l.49). All 716 occurrences are word-final.
  The README states (l.49) that the edition labels are lower case and without punctuation. None of the 716
  pt-bearing words has an edition label ending in `pt` (716 aligned, 0 ending in `pt`).
- Latin has very few words ending in *-pt*. The Latin LM text (`synth/text_la.txt`) has 33 word-final `pt` in
  474,310 words (0.7 per 10k), mostly modern words such as *concept* and *Egypt*. Before the fix, 18 % of Saint Gall training words (717 / 3,933) ended in `pt`.
- Evaluation references contain the letters `p,t` inside 19 valid and 57 test words (*scriptum*, *propter*,
  *accepta*, *nuptiae*, ...). The only valid/test word ending in `pt` is `ppt` (one occurrence). The same
  inner letters appear in 31 training words (*scripturas*, *contemptu*, *septem*, ...). All 31 are kept: 31 `pt` substrings
  remain in the marker-free training labels.
- Diff of the training labels, old -> new (both canvases give the same result):

| rows | labels changed | words shortened | characters removed | each removal = a raw word-final `pt` token | violations |
|---:|---:|---:|---:|:-:|---:|
| 468 | **439** | 716 | 1,432 (= 2 x 716) | yes | 0 |

  For each word, the check is that the new word equals the old word minus a final `pt` only if the raw word's last token is `pt`. It also checks that the letters equal the raw tokens with `pt` removed.

## e. Byte identity (SHA-256 per row, index-wise)

| pair | split | images identical | labels identical | split digest equal |
|---|---|---:|---:|:-:|
| saintgall vs saintgall_nopt | train | 468/468 | 29/468 (439 changed) | — |
| | valid | 235/235 | 235/235 | yes |
| | test | 707/707 | 707/707 | yes |
| saintgall_128x1632 vs saintgall_nopt_128x1632 | train | 468/468 | 29/468 | — |
| | valid | 235/235 | 235/235 | yes |
| | test | 707/707 | 707/707 | yes |

Some differences do not affect content:
- The nopt archives store `gt` with the HDF5 string tag `ascii` instead of `utf-8`. Every label is ASCII, so the bytes are the same.
- The nopt archives are uncompressed (83 MB vs 12 MB).
- The 68x864 and 128x1632 archives list rows in different orders (glob vs sorted order), although the multisets are equal.

Every prediction TSV used below has references that equal one of the archives row for row.

## f. Results: old (pt-trained) vs marker-free, and the pt-insertion share

CER %. "old" is the validation-best candidate among the pt-trained models. "pt deleted" is the old model's
CER after deleting one word-final `pt` per emitted word (except the training word `ppt`) and any stand-alone `pt`.
That post-hoc deletion is diagnostic only and is not a reported system.

| system | old valid | old test | old valid / test, pt deleted | share of old char errors that are `pt` (valid / test) | new valid | new test | chosen on valid |
|---|---:|---:|---:|---:|---:|---:|:-:|
| Teacher raw CTC (best single) | 7.11 | 7.10 | 1.68 / 1.73 | 76 % / 76 % | 1.56 | 1.74 | new |
| Teacher ensemble (no LM) | 6.95 | 6.88 | 1.45 / 1.47 | 79 % / 79 % | 1.37 | 1.36 | new |
| Teacher ensemble + LM | 6.66 | 6.46 | 1.24 / 1.12 | 81 % / 83 % | **1.20** | **1.11** | new |
| Student raw CTC, seed 1 | 7.88 | 7.70 | 2.75 / 2.74 | 65 % / 64 % | 2.55 | 2.49 | new |
| Student + LM, seed 1 | 7.03 | 6.68 | 2.06 / 2.03 | 71 % / 70 % | **1.88** | **1.77** | new |
| Student raw CTC, seed 2 | 7.57 | 7.59 | 2.25 / 2.36 | 70 % / 69 % | 2.14 | 2.30 | new |
| Student + LM, seed 2 | 6.95 | 6.69 | 1.81 / 1.67 | 74 % / 75 % | **1.66** | **1.57** | new |

Model ids:
- old: `ft_saintgall_c128_syn_teacher_s1`, `ens_saintgall_ensft_nolm`, `ens_saintgall_ensft_wiki`,
  `ft_saintgall_c128_student_s1`, `ens_saintgall_ft_saintgall_c128_student_s1_wiki`, `ft_saintgall_c128_student_kd_s2`,
  `ens_saintgall_ft_saintgall_c128_student_kd_s2_wiki`
- new: `nopt_sg_combined_final_teacher_s1`, `ens_saintgall_nopt_ens_nolm`,
  `ens_saintgall_nopt_ens_wiki`, `nopt_sg_student_s1`, `ens_saintgall_nopt_student_wiki`, `nopt_sg_student_s2`,
  `ens_saintgall_nopt_student_s2_wiki`

The old models emit 327-361 word-final `pt` on the 235 validation lines and 941-1,092 on the 707 test lines.
The new models emit none.

**Label effect vs model effect.** After the emitted `pt` is deleted, the old models remain 0.04-0.20 pp
(valid) and -0.01 to +0.26 pp (test) above the retrained ones. On test, the old raw CTC Teacher is 0.01 pp lower. The training-label convention explains
95-100 % of each drop: 99.8 % of the Teacher ens. + LM drop (6.46 -> 1.11) and 95 % of the Student + LM seed-1 drop (6.68 -> 1.77). The remainder comes from fine-tuning without the conflicting target, plus the
smaller new ensemble pool: 6 nopt Teachers instead of 9 pt-trained ones.

**Selection.** Old and new models are candidates in the same pool (`scripts/final_table.py:candidates` loads both
`saintgall` and `saintgall_nopt` results; nopt ids are in `STUDENTS` / `TEACHERS`, lines 21-25), and the pick is by validation CER. `docs/FINAL_RESULTS_T5.md` (§saintgall table)
and `results/final_selection.json` select the nopt model for all nine Saint Gall system x stage cells. For example, Teacher
ens. + LM is `ens_saintgall_nopt_ens_wiki` (v 1.20 / t 1.11), and Student + LM is `ens_saintgall_nopt_student_wiki` (v 1.88 / t 1.77).
Validation alone suffices for the choice. Old validation CER is at least 6.66 and new is at most 2.55, and the convention follows from the README and the validation labels.
The LM was refitted: `ensemble_rescore.py:151` takes LM text from the `--dataset` archive's train split, so the
corrected labels, plus `synth/text_la.txt`.

**Seed 2.** `results/seed2_student.json` still records the pt-trained Saint Gall Student
(`ft_saintgall_c128_student_kd_s2`, selected test 6.69). The 1.57 in `docs/FINAL_REPORT_T5.md` §14b comes from
`ens_saintgall_nopt_student_s2_wiki.json` (v 1.66). Its validation-selected path is char LM over raw CTC (v 2.14). No T5 arm
was run for nopt seed 2. `seed2_student.json` should be regenerated, or the provenance noted.

**Minor discrepancies in `docs/FINAL_REPORT_T5.md` §4b.**
- "717 training words end in pt" is right as an archive count, but one of them is the genuine `ppt`.
- "no validation or test reference contains [pt]" holds for the special letter. One validation word (`ppt`) ends in the letters p,t.
- "317 insertions on the 235 validation lines of the ensemble" was not reproduced. This audit counts 356 emitted
  word-final `pt` for `ens_saintgall_ensft_wiki` (328 on words that are otherwise correct) and 361 without the LM.

Cross-check: `handwritten-text-recognition/htrjand-jan/results/small_corpus_diagnosis.json` agrees. It reports 716 tokens on 20/20
train pages and 0 on valid/test, `pt_word_final_train` 717, Washington 1 / 5, and for H_sg arms CER 7.17 -> 1.98
with `pt` stripped.

Recommended presentation: state the change as **training-label normalisation (annotation-convention alignment)**: the raw "pt" punctuation letter, which the corpus annotates on training pages only, is removed from the training labels; evaluation labels and images are unchanged. Report both conventions: old-convention models (6.46 / 6.68) beside marker-free ones (1.11 / 1.77, seed 2 1.57), optionally with the diagnostic "old model, pt deleted" (1.12) that shows the gain is the label effect.

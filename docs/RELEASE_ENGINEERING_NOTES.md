# Release engineering notes (code + models, v1.0)

Public export of HTR-JAND, the line-level handwriting recognizer, prepared from the internal working repository
(public release tag `v1.0`; private research commits `ebbfbcd` for code and results, `a30a640` for audits and
controls, `0fc8721` for figures; both recorded in `htrjand/release.py` and in every `config.json`). Nothing is
uploaded; the Hugging Face repositories are staged in `../htrjand-release-hf/`. Repository names are
defined once in `htrjand/release.py` (namespace `MHamdan`, prefix `htrjand`).

## 1. Contents

- `htrjand/`: the package (models, data, training, evaluation, decoding, T5 post-processing) plus the release API
  (`hub.py` `HTRJANDModel`, `inference.py` preprocessing and line check, `segment_lines.py` optional naive splitter,
  `lmdecode.py`, `t5correct.py`, `release.py`).
- `predict.py` (also `htrjand-predict`), `app.py` (Gradio), `notebooks/HTR_JAND_Demo.ipynb`, `examples/` (12 public-domain /
  CC0 out-of-distribution line crops, `examples/SOURCES.md`).
- `configs/htrjand_base.yaml` (the five-corpus 68 x 864 base configuration with measured parameter counts, plus the
  `train.py` package defaults), `configs/queues/` (exact commands of every run; `configs/queues/README.md` indexes
  the queues and the runs made by scripts; commands that no longer ran with the current code are fixed in place and
  annotated), `configs/synth_fonts.json` and `configs/synth_text_manifest.json` (inputs of the synthetic lines).
- `scripts/`: data builders, run queue, registry, leakage audit, final tables, controls, figures, release tools
  (`export_hf.py`, `verify_export.py`, `build_release_lm.py`, `build_corpus_lm.py`, `eval_release_decoders.py`,
  `measure_latency.py`, `write_model_cards.py`, `write_release_lm_eval.py`, `release_hygiene.py`) and
  `fetch_synth_resources.py` (synthetic-line text from the `htrjand-charlm` repo, the 106 fonts from the pinned
  google/fonts commit, SHA-256 checked; called by `build_c128.sh`).
- `results/`: metrics JSON only (no per-line predictions, no T5 caches, no benchmark transcriptions; two audit JSONs
  carry redacted example strings), `results/release_eval/` (validation-only evaluation of the released decoders).
- `docs/`: `FINAL_REPORT_T5.md`, `FINAL_RESULTS_T5.md` (validation-selected results), `RESULTS_RECONSTRUCTION.md`
  (base-configuration runs), `EXPERIMENT_REGISTRY.md`, `DECISION_LOG.md`, `DATA_SPLITS.md`, `RELEASE_LM_EVAL.md`,
  `audit_2026_10/{CONTROLS,SAINTGALL_AUDIT,LICENSING}.md`, `figures/`, `assets/ood_examples.{png,json}`.

Machine-specific paths were removed (data: `$HTRJAND_DATA_DIR`, raw corpora: `$HTRJAND_RAW_DIR`; GPUs via
`CUDA_VISIBLE_DEVICES` and `--gpu` / `--device`); inference needs only torch, numpy, opencv, safetensors,
huggingface_hub and Pillow (lazy imports for h5py, editdistance, transformers).

## 2. Models

Checkpoints held only `model_state_dict` plus metadata (no optimiser, scheduler, AMP-scaler or EMA state). Each variant
is exported as `model.safetensors` (state dict, auxiliary head kept for a strict load) + `config.json` (architecture,
preprocessing, tokenizer, decoder settings, provenance, metrics, SHA-256); `from_pretrained` builds the network from the
config alone. Selections: seed-2 general Student `c128_student_kd_s2` (combined validation CER 4.951 % vs 5.079 %), Teacher `c128_syn_teacher_s2` (3.131 % vs 3.141 %). Parameter counts: Student 806,287 deployed (863,230
total), Teacher 7,220,911 (7,448,350); presets `htrjand_student` / `htrjand_teacher` in `htrjand/models.py`.
Metrics in `config.json` and the cards: for each path the T5 numbers are those of the validation-selected corrector
of `results/final_selection.json` (seed 2: `results/seed2_student.json`); where that corrector (`t5_ctx0`, `t5_ctx1`)
is not released, it is flagged and the numbers of the released `t5_ext_ctx1` are given separately.

Verification (`scripts/verify_export.py`): 25 (model, corpus) pairs x 64 validation lines; inputs, logits and greedy strings bit-identical to the training checkpoints loaded with the original code (PASS); raw-image preprocessing equals the archive builder on 63/63 IAM lines.

## 3. Hugging Face staging

| path | bytes | sha256 |
|---|---:|---|
| `htrjand-charlm/LICENSE` | 1,085 | `bdd1763eee109e67d1b59944532d8a4c4fea672081453a6514205e7cd02ba155` |
| `htrjand-charlm/README.md` | 5,212 | `a8081d9cea1b67c3237b7714bdbf49db7d724028ac49e510a2bc2fe5fd23e738` |
| `htrjand-charlm/charlm_en.gz` | 39,887,967 | `e34475aabd588c1804e8f2e335cc1c2bec32e37519458145ad35e33f0d82b2a0` |
| `htrjand-charlm/charlm_fr.gz` | 25,922,735 | `a388836b765a2ac4347eb253045bc36aae15b86b5f63de1b0c80b06b45e013dd` |
| `htrjand-charlm/charlm_la.gz` | 21,552,918 | `6ac5a062461fb86e689663a4bdb5e7a035fc92d68a3ba133dc9556f04c01e64a` |
| `htrjand-charlm/fonts.json` | 36,579 | `e290a147c7f621a93a4160c0c2f14d0bcdb29639ff44aff05809057addeee1a0` |
| `htrjand-charlm/lexicon/wiki_en.txt.gz` | 175,849 | `835af1615486d8db38e209f094092a839a56cd1d3a8a9823be3ed83a245de71d` |
| `htrjand-charlm/lexicon/wiki_fr.txt.gz` | 140,041 | `8672fd8ded18ddd24fb604e8bae4090e5c619f7a91febdec4c7e08ce4cbd6801` |
| `htrjand-charlm/lexicon/wiki_la.txt.gz` | 108,660 | `26637819b87a1d3cfae82c440cb56c6d8519df89f0206e6e4bf37b39bd049f9d` |
| `htrjand-charlm/synth_text/manifest.json` | 1,278 | `0dc76bb2df6559cc99e6c217cc8304d2bd1cdb9aa716038d66018c4b0149821e` |
| `htrjand-charlm/synth_text/text_en.txt.gz` | 5,611,200 | `7b50d0c71b77aaf6cc11830f01065e186825ff36a4fe86cdb14de60e94c658b2` |
| `htrjand-charlm/synth_text/text_fr.txt.gz` | 3,049,484 | `a46091668b53aa0541d09299ce8bedcfb1ad7b58a28de00dc5ca3195e60e8f98` |
| `htrjand-charlm/synth_text/text_la.txt.gz` | 1,553,624 | `3de87f5e446e6766f72b51e044bb59fa5675a97e879a50bc5b53671de05cfb9b` |
| `htrjand-demo/LICENSE` | 1,601 | `dad40ea1b5b86c4e922983c0801344738bf5f67a17ad6bbda83db186874d61d2` |
| `htrjand-demo/README.md` | 1,440 | `1eea4fb13674de72b407dd16c2c987a84d51ae1f1323a2a7e3576a7b45f4682e` |
| `htrjand-demo/app.py` | 8,588 | `1c971f5a734499ac7682dc4e1b7d1d1d448f9d5be071bf974e8d008dae301775` |
| `htrjand-demo/requirements.txt` | 283 | `b1680fedad8058c7f326c512e00de756e73f4948420248bdb94d71e12db99db1` |
| `htrjand-demo/examples/SOURCES.md` | 9,361 | `5375d9beb2fd81a0c5736059ea6c663f89517d2e2d46016558fc50d4e0b73bf7` |
| `htrjand-demo/examples/ood_1_lincoln_a.png` | 146,309 | `2017a8a8d8aae9df5db29c6e37e69340f4c1032d44fb2b1696dc368ad8bddbb3` |
| `htrjand-demo/examples/ood_1_lincoln_b.png` | 151,770 | `fceda9814672e041efc3ad9e165837c9c40850e2c5bbfa242f5673302ffbe03b` |
| `htrjand-demo/examples/ood_2_gettysburg_a.png` | 121,242 | `90e98fd23e2c98d454735412e1bf5396f288e1afcd53fa29dd8f94fb94c261e9` |
| `htrjand-demo/examples/ood_2_gettysburg_b.png` | 125,383 | `f7c7539c510844a5e4af3c9281e5d92528b38af1d613052230fde7efe9a4f761` |
| `htrjand-demo/examples/ood_3_darwin_a.png` | 150,879 | `1fdd1ac907a3e5a1767fd308e6a1759d442cbf1d97b45f751ba4d5db3b91a5bb` |
| `htrjand-demo/examples/ood_3_darwin_b.png` | 128,798 | `b874e94b9d8270034f84fa63f4f86d0b106bba43dd7d7de314c71bb37eada817` |
| `htrjand-demo/examples/ood_4_balzac_a.png` | 228,830 | `7707e872dca3f749a4a163ee143db40f11332bcc3b84cc75a76903debfe17548` |
| `htrjand-demo/examples/ood_4_balzac_b.png` | 267,694 | `d3656ade6226a500d2b626eb0858dbe179219b0cdd425fec9da320707a89e7b1` |
| `htrjand-demo/examples/ood_5_marieantoinette_a.png` | 161,552 | `fb6c303cdacf2331475fe9c24390827bef255d3c60b2fc4e95192f4ac6fbafee` |
| `htrjand-demo/examples/ood_7_modern_de_a.png` | 64,805 | `7c32cf546660aa38cc0e188847fc220832c428d13678ca83cab355b6bbbf7607` |
| `htrjand-demo/examples/ood_8_modern_en_a.png` | 43,212 | `21d396ad1f42539c097fa5ae8078550ac79ed118084ecf17fc31ec4f130550af` |
| `htrjand-demo/examples/ood_8_modern_en_b.png` | 46,498 | `df1faf642ef49029230232d2f4c38f6a7529ba7b6a8921e5273195f4b883c609` |
| `htrjand-student/LICENSE` | 734 | `dfe1344603983fb8353263fab76aafca136d79ea9f53f48e48ebaa00fb331793` |
| `htrjand-student/README.md` | 21,154 | `7ee052c3ac8e435d1edc290121616fb74c227dc82e9d2b129b6b90643f2b9162` |
| `htrjand-student-seed2/LICENSE` | 734 | `dfe1344603983fb8353263fab76aafca136d79ea9f53f48e48ebaa00fb331793` |
| `htrjand-student-seed2/README.md` | 20,697 | `69befd0830b09ace047dd3dfc44464bf02bf1a77cc55889ba0f3a209f1adae40` |
| `htrjand-student-seed2/bentham/config.json` | 10,827 | `38f2b80073c6d105e302c90be4ff31701041e80c7b1f56a4ee5da6bc50127481` |
| `htrjand-student-seed2/bentham/model.safetensors` | 3,461,344 | `f89b9526f1f1565ae46ac97b10b0a1e92d48f3cbcd679d0fcfe0c37f140ae19c` |
| `htrjand-student-seed2/general/config.json` | 14,784 | `72de670f768e1e7601d4fa90bfd4b97bf21732ccca45620571f06e6f42556590` |
| `htrjand-student-seed2/general/model.safetensors` | 3,461,344 | `290ad8f831e5861c93874a8f1886036205dc050780ec486b313d96c3296f16ed` |
| `htrjand-student-seed2/iam/config.json` | 10,685 | `14d97f64a77fc07117375d3c00dbc56b983970b17b406a3a3e11fbb555ac972c` |
| `htrjand-student-seed2/iam/model.safetensors` | 3,461,344 | `e5d7337af2e745edae16234d85fe2dd34d81983a5b4d61323721e0f985cea119` |
| `htrjand-student-seed2/rimes/config.json` | 10,761 | `9950128d6164c87184e248eaf927e8c8dbdb7170fe7031566501a83f15cfc260` |
| `htrjand-student-seed2/rimes/model.safetensors` | 3,461,344 | `be780cb90beb7e66c15d68b4d5c6b2c1b3a959ddffdfd9fe6aafbd5c41fe8987` |
| `htrjand-student-seed2/saintgall/config.json` | 10,880 | `19ac26ee451c195b4bb57de4d81883f19634b576e4b04b91c2619da73f4cb51b` |
| `htrjand-student-seed2/saintgall/model.safetensors` | 3,461,344 | `2ecc72f90f9f9fa0cc083205cc1a994cf57a1ec004a157d92bfb6e20275e5465` |
| `htrjand-student-seed2/washington/config.json` | 10,861 | `62f2d28f3832ab5baf4e3f889b0cb08d8900ef72af8beea65e1d0331b19ce83a` |
| `htrjand-student-seed2/washington/model.safetensors` | 3,461,344 | `2d58f60ad1884d33af791041014228c3bbaa751050c4ddf7b518d62330fcb9eb` |
| `htrjand-student/bentham/config.json` | 10,779 | `4e3f62cbab24416316ff8060d0576e33923bf4984ad4027614c18ba2502db49b` |
| `htrjand-student/bentham/model.safetensors` | 3,461,344 | `191a37e09c13d9c6282f665ecf5e956c8a68045dde8e900707cc38fdb53c2a41` |
| `htrjand-student/general/config.json` | 20,541 | `ca2385108ff4cb0a3ae83d412788186d8990d7da54f61dc95e75fe5dec01ef1f` |
| `htrjand-student/general/model.safetensors` | 3,461,344 | `0150a3aa2545c5890a333e84c8269a1ad4ff60d0e2e357f192647a19ef5b9878` |
| `htrjand-student/iam/config.json` | 10,668 | `c288e86d90dcd61d13c4f59f2f706c9118c6c029aa40835f9aebed42d36f946e` |
| `htrjand-student/iam/model.safetensors` | 3,461,344 | `6cfc03c7a96d7a6615545a170333c6e066305de4d0485162f38667313126093c` |
| `htrjand-student/rimes/config.json` | 10,735 | `268eceb1015211ea58dee1155191b91b9560694162a4425af08fc7defe54d30f` |
| `htrjand-student/rimes/model.safetensors` | 3,461,344 | `6f289b5f970da4c9764570afe46bb12da65ba3f3df7457cb2195c0b16e5d2e8d` |
| `htrjand-student/saintgall/config.json` | 12,351 | `4e704cf80eb6582c0e9dfeb923e770e9cd3f575933e26c0bf00a5935b69f806a` |
| `htrjand-student/saintgall/model.safetensors` | 3,461,344 | `08b824b0e6d7a8c2dd082dc7b0bc78259e1d90e4147d68025bc6052967716106` |
| `htrjand-student/washington/config.json` | 11,538 | `ef2a45f967ec2145fbbcddfa94ac27300d6718f47875b0e074a2562dc9cddbba` |
| `htrjand-student/washington/model.safetensors` | 3,461,344 | `a97744c2bc9f1d4eaf5d3641abd3edeec7df1df0ef4e4f7fc34edba1597a450d` |
| `htrjand-t5-corrector/LICENSE-APACHE-2.0` | 10,173 | `a6cba85bc92e0cff7a450b1d873c0eaa2e9fc96bf472df0247a26bec77bf3ff9` |
| `htrjand-t5-corrector/NOTICE` | 1,004 | `847012be216de630c954675bb6611e877d14612debc4ddff8bc87f8eb379e949` |
| `htrjand-t5-corrector/README.md` | 8,133 | `db102549c348b4dd8d08b20a2d5c96baae75e8a19a456d14a7002256cd525bc9` |
| `htrjand-t5-corrector/config.json` | 1,552 | `db51bfac879bda65227ae6e9a80909fcaceeee8c23a1c42ca1f6e210acf01dc1` |
| `htrjand-t5-corrector/generation_config.json` | 142 | `c59c4c6e64f71546d7bf1d9e24bed776a1e722a03dc81e12be21ba878dec3777` |
| `htrjand-t5-corrector/htrjand_corrector.json` | 2,908 | `c32340f4675dd0adb9a712953ce30a8127ace327ac057c222546544983d018a2` |
| `htrjand-t5-corrector/model.safetensors` | 242,002,984 | `45e2437160f75b4fb838c3850ded8c76ac173cefd66fa5ce503f5c624a0c86df` |
| `htrjand-t5-corrector/tokenizer.json` | 2,425,923 | `c87593afeab511b501dc863aa821fb09a8ed73b21067f90f90a40aee5b73bc01` |
| `htrjand-t5-corrector/tokenizer_config.json` | 2,396 | `4cc3e49b271417224347dc37ce50f356a981fb01e036fe69f28c7b66037b3edc` |
| `htrjand-t5-corrector/train_meta.json` | 992 | `4f78d5147da86c574d837297ea3a20724cd2b91199e6fe7a479695130851ab86` |
| `htrjand-teacher/LICENSE` | 734 | `dfe1344603983fb8353263fab76aafca136d79ea9f53f48e48ebaa00fb331793` |
| `htrjand-teacher/README.md` | 13,889 | `befd45f80ae2fba5f1417f8493c79b091b9a546961fca4933cc341e1ac4de126` |
| `htrjand-teacher/general/config.json` | 14,611 | `d40b918bde5df8878ba850d09496e3328b02fa697d24511b626d36e070f1cefe` |
| `htrjand-teacher/general/model.safetensors` | 29,810,584 | `0fa6addfb56676ce644d9d0abd7906c02c47d3b71dfc642df617e0216f7c4cf1` |

Licences: recognisers CC BY-NC 4.0; T5 corrector CC BY-NC 4.0 with `NOTICE` (modified from google-t5/t5-small) and
`LICENSE-APACHE-2.0`; `htrjand-charlm` CC BY-SA 4.0 (Wikipedia-only LMs and lexicons); Space MIT (code). Model cards are
generated from JSON (`scripts/write_model_cards.py`); citation `hamdan2026htrjand`, manuscript under review. The
`htrjand-charlm` repo also carries the synthetic-line text (`synth_text/`) and the font list (`fonts.json`).

## 4. Released decoders (validation; details in `docs/RELEASE_LM_EVAL.md`)

| corpus (seed-1 Student) | greedy | released Wikipedia LM | corpus + Wikipedia LM of the experiments | released LM + T5 |
|---|---:|---:|---:|---:|
| IAM | 5.16 | 3.68 | 3.42 | 3.57 |
| RIMES | 3.58 | 2.57 | 2.21 | 2.51 |
| Bentham | 3.74 | 3.13 | 2.90 | 3.11 |
| Saint Gall | 2.55 | 2.04 | 1.88 | 2.04 |
| Washington | 5.03 | 3.58 | 3.54 | 3.50 |

The released LM is weaker than the experiments' LM because most of the gain comes from the corpus training text
(`results/control_lm_source.json`), which is not redistributed.

## 5. CPU latency (batch 1, one line, preprocessing included)

Intel(R) Xeon(R) 6507P, 4 threads, median of 15; shared machine.

| model | greedy | char LM | greedy + T5 | LM + T5 |
|---|---:|---:|---:|---:|
| htrjand-student/general | 32 ms | 59 ms | 129 ms | 188 ms |
| htrjand-student/iam | 47 ms | 73 ms | 165 ms | 184 ms |
| htrjand-student/rimes | 34 ms | 54 ms | 212 ms | 230 ms |
| htrjand-student/bentham | 34 ms | 48 ms | 141 ms | 152 ms |
| htrjand-student/saintgall | 25 ms | 43 ms | 191 ms | 186 ms |
| htrjand-student/washington | 32 ms | 48 ms | 140 ms | 164 ms |
| htrjand-student-seed2/general | 28 ms | 37 ms | 157 ms | 189 ms |
| htrjand-student-seed2/iam | 23 ms | 44 ms | 146 ms | 177 ms |
| htrjand-student-seed2/rimes | 23 ms | 41 ms | 175 ms | 205 ms |
| htrjand-student-seed2/bentham | 32 ms | 46 ms | 151 ms | 168 ms |
| htrjand-student-seed2/saintgall | 23 ms | 50 ms | 152 ms | 170 ms |
| htrjand-student-seed2/washington | 24 ms | 46 ms | 154 ms | 159 ms |
| htrjand-teacher/general | 97 ms | 115 ms | 211 ms | 252 ms |

## 6. Checks (last run 2026-10-23)

| check | result |
|---|---|
| clean venv `pip install -e ".[dev]"`; `python -m pytest -q` (no archives) | 87 passed, 14 skipped (13 archive tests, which pass with `HTRJAND_DATA_DIR` set; the app test needs `gradio`, not in `[dev]`, and passes where it is installed) (archive tests; they pass with `HTRJAND_DATA_DIR` set) |
| `scripts/verify_export.py --n 64` | PASS |
| every `htrjand/train.py` command in `configs/queues/` parsed with the current argument parser and presets | 0 mismatches |
| Space app smoke test (`HTRJAND_MODEL_ROOT`, greedy / LM / T5, Latin rerouting, 1 px image) | no exception; 1 px image gets a message |
| `scripts/registry.py --runs-dir <runs>` | 318 rows, written to `results/registry.jsonl`, `docs/EXPERIMENT_REGISTRY.md` |
| `scripts/leakage_audit.py` (archives, runs, T5 pairs, Wikipedia text) | PASS |
| `scripts/final_table.py --out-dir /tmp/x` | identical to the shipped `results/final_selection.json` and `docs/FINAL_RESULTS_T5.md` |
| clean venv, `pip install <repo>`; `jupyter nbconvert --execute notebooks/HTR_JAND_Demo.ipynb` (models from the staging dir) | 18 cells, 0 errors |
| `scripts/release_hygiene.py --hf-root ../htrjand-release-hf --data-dir $HTRJAND_DATA_DIR` | PASS: no file > 5 MB, no weights / archives / predictions / caches, no machine path, host name or token, no notebook outputs, images only in `examples/`, `docs/figures/`, `docs/assets/`, no benchmark line of >= 25 characters in any text file, no reference to the earlier pre-print |

## 7. Commands

```bash
pip install -e ".[dev]" && python -m pytest -q
python predict.py --model ../htrjand-release-hf/htrjand-student --variant general --image examples/ood_2_gettysburg_a.png --decoder lm --t5 --device cpu
HTRJAND_MODEL_ROOT=../htrjand-release-hf python app.py
HTRJAND_MODEL_ID=../htrjand-release-hf/htrjand-student HTRJAND_EXAMPLES_DIR=$PWD/examples \
    jupyter nbconvert --to notebook --execute notebooks/HTR_JAND_Demo.ipynb --output-dir /tmp --output nb_run
python scripts/export_hf.py --runs-dir <runs> --out-root ../htrjand-release-hf
HTRJAND_DATA_DIR=<archives> python scripts/verify_export.py --hf-root ../htrjand-release-hf --runs-dir <runs> --old-code <source checkout> --n 64
python scripts/write_model_cards.py --hf-root ../htrjand-release-hf
python scripts/release_hygiene.py --hf-root ../htrjand-release-hf
```

## 8. Open items

1. Nothing is uploaded; the Hub download path is tested offline only. Upload all five repos and the Space, then run
   `predict.py` without `--model` and the notebook without the environment override.
2. The dump / snapshot dates of the Wikipedia text are not recorded (cards say so).
3. Paper venue / DOI / URL for `hamdan2026htrjand` (cards and `CITATION.cff`).
4. Preset and run identifiers (`paper_teacher`, `paper_student`, `historical_teacher`, `iam_paper_*`, normalisation and
   fusion value `paper`) are stored in checkpoints, configs and queue commands and were kept unchanged; the released
   networks are also available as `htrjand_student` / `htrjand_teacher`.


# Release engineering notes (code + models, v1.0)

Public export of HTR-JAND, the line-level handwriting recognizer, prepared from the internal working repository
(source commit `ebbfbcd`, synchronised with later commits for audits, controls and figures). Nothing is committed,
pushed or uploaded; the Hugging Face repositories are staged in `../htrjand-release-hf/`. Repository names are
defined once in `htrjand/release.py` (namespace `MHamdan`, prefix `htrjand`).

## 1. Contents

- `htrjand/`: the package (models, data, training, evaluation, decoding, T5 post-processing) plus the release API
  (`hub.py` `HTRJANDModel`, `inference.py` preprocessing and line check, `segment_lines.py` optional naive splitter,
  `lmdecode.py`, `t5correct.py`, `release.py`).
- `predict.py` (also `htrjand-predict`), `app.py` (Gradio), `notebooks/HTR_JAND_Demo.ipynb`, `examples/` (12 public-domain /
  CC0 out-of-distribution line crops, `examples/SOURCES.md`).
- `configs/htrjand_base.yaml` (base configuration, 68 x 864) and `configs/queues/` (exact commands of every run).
- `scripts/`: data builders, run queue, registry, leakage audit, final tables, controls, figures, release tools
  (`export_hf.py`, `verify_export.py`, `build_release_lm.py`, `build_corpus_lm.py`, `eval_release_decoders.py`,
  `measure_latency.py`, `write_model_cards.py`, `write_release_lm_eval.py`, `release_hygiene.py`).
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
total), Teacher 7,220,911 (7,448,350).

Verification (`scripts/verify_export.py`): 25 (model, corpus) pairs x 64 validation lines; inputs, logits and greedy strings bit-identical to the training checkpoints loaded with the original code (PASS); raw-image preprocessing equals the archive builder on 63/63 IAM lines.

## 3. Hugging Face staging

| path | bytes | sha256 |
|---|---:|---|
| `htrjand-charlm/LICENSE` | 826 | `f0107962c66316b95b4c14b07a31453eab2a12cc73fdcc632b53d42cb6a66b56` |
| `htrjand-charlm/README.md` | 3,611 | `a4bad218f1751b167c4c1506f2519713301d62f04f1f40ed531538730d70e7bb` |
| `htrjand-charlm/charlm_en.gz` | 39,887,967 | `e34475aabd588c1804e8f2e335cc1c2bec32e37519458145ad35e33f0d82b2a0` |
| `htrjand-charlm/charlm_fr.gz` | 25,922,735 | `a388836b765a2ac4347eb253045bc36aae15b86b5f63de1b0c80b06b45e013dd` |
| `htrjand-charlm/charlm_la.gz` | 21,552,918 | `6ac5a062461fb86e689663a4bdb5e7a035fc92d68a3ba133dc9556f04c01e64a` |
| `htrjand-charlm/lexicon/wiki_en.txt.gz` | 175,849 | `835af1615486d8db38e209f094092a839a56cd1d3a8a9823be3ed83a245de71d` |
| `htrjand-charlm/lexicon/wiki_fr.txt.gz` | 140,041 | `8672fd8ded18ddd24fb604e8bae4090e5c619f7a91febdec4c7e08ce4cbd6801` |
| `htrjand-charlm/lexicon/wiki_la.txt.gz` | 108,660 | `26637819b87a1d3cfae82c440cb56c6d8519df89f0206e6e4bf37b39bd049f9d` |
| `htrjand-demo/LICENSE` | 1,851 | `921de3c1a9cb2059ea9dd8cb04e72fe84b86f52666c339a5cbad40ba06c9fb56` |
| `htrjand-demo/README.md` | 1,166 | `3ab112289219a628e42eecf2b812aec0706af6f7d3f120ea7c3f70e4c5fa91d8` |
| `htrjand-demo/app.py` | 7,347 | `ecc128e4fef54642a69275d473ef29857d4d52a3e0ccd0f1b812be203c7b916c` |
| `htrjand-demo/requirements.txt` | 191 | `15532aa3b6c57c6114abc6c385d33543c2044308528af9f4b3ccd82dec367328` |
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
| `htrjand-student/README.md` | 18,530 | `097586422fffe90e6d19d794ae04c3385ea5596c90edf7e5cdf5ee7395779213` |
| `htrjand-student-seed2/LICENSE` | 734 | `dfe1344603983fb8353263fab76aafca136d79ea9f53f48e48ebaa00fb331793` |
| `htrjand-student-seed2/README.md` | 18,452 | `1bbab2400e6a1c21e7ea8de5ef2db3b71d3b0b9ce1f940f7262210348944a953` |
| `htrjand-student-seed2/bentham/config.json` | 10,242 | `dc7a11aa18a4140ee4b5d47bb4488e4e7960421d530b298c7a5b7d95b3a4a80c` |
| `htrjand-student-seed2/bentham/model.safetensors` | 3,461,344 | `f89b9526f1f1565ae46ac97b10b0a1e92d48f3cbcd679d0fcfe0c37f140ae19c` |
| `htrjand-student-seed2/general/config.json` | 14,562 | `8fb9c40ca94250c80d2c49b4222bc93fd825303b2b5ed18db4d519f2ab236763` |
| `htrjand-student-seed2/general/model.safetensors` | 3,461,344 | `290ad8f831e5861c93874a8f1886036205dc050780ec486b313d96c3296f16ed` |
| `htrjand-student-seed2/iam/config.json` | 10,108 | `ec4c0dcb17bea28ff31db844640bd47238c35ce9d6963af718b45cd6ed50b2d8` |
| `htrjand-student-seed2/iam/model.safetensors` | 3,461,344 | `e5d7337af2e745edae16234d85fe2dd34d81983a5b4d61323721e0f985cea119` |
| `htrjand-student-seed2/rimes/config.json` | 10,180 | `138724c52520279bbe9c2146521dae8cb8bcab84bf99217fdf323e421f8ab338` |
| `htrjand-student-seed2/rimes/model.safetensors` | 3,461,344 | `be780cb90beb7e66c15d68b4d5c6b2c1b3a959ddffdfd9fe6aafbd5c41fe8987` |
| `htrjand-student-seed2/saintgall/config.json` | 10,281 | `d02ddf999fb521b4e39c37084ee5ddbb9585165b8b100b757c9d722f5bd705c4` |
| `htrjand-student-seed2/saintgall/model.safetensors` | 3,461,344 | `2ecc72f90f9f9fa0cc083205cc1a994cf57a1ec004a157d92bfb6e20275e5465` |
| `htrjand-student-seed2/washington/config.json` | 10,270 | `b1a3b63aea6ea25da995b26f4cc612e6122d712c220847e1a24ce75381667ac4` |
| `htrjand-student-seed2/washington/model.safetensors` | 3,461,344 | `2d58f60ad1884d33af791041014228c3bbaa751050c4ddf7b518d62330fcb9eb` |
| `htrjand-student/bentham/config.json` | 10,138 | `2cdeb2d4d894240c4c03a480fcb1a3f552cb8e05990302e67310f58f8fc6e47e` |
| `htrjand-student/bentham/model.safetensors` | 3,461,344 | `191a37e09c13d9c6282f665ecf5e956c8a68045dde8e900707cc38fdb53c2a41` |
| `htrjand-student/general/config.json` | 19,381 | `87f0289c998f4f1fd8f298e4f4d7caebddfd6af00229b799f111e493af412a3b` |
| `htrjand-student/general/model.safetensors` | 3,461,344 | `0150a3aa2545c5890a333e84c8269a1ad4ff60d0e2e357f192647a19ef5b9878` |
| `htrjand-student/iam/config.json` | 10,035 | `6d115d508c5a3d63fe95eecd4b9eff891177086fd59408cb38706ad203bb0285` |
| `htrjand-student/iam/model.safetensors` | 3,461,344 | `6cfc03c7a96d7a6615545a170333c6e066305de4d0485162f38667313126093c` |
| `htrjand-student/rimes/config.json` | 10,098 | `5df074e8976c1949367670b66fba99628ff83f730a45da21b3179f7a04f3bff5` |
| `htrjand-student/rimes/model.safetensors` | 3,461,344 | `6f289b5f970da4c9764570afe46bb12da65ba3f3df7457cb2195c0b16e5d2e8d` |
| `htrjand-student/saintgall/config.json` | 10,265 | `4ad7c62527f6c926099b7374721d9694e55fda751c068dc620ca2ef529e0cebe` |
| `htrjand-student/saintgall/model.safetensors` | 3,461,344 | `08b824b0e6d7a8c2dd082dc7b0bc78259e1d90e4147d68025bc6052967716106` |
| `htrjand-student/washington/config.json` | 10,184 | `eed5d7aaa7a4fd1290604d4bfcc23671c1cabf6e32215bdbab0906c12cccb455` |
| `htrjand-student/washington/model.safetensors` | 3,461,344 | `a97744c2bc9f1d4eaf5d3641abd3edeec7df1df0ef4e4f7fc34edba1597a450d` |
| `htrjand-t5-corrector/LICENSE-APACHE-2.0` | 10,173 | `a6cba85bc92e0cff7a450b1d873c0eaa2e9fc96bf472df0247a26bec77bf3ff9` |
| `htrjand-t5-corrector/NOTICE` | 1,004 | `847012be216de630c954675bb6611e877d14612debc4ddff8bc87f8eb379e949` |
| `htrjand-t5-corrector/README.md` | 7,641 | `17c1f3f7297a43daf5968db1ba3f393c36c267c7e5e92df60531c8db6b7f8b2a` |
| `htrjand-t5-corrector/config.json` | 1,552 | `db51bfac879bda65227ae6e9a80909fcaceeee8c23a1c42ca1f6e210acf01dc1` |
| `htrjand-t5-corrector/generation_config.json` | 142 | `c59c4c6e64f71546d7bf1d9e24bed776a1e722a03dc81e12be21ba878dec3777` |
| `htrjand-t5-corrector/htrjand_corrector.json` | 2,908 | `c32340f4675dd0adb9a712953ce30a8127ace327ac057c222546544983d018a2` |
| `htrjand-t5-corrector/model.safetensors` | 242,002,984 | `45e2437160f75b4fb838c3850ded8c76ac173cefd66fa5ce503f5c624a0c86df` |
| `htrjand-t5-corrector/tokenizer.json` | 2,425,923 | `c87593afeab511b501dc863aa821fb09a8ed73b21067f90f90a40aee5b73bc01` |
| `htrjand-t5-corrector/tokenizer_config.json` | 2,396 | `4cc3e49b271417224347dc37ce50f356a981fb01e036fe69f28c7b66037b3edc` |
| `htrjand-t5-corrector/train_meta.json` | 992 | `4f78d5147da86c574d837297ea3a20724cd2b91199e6fe7a479695130851ab86` |
| `htrjand-teacher/LICENSE` | 734 | `dfe1344603983fb8353263fab76aafca136d79ea9f53f48e48ebaa00fb331793` |
| `htrjand-teacher/README.md` | 12,995 | `0bc426925a23989ab1ffaad11288d2b87011535595db8f4124d19ea9ab421408` |
| `htrjand-teacher/general/config.json` | 14,391 | `e6fd69935161836f4ea49c28f86008264bedb60f28e29b9875c44d013a721e39` |
| `htrjand-teacher/general/model.safetensors` | 29,810,584 | `0fa6addfb56676ce644d9d0abd7906c02c47d3b71dfc642df617e0216f7c4cf1` |

Licences: recognisers CC BY-NC 4.0; T5 corrector CC BY-NC 4.0 with `NOTICE` (modified from google-t5/t5-small) and
`LICENSE-APACHE-2.0`; `htrjand-charlm` CC BY-SA 4.0 (Wikipedia-only LMs and lexicons); Space MIT (code). Model cards are
drafts generated from JSON; the only TODO is the venue / DOI / URL of the paper (`hamdan2026htrjand`, under review).

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
| clean venv `pip install -e ".[dev]"`; `python -m pytest -q` (no archives) | 79 passed, 13 skipped (archive tests; they pass with `HTRJAND_DATA_DIR` set) |
| `scripts/verify_export.py --n 64` | PASS |
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
   fusion value `paper`) are stored in checkpoints, configs and queue commands and were kept unchanged.


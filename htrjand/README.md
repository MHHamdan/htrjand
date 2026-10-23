# `htrjand` package

HTR-JAND, the line-level handwriting recognizer: gated CNN + squeeze-and-excitation, BiLSTM, Combined
Attention (multi-head self-attention + Proxima attention), CTC, with Teacher-Student knowledge distillation,
an auxiliary CTC head, synthetic lines and gated T5 post-processing. Inference API: `HTRJANDModel`
(`hub.py`); command line: `predict.py` at the repository root.

| module | contents |
|---|---|
| `models.py` | `HTRJAND`, presets (`paper_teacher`, `paper_student`, ablations), `from_checkpoint` |
| `data.py`, `prepare_data.py` | official-partition HDF5 archives, tokenizer (dedicated CTC blank), augmentation |
| `losses.py` | CTC, KD (τ²·KL(P_Teacher ‖ P_Student)), loss-weight schedule |
| `train.py`, `evaluate.py`, `metrics.py` | training (Teacher / Student / KD), greedy evaluation, CER / WER / SER |
| `synthetic.py`, `synth_archive.py` | synthetic-line archives (Alg. 2, Eq. 33 mixing) |
| `decode.py`, `lmdecode.py`, `enhance_decode.py`, `ensemble_rescore.py` | CTC prefix beam search, character LM, n-best rescoring, Teacher ensembles |
| `t5post.py`, `t5correct.py`, `postprocess.py` | T5 corrector (pairs, training, gated application) |
| `hub.py`, `inference.py`, `segment_lines.py`, `release.py` | released-model API, user-image preprocessing, optional line splitter, repo names |

The preset names `paper_teacher` / `paper_student` are the identifiers stored in every checkpoint and queue
command; `configs/htrjand_base.yaml` is the base configuration (68 x 864) with a provenance tag per value, and
`configs/queues/` holds the exact commands of the enhanced configuration (128 x 1632).

## Data

The corpora are not redistributed. Place each under `raw/<corpus>/` (layout in the docstring of
`prepare_data.py`) and build the official-partition archives:

```bash
export HTRJAND_DATA_DIR=/path/to/archives          # default: <repo>/data
for d in iam rimes bentham saintgall washington; do
  python -m htrjand.prepare_data --dataset $d --raw-dir raw                       # 68 x 864
  python -m htrjand.prepare_data --dataset $d --raw-dir raw --size 128 1632 \
      --out $HTRJAND_DATA_DIR/${d}_128x1632.hdf5                                  # 128 x 1632
done
```

Splits and the overlap check: `docs/DATA_SPLITS.md`. Saint Gall marker-free labels:
`scripts/build_saintgall_nopt.py`, `docs/audit_2026_10/SAINTGALL_AUDIT.md`.

## Training and evaluation

```bash
# Teacher (L = L_ctc + 0.1 L_aux), base configuration defaults
python -m htrjand.train --dataset iam --preset paper_teacher --seed 1 --gpu 0
# Student with KD from a trained Teacher
python -m htrjand.train --dataset iam --preset paper_student --seed 1 --gpu 0 \
    --kd --teacher-ckpt runs/iam_paper_teacher_s1/best.pth
# greedy evaluation (test decoded once per checkpoint; selection always on validation)
python -m htrjand.evaluate --ckpt runs/iam_paper_teacher_s1/best.pth --split test --gpu 0 --latency
```

`--gpu N` is a device index inside `CUDA_VISIBLE_DEVICES` (`-1` = CPU). The enhanced configuration's exact
commands are in `configs/queues/` (run with `python scripts/run_queue.py <queue> <log_dir>`); the reproduction
order is in `docs/FINAL_REPORT_T5.md` §16.

A checkpoint is a dict with `model_state_dict`, `model_cfg` (full constructor configuration), `charset`,
`dataset`, `seed`, `epoch`, `val_cer`, `normalisation`, `args`, `result_class`; load with
`models.from_checkpoint(path)`. Released models use the safetensors + `config.json` format of `hub.py`.

## License

MIT for the code (`LICENSE`). The corpora remain under their providers' licences.

#!/bin/bash
# Validation + test predictions of every base path the T5 stage is applied to.
set -e
cd "$(dirname "$0")/.."
: "${HTRJAND_DATA_DIR:?set HTRJAND_DATA_DIR to the directory holding the archives}"
export HTRJAND_DATA_DIR
# 1. raw CTC (greedy), both candidate Students, every corpus, validation split (test files exist)
for m in c128_student_s1 c128_student_kd_s1; do for ds in iam rimes bentham saintgall washington; do
  [ -f results/${m}_${ds}_valid_preds.tsv ] || python -m htrjand.evaluate --gpu 0 --split valid --ckpt runs/$m/best.pth --dataset $ds --tag ${m}_$ds > /dev/null
done; done
echo GREEDY DONE

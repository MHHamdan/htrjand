#!/bin/bash
# Chained T5 stage: out-of-fold pairs -> two T5-small correctors (context 1 / 0) -> Algorithm 5 on every
# validation-selected base path. Every selection inside uses validation only.
set -e
cd "$(dirname "$0")/.."
: "${HTRJAND_DATA_DIR:?set HTRJAND_DATA_DIR to the directory holding the archives}"
export HTRJAND_DATA_DIR
until [ -f runs/c128_fold0of2/best.pth ] && [ -f runs/c128_fold1of2/best.pth ] && grep -q '"finished": true' runs/c128_fold0of2/train_log.json && grep -q '"finished": true' runs/c128_fold1of2/train_log.json; do sleep 120; done
echo "[$(date +%T)] folds finished"
[ -f postproc/t5data/pairs_train.tsv ] || python -m htrjand.t5post pairs --combined $HTRJAND_DATA_DIR/combined_128x1632.hdf5 \
   --fold-ckpts runs/c128_fold0of2/best.pth runs/c128_fold1of2/best.pth --synthetic-per-line 2 --out-dir postproc/t5data
echo "[$(date +%T)] pairs done"
python -m htrjand.t5post train --pairs-dir postproc/t5data --out postproc/t5_ctx1 --context 1 > logs/t5_ctx1.log 2>&1 &
python -m htrjand.t5post train --pairs-dir postproc/t5data --out postproc/t5_ctx0 --context 0 > logs/t5_ctx0.log 2>&1 &
wait
echo "[$(date +%T)] T5 trained"
# base paths, selected on validation: student_ctc (greedy), student_lm (Wikipedia char LM), teacher_ens (ensemble + LM)
declare -A SG=( [iam]=c128_student_kd_s1 [rimes]=c128_student_kd_s1 [bentham]=c128_student_kd_s1 [saintgall]=c128_student_s1 [washington]=c128_student_s1 )
declare -A SL=( [iam]=c128_student_kd_s1 [rimes]=c128_student_s1 [bentham]=c128_student_kd_s1 [saintgall]=c128_student_s1 [washington]=c128_student_s1 )
declare -A TE=( [iam]=iam_ens10_wiki400 [rimes]=rimes_ens7_wiki [bentham]=bentham_ens7_wiki [saintgall]=saintgall_ens7_wiki [washington]=washington_ens7_wiki )
for ctx in 1 0; do
  for ds in iam rimes bentham saintgall washington; do
    m=${SG[$ds]}; python -m htrjand.t5post apply --dataset $ds --corrector postproc/t5_ctx$ctx --tag ${ds}_student_ctc_ctx$ctx \
      --val-preds results/${m}_${ds}_valid_preds.tsv --test-preds results/${m}_${ds}_test_preds.tsv --out results/t5_${ds}_student_ctc_ctx$ctx.json 2>&1 | grep '^\[' &
    m=${SL[$ds]}; python -m htrjand.t5post apply --dataset $ds --corrector postproc/t5_ctx$ctx --tag ${ds}_student_lm_ctx$ctx \
      --val-preds results/ens_${ds}_${m}_wiki_valid_preds.tsv --test-preds results/ens_${ds}_${m}_wiki_test_preds.tsv --out results/t5_${ds}_student_lm_ctx$ctx.json 2>&1 | grep '^\[' &
    e=${TE[$ds]}; python -m htrjand.t5post apply --dataset $ds --corrector postproc/t5_ctx$ctx --tag ${ds}_teacher_ens_ctx$ctx \
      --val-preds results/ens_${e}_valid_preds.tsv --test-preds results/ens_${e}_test_preds.tsv --out results/t5_${ds}_teacher_ens_ctx$ctx.json 2>&1 | grep '^\[' &
    wait
  done
done
echo "[$(date +%T)] T5 PIPELINE DONE"

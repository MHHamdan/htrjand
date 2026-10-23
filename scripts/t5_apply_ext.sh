#!/bin/bash
# Algorithm 5 with the external-text corrector (postproc/t5_ext_ctx1) on every validation-selected base path.
cd "$(dirname "$0")/.."
: "${HTRJAND_DATA_DIR:?set HTRJAND_DATA_DIR to the directory holding the archives}"
export HTRJAND_DATA_DIR
declare -A SG=( [iam]=c128_student_kd_s1 [rimes]=c128_student_kd_s1 [bentham]=c128_student_kd_s1 [saintgall]=c128_student_s1 [washington]=c128_student_s1 )
declare -A SL=( [iam]=c128_student_kd_s1 [rimes]=c128_student_s1 [bentham]=c128_student_kd_s1 [saintgall]=c128_student_s1 [washington]=c128_student_s1 )
declare -A TE=( [iam]=iam_ens10_wiki400 [rimes]=rimes_ens7_wiki [bentham]=bentham_ens7_wiki [saintgall]=saintgall_ens7_wiki [washington]=washington_ens7_wiki )
C=postproc/t5_ext_ctx1
for ds in iam rimes bentham saintgall washington; do
  m=${SG[$ds]}; python -m htrjand.t5post apply --dataset $ds --corrector $C --tag ${ds}_student_ctc_ext \
    --val-preds results/${m}_${ds}_valid_preds.tsv --test-preds results/${m}_${ds}_test_preds.tsv --out results/t5_${ds}_student_ctc_ext.json 2>&1 | grep '^\[' &
  m=${SL[$ds]}; python -m htrjand.t5post apply --dataset $ds --corrector $C --tag ${ds}_student_lm_ext \
    --val-preds results/ens_${ds}_${m}_wiki_valid_preds.tsv --test-preds results/ens_${ds}_${m}_wiki_test_preds.tsv --out results/t5_${ds}_student_lm_ext.json 2>&1 | grep '^\[' &
  e=${TE[$ds]}; python -m htrjand.t5post apply --dataset $ds --corrector $C --tag ${ds}_teacher_ens_ext \
    --val-preds results/ens_${e}_valid_preds.tsv --test-preds results/ens_${e}_test_preds.tsv --out results/t5_${ds}_teacher_ens_ext.json 2>&1 | grep '^\[' &
  wait
done
echo EXT APPLY DONE

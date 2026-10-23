#!/bin/bash
# Fine-tuned Students: Wikipedia-LM decoding, then T5 post-processing with each corrector on both paths.
cd "$(dirname "$0")/.."
: "${HTRJAND_DATA_DIR:?set HTRJAND_DATA_DIR to the directory holding the archives}"
export HTRJAND_DATA_DIR
declare -A M=( [iam]=ft_iam_c128_student_kd_s1 [rimes]=ft_rimes_c128_student_kd_s1 [bentham]=ft_bentham_c128_student_kd_s1 [saintgall]=ft_saintgall_c128_student_s1 [washington]=ft_washington_c128_student_s1 )
lm() { case $1 in iam) echo "--lm-text synth/text_en.txt --lm-max-lines 400000 --order 10";; rimes) echo "--lm-text synth/text_fr.txt --lm-max-lines 200000 --order 9";; saintgall) echo "--lm-text synth/text_la.txt --lm-max-lines 200000 --order 9";; *) echo "--lm-text synth/text_en.txt --lm-max-lines 200000 --order 9";; esac; }
for ds in iam rimes bentham saintgall washington; do
  m=${M[$ds]}
  python -m htrjand.ensemble_rescore --dataset $ds --tag ${ds}_${m}_wiki --members runs/$m/best.pth $(lm $ds) --betas 0.0,0.5,1.0,1.5,2.0,3.0,4.0,5.0 --workers 6 2>&1 | tail -n 1 &
done; wait
for C in t5_ctx1 t5_ctx0 t5_ext_ctx1; do for ds in iam rimes bentham saintgall washington; do
  m=${M[$ds]}
  python -m htrjand.t5post apply --dataset $ds --corrector postproc/$C --tag ${ds}_ftstudent_ctc_$C --val-preds results/${m}_${ds}_valid_preds.tsv --test-preds results/${m}_${ds}_test_preds.tsv --out results/t5_${ds}_ftstudent_ctc_$C.json 2>&1 | grep '^\[' &
  python -m htrjand.t5post apply --dataset $ds --corrector postproc/$C --tag ${ds}_ftstudent_lm_$C --val-preds results/ens_${ds}_${m}_wiki_valid_preds.tsv --test-preds results/ens_${ds}_${m}_wiki_test_preds.tsv --out results/t5_${ds}_ftstudent_lm_$C.json 2>&1 | grep '^\[' &
done; wait; done
echo FT STUDENT PATHS DONE

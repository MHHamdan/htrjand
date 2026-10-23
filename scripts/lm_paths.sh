#!/bin/bash
# Wikipedia-LM decoding (Student, both candidates) and Teacher ensembles, writing validation predictions too.
cd "$(dirname "$0")/.."
: "${HTRJAND_DATA_DIR:?set HTRJAND_DATA_DIR to the directory holding the archives}"
export HTRJAND_DATA_DIR
lm() { case $1 in iam) echo "--lm-text synth/text_en.txt --lm-max-lines 400000 --order 10";; rimes) echo "--lm-text synth/text_fr.txt --lm-max-lines 200000 --order 9";; saintgall) echo "--lm-text synth/text_la.txt --lm-max-lines 200000 --order 9";; *) echo "--lm-text synth/text_en.txt --lm-max-lines 200000 --order 9";; esac; }
for ds in iam rimes bentham saintgall washington; do
  for m in c128_student_s1 c128_student_kd_s1; do
    python -m htrjand.ensemble_rescore --dataset $ds --tag ${ds}_${m}_wiki --members runs/$m/best.pth $(lm $ds) --betas 0.0,0.5,1.0,1.5,2.0,3.0,4.0,5.0 --workers 6 2>&1 | tail -n 1 &
  done; wait
done
python -m htrjand.ensemble_rescore --dataset iam --tag iam_ens10_wiki400 --members runs/combined_final_teacher_s1/best.pth runs/combined_final_teacher_s2/best.pth runs/combined_syn_teacher_s1/best.pth runs/combined_syn_teacher_s2/best.pth runs/c128_syn_teacher_s1/best.pth runs/c128_syn_teacher_s2/best.pth runs/iam_final_teacher_s1/best.pth runs/iam_final_teacher_s2/best.pth runs/iam_final_teacher_s3/best.pth runs/combined_synft_teacher_s1/best.pth $(lm iam) --betas 0.0,0.5,1.0,1.5,2.0,3.0,4.0,5.0 --workers 10 2>&1 | tail -n 1 &
python -m htrjand.ensemble_rescore --dataset rimes --tag rimes_ens7_wiki --members runs/combined_final_teacher_s1/best.pth runs/combined_final_teacher_s2/best.pth runs/combined_syn_teacher_s1/best.pth runs/combined_syn_teacher_s2/best.pth runs/c128_syn_teacher_s1/best.pth runs/c128_syn_teacher_s2/best.pth runs/rimes_tuned_teacher_s1/best.pth $(lm rimes) --betas 0.0,0.5,1.0,1.5,2.0,3.0,4.0,5.0 --workers 6 2>&1 | tail -n 1 &
wait
python -m htrjand.ensemble_rescore --dataset bentham --tag bentham_ens7_wiki --members runs/combined_final_teacher_s1/best.pth runs/combined_final_teacher_s2/best.pth runs/combined_syn_teacher_s1/best.pth runs/combined_syn_teacher_s2/best.pth runs/c128_syn_teacher_s1/best.pth runs/c128_syn_teacher_s2/best.pth runs/bentham_tuned_teacher_s1/best.pth $(lm bentham) --betas 0.0,0.5,1.0,1.5,2.0,3.0,4.0,5.0 --workers 6 2>&1 | tail -n 1 &
python -m htrjand.ensemble_rescore --dataset saintgall --tag saintgall_ens7_wiki --members runs/combined_final_teacher_s1/best.pth runs/combined_final_teacher_s2/best.pth runs/combined_syn_teacher_s1/best.pth runs/combined_syn_teacher_s2/best.pth runs/c128_syn_teacher_s1/best.pth runs/c128_syn_teacher_s2/best.pth runs/saintgall_tuned_teacher_s1/best.pth $(lm saintgall) --betas 0.0,0.5,1.0,1.5,2.0,3.0,4.0,5.0 --workers 5 2>&1 | tail -n 1 &
python -m htrjand.ensemble_rescore --dataset washington --tag washington_ens7_wiki --members runs/combined_final_teacher_s1/best.pth runs/combined_final_teacher_s2/best.pth runs/combined_syn_teacher_s1/best.pth runs/combined_syn_teacher_s2/best.pth runs/c128_syn_teacher_s1/best.pth runs/c128_syn_teacher_s2/best.pth runs/washington_tuned_teacher_s1/best.pth $(lm washington) --betas 0.0,0.5,1.0,1.5,2.0,3.0,4.0,5.0 --workers 5 2>&1 | tail -n 1 &
wait
echo LM DONE

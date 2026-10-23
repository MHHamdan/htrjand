#!/bin/bash
# After per-corpus Teacher fine-tuning: ensembles that add the two fine-tuned Teachers, then T5 post-processing on them.
cd "$(dirname "$0")/.."
: "${HTRJAND_DATA_DIR:?set HTRJAND_DATA_DIR to the directory holding the archives}"
export HTRJAND_DATA_DIR
until grep -q 'QUEUE DONE' logs/queue_FTT1.out 2>/dev/null && grep -q 'QUEUE DONE' logs/queue_FTT2.out 2>/dev/null; do sleep 120; done
echo "[$(date +%T)] teacher fine-tuning done"
C="runs/combined_final_teacher_s1/best.pth runs/combined_final_teacher_s2/best.pth runs/combined_syn_teacher_s1/best.pth runs/combined_syn_teacher_s2/best.pth runs/c128_syn_teacher_s1/best.pth runs/c128_syn_teacher_s2/best.pth"
declare -A X=( [iam]="runs/iam_final_teacher_s1/best.pth runs/iam_final_teacher_s2/best.pth runs/iam_final_teacher_s3/best.pth runs/combined_synft_teacher_s1/best.pth" [rimes]=runs/rimes_tuned_teacher_s1/best.pth [bentham]=runs/bentham_tuned_teacher_s1/best.pth [saintgall]=runs/saintgall_tuned_teacher_s1/best.pth [washington]=runs/washington_tuned_teacher_s1/best.pth )
lm() { case $1 in iam) echo "--lm-text synth/text_en.txt --lm-max-lines 400000 --order 10";; rimes) echo "--lm-text synth/text_fr.txt --lm-max-lines 200000 --order 9";; saintgall) echo "--lm-text synth/text_la.txt --lm-max-lines 200000 --order 9";; *) echo "--lm-text synth/text_en.txt --lm-max-lines 200000 --order 9";; esac; }
for ds in iam rimes bentham saintgall washington; do
  F="runs/ft_${ds}_c128_syn_teacher_s1/best.pth runs/ft_${ds}_c128_syn_teacher_s2/best.pth"
  python -m htrjand.ensemble_rescore --dataset $ds --tag ${ds}_ensft_wiki --members $F $C ${X[$ds]} $(lm $ds) --betas 0.0,0.5,1.0,1.5,2.0,3.0,4.0,5.0 --workers 6 2>&1 | tail -n 1 &
done; wait
for Cr in t5_ctx1 t5_ctx0 t5_ext_ctx1; do for ds in iam rimes bentham saintgall washington; do
  python -m htrjand.t5post apply --dataset $ds --corrector postproc/$Cr --tag ${ds}_teacher_ensft_$Cr --val-preds results/ens_${ds}_ensft_wiki_valid_preds.tsv --test-preds results/ens_${ds}_ensft_wiki_test_preds.tsv --out results/t5_${ds}_teacher_ensft_$Cr.json 2>&1 | grep '^\[' &
done; wait; done
python scripts/final_table.py
echo "FT TEACHER PATHS DONE"

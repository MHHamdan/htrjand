#!/bin/bash
# Paper-closest Teacher path (no character LM anywhere): sequence-level ensemble of the same members
# (gen-alpha 0, alpha 0: candidates from plain CTC beam search, scored by the members' CTC only),
# then Algorithm 5 with each corrector.
cd "$(dirname "$0")/.."
: "${HTRJAND_DATA_DIR:?set HTRJAND_DATA_DIR to the directory holding the archives}"
export HTRJAND_DATA_DIR
C="runs/combined_final_teacher_s1/best.pth runs/combined_final_teacher_s2/best.pth runs/combined_syn_teacher_s1/best.pth runs/combined_syn_teacher_s2/best.pth runs/c128_syn_teacher_s1/best.pth runs/c128_syn_teacher_s2/best.pth"
declare -A X=( [iam]="runs/iam_final_teacher_s1/best.pth runs/iam_final_teacher_s2/best.pth runs/iam_final_teacher_s3/best.pth runs/combined_synft_teacher_s1/best.pth" [rimes]=runs/rimes_tuned_teacher_s1/best.pth [bentham]=runs/bentham_tuned_teacher_s1/best.pth [saintgall]=runs/saintgall_tuned_teacher_s1/best.pth [washington]=runs/washington_tuned_teacher_s1/best.pth )
for ds in iam rimes bentham saintgall washington; do
  F="runs/ft_${ds}_c128_syn_teacher_s1/best.pth runs/ft_${ds}_c128_syn_teacher_s2/best.pth"
  python -m htrjand.ensemble_rescore --dataset $ds --tag ${ds}_ensft_nolm --members $F $C ${X[$ds]} --gen-alpha 0 --gen-beta 0 --alphas 0.0 --betas 0.0,0.25,0.5,1.0,1.5,2.0 --workers 6 2>&1 | tail -n 1 &
done; wait
for Cr in t5_ctx1 t5_ctx0 t5_ext_ctx1; do for ds in iam rimes bentham saintgall washington; do
  python -m htrjand.t5post apply --dataset $ds --corrector postproc/$Cr --tag ${ds}_teacher_ensnolm_$Cr --val-preds results/ens_${ds}_ensft_nolm_valid_preds.tsv --test-preds results/ens_${ds}_ensft_nolm_test_preds.tsv --out results/t5_${ds}_teacher_ensnolm_$Cr.json 2>&1 | grep '^\[' &
done; wait; done
echo PAPER PATH DONE

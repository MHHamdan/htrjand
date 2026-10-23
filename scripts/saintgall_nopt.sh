#!/bin/bash
# Saint Gall without the "pt" special letter (annotated on training pages only): fine-tune every Saint Gall
# model on the corrected training labels, then redo LM / ensemble / T5 paths. Every choice on validation.
cd "$(dirname "$0")/.."
: "${HTRJAND_DATA_DIR:?set HTRJAND_DATA_DIR to the directory holding the archives}"
export HTRJAND_DATA_DIR
D=$HTRJAND_DATA_DIR; G=${GPU:-0}   # device index inside CUDA_VISIBLE_DEVICES (the original run used 1)
ST="--preset paper_student --set pools=[[2,2],[2,2],[4,2]] --dropout 0.05 --dropout-cnn 0.0 --dropout-rnn 0.0 --weight-decay 1e-4"
TE="--preset paper_teacher --dropout 0.2 --dropout-cnn 0.1 --dropout-rnn 0.2 --weight-decay 5e-4"
COMMON="--gpu $G --workers 6 --sched cosine --lr 3e-4 --epochs 30 --warmup-epochs 1 --patience 0 --aug-set strong --ema 0.999 --batch-size 16 --seed 1 --keep-init-charset"
ft() { # tag parent archive presetflags
  [ -f runs/$1/train_log.json ] && grep -q '"finished": true' runs/$1/train_log.json || python -m htrjand.train $COMMON $4 --dataset $3 --init-ckpt runs/$2/best.pth --tag $1 > logs/$1.log 2>&1
  for sp in valid test; do python -m htrjand.evaluate --gpu $G --split $sp --ckpt runs/$1/best.pth --dataset saintgall_nopt --tag $1_saintgall_nopt > /dev/null 2>&1; done
  echo "$1 valid $(python -c "import json;print(round(100*json.load(open('results/$1_saintgall_nopt_valid.json'))['cer'],2))")"
}
ft nopt_sg_student_s1 c128_student_s1 $D/saintgall_nopt_128x1632.hdf5 "$ST"
ft nopt_sg_c128_teacher_s1 c128_syn_teacher_s1 $D/saintgall_nopt_128x1632.hdf5 "$TE"
ft nopt_sg_c128_teacher_s2 c128_syn_teacher_s2 $D/saintgall_nopt_128x1632.hdf5 "$TE"
for m in combined_final_teacher_s1 combined_final_teacher_s2 combined_syn_teacher_s1 combined_syn_teacher_s2; do
  ft nopt_sg_$m $m $D/saintgall_nopt.hdf5 "$TE"
done
LM="--lm-text synth/text_la.txt --lm-max-lines 200000 --order 9 --betas 0.0,0.5,1.0,1.5,2.0,3.0,4.0,5.0 --workers 12"
M="runs/nopt_sg_c128_teacher_s1/best.pth runs/nopt_sg_c128_teacher_s2/best.pth runs/nopt_sg_combined_final_teacher_s1/best.pth runs/nopt_sg_combined_final_teacher_s2/best.pth runs/nopt_sg_combined_syn_teacher_s1/best.pth runs/nopt_sg_combined_syn_teacher_s2/best.pth"
python -m htrjand.ensemble_rescore --gpu $G --dataset saintgall_nopt --tag saintgall_nopt_student_wiki --members runs/nopt_sg_student_s1/best.pth $LM 2>&1 | tail -n 1
python -m htrjand.ensemble_rescore --gpu $G --dataset saintgall_nopt --tag saintgall_nopt_ens_wiki --members $M $LM 2>&1 | tail -n 1
python -m htrjand.ensemble_rescore --gpu $G --dataset saintgall_nopt --tag saintgall_nopt_ens_nolm --members $M --gen-alpha 0 --gen-beta 0 --alphas 0.0 --betas 0.0,0.25,0.5,1.0,1.5,2.0 --workers 12 2>&1 | tail -n 1
for C in t5_ctx1 t5_ctx0 t5_ext_ctx1; do
  for b in "student_ctc:results/nopt_sg_student_s1_saintgall_nopt_valid_preds.tsv:results/nopt_sg_student_s1_saintgall_nopt_test_preds.tsv" \
           "student_lm:results/ens_saintgall_nopt_student_wiki_valid_preds.tsv:results/ens_saintgall_nopt_student_wiki_test_preds.tsv" \
           "teacher_ens_lm:results/ens_saintgall_nopt_ens_wiki_valid_preds.tsv:results/ens_saintgall_nopt_ens_wiki_test_preds.tsv" \
           "teacher_ensnolm:results/ens_saintgall_nopt_ens_nolm_valid_preds.tsv:results/ens_saintgall_nopt_ens_nolm_test_preds.tsv"; do
    IFS=: read name vp tp <<< "$b"
    python -m htrjand.t5post apply --gpu $G --dataset saintgall_nopt --corrector postproc/$C --tag saintgall_nopt_${name}_$C --val-preds $vp --test-preds $tp --out results/t5_saintgall_nopt_${name}_$C.json 2>&1 | grep '^\[' &
  done; wait
done
echo "SAINTGALL NOPT DONE"

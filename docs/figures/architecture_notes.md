# Architecture figure — verified dimensions

Figure: `architecture.{pdf,png,svg}` (`scripts/fig_architecture.py`). Every number in it is read from
`architecture_shapes.json`, which `scripts/verify_architecture.py` writes by loading the checkpoints
with `htrjand.models.from_checkpoint` and running IAM test line 0 through forward hooks.
The existing Archify diagram (`.archify/architecture-htr-jand-20261016-212543/`) was not modified.

Checkpoints:
* Student, deployed: `runs/ft_iam_c128_student_kd_s1/best.pth`. This is the IAM fine-tune, with
  `kd=False`, initialised from the parent below.
* Student, KD stage: `runs/c128_student_kd_s1/best.pth`. It was trained on the five-corpus archive plus
  120k synthetic lines with KD from `runs/c128_syn_teacher_s1`. It has the same shapes as the deployed Student.
* Teacher: `runs/c128_syn_teacher_s1/best.pth`.

## Student (preset `paper_student`, `set pools=[[2,2],[2,2],[4,2]]`)
| stage | output shape (B=1) | notes |
|---|---|---|
| input | 1 × 1 × 128 × 1632 | uint8 archive image, per-image min–max to [−1, 1] (`normalisation=paper`) |
| block 1: gated conv 3×3 → BN → ReLU → MaxPool(2,2) → SE(r=8) | 1 × 16 × 64 × 816 | gated conv = one conv with 2·C outputs, `a·σ(b)` (conv weight 32×1×3×3) |
| block 2, MaxPool(2,2) | 1 × 32 × 32 × 408 | conv weight 64×16×3×3 |
| block 3, MaxPool(4,2) | 1 × 64 × 8 × 204 | conv weight 128×32×3×3 |
| columns (concat rows) | 1 × 204 × 512 | 64 channels × 8 rows; **T = 204 frames**, 8 input px per frame |
| 4 × BiLSTM + LayerNorm | 1 × 204 × 128 | 64 units per direction (`weight_hh` 256×64) |
| MHSA | 1 × 204 × 128, weights 1 × 204 × 204 | **1 head**, `in_proj` 384×128 |
| Proxima | 1 × 204 × 128 (one vector broadcast), weights 1 × 1 × 204 | learned query `q` 1×1×1×128 |
| Combined Attention `LN(W_f[A;P] + b + X)` | 1 × 204 × 128 | `W_f` 128×256; fusion = "paper" (Eq. 18 as written) |
| linear CTC head | 1 × 204 × 111 | V = 111 = 108 characters + PAD + UNK + blank |
| auxiliary CTC head (training only) | 1 × 204 × 111 | linear 512 → 111 on the CNN columns |

Parameter counts (Student):

| part | parameters |
|---|---:|
| CNN | 48,160 |
| BiLSTM + LN | 594,944 |
| MHSA | 66,048 |
| Proxima | 49,664 |
| W_f + LN | 33,152 |
| CTC head | 14,319 |
| **Deployed** | **806,287** ✓ |
| Auxiliary head (training only) | 56,943 |
| Total | 863,230 |

## Teacher (preset `paper_teacher`)
| stage | output shape |
|---|---|
| 5 gated conv + SE(r=16) blocks, channels 32/64/128/256/512, pools (2,2),(2,2),(2,2),(2,1),(2,1) | 1×32×64×816 → 1×64×32×408 → 1×128×16×204 → 1×256×8×204 → 1×512×4×204 |
| columns | 1 × 204 × 2048 (512 ch × 4 rows) |
| 4 × BiLSTM + LN, 128 per direction | 1 × 204 × 256 |
| MHSA, 2 heads (weights averaged in the returned map), and Proxima, 2 heads (weights 1 × 2 × 204) | 1 × 204 × 256 |
| CTC head | 1 × 204 × 111 |

Teacher parameters: **7,220,911 deployed** ✓ (7,448,350 total including the 227,439-parameter auxiliary head).
By part: CNN 3,181,632; BiLSTM 3,418,112; MHSA 263,168; Proxima 197,632; W_f + LN 131,840; head 28,527.

## Training-only elements (taken from the checkpoint `args`)
* KD (KD-stage Student): τ = 2. The loss is τ²·KL(softmax(z_T/τ) ‖ softmax(z_S/τ)), averaged over frames
  (`losses.kd_loss`). The weights move linearly over training: γ from 0.2 to 0.5 and α from 0.7 to 0.4,
  with δ = 0.1 (`losses.weights`). The frozen Teacher and the Student both emit 204 frames, so the
  Eq. 35 interpolation is the identity.
* Synthetic lines: `synth/synth_120k_128x1632.hdf5`, with share r rising from 0.1 to 0.4 (Eq. 33),
  redrawn each epoch. The KD-stage Student ran 150 epochs and the Teacher 120.
* Auxiliary CTC head, δ = 0.1, in every stage.
* The per-corpus fine-tuning (30 epochs, IAM) uses no KD and no synthetic lines: L = L_CTC + δ·L_aux.

## Decoding (outside the network)
* Greedy CTC.
* CTC prefix beam search with an external character n-gram LM, fitted on the training transcriptions
  plus Wikipedia text (`ensemble_rescore.py`). The order is 10 for IAM and 9 for the other corpora.
  α and β are chosen on validation (IAM: α = 0.3, β = 2.0).
* Optional gated T5-small corrector (`t5post.py`). It applies a candidate only when a word is
  out-of-vocabulary (IAM selection) and the confidence exceeds θ (IAM: 0.98). Context is ±1 line.
  All of these settings are chosen on validation.

## Differences from the description in the request
None in the dimensions. Worth stating explicitly: the Student's BiLSTM width is 64 per direction, so
d_model = 128. The Teacher's is 128 per direction, so d_model = 256. The Student has 1 attention head
and the Teacher 2. KD is applied in the five-corpus stage, not in the per-corpus fine-tune that
produced the deployed weights.

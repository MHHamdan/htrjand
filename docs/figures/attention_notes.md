# Attention / interpretability figure — what each panel shows and does not show

Figure: `attention.{pdf,png,svg}`. Model: the validation-selected IAM Student
`runs/ft_iam_c128_student_kd_s1/best.pth` (806,287 deployed parameters, 1 attention head).
Line: IAM test index 1849 (`p06-096-09`), the `lm_correction` sample of
`results/qualitative_examples.json`, chosen by rule, not by how the maps look.
All activations come from forward hooks on the real modules, eval mode, on the archived
128×1632 input (`scripts/attention_activations.py`). Check that the hooks see the deployed
computation: the greedy decode of the hooked forward pass equals the published raw-CTC
prediction on **1861/1861** IAM test lines (`attention_stats.json`, `greedy_matches_published_tsv`).
The test-set numbers below come from `attention_stats.json` and cover all 1861 IAM test lines.

## Geometry
The CNN reduces 128×1632 to 64 channels × 8 rows × **204 frames**. Frame f is computed from
input columns 8f to 8f+7, but its receptive field is wider than that (3×3 convolutions in
three blocks, then four BiLSTMs that see the whole line). The x-positions in panels (b)–(f)
show where each frame sits in the input. They do not show which pixels a frame depends on.
Panels (b)–(f) are zoomed to the line's content span. The rest of the canvas is padding.

## Panels
**(a) Input.** The archived network input, re-inverted for display.

**(b) CTC alignment.** The greedy path's emissions (first frame of each non-blank run), drawn at their
frames. Red marks an emission that the Levenshtein alignment against the GT calls wrong. CTC
emissions are peaky, so a character's emission frame is usually near one edge of the glyph, not at its
centre. The panel shows timing, not segmentation.

**(c) MHSA, the attention each frame receives.** This is the column mean of the 204×204 matrix over
the queries in the span, and panel (g) shows the full matrix. **Finding: MHSA is neither local nor
diagonal.** Across the test set, the mass within ±k frames of the query equals what uniform
attention would give: 4.3 % vs 4.7 % at k = 4, and 17.8 % vs 16.6 % at k = 16. The self-weight is
0.4 %. The attention is close to **independent of the query**: (g) shows vertical stripes, and the
mean Jensen–Shannon divergence between a row and the mean row is 0.17 nats, against a maximum of 0.69.
The stripes fall on **word gaps**. Frames within ±2 of an emitted space receive **2.3×** their share
of the attention, and character-emission frames receive 0.75×. On this line the two strongest stripes
sit on the first two emitted spaces, and the third emitted space (before "?") gets a weak one.
The space the model missed in "conspired so" gets no stripe. So in the
Student, MHSA acts as a line-level, query-independent pooling of word-boundary frames. It does not
act as a contextual lookup between characters. Because the stripes follow the model's *own* space
emissions, they are not evidence that attention *finds* word boundaries. Cause and effect cannot be
separated from these maps.

**(d) Proxima.** One learned query takes a softmax over the 204 frames, as in `models.py`
`ProximaAttention`. Its single output vector is broadcast to every frame. **Finding: it is a soft
global pooling that is far from uniform.** On the test set the effective number of frames (exp of
the entropy) is 122 out of a 193-frame span (p10–p90: 112–131). The weights put 98 % of their mass
on the content span and favour character-emission frames (**2.5×**). They are neutral on space frames
(1.0×). Because the output is identical for every frame, Proxima can only supply line-level context,
such as writer style or ink statistics, through the fusion. It cannot help localise characters. The
name suggests a proximity operator, but the computation has no locality.

**(e) Fusion (Eq. 18).** The panel plots the per-frame L2 norms of the three summands inside the
LayerNorm, `O = LN(W_f[:, :d]·A + (W_f[:, d:]·P + b) + X)`. The Proxima term is flat by construction.
Averaged over content frames on the test set, the MHSA term is 2.21, the Proxima term 2.25 and the
residual X (the BiLSTM output) 5.50. Both attention branches therefore add vectors of comparable
size, each about 40 % of the residual's norm. The **limitation** is that a norm is not an importance
measure. The LayerNorm is non-linear, and the terms can cancel or align. The panel shows that neither
branch is numerically negligible. It does not show how much each branch matters for the output. The
68×864 component ablations bear on that question: in the IAM Teacher family, the no-attention
ablation reached 6.99 % CER against 7.29 % for the full tuned Teacher
(`iam_tuned_abl_no_attention` vs `iam_tuned_teacher`). That is a different model, so it is context and not a Student result.

**(f) Confidence strip.** The panel shows the per-frame maximum posterior. The three wrong emissions
on this line fall on visibly lower-confidence frames. **Test-set result (h):** the confidence at the
emitting frame is a median of 0.999 for the 77,235 correct emissions and 0.82 for the 3,950 wrong
ones. As a detector of wrong emissions, low confidence reaches an AUROC of 0.87. Deletions have no
emitting frame, so this signal cannot see them.

**(g) MHSA matrix**, restricted to the span, with rows renormalised.

**(h) Confidence histograms** over all IAM test emissions, as normalised shares on a log scale.

## Panels considered and dropped
* **The MHSA attention row of a chosen character overlaid on the image.** Rows are close to
  identical (the query-independence above), so any single row looks like (c). The panel would repeat (c).
* **Per-head maps.** The Student has one head. There is nothing to compare.
* **Proxima as a spatial heat-map overlaid on the image.** It would suggest a localisation that the
  broadcast output does not use. (d) shows the same weights as a 1-D profile on the shared frame axis.

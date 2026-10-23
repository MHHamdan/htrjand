#!/usr/bin/env python3
"""Real activations of the selected IAM Student (runs/ft_iam_c128_student_kd_s1/best.pth) on the IAM test set.

    CUDA_VISIBLE_DEVICES=0 python scripts/attention_activations.py
    -> results/attention_student_iam_test.npz   (per-line activations for the figure lines)
       docs/figures/attention_stats.json (test-set statistics)

For every test line (batch inference, eval mode, no augmentation, the archive's 128x1632 image with the
checkpoint's normalisation) it records, from hooks on the real modules:
  * frame log-posteriors (B, T=204, V) -> greedy CTC path; the decoded string is checked against the
    published raw-CTC TSV (results/ft_iam_c128_student_kd_s1_iam_test_preds.tsv) line by line;
  * the MHSA attention matrix (T x T; one head in the Student, so no head averaging is involved);
  * the Proxima weights over frames (1 x T, one learned query, one head);
  * the fusion terms of Eq. 18, O = LN(W_f [A ; P] + b + X): per frame the norms of W_f[:, :d] A,
    W_f[:, d:] P + b (Proxima's output is one vector broadcast to every frame) and of the residual X.
Test-set statistics (all 1861 lines): MHSA locality (attention mass within +-k frames of the query,
mean |i-j|, normalised entropy), Proxima normalised entropy, effective number of frames (exp entropy) and
the share of its mass on ink columns; fusion norm ratios; per-emission confidence (max posterior at the
emitting frame) for correct vs. erroneous emissions.
"""
import json
import os
import sys

import editdistance
import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refined_figs_common as C  # noqa: E402
from htrjand import data as D  # noqa: E402
from htrjand import models as M  # noqa: E402

RUN = "ft_iam_c128_student_kd_s1"


def greedy_path(lp, blank):
    am = lp.argmax(-1)
    emit = [(f, int(a)) for f, a in enumerate(am) if a != blank and (f == 0 or a != am[f - 1])]
    return am, emit


def main(keep=(13, 67, 193, 1431, 1531, 1849, 913)):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = M.from_checkpoint(os.path.join(C.ROOT, "runs", RUN, "best.pth"))
    model.to(dev).eval()
    tok = D.Tokenizer(ck["charset"])
    norm = ck.get("normalisation", "paper")
    ca = model.attention
    cap = {}
    ca.register_forward_hook(lambda m, i, o: cap.__setitem__("ca_in", i[0].detach()))
    ca.mha.register_forward_hook(lambda m, i, o: cap.update(mha_out=o[0].detach(), mha_w=o[1].detach()))
    ca.prox.register_forward_hook(lambda m, i, o: cap.update(prox_out=o[0].detach(), prox_w=o[1].detach()))
    d = model.d_model
    Wf, bf = ca.W_f.weight.detach(), ca.W_f.bias.detach()
    tsv = C.read_tsv(os.path.join(C.RES, f"{RUN}_iam_test_preds.tsv"))
    with h5py.File(os.path.join(C.DATA, "iam_128x1632.hdf5")) as f:
        imgs = f["test"]["dt"][:]
        gt = [g.decode() if isinstance(g, bytes) else g for g in f["test"]["gt"][:]]
    T = model.n_frames
    px = 1632 / T
    st = dict(mha_within=[], mha_dist=[], mha_ent=[], mha_self=[], prox_ent=[], prox_eff=[], prox_ink=[],
              prox_max=[], span=[], eff_over_span=[], lift_mha_space=[], lift_prox_space=[], lift_mha_char=[], lift_prox_char=[], row_js=[], uni_within=[], n_mha=[], n_prox=[], n_res=[], conf_ok=[], conf_err=[], frame_conf=[])
    match = 0
    kept = {}
    bs = 32
    with torch.no_grad():
        for s in range(0, len(gt), bs):
            x = np.stack([D.normalise(im, norm) for im in imgs[s:s + bs]])[:, None]
            logits = model(torch.from_numpy(x).to(dev))
            lp = logits.log_softmax(-1).cpu().numpy()
            A, wa = cap["mha_out"], cap["mha_w"].cpu().numpy()            # (B,T,d), (B,T,T)
            P, wp = cap["prox_out"], cap["prox_w"].cpu().numpy()[:, 0]     # (B,T,d), (B,T)
            X = cap["ca_in"]
            cA = (A @ Wf[:, :d].T).norm(dim=-1).cpu().numpy()
            cP = (P @ Wf[:, d:].T + bf).norm(dim=-1).cpu().numpy()
            cX = X.norm(dim=-1).cpu().numpy()
            for b in range(lp.shape[0]):
                i = s + b
                am, emit = greedy_path(lp[b], tok.BLANK)
                hyp = tok.decode([a for _, a in emit])
                match += hyp == tsv[i]["hyp"]
                ink = imgs[i].reshape(128, T, int(px)).max(axis=(0, 2)) > 0       # frame has non-pad pixels
                ink_f = np.where(ink)[0]
                lo, hi = ink_f[0], ink_f[-1] + 1
                W = wa[b][lo:hi, lo:hi]
                W = W / W.sum(1, keepdims=True)
                ii, jj = np.meshgrid(np.arange(hi - lo), np.arange(hi - lo), indexing="ij")
                dist = np.abs(ii - jj)
                st["mha_within"].append([float((W * (dist <= k)).sum(1).mean()) for k in (0, 1, 2, 4, 8, 16)])
                st["mha_dist"].append(float((W * dist).sum(1).mean()))
                st["mha_self"].append(float(np.diag(W).mean()))
                st["mha_ent"].append(float((-(W * np.log(W + 1e-12)).sum(1) / np.log(hi - lo)).mean()))
                st["span"].append(hi - lo)
                # where does the attention go? lift = mass on a frame set / share of span frames in that set
                sp_id = tok.char_to_index.get(" ")
                sp = np.zeros(T, bool); chf = np.zeros(T, bool)
                for fr, a in emit:
                    if a == sp_id:
                        sp[max(lo, fr - 2):min(hi, fr + 3)] = True
                    else:
                        chf[fr] = True
                colm = wa[b][lo:hi, lo:hi].sum(0) / (hi - lo)
                colm = colm / colm.sum()
                pin = wp[b][lo:hi] / wp[b][lo:hi].sum()
                for key, msk in (("space", sp[lo:hi]), ("char", chf[lo:hi])):
                    if msk.any():
                        share = msk.mean()
                        st[f"lift_mha_{key}"].append(float(colm[msk].sum() / share))
                        st[f"lift_prox_{key}"].append(float(pin[msk].sum() / share))
                mrow = W.mean(0)
                mm = (W + mrow) / 2
                st["row_js"].append(float((0.5 * (W * np.log((W + 1e-12) / mm)).sum(1)
                                           + 0.5 * (mrow * np.log((mrow + 1e-12) / mm)).sum(1)).mean()))
                st["uni_within"].append([float(((dist <= k).sum(1) / (hi - lo)).mean()) for k in (0, 1, 2, 4, 8, 16)])
                p = wp[b]
                ent = float(-(p * np.log(p + 1e-12)).sum())
                st["prox_ent"].append(ent / np.log(T)); st["prox_eff"].append(float(np.exp(ent))); st["eff_over_span"].append(float(np.exp(ent)) / (hi - lo))
                st["prox_ink"].append(float(p[lo:hi].sum())); st["prox_max"].append(float(p.max()))
                st["n_mha"].append(float(cA[b][lo:hi].mean())); st["n_prox"].append(float(cP[b][lo:hi].mean()))
                st["n_res"].append(float(cX[b][lo:hi].mean()))
                conf = np.exp(lp[b].max(-1))
                ops = C.align(gt[i], hyp)
                k = 0
                for op, _, _ in ops:
                    if op == "D":
                        continue
                    (st["conf_ok"] if op == "=" else st["conf_err"]).append(float(conf[emit[k][0]])); k += 1
                if i in keep:
                    kept[i] = dict(lp=lp[b], mha=wa[b], prox=p, cA=cA[b], cP=cP[b], cX=cX[b], hyp=hyp,
                                   emit=np.array(emit), ops=ops, ink=(lo, hi))
    def m(v): return float(np.mean(v))
    co, ce = np.array(st["conf_ok"]), np.array(st["conf_err"])
    # AUROC of (1 - confidence) for detecting an erroneous emission
    allc = np.concatenate([co, ce]); lab = np.r_[np.zeros(len(co)), np.ones(len(ce))]
    order = np.argsort(-(1 - allc), kind="stable"); r = np.empty(len(allc)); r[order] = np.arange(len(allc), 0, -1)
    auroc = float((r[lab == 1].sum() - len(ce) * (len(ce) + 1) / 2) / (len(ce) * len(co)))
    within = np.array(st["mha_within"]).mean(0)
    out = dict(model=RUN, ckpt=f"runs/{RUN}/best.pth", n_lines=len(gt), frames=T, px_per_frame=px,
               greedy_matches_published_tsv=f"{match}/{len(gt)}",
               mha=dict(heads=model.cfg["heads"], mass_within_k_frames={f"k={k}": float(v) for k, v in zip((0, 1, 2, 4, 8, 16), within)},
                        same_mass_if_uniform_over_span={f"k={k}": float(v) for k, v in zip((0, 1, 2, 4, 8, 16), np.array(st["uni_within"]).mean(0))},
                        mean_abs_offset_frames=m(st["mha_dist"]),
                        mean_js_divergence_row_vs_mean_row_nats=m(st["row_js"]),
                        lift_on_space_frames=m(st["lift_mha_space"]), lift_on_char_emission_frames=m(st["lift_mha_char"]), mean_self_weight=m(st["mha_self"]),
                        normalised_entropy=m(st["mha_ent"]),
                        note="rows restricted to frames with non-padding pixels and renormalised; lift = attention mass on a frame set divided by that set's share of the span (1 = no preference); space frames = +-2 frames around a greedy emission of a space; char frames = emission frames of non-space characters; JS divergence 0 = every query row identical (query-independent attention), ln2 = 0.693 max"),
               proxima=dict(normalised_entropy_over_204=m(st["prox_ent"]), effective_frames=m(st["prox_eff"]), content_span_frames=m(st["span"]),
                            effective_frames_over_span=m(st["eff_over_span"]),
                            effective_frames_p10_p90=[float(np.percentile(st["prox_eff"], 10)), float(np.percentile(st["prox_eff"], 90))],
                            mass_on_ink_frames=m(st["prox_ink"]),
                            lift_on_space_frames=m(st["lift_prox_space"]), lift_on_char_emission_frames=m(st["lift_prox_char"]), max_weight=m(st["prox_max"])),
               fusion_norms_mean_over_ink_frames=dict(mha_term=m(st["n_mha"]), proxima_term=m(st["n_prox"]), residual_X=m(st["n_res"])),
               confidence=dict(n_correct_emissions=len(co), n_error_emissions=len(ce),
                               mean_conf_correct=float(co.mean()), mean_conf_error=float(ce.mean()),
                               median_conf_correct=float(np.median(co)), median_conf_error=float(np.median(ce)),
                               auroc_low_conf_detects_error=auroc,
                               note="emission = first frame of each greedy-CTC run; deletions have no emission"))
    edges = np.linspace(0, 1, 21)
    out["confidence_hist"] = dict(edges=edges.tolist(), correct=(np.histogram(co, edges)[0] / len(co)).tolist(),
                                  error=(np.histogram(ce, edges)[0] / len(ce)).tolist())
    json.dump(out, open(os.path.join(C.OUT, "attention_stats.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(C.RES, "attention_student_iam_test.npz"),
                        **{f"{i}_{k}": (v if not isinstance(v, (list, tuple, str)) else np.array(v, dtype=object))
                           for i, e in kept.items() for k, v in e.items()})
    print(json.dumps({k: v for k, v in out.items() if k != "confidence_hist"}, indent=1))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Verify every dimension drawn in docs/figures/architecture.* against models.py and the actual
checkpoints, by running one real IAM test line through each network with forward hooks.

    python scripts/verify_architecture.py   -> docs/figures/architecture_shapes.json (+ stdout)

Student: runs/ft_iam_c128_student_kd_s1/best.pth (the selected IAM Student; KD happened in its parent
runs/c128_student_kd_s1, whose Teacher was runs/c128_syn_teacher_s1). Teacher: runs/c128_syn_teacher_s1/best.pth.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refined_figs_common as C  # noqa: E402
from htrjand import data as D  # noqa: E402
from htrjand import models as M  # noqa: E402


def trace(run):
    path = os.path.join(C.ROOT, "runs", run, "best.pth")
    model, ck = M.from_checkpoint(path)
    shapes = []
    def hook(name):
        def f(m, i, o):
            o0 = o[0] if isinstance(o, tuple) else o
            shapes.append((name, list(i[0].shape), list(o0.shape)))
        return f
    for k, blk in enumerate(model.cnn):
        blk.register_forward_hook(hook(f"cnn block {k + 1} (gated conv 3x3 -> BN -> ReLU -> MaxPool{tuple(model.cfg['pools'][k])} -> SE)"))
    for k, l in enumerate(model.lstms):
        l.register_forward_hook(hook(f"BiLSTM {k + 1} (+LayerNorm)"))
    ca = model.attention
    ca.mha.register_forward_hook(hook("MHSA (output, weights)"))
    ca.prox.register_forward_hook(hook("Proxima (output broadcast, weights)"))
    ca.register_forward_hook(hook("Combined Attention LN(W_f[A;P]+b+X)"))
    model.fc.register_forward_hook(hook("linear CTC head"))
    model.aux.register_forward_hook(hook("auxiliary CTC head on CNN columns (training only)"))
    x = torch.from_numpy(D.normalise(C.test_image("iam", 0), ck.get("normalisation", "paper"))[None, None])
    with torch.no_grad():
        logits, aux, (wa, wp) = model(x, return_aux=True, return_attention=True)
    extra = dict(mha_weights=list(wa.shape), proxima_weights=list(wp.shape), columns=[1, model.n_frames, model.column_dim])
    groups = {"cnn": model.cnn, "lstm": torch.nn.ModuleList([model.lstms, model.lstm_norms]), "mhsa": ca.mha,
              "proxima": ca.prox, "fusion W_f + LN": torch.nn.ModuleList([ca.W_f, ca.norm]), "ctc head": model.fc,
              "aux head": model.aux}
    params = {k: int(sum(p.numel() for p in v.parameters())) for k, v in groups.items()}
    sd = {k: list(v.shape) for k, v in ck["model_state_dict"].items()
          if any(s in k for s in ("cnn.0.0.conv.weight", "cnn.2.0.conv.weight", "cnn.4.0.conv.weight", "se", "lstms.0.weight_ih",
                                  "lstms.3.weight_hh", "attention.mha.attn.in_proj_weight", "attention.prox.q",
                                  "attention.W_f.weight", "fc.weight", "aux.weight"))}
    return dict(run=run, ckpt=os.path.relpath(path, C.ROOT), cfg=ck["model_cfg"], input=list(x.shape),
                frames=model.n_frames, feature_rows=model.feature_rows, column_dim=model.column_dim, d_model=model.d_model,
                vocab_incl_blank=ck["model_cfg"]["vocab_size"], charset_size=len(ck["charset"]),
                params_total=int(sum(p.numel() for p in model.parameters())), params_deployed=int(model.deployed_parameters()),
                params_by_part=params, hooks=shapes, extra=extra, selected_state_dict_shapes=sd,
                train_args={k: ck["args"].get(k) for k in ("dataset", "kd", "tau", "alpha", "gamma", "aux_weight", "synthetic",
                                                            "synth_ratio", "init_ckpt", "teacher_ckpt", "epochs", "set")})


def main():
    out = {}
    for name, run in (("student", "ft_iam_c128_student_kd_s1"), ("student_parent_kd", "c128_student_kd_s1"),
                      ("teacher", "c128_syn_teacher_s1")):
        r = trace(run)
        out[name] = r
        print(f"== {name}: {run}  total {r['params_total']:,}  deployed {r['params_deployed']:,}  T={r['frames']} "
              f"rows={r['feature_rows']} col_dim={r['column_dim']} d={r['d_model']} V={r['vocab_incl_blank']}")
        for n, i, o in r["hooks"]:
            print(f"   {n:70s} {str(i):22s} -> {o}")
        print("   attention weights:", r["extra"])
        print("   params by part:", r["params_by_part"])
        print("   train args:", r["train_args"])
    json.dump(out, open(os.path.join(C.OUT, "architecture_shapes.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

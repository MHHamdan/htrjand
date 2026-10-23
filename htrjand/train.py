"""Train a Teacher, or a Student with knowledge distillation (Sections III-C, IV).

    # Teacher (loss = L_ctc + delta * L_aux)
    python -m htrjand.train --dataset iam --preset paper_teacher --seed 1 --gpu 0

    # Student with KD from a trained teacher (Eq. 19 with the Section IV-A schedule)
    python -m htrjand.train --dataset iam --preset paper_student --seed 1 --gpu 0 \
        --teacher-ckpt runs/iam_paper_teacher_s1/best.pth --kd

Every argument default is the value declared, with its evidence class, in
configs/htrjand_base.yaml; tests/test_config.py asserts the two agree. Nothing in
this script reads the test split.
"""
import argparse
import contextlib
import json
import os
import platform
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import data as D
from . import losses as L
from . import metrics as Me
from . import models as M
from . import synthetic as SY

from ._compat import REPO_ROOT  # noqa: E402  (source checkout, else the working directory)


class EMA:
    """Exponential moving average of the float parameters and buffers. It is an
    average over one optimisation trajectory, not a second model or extra data."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: (v.detach().clone().float() if v.is_floating_point() else v.detach().clone())
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.is_floating_point():
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v.detach())

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    @contextlib.contextmanager
    def applied(self, model):
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict({k: v.to(backup[k].dtype) for k, v in self.shadow.items()})
        try:
            yield
        finally:
            model.load_state_dict(backup)


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not deterministic


def worker_init(worker_id):
    """Give each DataLoader worker its own NumPy / Python stream derived from
    the torch seed of that worker (torch seeds workers itself)."""
    s = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(s)
    random.seed(s)


def environment():
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=REPO_ROOT,
                                           stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return None
    return dict(python=platform.python_version(), torch=torch.__version__,
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
                hostname=platform.node(), git_commit=git("rev-parse", "HEAD"),
                git_dirty=bool(git("status", "--porcelain")), command=" ".join(sys.argv))


@torch.no_grad()
def evaluate(model, loader, labels, tok, device, blank):
    """Validation pass: mean CTC loss, corpus CER and WER. References are the
    archive transcriptions themselves, not a decode of their encoding."""
    model.eval()
    hyps, losses = [], []
    for img, tgt, tlen in loader:
        img = img.to(device, non_blocking=True)
        logits = model(img)
        losses.append(float(L.ctc_loss(logits, tgt.to(device), tlen.to(device), blank)))
        hyps += tok.ctc_greedy_decode(logits)
    refs = labels[: len(hyps)]
    return float(np.mean(losses)), Me.cer(refs, hyps), Me.wer(refs, hyps)


def load_teacher(path, device):
    teacher, ck = M.from_checkpoint(path, map_location="cpu")
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher, ck


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("data")
    g.add_argument("--dataset", required=True, help=f"one of {sorted(D.OFFICIAL)} or an archive path")
    g.add_argument("--data-dir", default=None, help="archive directory (default $HTRJAND_DATA_DIR or <repo>/data)")
    g.add_argument("--normalisation", default="paper", choices=D.NORMALISATIONS,
                   help="paper = Eq. (1) min-max to [-1,1]; unit = historical /255")
    g.add_argument("--augment", dest="augment", action="store_true", default=True,
                   help="train-split augmentation (Eq. 2); on by default")
    g.add_argument("--no-augment", dest="augment", action="store_false")
    g.add_argument("--aug-set", default="base", choices=D.AUG_SETS)
    g.add_argument("--limit", type=int, default=None, help="lines per split, smoke tests only")
    g.add_argument("--synthetic", default=None, metavar="HDF5",
                   help="synthetic-line archive (synth_archive.py); mixed in per Eq. 33")
    g.add_argument("--synth-ratio", type=float, nargs=2, default=(0.1, 0.4), metavar=("R0", "RMAX"),
                   help="Eq. 33 r_0 and r_max (paper: 0.1, 0.4)")
    g.add_argument("--synth-only", action="store_true",
                   help="train on the synthetic archive alone (pretraining stage); validation stays real")
    g.add_argument("--curriculum", default="none", choices=("none", "acp"),
                   help="acp = Algorithm 3 (Adaptive Curriculum Progression), five stages, see ACP_STAGES")
    g.add_argument("--acp-t0", type=float, default=0.70, help="Alg. 3 initial threshold T on 1 - validation CER")
    g.add_argument("--acp-dt", type=float, default=0.05, help="Alg. 3 threshold increment Delta_T")
    g.add_argument("--acp-max-stage-epochs", type=int, default=15, help="advance anyway after this many epochs in a stage")
    g.add_argument("--init-ckpt", default=None, help="warm-start every matching tensor from this checkpoint")
    g.add_argument("--keep-init-charset", action="store_true",
                   help="use the --init-ckpt charset (per-corpus fine-tuning of a multi-corpus model)")
    g.add_argument("--fold", default=None, metavar="k/K",
                   help="train on the K-fold complement of fold k of the training split "
                        "(cross-decoding models for post-OCR correction pairs)")
    g = ap.add_argument_group("model")
    g.add_argument("--preset", default="paper_teacher", choices=sorted(M.PRESETS))
    g.add_argument("--dropout", type=float, default=0.01, help="before the output layer and inside attention")
    g.add_argument("--dropout-cnn", type=float, default=0.0, help="Dropout2d after each CNN block")
    g.add_argument("--dropout-rnn", type=float, default=0.0, help="between BiLSTM layers")
    g.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a constructor argument of the preset, e.g. --set fusion=historical")
    g = ap.add_argument_group("optimisation")
    g.add_argument("--epochs", type=int, default=200)
    g.add_argument("--batch-size", type=int, default=32)
    g.add_argument("--lr", type=float, default=5e-4)
    g.add_argument("--weight-decay", type=float, default=1e-4)
    g.add_argument("--clip", type=float, default=1.0)
    g.add_argument("--sched", default="plateau", choices=("plateau", "cosine"),
                   help="plateau = ReduceLROnPlateau on val loss (historical); cosine = linear "
                        "warmup then cosine decay over --epochs (needs the full budget)")
    g.add_argument("--warmup-epochs", type=int, default=5, help="warmup length for --sched cosine")
    g.add_argument("--min-lr-frac", type=float, default=0.01, help="cosine floor as a fraction of --lr")
    g.add_argument("--ema", type=float, default=0.0,
                   help="decay of an exponential moving average of the weights, evaluated beside "
                        "the raw model and checkpointed when it is the better one (0 = off)")
    g.add_argument("--plateau-factor", type=float, default=0.5)
    g.add_argument("--plateau-patience", type=int, default=3)
    g.add_argument("--min-lr", type=float, default=1e-6)
    g.add_argument("--patience", type=int, default=10, help="early stopping, epochs without improvement (paper: 10); 0 = off")
    g.add_argument("--min-delta", type=float, default=1e-3, help="early-stopping improvement threshold on val loss (paper: 0.001)")
    g.add_argument("--es-metric", default="loss", choices=("loss", "cer"), help="early-stopping quantity (paper: validation loss)")
    g.add_argument("--aux-weight", type=float, default=0.1, help="delta of Eq. 19 (paper: 0.1)")
    g.add_argument("--amp", action="store_true", help="bfloat16 autocast; CTC and KD stay in fp32")
    g = ap.add_argument_group("distillation")
    g.add_argument("--teacher-ckpt", default=None)
    g.add_argument("--kd", action="store_true", help="add gamma * L_kd from --teacher-ckpt")
    g.add_argument("--tau", type=float, default=2.0, help="temperature (not given in the paper; decision D-004)")
    g.add_argument("--alpha", type=float, nargs=2, default=(0.7, 0.4), metavar=("START", "END"))
    g.add_argument("--gamma", type=float, nargs=2, default=(0.2, 0.5), metavar=("START", "END"))
    g.add_argument("--init-from-teacher", action="store_true",
                   help="warm-start every student tensor whose shape matches the teacher's")
    g = ap.add_argument_group("run")
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--deterministic", action="store_true", help="cudnn deterministic kernels (slower)")
    g.add_argument("--gpu", type=int, default=0, help="CUDA device index; -1 = CPU")
    g.add_argument("--workers", type=int, default=4)
    g.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "runs"))
    g.add_argument("--tag", default=None, help="run name (default <dataset>_<preset>_s<seed>)")
    g.add_argument("--overwrite", action="store_true", help="retrain over an existing finished run")
    args = ap.parse_args(argv)

    if args.kd and not args.teacher_ckpt:
        ap.error("--kd needs --teacher-ckpt")
    tag = args.tag or f"{args.dataset}_{args.preset}_s{args.seed}" + ("_kd" if args.kd else "")
    outdir = os.path.join(args.out_dir, tag)
    log_path = os.path.join(outdir, "train_log.json")
    if os.path.exists(log_path) and not args.overwrite:
        prev = json.load(open(log_path))
        if prev.get("finished"):
            print(f"[{tag}] finished run exists (best val CER {100*prev['best_val_cer']:.2f} "
                  f"at epoch {prev['best_epoch']}); pass --overwrite to retrain", flush=True)
            return prev
    os.makedirs(outdir, exist_ok=True)
    set_seed(args.seed, args.deterministic)
    device = torch.device("cpu" if args.gpu < 0 or not torch.cuda.is_available() else f"cuda:{args.gpu}")

    charset = D.build_charset(args.dataset, args.data_dir)       # train split only
    if args.init_ckpt and args.keep_init_charset:
        # per-corpus fine-tuning of a multi-corpus model keeps the model's output layer;
        # its charset was itself built from training transcriptions only
        charset = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)["charset"]
    tok = D.Tokenizer(charset)
    fold = None
    if args.fold:
        k, K = (int(x) for x in args.fold.split("/"))
        fold = ("train", k, K)
    tr = D.LineDataset(args.dataset, "train", tok, augment=args.augment, seed=args.seed,
                       aug_set=args.aug_set, normalisation=args.normalisation,
                       root=args.data_dir, limit=args.limit, fold=fold)
    va = D.LineDataset(args.dataset, "valid", tok, normalisation=args.normalisation,
                       root=args.data_dir, limit=args.limit)
    gen = torch.Generator()
    gen.manual_seed(args.seed)
    syn = None
    if args.synthetic:
        syn = D.LineDataset(args.synthetic, "train", tok, augment=args.augment, seed=args.seed + 7919,
                            aug_set=args.aug_set, normalisation=args.normalisation)
        if args.synth_only:
            tr = syn
    syn_rng = np.random.default_rng(args.seed)
    # Algorithm 3 (ACP). The paper names five stages "from basic character recognition to full
    # complexity" but not their content; the declared reading (documented in docs/FINAL_REPORT_T5.md):
    # stage -> (max line length as a quantile of TRAINING label lengths, augmentation set, synthetic on)
    ACP_STAGES = {1: (0.25, "base", False), 2: (0.50, "base", False), 3: (0.75, "base", True),
                  4: (1.00, "base", True), 5: (1.00, args.aug_set, True)}
    acp = dict(stage=1, T=args.acp_t0, since=0) if args.curriculum == "acp" else None
    lengths = np.array([len(t) for t in tr.labels]) if acp else None

    def make_loader(ep):
        """Eq. 33 / Alg. 2: n_s = |D| r / (1 - r) synthetic lines, redrawn every epoch."""
        ds = tr
        use_syn = syn is not None and not args.synth_only
        if acp:
            q, aug, syn_on = ACP_STAGES[acp["stage"]]
            keep = np.where(lengths <= np.quantile(lengths, q))[0]
            tr.aug_set = aug
            if syn is not None:
                syn.aug_set = aug
            ds = torch.utils.data.Subset(tr, keep.tolist())
            use_syn = use_syn and syn_on
        if use_syn:
            r = SY.synthetic_ratio(ep, args.epochs, *args.synth_ratio)
            n_s = min(SY.n_synthetic(len(ds), r), len(syn))
            idx = syn_rng.choice(len(syn), n_s, replace=False)
            ds = torch.utils.data.ConcatDataset([ds, torch.utils.data.Subset(syn, idx.tolist())])
        return DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                          pin_memory=device.type == "cuda", collate_fn=D.collate, generator=gen,
                          worker_init_fn=worker_init)   # workers are re-forked each epoch so set_epoch reaches them
    trl = make_loader(0)
    val = DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                     pin_memory=device.type == "cuda", collate_fn=D.collate)

    override = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        override[k] = json.loads(v) if v[:1] in "[{0123456789-tf" else v
    input_size = tuple(int(x) for x in va.images.shape[1:])
    model = M.build(args.preset, tok.vocab_size, input_size=input_size, dropout=args.dropout,
                    dropout_cnn=args.dropout_cnn, dropout_rnn=args.dropout_rnn, **override).to(device)
    n_total, n_deployed = M.count_params(model), model.deployed_parameters()

    teacher, teacher_ck = None, None
    if args.teacher_ckpt:
        teacher, teacher_ck = load_teacher(args.teacher_ckpt, device)
        if teacher_ck["charset"] != charset:
            raise SystemExit("teacher charset differs from this dataset's training charset")
        if args.init_from_teacher:
            own, n_loaded = model.state_dict(), 0
            for k, v in teacher.state_dict().items():
                if k in own and own[k].shape == v.shape:
                    own[k] = v.clone()
                    n_loaded += 1
            model.load_state_dict(own)
            print(f"[{tag}] warm-started {n_loaded} tensors from the teacher", flush=True)
    if args.init_ckpt:
        src = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        if src.get("charset") != charset:
            raise SystemExit("--init-ckpt charset differs from this dataset's training charset")
        own, n_loaded = model.state_dict(), 0
        for k, v in src["model_state_dict"].items():
            if k in own and own[k].shape == v.shape:
                own[k] = v.to(own[k].dtype)
                n_loaded += 1
        model.load_state_dict(own)
        print(f"[{tag}] warm-started {n_loaded}/{len(own)} tensors from {args.init_ckpt}", flush=True)
    print(f"[{tag}] {args.preset}: {n_total:,} parameters ({n_deployed:,} deployed), "
          f"T={model.n_frames}, vocab={tok.vocab_size}, device={device}"
          + (f", teacher {M.count_params(teacher):,}" if teacher else ""), flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.sched == "cosine":
        import math
        wu, tot_ep, floor = args.warmup_epochs, max(args.epochs, 1), args.min_lr_frac

        def _lr(ep):
            if ep < wu:
                return (ep + 1) / max(wu, 1)
            t = (ep - wu) / max(tot_ep - wu, 1)
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr)
    else:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=args.plateau_factor,
                                                           patience=args.plateau_patience, min_lr=args.min_lr)
    ema = EMA(model, args.ema) if args.ema else None
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if args.amp and device.type == "cuda" \
        else contextlib.nullcontext

    hist, best_cer, best_epoch = [], float("inf"), -1
    es_best, es_bad = float("inf"), 0
    t0 = time.time()
    log = dict(tag=tag, args=vars(args), model_cfg=model.cfg, params_total=n_total,
               params_deployed=n_deployed, params_teacher=(M.count_params(teacher) if teacher else None),
               charset=charset, vocab_size=tok.vocab_size, n_train=len(tr), n_valid=len(va),
               environment=environment(), finished=False, history=hist)
    for ep in range(args.epochs):
        model.train()
        tr.set_epoch(ep)
        if syn is not None:
            syn.set_epoch(ep)
        if ep > 0 and ((syn is not None and not args.synth_only) or acp):
            trl = make_loader(ep)
        w = L.weights(ep, args.epochs, tuple(args.alpha), tuple(args.gamma), args.aux_weight)
        if not args.kd:                       # teacher: L = L_ctc + delta L_aux (gamma = 0, alpha = 1)
            w = dict(alpha=1.0, beta=0.0, gamma=0.0, delta=args.aux_weight)
        sums = dict(loss=0.0, ctc=0.0, kd=0.0, aux=0.0)
        n_batches, n_infeasible = 0, 0
        for img, tgt, tlen in trl:
            img, tgt, tlen = img.to(device, non_blocking=True), tgt.to(device), tlen.to(device)
            opt.zero_grad(set_to_none=True)
            with amp():
                logits, aux_logits = model(img, return_aux=True)
            l_ctc = L.ctc_loss(logits, tgt, tlen, tok.BLANK)
            loss = w["alpha"] * l_ctc
            l_aux = l_kd = None
            if aux_logits is not None and w["delta"] > 0:
                l_aux = L.ctc_loss(aux_logits, tgt, tlen, tok.BLANK)
                loss = loss + w["delta"] * l_aux
            if args.kd:
                with torch.no_grad(), amp():
                    z_t = teacher(img)
                l_kd = L.kd_loss(L.align_student_logits(logits, z_t.size(1)), z_t, args.tau)
                loss = loss + w["gamma"] * l_kd
            loss.backward()
            if args.clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            if ema is not None:
                ema.update(model)
            n_batches += 1
            n_infeasible += L.ctc_infeasible(tgt, tlen, logits.size(1))
            sums["loss"] += loss.item()
            sums["ctc"] += l_ctc.item()
            sums["aux"] += l_aux.item() if l_aux is not None else 0.0
            sums["kd"] += l_kd.item() if l_kd is not None else 0.0
        vloss, vcer, vwer = evaluate(model, val, va.labels, tok, device, tok.BLANK)
        kept, ema_cer = "raw", None
        if ema is not None:
            with ema.applied(model):
                evl, ecer, ewer = evaluate(model, val, va.labels, tok, device, tok.BLANK)
            ema_cer = ecer
            if ecer < vcer:
                vloss, vcer, vwer, kept = evl, ecer, ewer, "ema"
        if args.sched == "plateau":
            sched.step(vloss)
        else:
            sched.step()
        rec = dict(epoch=ep, **{f"train_{k}": v / max(n_batches, 1) for k, v in sums.items()},
                   ctc_infeasible=n_infeasible, val_loss=vloss, val_cer=vcer, val_wer=vwer,
                   val_cer_ema=ema_cer, kept=kept,
                   lr=opt.param_groups[0]["lr"], weights=w, elapsed_s=time.time() - t0)
        if acp:
            rec["acp_stage"], rec["acp_T"] = acp["stage"], acp["T"]
            acp["since"] += 1
            if acp["stage"] < 5 and (1.0 - vcer > acp["T"] or acp["since"] >= args.acp_max_stage_epochs):
                acp["stage"] += 1; acp["T"] += args.acp_dt; acp["since"] = 0     # Alg. 3 lines 5-7
        hist.append(rec)
        star = ""
        if vcer < best_cer:
            best_cer, best_epoch, star = vcer, ep, " *"
            state = ema.state_dict() if kept == "ema" else model.state_dict()
            torch.save(dict(model_state_dict=state, model_cfg=model.cfg, preset=args.preset,
                            dataset=args.dataset, seed=args.seed, epoch=ep, val_cer=vcer, val_loss=vloss,
                            charset=charset, vocab_size=tok.vocab_size, normalisation=args.normalisation,
                            input_size=list(input_size), args=vars(args),
                            teacher_ckpt=args.teacher_ckpt, result_class="HTRJAND_RUN"),
                       os.path.join(outdir, "best.pth"))
        es_value = vloss if args.es_metric == "loss" else vcer
        if es_value < es_best - args.min_delta:
            es_best, es_bad = es_value, 0
        else:
            es_bad += 1
        print(f"[{tag}] ep{ep:3d} loss={rec['train_loss']:.4f} ctc={rec['train_ctc']:.4f} "
              f"kd={rec['train_kd']:.4f} aux={rec['train_aux']:.4f} val={vloss:.4f} "
              f"cer={100*vcer:.2f} wer={100*vwer:.2f} lr={rec['lr']:.1e}{star}", flush=True)
        log.update(best_val_cer=best_cer, best_epoch=best_epoch, epochs_run=ep + 1,
                   elapsed_s=time.time() - t0)
        json.dump(log, open(log_path, "w"), indent=1)
        if args.patience and es_bad >= args.patience:
            print(f"[{tag}] early stop: no val {args.es_metric} improvement > {args.min_delta} "
                  f"for {args.patience} epochs", flush=True)
            break
    log["finished"] = True
    json.dump(log, open(log_path, "w"), indent=1)
    print(f"[{tag}] done: best val CER {100*best_cer:.2f} at epoch {best_epoch}, "
          f"{(time.time()-t0)/60:.1f} min", flush=True)
    return log


if __name__ == "__main__":
    main()

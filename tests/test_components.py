"""Scientific correctness tests of the HTR-JAND components.
CPU only, no archives needed:  python -m pytest tests -q
"""
import math
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from htrjand import data as D          # noqa: E402
from htrjand import losses as L        # noqa: E402
from htrjand import metrics as Me      # noqa: E402
from htrjand import models as M        # noqa: E402

torch.manual_seed(0)


# ------------------------------------------------------------------ blocks
def test_full_gated_conv_is_eq6():
    """(W1 * X) ⊙ σ(W2 * X): recompute from the two halves of the kernel."""
    g = M.FullGatedConv2d(3, 5)
    x = torch.randn(2, 3, 12, 20)
    w, b = g.conv.weight, g.conv.bias
    lin = F.conv2d(x, w[:5], b[:5], padding=1)
    gate = torch.sigmoid(F.conv2d(x, w[5:], b[5:], padding=1))
    assert g(x).shape == (2, 5, 12, 20)
    assert torch.allclose(g(x), lin * gate, atol=1e-6)


def test_se_block_is_eq5_and_preserves_shape():
    se = M.SEBlock(16, reduction=4)
    x = torch.randn(2, 16, 7, 9)
    y = se(x)
    assert y.shape == x.shape
    s = se.fc(x.mean(dim=(2, 3)))
    assert torch.allclose(y, x * s[:, :, None, None])
    assert (s >= 0).all() and (s <= 1).all()


def test_conv_block_order_matches_eq4():
    blk = M.conv_block(1, 8, (2, 1), gated=True, se=True)
    types = [type(m) for m in blk]
    assert types == [M.FullGatedConv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, M.SEBlock]
    assert blk(torch.randn(1, 1, 10, 16)).shape == (1, 8, 5, 16)      # height-only pooling


# ------------------------------------------------------------------ attention
class _HistoricalAaren(nn.Module):
    """Verbatim O(T^2) form of src/mainTrainer_test__.py:427."""

    def __init__(self, src):
        super().__init__()
        self.src = src

    def forward(self, x):
        s = self.src
        b, l, _ = x.size()
        k = s.k_proj(x).view(b, l, s.heads, s.head_dim).transpose(1, 2)
        v = s.v_proj(x).view(b, l, s.heads, s.head_dim).transpose(1, 2)
        q = s.q.expand(b, -1, l, -1)
        w = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / (s.head_dim ** 0.5), dim=-1)
        c = torch.matmul(w, v).transpose(1, 2).contiguous().view(b, l, s.d)
        return s.out_proj(c)


def test_proxima_matches_historical_form_and_is_constant_over_time():
    p = M.ProximaAttention(32, 2)
    x = torch.randn(3, 40, 32)
    o, w = p(x)
    assert o.shape == (3, 40, 32) and w.shape == (3, 2, 40)
    assert torch.allclose(o, _HistoricalAaren(p)(x), atol=1e-6)
    assert torch.allclose(o[:, 0], o[:, 23])             # one summary broadcast to all frames
    assert torch.allclose(w.sum(-1), torch.ones(3, 2))


def test_combined_attention_is_eq18():
    """LN(W_f [O_MHA ; O_Proxima] + X) with dropout off."""
    ca = M.CombinedAttention(32, 2, dropout=0.0, fusion="paper").eval()
    x = torch.randn(2, 11, 32)
    out, (wa, wp) = ca(x)
    a, _ = ca.mha(x)
    p, _ = ca.prox(x)
    ref = ca.norm(ca.W_f(torch.cat([a, p], -1)) + x)
    assert torch.allclose(out, ref, atol=1e-6)
    assert out.shape == x.shape and wa.shape == (2, 11, 11)


def test_mha_head_count_and_dims():
    mha = M.MultiHeadSelfAttention(64, 2)
    assert mha.attn.num_heads == 2 and mha.attn.head_dim == 32


# ------------------------------------------------------------------ network
@pytest.mark.parametrize("preset", sorted(M.PRESETS))
def test_every_preset_builds_runs_and_backpropagates(preset):
    m = M.build(preset, 50)
    x = torch.randn(2, 1, 68, 864)
    y, aux, w = m(x, return_aux=True, return_attention=True)
    assert y.shape == (2, 108, 50)
    assert aux is None or aux.shape == (2, 108, 50)
    y.sum().backward()
    assert all(p.grad is not None for n, p in m.named_parameters() if not n.startswith("aux."))


def test_paper_teacher_and_student_geometry():
    t, s = M.build("paper_teacher", 104), M.build("paper_student", 104)
    assert t.cfg["channels"] == [32, 64, 128, 256, 512] and s.cfg["channels"] == [16, 32, 64]
    assert t.cfg["heads"] == 2 and s.cfg["heads"] == 1
    assert t.d_model == 256 and s.d_model == 128           # hidden 128 vs 64 per direction
    assert t.n_frames == s.n_frames == 108                 # logit interpolation is the identity
    assert t.feature_rows == 2 and t.column_dim == 1024    # 108 x 1024 columns after the reshape
    assert M.count_params(s) < M.count_params(t)


def test_historical_teacher_parameter_count():
    """2,634,544 parameters at the historical 112-symbol vocabulary (model_summary.txt)."""
    assert M.count_params(M.build("historical_teacher", 112)) == 2_634_544


def test_construction_leaves_batchnorm_statistics_untouched():
    m = M.build("paper_teacher", 50)
    bn = m.cnn[0][1]
    assert int(bn.num_batches_tracked) == 0 and torch.allclose(bn.running_var, torch.ones_like(bn.running_var))


def test_checkpoint_roundtrip(tmp_path):
    m = M.build("paper_student", 30)
    ck = dict(model_state_dict=m.state_dict(), model_cfg=m.cfg, preset="paper_student",
              dataset="x", charset="abc", seed=1)
    torch.save(ck, tmp_path / "best.pth")
    m2, ck2 = M.from_checkpoint(str(tmp_path / "best.pth"))
    x = torch.randn(1, 1, 68, 864)
    m.eval()
    assert torch.allclose(m(x), m2(x))


# ------------------------------------------------------------------ losses / KD
def test_kd_direction_is_kl_teacher_to_student_times_tau2():
    zs, zt, tau = torch.randn(2, 5, 7), torch.randn(2, 5, 7), 3.0
    pt, ps = F.softmax(zt / tau, -1), F.softmax(zs / tau, -1)
    by_hand = (pt * (pt.log() - ps.log())).sum(-1).mean() * tau ** 2
    assert torch.allclose(L.kd_loss(zs, zt, tau), by_hand, atol=1e-5)
    assert L.kd_loss(zt, zt, tau).abs() < 1e-6


def test_teacher_receives_no_gradient_from_kd():
    teacher, student = M.build("paper_teacher", 20), M.build("paper_student", 20)
    for p in teacher.parameters():
        p.requires_grad_(False)
    x = torch.randn(1, 1, 68, 864)
    with torch.no_grad():
        zt = teacher(x)
    loss = L.kd_loss(student(x), zt, 2.0)
    loss.backward()
    assert all(p.grad is None for p in teacher.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters())


def test_logit_interpolation_eq35():
    z = torch.randn(2, 54, 9)
    assert L.align_student_logits(z, 54) is z
    assert L.align_student_logits(z, 108).shape == (2, 108, 9)


def test_loss_weight_schedule_section_iv_a():
    w0, w1 = L.weights(0, 100), L.weights(99, 100)
    assert math.isclose(w0["alpha"], 0.7) and math.isclose(w0["gamma"], 0.2) and w0["delta"] == 0.1
    assert math.isclose(w1["alpha"], 0.4) and math.isclose(w1["gamma"], 0.5) and w1["delta"] == 0.1
    assert abs(w0["beta"]) < 1e-9 and abs(w1["beta"]) < 1e-9          # beta is zero at both ends


def test_ctc_loss_and_infeasibility_count():
    tok = D.Tokenizer("ab")
    logits = torch.randn(1, 4, tok.vocab_size)
    tgt = torch.tensor([tok.encode("aab")])
    assert L.ctc_loss(logits, tgt, torch.tensor([3]), tok.BLANK) > 0
    assert L.ctc_infeasible(tgt, torch.tensor([3]), 4) == 0           # needs 4 frames
    assert L.ctc_infeasible(tgt, torch.tensor([3]), 3) == 1


# ------------------------------------------------------------------ tokenizer / decoding / metrics
def test_tokenizer_roundtrip_dedicated_blank_and_unk():
    tok = D.Tokenizer("hello wrd")
    assert tok.decode(tok.encode("hello world")) == "hello world"
    assert tok.BLANK == tok.vocab_size - 1 and tok.BLANK not in tok.index_to_char
    assert tok.decode(tok.encode("hex")) == "he�"


def test_greedy_decode_collapses_repeats_then_blanks():
    tok = D.Tokenizer("ab")
    a, b, bl = tok.char_to_index["a"], tok.char_to_index["b"], tok.BLANK
    seq = [a, a, bl, a, b, b, bl, bl, b]
    logits = torch.full((1, len(seq), tok.vocab_size), -10.0)
    for t, i in enumerate(seq):
        logits[0, t, i] = 10.0
    assert tok.ctc_greedy_decode(logits) == ["aabb"]


def test_metrics_are_corpus_aggregated():
    refs, hyps = ["abcd", "x"], ["abcd", "y"]
    assert math.isclose(Me.cer(refs, hyps), 1 / 5)                  # not mean(0, 1) = 0.5
    assert Me.wer(["a b", "c"], ["a b", "d"]) == 1 / 3
    assert Me.ser(refs, hyps) == 0.5


def test_normalisation_eq1_range():
    img = np.array([[0, 128, 255]], dtype=np.uint8)
    y = D.normalise(img, "paper")
    assert y.min() == -1.0 and y.max() == 1.0
    assert D.normalise(img, "unit").max() == 1.0


# ------------------------------------------------------------------ synthetic data (synthetic-line generation, synthetic-ratio schedule)
def test_synthetic_ratio_and_count():
    from htrjand import synthetic as S
    assert S.synthetic_ratio(0, 100) == 0.1 and math.isclose(S.synthetic_ratio(100, 100), 0.4)
    assert S.n_synthetic(1000, 0.2) == 250                    # n = |D| r / (1 - r)


def test_synthetic_lines_have_archive_geometry(tmp_path):
    import glob
    fonts = glob.glob("/usr/share/fonts/**/DejaVuSans.ttf", recursive=True) or \
        glob.glob("/usr/share/fonts/**/*.ttf", recursive=True)
    if not fonts:
        pytest.skip("no system font")
    from htrjand import synthetic as S
    imgs, texts = S.generate("abc de", os.path.dirname(fonts[0]), 2, seed=1)
    assert imgs.shape == (2, 68, 864) and imgs.dtype == np.uint8 and len(texts) == 2
    assert imgs.mean() < 128                                  # stored ink-bright on dark


# ------------------------------------------------------------------ tuning options
@pytest.mark.parametrize("aug_set", D.AUG_SETS)
def test_every_augmentation_set_keeps_geometry_and_range(aug_set):
    rng = np.random.default_rng(0)
    img = np.zeros((68, 864), np.float32)
    img[20:40, 100:700] = 1.0
    out = D.augment(img, rng, aug_set)
    assert out.shape == (68, 864) and out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_maxpool_column_reduction_shapes():
    m = M.build("paper_teacher", 40, flatten="maxpool")
    assert m.column_dim == 512 and m.cfg["flatten"] == "maxpool"
    assert m(torch.randn(1, 1, 68, 864)).shape == (1, 108, 40)


# ------------------------------------------------------------------ post-processing (T5 corrector, lexicon-based correction)
def test_confusions_and_corruption_follow_observed_errors():
    from htrjand import postprocess as P
    st = P.confusions([("hallo world", "hello world"), ("helo wrld", "hello world")])
    assert st["subs"]["e"]["a"] == 1
    assert sum(st["dels"].values()) + sum(sum(c.values()) for c in st["subs"].values()) >= 3
    assert 0 < st["rate"] < 1
    import random
    rng = random.Random(0)
    out = [P.corrupt("hello world", st, rng, rate=0.5) for _ in range(20)]
    assert any(o != "hello world" for o in out)


def test_lexicon_is_train_only_and_conservative():
    from htrjand import postprocess as P
    lex = P.Lexicon(["the quick brown fox", "the lazy dog", "the fox"])
    assert lex.correct("the quic fox") == "the quick fox"           # distance-1 fix
    assert lex.correct("the zebra") == "the zebra"                   # no candidate -> unchanged
    assert lex.correct("the fox.") == "the fox."                     # punctuation kept
    assert lex.correct("a 1234") == "a 1234"                         # digits and 1-letter words untouched

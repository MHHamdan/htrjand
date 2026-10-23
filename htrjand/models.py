"""HTR-JAND network: gated CNN + SE -> BiLSTM -> Combined
Attention (multi-head self-attention + Proxima attention) -> CTC head, with a
training-only auxiliary head.

Every choice the method description does not fix is a keyword argument whose default is declared, with
its evidence class, in configs/htrjand_base.yaml.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FullGatedConv2d(nn.Module):
    """gated convolution: FullGatedConv2d(X) = (W1 * X) ⊙ σ(W2 * X).

    One convolution with 2·out channels, split in half: the first half is the
    linear path W1, the second the gate W2. This is the same parametrisation as
    the earlier implementation of this project and Bluche's / the
    Keras FullGatedConv2D the paper cites.
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 2, kernel_size, stride, padding)
        self.out_ch = out_ch

    def forward(self, x):
        a, b = self.conv(x).split(self.out_ch, dim=1)
        return a * torch.sigmoid(b)


class SEBlock(nn.Module):
    """SE block: f · σ(W2 ReLU(W1 GAP(f))). The reduction ratio r is not in the
    paper; 16 is the historical teacher's, 8 the historical student's."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc = nn.Sequential(nn.Linear(channels, hidden, bias=False), nn.ReLU(inplace=True),
                                nn.Linear(hidden, channels, bias=False), nn.Sigmoid())

    def forward(self, x):
        s = self.fc(x.mean(dim=(2, 3)))
        return x * s[:, :, None, None]


def conv_block(in_ch, out_ch, pool, gated=True, se=True, se_reduction=16, dropout=0.0):
    """CNN block: f_l = SE(MaxPool(ReLU(BN(W_l * f_{l-1} + b_l)))), with W_l the
    gated convolution when `gated` (the paper's model) or a plain
    3x3 convolution (the historical checkpoints, ablation `conv: plain`).
    `pool` is (kh, kw); (2, 1) is height-only pooling."""
    layers = [FullGatedConv2d(in_ch, out_ch) if gated else nn.Conv2d(in_ch, out_ch, 3, 1, 1),
              nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.MaxPool2d(pool, pool)]
    if se:
        layers.append(SEBlock(out_ch, se_reduction))
    if dropout:
        layers.append(nn.Dropout2d(dropout))
    return nn.Sequential(*layers)


class MultiHeadSelfAttention(nn.Module):
    """multi-head self-attention: h heads of softmax(QK^T/sqrt(d_k))V, concatenated, W^O."""

    def __init__(self, d, heads, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        o, w = self.attn(x, x, x, need_weights=True, average_attn_weights=True)
        return o, w                                   # (B, T, d), (B, T, T)


class ProximaAttention(nn.Module):
    """"Proxima Attention" as it exists in every historical implementation
    (the earlier implementations of this project): Proxima attention
    K = X W^K, V = X W^V, and a single LEARNED query vector per head that does
    not depend on the position. The softmax runs over the whole line, so the
    result is one content-weighted summary of V per head, broadcast to every
    frame; it is global pooling at O(T) cost, not a local or proximity-based
    operator, and there is no "dynamic query update" in any version of the
    code. The historical O(T^2) form (query expanded to T identical rows) is
    numerically identical; `tests/test_components.py` checks that.
    """

    def __init__(self, d, heads):
        super().__init__()
        self.d, self.heads, self.head_dim = d, heads, d // heads
        self.q = nn.Parameter(torch.randn(1, heads, 1, self.head_dim))
        self.k_proj, self.v_proj, self.out_proj = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)

    def forward(self, x):
        B, T, _ = x.shape
        k = self.k_proj(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)   # (B, h, T, dk)
        v = self.v_proj(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        q = self.q.expand(B, -1, 1, -1)                                           # (B, h, 1, dk)
        w = F.softmax(q @ k.transpose(-2, -1) / self.head_dim ** 0.5, dim=-1)       # (B, h, 1, T)
        ctx = (w @ v).transpose(1, 2).reshape(B, 1, self.d)                        # (B, 1, d)
        return self.out_proj(ctx).expand(B, T, self.d), w.squeeze(2)              # (B, T, d), (B, h, T)


class CombinedAttention(nn.Module):
    """Combined Attention fusion: O = LayerNorm(W_f [O_MHA ; O_Proxima] + X).

    fusion="paper": the equation as written (one residual, one LayerNorm).
    fusion="historical": the form in the historical code and in the 2026
    reimplementation, LN(X + W_f[LN(X + MHA(X)) ; LN(X + Proxima(X))]), i.e.
    each branch carries its own post-norm residual before the fusion.
    """

    def __init__(self, d, heads, dropout=0.0, fusion="paper"):
        super().__init__()
        if fusion not in ("paper", "historical"):
            raise ValueError(fusion)
        self.fusion = fusion
        self.mha = MultiHeadSelfAttention(d, heads, dropout)
        self.prox = ProximaAttention(d, heads)
        self.W_f = nn.Linear(2 * d, d)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)
        if fusion == "historical":
            self.norm_mha, self.norm_prox = nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, x):
        a, wa = self.mha(x)
        p, wp = self.prox(x)
        if self.fusion == "historical":
            a = self.norm_mha(x + self.drop(a))
            p = self.norm_prox(x + self.drop(p))
        out = self.norm(self.drop(self.W_f(torch.cat([a, p], dim=-1))) + x)
        return out, (wa, wp)


class HTRJAND(nn.Module):
    """The HTR-JAND network.

    image (B, 1, 68, 864)
      -> CNN blocks (gated conv, BN, ReLU, pooling, SE) (B, C_L, H_f, T)
      -> columns: one vector per frame               (B, T, C_L * H_f)
      -> [attention here when attn_position="before_lstm"]
      -> BiLSTM x lstm_layers, LayerNorm after each  (B, T, 2 * lstm_hidden)
      -> Combined Attention (MHSA + Proxima, fusion)   (B, T, 2 * lstm_hidden)
      -> dropout -> Linear -> CTC logits             (B, T, V)
      -> auxiliary dense head on the CNN columns     (B, T, V)   [training only]
    """

    def __init__(self, vocab_size, input_size=(68, 864), channels=(32, 64, 128, 256, 512),
                 pools=((2, 2), (2, 2), (2, 2), (2, 1), (2, 1)), gated=True, se=True,
                 se_reduction=16, lstm_hidden=64, lstm_layers=4, heads=2,
                 attn="combined", attn_position="after_lstm", fusion="paper",
                 dropout=0.01, dropout_cnn=0.0, dropout_rnn=0.0, aux_head=True,
                 flatten="concat"):
        super().__init__()
        if flatten not in ("concat", "maxpool"):
            raise ValueError(flatten)
        if len(channels) != len(pools):
            raise ValueError("one pooling kernel per CNN block")
        if attn not in ("combined", "mha", "proxima", "none"):
            raise ValueError(attn)
        if attn_position not in ("after_lstm", "before_lstm"):
            raise ValueError(attn_position)
        self.cfg = dict(vocab_size=vocab_size, input_size=list(input_size), channels=list(channels),
                        pools=[list(p) for p in pools], gated=gated, se=se, se_reduction=se_reduction,
                        lstm_hidden=lstm_hidden, lstm_layers=lstm_layers, heads=heads, attn=attn,
                        attn_position=attn_position, fusion=fusion, dropout=dropout,
                        dropout_cnn=dropout_cnn, dropout_rnn=dropout_rnn, aux_head=aux_head,
                        flatten=flatten)
        blocks, in_ch = [], 1
        for out_ch, pool in zip(channels, pools):
            blocks.append(conv_block(in_ch, out_ch, tuple(pool), gated, se, se_reduction, dropout_cnn))
            in_ch = out_ch
        self.cnn = nn.Sequential(*blocks)

        # geometry from the declared input, computed on a detached dummy so no
        # BatchNorm statistic is touched (the modules are in eval mode here)
        self.cnn.eval()
        with torch.no_grad():
            f = self.cnn(torch.zeros(1, 1, *input_size))
        self.cnn.train()
        self.feature_rows, self.n_frames = int(f.shape[2]), int(f.shape[3])
        # "concat" stacks the surviving feature rows into one vector per column
        # (historical); "maxpool" keeps the column maximum over rows, a
        # vertical-translation-invariant reduction (Retsinas et al., 2022) that
        # is the one architecture-level option outside the paper's description
        self.flatten = flatten
        self.column_dim = int(f.shape[1] * f.shape[2]) if flatten == "concat" else int(f.shape[1])
        d = 2 * lstm_hidden
        self.d_model = d

        self.attention, self.attn_in = None, None
        attn_dim = self.column_dim if attn_position == "before_lstm" else d
        if attn == "combined":
            self.attention = CombinedAttention(attn_dim, heads, dropout, fusion)
        elif attn == "mha":
            self.attention = nn.ModuleDict(dict(mha=MultiHeadSelfAttention(attn_dim, heads, dropout),
                                                norm=nn.LayerNorm(attn_dim)))
        elif attn == "proxima":
            self.attention = nn.ModuleDict(dict(prox=ProximaAttention(attn_dim, heads),
                                                norm=nn.LayerNorm(attn_dim)))

        self.lstms = nn.ModuleList([nn.LSTM(self.column_dim if i == 0 else d, lstm_hidden, 1,
                                            batch_first=True, bidirectional=True)
                                    for i in range(lstm_layers)])
        self.lstm_norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(lstm_layers)])
        self.rnn_drop = nn.Dropout(dropout_rnn) if dropout_rnn else None
        self.dropout_fc = nn.Dropout(dropout)
        self.fc = nn.Linear(d, vocab_size)
        # "Dense+Soft+Auxiliary" (architecture figure legend): a dense classifier on the CNN
        # columns, trained with its own CTC term (auxiliary-loss reading, A-17)
        self.aux = nn.Linear(self.column_dim, vocab_size) if aux_head else None

    # ------------------------------------------------------------------ helpers
    def _attend(self, x):
        if self.attention is None:
            return x, None
        if isinstance(self.attention, CombinedAttention):
            return self.attention(x)
        if "mha" in self.attention:
            o, w = self.attention["mha"](x)
        else:
            o, w = self.attention["prox"](x)
        return self.attention["norm"](x + self.dropout_fc(o)), w

    def columns(self, x):
        f = self.cnn(x)                                          # (B, C, H_f, T)
        if self.flatten == "maxpool":
            return f.max(dim=2).values.transpose(1, 2)           # (B, T, C)
        return f.permute(0, 3, 1, 2).reshape(f.shape[0], f.shape[3], -1)   # (B, T, C*H_f)

    def forward(self, x, return_aux=False, return_attention=False):
        cols = self.columns(x)
        h, w = cols, None
        if self.cfg["attn_position"] == "before_lstm":
            h, w = self._attend(h)
        for k, (lstm, norm) in enumerate(zip(self.lstms, self.lstm_norms)):
            h, _ = lstm(h)
            h = norm(h)
            if self.rnn_drop is not None and k < len(self.lstms) - 1:
                h = self.rnn_drop(h)
        if self.cfg["attn_position"] == "after_lstm":
            h, w = self._attend(h)
        logits = self.fc(self.dropout_fc(h))
        out = [logits]
        if return_aux:
            out.append(self.aux(cols) if self.aux is not None else None)
        if return_attention:
            out.append(w)
        return out[0] if len(out) == 1 else tuple(out)

    def deployed_parameters(self):
        """Parameters that exist at inference: everything but the auxiliary head."""
        n = sum(p.numel() for p in self.parameters())
        return n - (sum(p.numel() for p in self.aux.parameters()) if self.aux is not None else 0)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- presets
_T = dict(channels=(32, 64, 128, 256, 512), pools=((2, 2), (2, 2), (2, 2), (2, 1), (2, 1)))
_S = dict(channels=(16, 32, 64), pools=((2, 2), (2, 2), (2, 2)), se_reduction=8)

PRESETS = {
    # The Teacher as the method describes it: five gated+SE
    # blocks from 32 channels, four BiLSTM layers of hidden size 128 per
    # direction ("128" read as nn.LSTM hidden size; 256-d attention), two heads,
    # combined attention after the recurrent stage (text order), Combined Attention fusion.
    "paper_teacher":  dict(**_T, gated=True, se=True, lstm_hidden=128, lstm_layers=4, heads=2,
                           attn="combined", attn_position="after_lstm", fusion="paper", aux_head=True),
    # Same under the historical config's reading of "128" as the bidirectional
    # output (64 per direction, 128-d attention), i.e. the width of the
    # historical checkpoints. Decision D-001 in docs/DECISION_LOG.md.
    "paper_teacher_d128": dict(**_T, gated=True, se=True, lstm_hidden=64, lstm_layers=4, heads=2,
                               attn="combined", attn_position="after_lstm", fusion="paper", aux_head=True),
    # The Student: three blocks from 16 channels, one head,
    # LSTM hidden size 64 per direction ("64 instead of 128"); same depth.
    "paper_student":  dict(**_S, gated=True, se=True, lstm_hidden=64, lstm_layers=4, heads=1,
                           attn="combined", attn_position="after_lstm", fusion="paper", aux_head=True),
    "paper_student_d64": dict(**_S, gated=True, se=True, lstm_hidden=32, lstm_layers=4, heads=1,
                              attn="combined", attn_position="after_lstm", fusion="paper", aux_head=True),
    # The network that produced the historical checkpoints
    # (2,634,544 parameters at V = 112):
    # plain convolutions, no auxiliary head, nested-LN fusion.
    "historical_teacher": dict(**_T, gated=False, se=True, lstm_hidden=64, lstm_layers=4, heads=2,
                               attn="combined", attn_position="after_lstm", fusion="historical",
                               aux_head=False),
    # The earlier Student class of this project: plain
    # convolutions, SE r = 8, attention on the CNN columns before the LSTMs.
    "historical_student": dict(**_S, gated=False, se=True, lstm_hidden=64, lstm_layers=3, heads=1,
                               attn="combined", attn_position="before_lstm", fusion="historical",
                               aux_head=True),
    # The released 128 x 1632 networks (input_size=(128, 1632)): the Teacher is paper_teacher; the
    # Student is paper_student with a (4, 2) third pool, which keeps 8 feature rows x 64 channels =
    # 512-d columns, 204 frames and the same 806,287 deployed parameters. The training queues pass
    # `--preset paper_student --set pools=[[2,2],[2,2],[4,2]]`, which builds the same network.
    "htrjand_teacher": dict(**_T, gated=True, se=True, lstm_hidden=128, lstm_layers=4, heads=2,
                            attn="combined", attn_position="after_lstm", fusion="paper", aux_head=True),
    "htrjand_student": dict(channels=(16, 32, 64), pools=((2, 2), (2, 2), (4, 2)), se_reduction=8,
                            gated=True, se=True, lstm_hidden=64, lstm_layers=4, heads=1,
                            attn="combined", attn_position="after_lstm", fusion="paper", aux_head=True),
    # Component ablations of the Teacher (all else equal)
    "abl_plain_conv":   dict(**_T, gated=False, se=True,  lstm_hidden=64, lstm_layers=4, heads=2, attn="combined"),
    "abl_no_se":        dict(**_T, gated=True,  se=False, lstm_hidden=64, lstm_layers=4, heads=2, attn="combined"),
    "abl_mha_only":     dict(**_T, gated=True,  se=True,  lstm_hidden=64, lstm_layers=4, heads=2, attn="mha"),
    "abl_proxima_only": dict(**_T, gated=True,  se=True,  lstm_hidden=64, lstm_layers=4, heads=2, attn="proxima"),
    "abl_no_attention": dict(**_T, gated=True,  se=True,  lstm_hidden=64, lstm_layers=4, heads=2, attn="none"),
    "abl_cnn_lstm":     dict(**_T, gated=False, se=False, lstm_hidden=64, lstm_layers=4, heads=2, attn="none"),
}


def build(preset, vocab_size, input_size=(68, 864), **override):
    cfg = dict(PRESETS[preset])
    cfg.update(override)
    return HTRJAND(vocab_size, input_size=input_size, **cfg)


def from_checkpoint(path, map_location="cpu"):
    """Rebuild a model from a checkpoint written by this package. Returns
    (model, checkpoint dict). The checkpoint carries the full constructor
    configuration, so no archive or preset table is needed to load it."""
    ck = torch.load(path, map_location=map_location, weights_only=False)
    cfg = dict(ck["model_cfg"])
    model = HTRJAND(**cfg)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, ck

"""Loss terms of HTR-JAND (CTC, KD, auxiliary CTC) and the loss-weight schedule.

    L_total = alpha * L_ctc + beta * L_ce + gamma * L_kd + delta * L_aux      (total loss)

* L_ctc  (CTC loss): `torch.nn.CTCLoss` on log-softmax frames.
* L_kd   (KD loss, with the tau^2 factor):
             tau^2 * KL( softmax(z_T / tau) || softmax(z_S / tau) )
         averaged over every (sample, frame). PyTorch's `kl_div(input=log P_S,
         target=P_T)` computes KL(P_T || P_S), the Teacher-to-Student direction
         of the KD loss (P = Teacher, Q = Student).
* z_S^interp (logit interpolation): the student logits are linearly interpolated along the
         time axis to the teacher's frame count when the two differ. For the
         presets in this package both emit T = W/8 frames, so it is the identity.
* L_aux  (auxiliary loss): a cross-entropy on an auxiliary classifier would need
         per-frame labels, which CTC does not have. The only alignment-free
         reading is a CTC loss on the auxiliary head, and that is what is
         implemented (assumption A-17).
* L_ce   (frame-wise cross-entropy): needs an alignment that is never defined,
         and the loss weights (alpha + gamma + delta = 1 at both ends of the
         schedule) give it beta = 0. It is not implemented; `weights()` reports
         beta for completeness and it is always 0 under these values.

Weight schedule: alpha 0.7 -> 0.4 and gamma 0.2 -> 0.5 move
linearly with the epoch fraction; delta = 0.1 throughout; beta is the remainder.
"""
import torch
import torch.nn.functional as F


def ctc_loss(logits, targets, target_lengths, blank, reduction="mean"):
    """logits (B, T, V) in any dtype; the CTC dynamic programme runs in fp32."""
    lp = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)          # (T, B, V)
    il = torch.full((logits.size(0),), logits.size(1), dtype=torch.long,
                    device=logits.device)
    return F.ctc_loss(lp, targets, il, target_lengths, blank=blank,
                      reduction=reduction, zero_infinity=True)


def ctc_infeasible(targets, target_lengths, n_frames):
    """Number of targets CTC cannot align in `n_frames` frames: a target needs
    len(y) + (#adjacent repeated characters) frames. `zero_infinity=True`
    silently zeroes their loss, so the trainer counts them instead of hiding them."""
    bad = 0
    for y, n in zip(targets, target_lengths):
        y = y[: int(n)]
        need = int(n) + int((y[1:] == y[:-1]).sum()) if int(n) > 1 else int(n)
        bad += int(need > n_frames)
    return bad


def align_student_logits(z_s, n_frames_teacher):
    """logit interpolation: Interpolate(z_S, len(z_T)) along the time axis (linear)."""
    if z_s.size(1) == n_frames_teacher:
        return z_s
    return F.interpolate(z_s.transpose(1, 2), size=n_frames_teacher,
                         mode="linear", align_corners=False).transpose(1, 2)


def kd_loss(z_s, z_t, tau):
    """tau^2 * mean_{b,t} KL( softmax(z_T/tau) || softmax(z_S/tau) ).

    `z_t` is used detached: the teacher never receives a gradient.
    The mean is over samples x frames, not samples alone: with T = 108 frames
    `reduction="batchmean"` on the (B, T, V) tensor would leave the term 108
    times larger than the CTC loss (audit B3).
    """
    z_t = z_t.detach()
    B, T, V = z_s.shape
    log_p_s = F.log_softmax(z_s.reshape(B * T, V).float() / tau, dim=-1)
    p_t = F.softmax(z_t.reshape(B * T, V).float() / tau, dim=-1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (tau ** 2)


def weights(epoch, total_epochs, alpha=(0.7, 0.4), gamma=(0.2, 0.5), delta=0.1):
    """Loss weights at `epoch` (0-based) of `total_epochs`.

    Returns dict(alpha, beta, gamma, delta). beta = 1 - alpha - gamma - delta,
    clipped at zero; with the paper's values it is exactly 0 at both ends and
    never positive.
    """
    f = 0.0 if total_epochs <= 1 else min(max(epoch / (total_epochs - 1), 0.0), 1.0)
    a = alpha[0] + f * (alpha[1] - alpha[0])
    g = gamma[0] + f * (gamma[1] - gamma[0])
    b = max(0.0, 1.0 - a - g - delta)
    return dict(alpha=a, beta=b, gamma=g, delta=delta)

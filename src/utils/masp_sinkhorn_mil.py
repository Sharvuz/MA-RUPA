"""
Modality-Adaptive Self-Paced Sinkhorn Multiple-Instance Learning Loss (MASP-Sinkhorn-MIL).
Theoretical Prescription: Intern-S2 (Shanghai AI Lab, System-2 Reasoning)
PyTorch Code Surgery: DeepSeek v4 (Elite Principal PyTorch Coder)
Engineering Integration: Antigravity
"""

import math
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class MASPSinkhornMILLoss(nn.Module):
    """Modality-Adaptive Self-Paced Sinkhorn Multiple-Instance Learning Loss (MASP-MIL).
    
    Addresses weak supervision noise and audio-visual modality conflict on XD-Violence:
    1. Formulates snippet anomaly assignment as an Optimal Transport problem with cost:
       C = [-log(1 - p), -log p] in R^{B x T x 2}
    2. Enforces asymmetric marginal targets:
       - Normal videos: rho_neg = 1e-3 (forces all snippets to normal)
       - Anomalous videos: rho_pos = 0.50 (targets upper 50% violent snippets without background contamination)
    3. Differentiable Sinkhorn log-kernel solver (T=30 iterations) providing stable, non-vanishing gradient
       back to both the multi-scale temporal backbone and modality fusion gates.
    4. Adaptive Entropy-Confidence Weighting across temporal scales.
    """

    def __init__(
        self,
        num_scales: int = 1,
        tau: float = 0.10,
        sinkhorn_iters: int = 30,
        pos_mass: float = 0.50,
        neg_mass: float = 1e-3,
        label_smoothing: float = 0.0,
        adaptive_scales: str = "entropy",   # "entropy" | "learnable" | "none"
        scale_temperature: float = 0.50,
        sparse_reg: float = 0.0,
        video_bce_weight: float = 0.0,
        eps: float = 1e-8,
        reduction: str = "mean",
    ):
        super().__init__()
        assert adaptive_scales in ("entropy", "learnable", "none")
        self.num_scales = num_scales
        self.tau = tau
        self.sinkhorn_iters = sinkhorn_iters
        self.pos_mass = pos_mass
        self.neg_mass = neg_mass
        self.label_smoothing = label_smoothing
        self.adaptive_scales = adaptive_scales
        self.scale_temperature = scale_temperature
        self.sparse_reg = sparse_reg
        self.video_bce_weight = video_bce_weight
        self.eps = eps
        self.reduction = reduction

        if adaptive_scales == "learnable" and num_scales > 1:
            self.scale_logits = nn.Parameter(torch.zeros(num_scales))
        else:
            self.register_parameter("scale_logits", None)

    def _build_ot(self, z: torch.Tensor, labels: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Constructs cost matrix C and marginal targets a, b."""
        # z: [B, T] snippet logits
        neg_log_p = -F.logsigmoid(z)         # -log(p) where p = prob anomaly
        neg_log_q = -F.logsigmoid(-z)        # -log(1 - p)
        C = torch.stack([neg_log_q, neg_log_p], dim=-1)  # [B, T, 2]
        B, T, _ = C.shape

        if mask is None:
            a = torch.full((B, T), 1.0 / T, device=z.device, dtype=z.dtype)
        else:
            m = mask.float()
            a = m / m.sum(dim=1, keepdim=True).clamp_min(1.0)
            a = a.clamp_min(self.eps)
            a = a / a.sum(dim=1, keepdim=True)

        # Column marginals (Target Class Mass)
        y = labels
        if self.label_smoothing > 0:
            y = y * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        rho_pos = self.pos_mass
        rho_neg = self.neg_mass
        b1 = y * (1.0 - rho_pos) + (1.0 - y) * (1.0 - rho_neg)
        b2 = y * rho_pos + (1.0 - y) * rho_neg
        b = torch.stack([b1, b2], dim=-1).clamp_min(self.eps)
        b = b / b.sum(dim=-1, keepdim=True)

        p = torch.sigmoid(z)
        return C, a, b, p

    def _sinkhorn(self, C: torch.Tensor, a: torch.Tensor, b: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Differentiable Log-domain Sinkhorn-Knopp algorithm."""
        K = -C / self.tau  # [B, T, 2]
        if mask is not None:
            K = K.masked_fill((~mask.bool()).unsqueeze(-1), -1e9)

        log_a = torch.log(a.clamp_min(self.eps))
        log_b = torch.log(b.clamp_min(self.eps))

        u = torch.zeros_like(a)
        v = torch.zeros_like(b)

        for _ in range(self.sinkhorn_iters):
            u = log_a - torch.logsumexp(K + v.unsqueeze(1), dim=2)
            v = log_b - torch.logsumexp(K + u.unsqueeze(2), dim=1)

        logP = K + u.unsqueeze(2) + v.unsqueeze(1)
        P = torch.exp(logP)
        return P, logP

    def _plan_confidence(self, P: torch.Tensor) -> torch.Tensor:
        """Measures sharpness/confidence of the positive anomaly transport plan."""
        # P: [B, T, 2]; analyze positive assignment column P[..., 1]
        m = P[..., 1]
        m = m / m.sum(dim=1, keepdim=True).clamp_min(self.eps)
        H = -(m * torch.log(m.clamp_min(self.eps))).sum(dim=1)
        T = max(2, m.shape[1])
        conf = 1.0 - H / math.log(T)  # in [0, 1], 1 = sharp, 0 = uniform/ambiguous
        return conf.clamp(0.0, 1.0)

    def forward(
        self,
        snippet_logits: Union[torch.Tensor, Sequence[torch.Tensor]],
        video_labels: torch.Tensor,
        masks: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
        return_scores: bool = False,
    ):
        """
        Args:
            snippet_logits: [B, T] or [B, T, 1] snippet logit tensor, or list of multi-scale tensors.
            video_labels: [B] binary labels (0 = normal, 1 = anomaly/violent).
            masks: [B, T] valid sequence boolean mask.
            return_scores: whether to also return predicted video-level scores.
        """
        if isinstance(snippet_logits, torch.Tensor):
            snippet_logits = [snippet_logits]

        B = video_labels.shape[0]
        labels = video_labels.float().view(-1)
        losses, weights, scores = [], [], []

        for s, z in enumerate(snippet_logits):
            if z.dim() == 3 and z.size(-1) == 1:
                z = z.squeeze(-1)
            z = z.float()

            mask = None
            if masks is not None:
                mask = masks[s] if isinstance(masks, (list, tuple)) else masks

            C, a, b, p = self._build_ot(z, labels, mask)
            P, _ = self._sinkhorn(C, a, b, mask)

            # OT assignment loss: <P, C>
            cost = (P * C).sum(dim=(1, 2))  # [B]
            losses.append(cost)

            # Video score from anomalous transport mass
            pos_col = P[..., 1]
            pos_mass = pos_col.sum(dim=1, keepdim=True).clamp_min(self.eps)
            v_score = (pos_col * p).sum(dim=1, keepdim=True) / pos_mass
            scores.append(v_score.squeeze(1))

            # Adaptive scale confidence
            with torch.no_grad():
                conf = self._plan_confidence(P)
            weights.append(conf)

        loss_stack = torch.stack(losses, dim=1)   # [B, S]
        score_stack = torch.stack(scores, dim=1)  # [B, S]

        # Multi-scale aggregation
        if len(losses) == 1:
            per_video_loss = loss_stack.squeeze(1)
            final_scores = score_stack.squeeze(1)
        elif self.adaptive_scales == "learnable" and self.scale_logits is not None:
            w = torch.softmax(self.scale_logits, dim=0)
            per_video_loss = (loss_stack * w.unsqueeze(0)).sum(dim=1)
            final_scores = (score_stack * w.unsqueeze(0)).sum(dim=1)
        elif self.adaptive_scales == "entropy":
            conf = torch.stack(weights, dim=1)  # [B, S]
            w = torch.softmax(conf / max(1e-4, self.scale_temperature), dim=1)
            per_video_loss = (loss_stack * w).sum(dim=1)
            final_scores = (score_stack * w).sum(dim=1)
        else:
            per_video_loss = loss_stack.mean(dim=1)
            final_scores = score_stack.mean(dim=1)

        # Video-level auxiliary BCE loss
        if self.video_bce_weight > 0.0:
            bce_loss = F.binary_cross_entropy(final_scores.clamp(self.eps, 1.0 - self.eps), labels)
            per_video_loss = per_video_loss + self.video_bce_weight * bce_loss

        if self.reduction == "mean":
            total_loss = per_video_loss.mean()
        elif self.reduction == "sum":
            total_loss = per_video_loss.sum()
        else:
            total_loss = per_video_loss

        if return_scores:
            return total_loss, final_scores
        return total_loss


def compute_masp_sinkhorn_mil_loss(
    snippet_logits: torch.Tensor,
    video_labels: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
    tau: float = 0.10,
    sinkhorn_iters: int = 30,
    pos_mass: float = 0.50,
) -> torch.Tensor:
    """Functional wrapper for MASP-Sinkhorn-MIL Loss."""
    criterion = MASPSinkhornMILLoss(
        tau=tau,
        sinkhorn_iters=sinkhorn_iters,
        pos_mass=pos_mass,
    )
    mask = None
    if lengths is not None:
        T = snippet_logits.shape[1]
        device = snippet_logits.device
        mask = torch.arange(T, device=device)[None, :] < lengths[:, None]
    return criterion(snippet_logits, video_labels, masks=mask)

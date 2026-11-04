"""Clean, Principled SOTA Loss Suite for MA-RUPA v3 (WS-VAD).

Ground-up clean mathematical formulation eliminating conflicting auxiliary losses:
  1. CLAS2: Top-k Multiple Instance Binary Cross-Entropy (Detector & Safe-Routing).
  2. CLASM: Top-k Multi-Class Vision-Language Alignment Cross-Entropy (logits2).
  3. RUPA_RESIDUAL_EVENT: Event-level Residual Semantic Alignment on Abnormal Bags (logits3).
  4. CLASM_BKG: Background-level Normal Semantic Alignment across All Bags (logits4).
  5. text_orthogonality_loss: Enforces Normal prompt orthogonality against all Anomaly classes.
  6. ConsistencyLoss & routing_anchor_loss: Stabilizes Counterfactual Prototype Reconstruction.
  7. edge_preserving_smoothness_loss: Charbonnier boundary-gated smoothness (DeSC / EPTS).
  8. audio_guided_mil_loss: Multimodal MIL + Normal-Bag Acoustic False-Alarm Suppression (XD-Violence).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler


class WarmCosineScheduler(_LRScheduler):
    """Linear warmup followed by cosine decay learning rate scheduler."""

    def __init__(self, optimizer, base_value: float, final_value: float,
                 total_iters: int, warmup_iters: int = 0, start_warmup_value: float = 0.0):
        self.final_value = final_value
        self.total_iters = max(1, int(total_iters))
        warmup_iters = max(0, min(int(warmup_iters), self.total_iters - 1))

        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
        rem_iters = np.arange(max(1, self.total_iters - warmup_iters))
        cosine_schedule = final_value + 0.5 * (base_value - final_value) * (
            1.0 + np.cos(np.pi * rem_iters / len(rem_iters))
        )
        self.schedule = np.concatenate((warmup_schedule, cosine_schedule))
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= len(self.schedule):
            return [self.final_value for _ in self.base_lrs]
        return [float(self.schedule[self.last_epoch]) for _ in self.base_lrs]


class ModelEMA:
    """Lightweight Polyak Exponential Moving Average of trainable weights for rock-solid evaluation."""

    def __init__(self, model: nn.Module, decay: float = 0.996):
        self.decay = decay
        src = model.module if isinstance(model, nn.DataParallel) else model
        self.shadow = {
            k: v.detach().clone()
            for k, v in src.state_dict().items()
            if v.is_floating_point() and not k.startswith('clip_adapter.clipmodel.')
        }

    @torch.no_grad()
    def update(self, model: nn.Module):
        src = model.module if isinstance(model, nn.DataParallel) else model
        for k, v in src.state_dict().items():
            if k in self.shadow and v.is_floating_point():
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> dict:
        """Temporarily swaps EMA weights into model and returns backup of live weights."""
        target = model.module if isinstance(model, nn.DataParallel) else model
        backup = {}
        state = target.state_dict()
        for k, ema_v in self.shadow.items():
            if k in state:
                backup[k] = state[k].detach().clone()
                state[k].copy_(ema_v.to(state[k].device, dtype=state[k].dtype))
        return backup

    @torch.no_grad()
    def restore(self, model: nn.Module, backup: dict):
        target = model.module if isinstance(model, nn.DataParallel) else model
        state = target.state_dict()
        for k, orig_v in backup.items():
            if k in state:
                state[k].copy_(orig_v.to(state[k].device, dtype=state[k].dtype))


def CLAS2(logits: torch.Tensor, labels: torch.Tensor, lengths: torch.Tensor,
          device: torch.device = None, topk_ratio: int = 16) -> torch.Tensor:
    """Top-k Binary Multiple-Instance Learning Cross-Entropy Loss."""
    dev = logits.device if device is None else device
    probs = torch.sigmoid(logits.float()).reshape(logits.shape[0], logits.shape[1])
    bag_labels = (1.0 - labels[:, 0].float()).reshape(-1).to(dev)

    bag_preds = []
    T = probs.shape[1]
    for i in range(probs.shape[0]):
        l_i = max(1, min(int(lengths[i]), T))
        k_i = max(1, min(int(l_i // topk_ratio + 1), l_i))
        top_vals, _ = torch.topk(probs[i, :l_i], k=k_i, largest=True)
        bag_preds.append(top_vals.mean())

    instance_probs = torch.stack(bag_preds, dim=0).clamp(1e-6, 1.0 - 1e-6)
    return F.binary_cross_entropy(instance_probs, bag_labels)


def CLASM(logits: torch.Tensor, labels: torch.Tensor, lengths: torch.Tensor,
          device: torch.device = None, topk_ratio: int = 16) -> torch.Tensor:
    """Top-k Multi-Class Vision-Language Alignment MIL Loss."""
    dev = logits.device if device is None else device
    logits_f = logits.float()
    norm_labels = (labels.float() / labels.float().sum(dim=1, keepdim=True).clamp_min(1e-6)).to(dev)

    bag_logits = []
    T = logits_f.shape[1]
    for i in range(logits_f.shape[0]):
        l_i = max(1, min(int(lengths[i]), T))
        k_i = max(1, min(int(l_i // topk_ratio + 1), l_i))
        top_vals, _ = torch.topk(logits_f[i, :l_i], k=k_i, largest=True, dim=0)
        bag_logits.append(top_vals.mean(dim=0))

    instance_logits = torch.stack(bag_logits, dim=0)
    return -torch.mean(torch.sum(norm_labels * F.log_softmax(instance_logits, dim=1), dim=1))


def CLASM_EVENT(logits: torch.Tensor, labels: torch.Tensor, lengths: torch.Tensor,
                device: torch.device = None, epsilon: float = 0.1) -> torch.Tensor:
    """Video-level Event Alignment Loss with label smoothing."""
    dev = logits.device if device is None else device
    logits_f = logits.float()
    num_classes = logits_f.shape[2]
    labels_f = labels.float().to(dev)
    labels_sm = (1.0 - epsilon) * (labels_f / labels_f.sum(dim=1, keepdim=True).clamp_min(1e-6)) + epsilon / num_classes

    bag_logits = []
    T = logits_f.shape[1]
    for i in range(logits_f.shape[0]):
        l_i = max(1, min(int(lengths[i]), T))
        top_vals, _ = torch.topk(logits_f[i, :l_i], k=1, largest=True, dim=0)
        bag_logits.append(top_vals.mean(dim=0))

    instance_logits = torch.stack(bag_logits, dim=0)
    return -torch.mean(torch.sum(labels_sm * F.log_softmax(instance_logits, dim=1), dim=1))


def RUPA_RESIDUAL_EVENT(logits: torch.Tensor, labels: torch.Tensor, lengths: torch.Tensor,
                        device: torch.device = None) -> torch.Tensor:
    """Aligns counterfactual residual features strictly on bags carrying an anomaly label."""
    anomaly_mask = labels[:, 1:].sum(dim=1) > 0
    if not torch.any(anomaly_mask):
        return logits.sum() * 0.0
    return CLASM_EVENT(logits[anomaly_mask], labels[anomaly_mask], lengths[anomaly_mask], device=device)


def CLASM_BKG(logits: torch.Tensor, labels: torch.Tensor, lengths: torch.Tensor,
              device: torch.device = None, epsilon: float = 0.1) -> torch.Tensor:
    """Aligns reconstructed normal background features with the 'normal' class across all videos."""
    dev = logits.device if device is None else device
    logits_f = logits.float()
    B, T, num_classes = logits_f.shape

    bkg_target = torch.full((B, num_classes), 0.01, device=dev, dtype=torch.float32)
    bkg_target[:, 0] = 1.0
    bkg_target = (1.0 - epsilon) * (bkg_target / bkg_target.sum(dim=1, keepdim=True)) + epsilon / num_classes

    bag_logits = []
    for i in range(B):
        l_i = max(1, min(int(lengths[i]), T))
        top_vals, _ = torch.topk(logits_f[i, :l_i], k=1, largest=True, dim=0)
        bag_logits.append(top_vals.mean(dim=0))

    instance_logits = torch.stack(bag_logits, dim=0)
    return -torch.mean(torch.sum(bkg_target * F.log_softmax(instance_logits, dim=1), dim=1))


def text_orthogonality_loss(text_features: torch.Tensor) -> torch.Tensor:
    """Enforces strict orthogonality between normalized 'Normal' prompt and all Anomaly prompts."""
    t_norm = F.normalize(text_features.float(), dim=-1)
    t_normal = t_norm[0]     # [D]
    t_abn = t_norm[1:]       # [C-1, D]
    return torch.abs(t_abn @ t_normal).mean()


class ConsistencyLoss(nn.Module):
    """Valid-masked MSE aligning detector probabilities with prototype reconstruction error."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, logits1: torch.Tensor, original_features: torch.Tensor,
                reconstructed_features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        recon_err = ((1.0 - F.cosine_similarity(original_features.float(), reconstructed_features.float(), dim=-1)) / 2.0).clamp(0.0, 1.0)
        det_prob = torch.sigmoid(logits1.float().squeeze(-1))
        B, N = det_prob.shape
        mask = torch.arange(N, device=logits1.device)[None, :] < lengths.to(logits1.device)[:, None]
        if not torch.any(mask):
            return logits1.sum() * 0.0
        return self.mse(det_prob[mask], recon_err[mask])


def routing_anchor_loss(dnp_dict: dict, lengths: torch.Tensor) -> torch.Tensor:
    """Continuation anchor keeping Safe-Gated routing anchored to reliable detector scores."""
    if 'routing_scores' not in dnp_dict or 'detector_scores' not in dnp_dict:
        return torch.tensor(0.0)
    routing_scores = dnp_dict['routing_scores'].float().squeeze(-1)
    detector_scores = dnp_dict['detector_scores'].detach().float().squeeze(-1)
    mask = torch.arange(routing_scores.shape[1], device=routing_scores.device)[None, :] < lengths.to(routing_scores.device)[:, None]
    if not torch.any(mask):
        return routing_scores.sum() * 0.0
    return F.mse_loss(routing_scores[mask], detector_scores[mask])


def edge_preserving_smoothness_loss(scores: torch.Tensor, lengths: torch.Tensor,
                                    eps: float = 1e-3, edge_sigma: float = 0.25) -> torch.Tensor:
    """Charbonnier Boundary-Gated Smoothness (EPTS / DeSC Principle).

    Smooths noisy background fluctuations while automatically turning off (`gate -> 0`)
    at large temporal jumps (`|s_{t+1} - s_t| > edge_sigma`) so explosive onset/offset
    peaks (Shooting, Explosion, RoadAccidents) are never clipped.
    """
    s = scores.float().squeeze(-1)  # [B, T]
    B, T = s.shape
    if T < 2:
        return s.sum() * 0.0

    diff = s[:, 1:] - s[:, :-1]  # [B, T-1]
    valid_pair = torch.arange(1, T, device=s.device)[None, :] < lengths.to(s.device)[:, None]
    if not torch.any(valid_pair):
        return s.sum() * 0.0

    with torch.no_grad():
        boundary_gate = torch.exp(-(diff ** 2) / (2.0 * (edge_sigma ** 2)))

    charbonnier = torch.sqrt(diff ** 2 + eps ** 2) - eps
    gated_penalty = boundary_gate * charbonnier
    return gated_penalty[valid_pair].mean()


def audio_guided_mil_loss(aux_logits: torch.Tensor, labels: torch.Tensor,
                          lengths: torch.Tensor, topk_ratio: int = 16,
                          normal_fa_weight: float = 0.50) -> torch.Tensor:
    """Trains the Decoupled Cross-Modal Audio-Visual Branch with MIL + Normal Acoustic Suppression.

    On XD-Violence, Normal movie/sports clips (`label_A`, e.g., The Bourne Ultimatum) often contain
    loud soundtrack drums, cheering, or dramatic SFX. Explicitly penalizing the top-8 acoustic peaks
    on Normal bags prevents false-alarm precision collapse while keeping visual features 100% pure.
    """
    if aux_logits is None:
        return torch.tensor(0.0)
    mil_loss = CLAS2(aux_logits, labels, lengths, device=aux_logits.device, topk_ratio=topk_ratio)

    normal_bag_mask = labels[:, 0] > 0.5
    if torch.any(normal_bag_mask):
        probs_norm = torch.sigmoid(aux_logits[normal_bag_mask].float().squeeze(-1))
        lens_norm = lengths[normal_bag_mask]
        T = probs_norm.shape[1]
        peaks = []
        for i in range(probs_norm.shape[0]):
            l_i = max(1, min(int(lens_norm[i]), T))
            k_i = max(1, min(8, l_i))
            peaks.append(torch.topk(probs_norm[i, :l_i], k=k_i).values.mean())
        fa_loss = torch.stack(peaks).mean()
        return mil_loss + normal_fa_weight * fa_loss

    return mil_loss

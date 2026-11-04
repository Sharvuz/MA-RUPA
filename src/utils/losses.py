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


def text_orthogonality_loss(text_features: torch.Tensor, num_classes: int = None) -> torch.Tensor:
    """Enforces strict orthogonality between normalized 'Normal' prompt and all Anomaly prompts.

    Handles `torch.nn.DataParallel` multi-GPU gathering automatically: when `DataParallel`
    concatenates `[C, 512]` from `G` GPUs into `[G * C, 512]`, slices the first `[C, 512]`
    replica so GPU 1's 'Normal' vector is never mistaken for an Anomaly class.
    """
    t_norm = F.normalize(text_features.float(), dim=-1)
    if num_classes is not None and t_norm.shape[0] > num_classes:
        t_norm = t_norm[:num_classes]
    elif t_norm.shape[0] > 1:
        # Auto-detect DataParallel replica boundary where GPU 1's Normal prompt repeats GPU 0's Normal prompt
        cos_with_normal = t_norm[1:] @ t_norm[0]
        dup_idx = torch.where(cos_with_normal > 0.9995)[0]
        if dup_idx.numel() > 0:
            c_inferred = int(dup_idx[0].item()) + 1
            if c_inferred > 1 and t_norm.shape[0] % c_inferred == 0:
                t_norm = t_norm[:c_inferred]

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
                          normal_fa_weight: float = 0.50,
                          aux_sem_logits: torch.Tensor = None) -> torch.Tensor:
    """Trains the Decoupled Cross-Modal Audio-Visual Branch with MIL + Semantic + Normal Suppression.

    On XD-Violence, Normal movie/sports clips (`label_A`, e.g., The Bourne Ultimatum) often contain
    loud soundtrack drums, cheering, or dramatic SFX. Explicitly penalizing the top-8 acoustic peaks
    on Normal bags prevents false-alarm precision collapse while keeping visual features 100% pure.
    When `aux_sem_logits` is provided (MA-RUPA v5 LAS-VAD), also aligns audio-visual features with
    the 7 CLIP+AAT semantic classes via top-k multi-class MIL (`CLASM`).
    """
    if aux_logits is None:
        return torch.tensor(0.0)
    dev = aux_logits.device
    mil_loss = CLAS2(aux_logits, labels, lengths, device=dev, topk_ratio=topk_ratio)

    if aux_sem_logits is not None:
        sem_loss = CLASM(aux_sem_logits, labels, lengths, device=dev, topk_ratio=topk_ratio)
        mil_loss = mil_loss + 0.50 * sem_loss

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


def iam_intention_loss(dnp_dict: dict, labels: torch.Tensor, lengths: torch.Tensor,
                       device: torch.device = None,
                       iam_prototypes: torch.Tensor = None) -> torch.Tensor:
    """LAS-VAD Intention Awareness Mechanism Loss (Sec. 3.2, Eq. 10-11).

    Combines:
      1. Top-k Multi-Class Intention MIL (`CLASM` on `iam_logits`) so intention prototypes
         discriminate behavioral trajectories (Velocity X_v & Acceleration X_a).
      2. Cross-Intention Contrastive Regularization (`L_cst`, Eq. 11) enforcing inter-prototype
         diversity across all C intention classes.
      3. Normal-Bag Kinematic False-Alarm Suppression penalizing isolated high-jerk spikes on
         Normal videos (`labels[:, 0] > 0.5`).
    """
    if 'iam_logits' not in dnp_dict or dnp_dict['iam_logits'] is None:
        return torch.tensor(0.0, device=device)

    iam_logits = dnp_dict['iam_logits']
    dev = iam_logits.device if device is None else device
    loss_iam = CLASM(iam_logits, labels, lengths, device=dev)

    # Cross-Intention Prototype Diversity (LAS-VAD Eq. 11)
    if iam_prototypes is not None:
        num_c = iam_logits.shape[-1]
        z = F.normalize(iam_prototypes.float(), dim=-1)
        if z.shape[0] > num_c:
            z = z[:num_c]
        sim = z @ z.t()
        eye_mask = torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
        loss_cst = F.relu(sim[~eye_mask]).mean() + torch.abs(z[1:] @ z[0]).mean()
        loss_iam = loss_iam + 0.25 * loss_cst

    # Normal-Bag Kinematic Jerk Spike Suppression
    normal_bag_mask = labels[:, 0] > 0.5
    if torch.any(normal_bag_mask) and 'iam_scores' in dnp_dict and dnp_dict['iam_scores'] is not None:
        iam_s_norm = dnp_dict['iam_scores'][normal_bag_mask].float().squeeze(-1)
        lens_norm = lengths[normal_bag_mask]
        T = iam_s_norm.shape[1]
        peaks = []
        for i in range(iam_s_norm.shape[0]):
            l_i = max(1, min(int(lens_norm[i]), T))
            k_i = max(1, min(4, l_i))
            peaks.append(torch.topk(iam_s_norm[i, :l_i], k=k_i).values.mean())
        loss_iam = loss_iam + 0.35 * torch.stack(peaks).mean()

    return loss_iam


def acc_pseudo_label_loss(dnp_dict: dict, lengths: torch.Tensor,
                          logits1: torch.Tensor = None) -> torch.Tensor:
    """LAS-VAD Anomaly-Connected Component Pseudo-Label & Agnostic Consistency Loss (Eq. 6 & Eq. 8).

    Combines:
      1. `L_aux` (Eq. 6): Valid-masked L1 distance between ACC graph-propagated pseudo-labels `g`
         (`acc_pseudo_labels.detach()`) and fine-grained consensus probabilities `p^f`
         (`fine_grained_probs`), enforcing intra-component temporal completeness.
      2. `L_reg` (Eq. 8): Agnostic-aware consistency `|1 - p^f_{i,0} - p^b_{i,1}|` aligning the
         fine-grained non-normal probability with the binary detector probability `p^b`.
    """
    if 'acc_pseudo_labels' not in dnp_dict or 'fine_grained_probs' not in dnp_dict:
        dev = logits1.device if logits1 is not None else None
        return torch.tensor(0.0, device=dev)

    g = dnp_dict['acc_pseudo_labels'].detach().float()   # [B, T, C]
    p_f = dnp_dict['fine_grained_probs'].float()         # [B, T, C]
    B, T, _ = p_f.shape
    dev = p_f.device
    valid_mask = torch.arange(T, device=dev)[None, :] < lengths.to(dev)[:, None]
    if not torch.any(valid_mask):
        return p_f.sum() * 0.0

    # Eq. (6): L_aux = (1 / T_valid) * sum |g_i - p^f_i|_1
    l_aux = F.l1_loss(p_f[valid_mask], g[valid_mask], reduction='mean')

    # Eq. (8): L_reg = (1 / T_valid) * sum |1 - p^f_{i,0} - p^b_{i,1}|
    if logits1 is not None:
        p_b_abn = torch.sigmoid(logits1.float().squeeze(-1))
    elif 'detector_scores' in dnp_dict and dnp_dict['detector_scores'] is not None:
        p_b_abn = dnp_dict['detector_scores'].float().squeeze(-1)
    else:
        return l_aux

    p_f_abn = (1.0 - p_f[:, :, 0]).clamp(0.0, 1.0)
    l_reg = F.l1_loss(p_f_abn[valid_mask], p_b_abn[valid_mask], reduction='mean')
    return l_aux + 0.50 * l_reg

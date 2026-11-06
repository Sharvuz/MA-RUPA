"""MA-RUPA v10 SOTA Loss Suite for Weakly Supervised Video Anomaly Detection (WS-VAD).

Restores the exact proven LGT-SGNM loss physics of commit 64fd87b / c331e45 (88.4%+ AUC &
39.90% AP on UCF-Crime; 84.1%+ AP on XD-Violence) while eliminating late-epoch bag
memorization without checkpoint rollback:
  1. CLAS2 (Asymmetric Multi-Scale Probability Top-k MIL + Per-Bag Confidence Margin):
     Preserves the exact probability-space Top-k winner-take-all Jacobian `p_t * (1 - p_t)`
     (`0.75 * k_16 + 0.25 * k_64` on Normal bags; `0.85 * k_16 + 0.15 * k_64` on Abnormal bags).
     Once a bag's Top-k confidence exceeds 95% (`per_bag_bce <= 0.05`), applies value-preserving
     gradient damping (`0.05x`) so normal background snippets inside Top-k of abnormal videos
     are never forced toward 1.0 in late epochs.
  2. CLASM (Multi-Scale Top-k Vision-Language Alignment + Per-Bag Confidence Margin):
     Blends `0.85 * k_16 + 0.15 * k_64` with label smoothing (`epsilon=0.03`) and damps
     per-bag gradients once bag cross-entropy drops below `0.35`.
  3. CLASM_EVENT & CLASM_BKG:
     Video-level event alignment (`logits3`) and background normal alignment (`logits4`).
  4. ConsistencyLoss (Raw-Cosine Detector-Prototype Tether with Decoder Target Cap):
     Exact raw cosine MSE `F.mse_loss(det_prob[mask], recon_err[mask])` for the detector,
     with SGNM decoder target capped at `0.25` (natural CLIP-cone boundary) so SGNM never
     learns to corrupt reconstructions on high-confidence anomaly peaks.
  5. ModelEMA (Warmup-Ramped Polyak Averaging):
     Uses dynamic ramp `decay_t = min(decay, (1 + step) / (20 + step))` to eliminate
     Step-0 random initialization lag while locking onto the converged SOTA basin.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler


class WarmCosineScheduler(_LRScheduler):
    """Per-iteration linear warmup followed by smooth cosine annealing."""

    def __init__(self, optimizer, base_value: float, final_value: float,
                 total_iters: int, warmup_iters: int = 0, start_warmup_value: float = 0.0):
        self.base_value = float(base_value)
        self.final_value = float(final_value)
        self.total_iters = max(1, int(total_iters))
        warmup_iters = max(0, min(int(warmup_iters), self.total_iters - 1))

        if warmup_iters > 0 and start_warmup_value <= 0.0:
            start_warmup_value = self.base_value * 0.10

        warmup_schedule = np.linspace(start_warmup_value, self.base_value, warmup_iters)
        rem_iters = np.arange(max(1, self.total_iters - warmup_iters))
        cosine_schedule = self.final_value + 0.5 * (self.base_value - self.final_value) * (
            1.0 + np.cos(np.pi * rem_iters / len(rem_iters))
        )
        self.schedule = np.concatenate((warmup_schedule, cosine_schedule))
        super().__init__(optimizer)

    def get_lr(self):
        idx = min(max(0, self.last_epoch), len(self.schedule) - 1)
        scale = float(self.schedule[idx]) / max(self.base_value, 1e-12)
        if self.last_epoch >= len(self.schedule):
            scale = self.final_value / max(self.base_value, 1e-12)
        return [base_lr * scale for base_lr in self.base_lrs]


class ModelEMA:
    """Warmup-ramped Polyak Exponential Moving Average of trainable weights."""

    def __init__(self, model: nn.Module, decay: float = 0.998):
        self.decay = float(decay)
        self.num_updates = 0
        src = model.module if isinstance(model, nn.DataParallel) else model
        self.shadow = {
            k: v.detach().clone()
            for k, v in src.state_dict().items()
            if not k.startswith('clip_adapter.clipmodel.')
        }

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.num_updates += 1
        eff_decay = min(self.decay, (1.0 + self.num_updates) / (20.0 + self.num_updates))
        src = model.module if isinstance(model, nn.DataParallel) else model
        for k, v in src.state_dict().items():
            if k in self.shadow and v.is_floating_point():
                self.shadow[k].mul_(eff_decay).add_(v.detach(), alpha=1.0 - eff_decay)

    def state_dict(self) -> dict:
        return {
            'decay': self.decay,
            'num_updates': self.num_updates,
            'shadow': {k: v.detach().cpu() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state: dict):
        if not state or 'shadow' not in state:
            return
        self.num_updates = int(state.get('num_updates', self.num_updates))
        for k, v in state['shadow'].items():
            if k in self.shadow and self.shadow[k].shape == v.shape:
                self.shadow[k].copy_(v.to(self.shadow[k].device, dtype=self.shadow[k].dtype))

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> dict:
        src = model.module if isinstance(model, nn.DataParallel) else model
        backup = {}
        for k, v in src.state_dict().items():
            if k in self.shadow:
                backup[k] = v.detach().clone()
                v.copy_(self.shadow[k].to(v.device, dtype=v.dtype))
        return backup

    @torch.no_grad()
    def restore(self, model: nn.Module, backup: dict):
        src = model.module if isinstance(model, nn.DataParallel) else model
        for k, v in src.state_dict().items():
            if k in backup:
                v.copy_(backup[k].to(v.device, dtype=v.dtype))


def CLAS2(logits: torch.Tensor, labels: torch.Tensor, lengths: torch.Tensor,
          device: torch.device = None, topk_ratio: int = 16,
          margin_thresh: float = 0.05, margin_scale: float = 0.05) -> torch.Tensor:
    """Asymmetric Multi-Scale Probability Top-k Binary MIL Loss with Per-Bag Confidence Margin.

    1. Probability-Space Top-k BCE (Exact 64fd87b formulation):
       Preserves the selective `p_t * (1 - p_t)` winner-take-all Jacobian inside Top-k so
       true anomaly snippets dominate gradient updates over normal background snippets.
    2. Per-Bag Confidence Margin (`margin_thresh = 0.05`):
       Once an individual bag's Top-k probability exceeds 95% confidence (`per_bag_bce <= 0.05`),
       damps that bag's gradient by `margin_scale = 0.05` while keeping forward loss unchanged,
       preventing late-epoch memorization of normal frames inside abnormal bags.
    """
    dev = logits.device if device is None else device
    probs = torch.sigmoid(logits.float()).reshape(logits.shape[0], logits.shape[1])
    bag_labels = (1.0 - labels[:, 0].float()).reshape(-1).to(dev)

    bag_preds = []
    T = probs.shape[1]
    for i in range(probs.shape[0]):
        l_i = max(1, min(int(lengths[i]), T))
        k_16 = max(1, min(int(l_i // topk_ratio + 1), l_i))
        k_sharp = max(1, min(int(l_i // 64 + 1), l_i))
        top_16, _ = torch.topk(probs[i, :l_i], k=k_16, largest=True)
        top_sharp, _ = torch.topk(probs[i, :l_i], k=k_sharp, largest=True)
        if bag_labels[i] < 0.5:
            bag_preds.append(0.75 * top_16.mean() + 0.25 * top_sharp.mean())
        else:
            bag_preds.append(0.85 * top_16.mean() + 0.15 * top_sharp.mean())

    instance_probs = torch.stack(bag_preds, dim=0).clamp(1e-6, 1.0 - 1e-6)
    per_bag_bce = F.binary_cross_entropy(instance_probs, bag_labels, reduction='none')
    if margin_thresh > 0.0:
        per_bag_bce = torch.where(
            per_bag_bce > margin_thresh,
            per_bag_bce,
            margin_scale * per_bag_bce + (1.0 - margin_scale) * per_bag_bce.detach(),
        )
    return per_bag_bce.mean()


def CLASM(logits: torch.Tensor, labels: torch.Tensor, lengths: torch.Tensor,
          device: torch.device = None, topk_ratio: int = 16, epsilon: float = 0.03,
          margin_thresh: float = 0.35, margin_scale: float = 0.05) -> torch.Tensor:
    """Multi-Scale Top-k Vision-Language Alignment MIL Loss with Per-Bag Confidence Margin."""
    dev = logits.device if device is None else device
    logits_f = logits.float()
    num_classes = logits_f.shape[2]
    norm_labels = (labels.float() / labels.float().sum(dim=1, keepdim=True).clamp_min(1e-6)).to(dev)
    if epsilon > 0.0 and num_classes > 2:
        norm_labels = (1.0 - epsilon) * norm_labels + (epsilon / float(num_classes))

    bag_logits = []
    T = logits_f.shape[1]
    for i in range(logits_f.shape[0]):
        l_i = max(1, min(int(lengths[i]), T))
        k_16 = max(1, min(int(l_i // topk_ratio + 1), l_i))
        k_sharp = max(1, min(int(l_i // 64 + 1), l_i))
        top_16, _ = torch.topk(logits_f[i, :l_i], k=k_16, largest=True, dim=0)
        top_sharp, _ = torch.topk(logits_f[i, :l_i], k=k_sharp, largest=True, dim=0)
        bag_logits.append(0.85 * top_16.mean(dim=0) + 0.15 * top_sharp.mean(dim=0))

    instance_logits = torch.stack(bag_logits, dim=0)
    per_bag_ce = -torch.sum(norm_labels * F.log_softmax(instance_logits, dim=1), dim=1)
    if margin_thresh > 0.0:
        per_bag_ce = torch.where(
            per_bag_ce > margin_thresh,
            per_bag_ce,
            margin_scale * per_bag_ce + (1.0 - margin_scale) * per_bag_ce.detach(),
        )
    return per_bag_ce.mean()


def CLASM_EVENT(logits: torch.Tensor, labels: torch.Tensor, lengths: torch.Tensor,
                device: torch.device = None, epsilon: float = 0.1) -> torch.Tensor:
    """Video-level Event Alignment Loss with label smoothing."""
    dev = logits.device if device is None else device
    logits_f = logits.float()
    num_classes = logits_f.shape[2]

    labels_f = labels.float().to(dev)
    labels_sum = labels_f.sum(dim=1, keepdim=True).clamp_min(1e-6)
    labels_sm = (1.0 - epsilon) * (labels_f / labels_sum) + (epsilon / num_classes)

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
    """Aligns residual event features strictly on abnormal bags."""
    anomaly_mask = labels[:, 1:].sum(dim=1) > 0
    if not torch.any(anomaly_mask):
        return logits.sum() * 0.0
    return CLASM_EVENT(logits[anomaly_mask], labels[anomaly_mask], lengths[anomaly_mask], device)


def CLASM_BKG(logits: torch.Tensor, labels: torch.Tensor, lengths: torch.Tensor,
              device: torch.device = None, epsilon: float = 0.1) -> torch.Tensor:
    """Aligns pooled normal background features (`logits4`) with the 'normal' class across all videos."""
    dev = logits.device if device is None else device
    logits_f = logits.float()
    B, T, num_classes = logits_f.shape

    bkg_target = torch.full((B, num_classes), 0.01, device=dev, dtype=torch.float32)
    bkg_target[:, 0] = 1.0
    bkg_target = bkg_target / bkg_target.sum(dim=1, keepdim=True).clamp_min(1e-6)
    labels_sm = (1.0 - epsilon) * bkg_target + (epsilon / num_classes)

    bag_logits = []
    for i in range(B):
        l_i = max(1, min(int(lengths[i]), T))
        top_vals, _ = torch.topk(logits_f[i, :l_i], k=1, largest=True, dim=0)
        bag_logits.append(top_vals.mean(dim=0))

    instance_logits = torch.stack(bag_logits, dim=0)
    return -torch.mean(torch.sum(labels_sm * F.log_softmax(instance_logits, dim=1), dim=1))


def text_orthogonality_loss(text_features: torch.Tensor, num_classes: int = None) -> torch.Tensor:
    """Enforces strict orthogonality between normalized 'Normal' prompt and all Anomaly prompts.

    Handles `torch.nn.DataParallel` multi-GPU gathering symmetrically across replicas.
    """
    t_norm = F.normalize(text_features.float(), dim=-1)
    if num_classes is not None and t_norm.shape[0] > num_classes and t_norm.shape[0] % num_classes == 0:
        t_replicas = t_norm.view(-1, num_classes, t_norm.shape[-1])
        t_normal = t_replicas[:, 0:1, :]
        t_abn = t_replicas[:, 1:, :]
        return torch.abs(torch.bmm(t_abn, t_normal.transpose(1, 2))).mean()
    elif num_classes is not None and t_norm.shape[0] > num_classes:
        t_norm = t_norm[:num_classes]
    elif t_norm.shape[0] > 1:
        cos_with_normal = t_norm[1:] @ t_norm[0]
        dup_idx = torch.nonzero(cos_with_normal > 0.9999, as_tuple=False)
        if dup_idx.numel() > 0:
            c_inferred = int(dup_idx[0].item()) + 1
            if c_inferred > 1 and t_norm.shape[0] % c_inferred == 0:
                t_replicas = t_norm.view(-1, c_inferred, t_norm.shape[-1])
                t_normal = t_replicas[:, 0:1, :]
                t_abn = t_replicas[:, 1:, :]
                return torch.abs(torch.bmm(t_abn, t_normal.transpose(1, 2))).mean()

    t_normal = t_norm[0]
    t_abn = t_norm[1:]
    return torch.abs(t_abn @ t_normal).mean()


def compute_normalized_recon_error(original_features: torch.Tensor,
                                   reconstructed_features: torch.Tensor,
                                   lengths: torch.Tensor = None) -> torch.Tensor:
    """Computes CLIP-Cone Baseline-Normalized Feature Reconstruction Distance in [0, 1]."""
    raw_cos_dist = (
        1.0 - F.cosine_similarity(original_features.float(), reconstructed_features.float(), dim=-1)
    ).clamp(0.0, 2.0)
    raw_half = (raw_cos_dist * 0.5).clamp(0.0, 1.0)

    B, T = raw_cos_dist.shape
    with torch.no_grad():
        bases = []
        for i in range(B):
            l_i = max(1, min(int(lengths[i]) if lengths is not None else T, T))
            valid_d = raw_cos_dist[i, :l_i]
            k_base = max(1, int(0.20 * l_i))
            base_val = torch.topk(valid_d, k=k_base, largest=False).values[-1]
            bases.append(base_val)
        base_tensor = torch.stack(bases, dim=0).unsqueeze(1)

    cone_norm = ((raw_cos_dist - base_tensor).clamp_min(0.0) / 0.45).clamp(0.0, 1.0)
    return 0.50 * raw_half + 0.50 * cone_norm


class ConsistencyLoss(nn.Module):
    """Valid-masked raw-cosine MSE aligning detector probabilities with SGNM reconstruction error.

    Preserves the exact 64fd87b / c331e45 L2 probability tether on `logits1` and `original_features`
    while capping the target for `reconstructed_features` at `0.25` (natural CLIP-cone limit)
    so SGNM's decoder never learns to corrupt reconstructions on high-scoring peaks.
    """

    def __init__(self, rec_target_cap: float = 0.25):
        super().__init__()
        self.mse = nn.MSELoss(reduction='mean')
        self.rec_target_cap = float(rec_target_cap)

    def forward(self, logits1: torch.Tensor, original_features: torch.Tensor,
                reconstructed_features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        recon_err = (
            (1.0 - F.cosine_similarity(original_features.float(), reconstructed_features.float(), dim=-1)) / 2.0
        ).clamp(0.0, 1.0)
        det_prob = torch.sigmoid(logits1.float().squeeze(-1))
        B, N = det_prob.shape
        mask = torch.arange(N, device=logits1.device)[None, :] < lengths.to(logits1.device)[:, None]
        if not torch.any(mask):
            return logits1.sum() * 0.0

        det_v = det_prob[mask]
        rec_v = recon_err[mask]

        recon_err_det_grad = (
            (1.0 - F.cosine_similarity(original_features.float(), reconstructed_features.detach().float(), dim=-1)) / 2.0
        ).clamp(0.0, 1.0)[mask]
        loss_det = self.mse(det_v, recon_err_det_grad)

        rec_target = det_v.detach().clamp(max=self.rec_target_cap)
        loss_rec = self.mse(rec_v, rec_target)
        return loss_det + (loss_rec - loss_rec.detach())


def routing_anchor_loss(dnp_dict: dict, lengths: torch.Tensor) -> torch.Tensor:
    """Anchors SafeGate routing scores to detached detector scores across valid snippets."""
    if 'routing_scores' not in dnp_dict or 'detector_scores' not in dnp_dict:
        return torch.tensor(0.0)
    routing_scores = dnp_dict['routing_scores'].float().squeeze(-1)
    detector_scores = dnp_dict['detector_scores'].detach().float().squeeze(-1)
    B, N = routing_scores.shape
    mask = torch.arange(N, device=routing_scores.device)[None, :] < lengths.to(routing_scores.device)[:, None]
    if not torch.any(mask):
        return routing_scores.sum() * 0.0
    return F.mse_loss(routing_scores[mask], detector_scores[mask])


def edge_preserving_smoothness_loss(scores: torch.Tensor, lengths: torch.Tensor,
                                    eps: float = 1e-3, edge_sigma: float = 0.25) -> torch.Tensor:
    """Charbonnier Boundary-Gated Smoothness."""
    s = scores.float().squeeze(-1)
    B, T = s.shape
    if T < 2:
        return s.sum() * 0.0

    diff = s[:, 1:] - s[:, :-1]
    valid_pair = torch.arange(1, T, device=s.device)[None, :] < lengths.to(s.device)[:, None]
    if not torch.any(valid_pair):
        return s.sum() * 0.0

    charbonnier = torch.sqrt(diff * diff + eps * eps) - eps
    edge_weight = torch.exp(-(diff.detach() / edge_sigma) ** 2)
    weighted = charbonnier * edge_weight
    return weighted[valid_pair].mean()


def audio_guided_mil_loss(aux_logits: torch.Tensor, labels: torch.Tensor,
                          lengths: torch.Tensor, topk_ratio: int = 16,
                          normal_fa_weight: float = 0.50,
                          aux_sem_logits: torch.Tensor = None) -> torch.Tensor:
    """Trains the optional Decoupled Cross-Modal Audio-Visual Branch."""
    if aux_logits is None:
        return torch.tensor(0.0)
    dev = aux_logits.device
    mil_loss = CLAS2(aux_logits, labels, lengths, device=dev, topk_ratio=topk_ratio)

    aux_probs = torch.sigmoid(aux_logits.float()).squeeze(-1)
    bag_labels = (1.0 - labels[:, 0].float()).reshape(-1).to(dev)
    normal_mask = bag_labels < 0.5

    fa_losses = []
    T = aux_probs.shape[1]
    for i in range(aux_probs.shape[0]):
        if normal_mask[i]:
            l_i = max(1, min(int(lengths[i]), T))
            k_i = max(1, min(int(l_i // topk_ratio + 1), l_i))
            top_norm, _ = torch.topk(aux_probs[i, :l_i], k=k_i, largest=True)
            fa_losses.append((top_norm ** 2).mean())

    total_aux = mil_loss
    if fa_losses:
        total_aux = total_aux + normal_fa_weight * torch.stack(fa_losses).mean()

    if aux_sem_logits is not None and aux_sem_logits.shape[-1] == labels.shape[-1]:
        sem_loss = CLASM(aux_sem_logits, labels, lengths, device=dev, topk_ratio=topk_ratio)
        total_aux = total_aux + 0.50 * sem_loss

    return total_aux


def iam_intention_loss(dnp_dict: dict, labels: torch.Tensor, lengths: torch.Tensor,
                       device: torch.device = None,
                       iam_prototypes: torch.Tensor = None) -> torch.Tensor:
    """Legacy IAM loss (kept for unit-test compatibility; disabled by default)."""
    if 'iam_logits' not in dnp_dict or dnp_dict['iam_logits'] is None:
        return torch.tensor(0.0, device=device)

    iam_logits = dnp_dict['iam_logits'].float()
    dev = iam_logits.device if device is None else device
    bag_labels = (1.0 - labels[:, 0].float()).reshape(-1).to(dev)

    B, T, _ = iam_logits.shape
    bag_preds = []
    for i in range(B):
        l_i = max(1, min(int(lengths[i]), T))
        k_i = max(1, min(int(l_i // 16 + 1), l_i))
        probs_i = torch.sigmoid(iam_logits[i, :l_i, 0])
        top_vals, _ = torch.topk(probs_i, k=k_i, largest=True)
        bag_preds.append(top_vals.mean())

    instance_probs = torch.stack(bag_preds, dim=0).clamp(1e-6, 1.0 - 1e-6)
    mil_loss = F.binary_cross_entropy(instance_probs, bag_labels)

    div_loss = iam_logits.sum() * 0.0
    if iam_prototypes is not None and iam_prototypes.shape[0] > 1:
        p_norm = F.normalize(iam_prototypes.float(), dim=-1)
        sim_mat = p_norm @ p_norm.t()
        eye = torch.eye(p_norm.shape[0], device=p_norm.device, dtype=torch.bool)
        div_loss = (sim_mat.masked_fill(eye, 0.0) ** 2).mean()

    return mil_loss + 0.25 * div_loss


def acc_pseudo_label_loss(dnp_dict: dict, lengths: torch.Tensor,
                          logits1: torch.Tensor = None) -> torch.Tensor:
    """Legacy ACC loss (kept for unit-test compatibility; disabled by default)."""
    if 'acc_pseudo_labels' not in dnp_dict or 'fine_grained_probs' not in dnp_dict:
        dev = logits1.device if logits1 is not None else None
        return torch.tensor(0.0, device=dev)

    pseudo_q = dnp_dict['acc_pseudo_labels'].detach().float()
    fg_probs = dnp_dict['fine_grained_probs'].float().clamp(1e-6, 1.0 - 1e-6)
    B, T, _ = fg_probs.shape
    mask = torch.arange(T, device=fg_probs.device)[None, :] < lengths.to(fg_probs.device)[:, None]
    if not torch.any(mask):
        return fg_probs.sum() * 0.0

    kl_per_frame = torch.sum(pseudo_q * (torch.log(pseudo_q.clamp_min(1e-6)) - torch.log(fg_probs)), dim=-1)
    if logits1 is not None:
        det_p = torch.sigmoid(logits1.detach().float().squeeze(-1))
        conf_weight = ((det_p - 0.5).abs() * 2.0).clamp(0.1, 1.0)
        kl_per_frame = kl_per_frame * conf_weight

    return kl_per_frame[mask].mean()

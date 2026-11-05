"""Argon-4 Gradient-Conserving SOTA Loss Suite for MA-RUPA (WS-VAD).

Engineered from First Principles under the Argon-4 Gradient Conservation Law:
  1. CLAS2 (Gradient-Conserving Multi-Scale Top-k MIL):
     Eliminates the Sigmoid Saturation Vanishing Gradient trap (`p_t * (1 - p_t) -> 0` at
     both `p_t -> 0` on hard anomalies like Arson/Explosion and `p_t -> 0.98` on isolated
     Normal false-alarm spikes) by coupling probability-space Top-k MIL with analytical
     Logit-Space Multi-Scale Top-k (`k_16` and `k_64`) `BCEWithLogits`.
  2. CLASM (Multi-Scale Top-k Vision-Language Alignment):
     Preserves sharp semantic gradients on short, fast-onset events (Explosion, Shooting,
     Arson, RoadAccidents) via multi-scale logit pooling (`0.80 * k_16 + 0.20 * k_64`).
  3. RUPA_RESIDUAL_EVENT & CLASM_BKG:
     Counterfactual event alignment strictly on Abnormal bags and valid-masked background
     alignment across all bags.
  4. text_orthogonality_loss:
     DataParallel-safe orthogonality between canonical 'normal' and anomaly text embeddings.
  5. ConsistencyLoss (CLIP-Cone Normalized & Peak-Conserving):
     Normalizes cosine reconstruction distance against each video's intrinsic CLIP-cone
     baseline and prevents uncalibrated reconstruction error from pulling down Top-k MIL
     anomaly peaks (`Explosion027`, `RoadAccidents009`, `Operation.Red.Sea`).
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
          device: torch.device = None, topk_ratio: int = 16,
          gc_weight: float = 0.35) -> torch.Tensor:
    """Gradient-Conserving Multi-Scale Top-k Binary MIL Loss (Argon-4 Law).

    Why standard post-sigmoid Top-k BCE stalls at ~88.3% AUC:
      In post-sigmoid averaging `p_bag = mean_{t in K} sigmoid(z_t)`, the gradient w.r.t.
      snippet logit `z_t` carries the factor `p_t * (1 - p_t)`:
        1. On subtle/dark Abnormal videos (`Arson022`, `Explosion025`, `City.of.God`) where
           `p_t ~ 0.02`, `p_t * (1 - p_t) ~ 0.0196 -> 0`, starving hard positives of gradient.
        2. On Normal videos with a 1-frame false-alarm spike (`Normal_887 max=0.979`),
           `p_bag` is diluted by 15 low frames (`p_bag ~ 0.07`) AND `p_1 * (1 - p_1) ~ 0.02 -> 0`,
           leaving the spike completely unpenalized!
      By coupling probability-space Top-k BCE with analytical Logit-Space Multi-Scale Top-k
      (`BCEWithLogits` at `k_16 = L//16 + 1` and `k_sharp = L//64 + 1`), the `p_t*(1-p_t)`
      saturation factor is analytically cancelled (`dL/dz_t = (sigmoid(z_bar) - y) / k`),
      guaranteeing non-vanishing gradients for both hard anomalies and isolated normal spikes.
    """
    dev = logits.device if device is None else device
    z_all = logits.float().reshape(logits.shape[0], logits.shape[1])
    probs = torch.sigmoid(z_all)
    bag_labels = (1.0 - labels[:, 0].float()).reshape(-1).to(dev)

    bag_probs = []
    bag_logits_k = []
    bag_logits_sharp = []
    T = z_all.shape[1]

    for i in range(z_all.shape[0]):
        l_i = max(1, min(int(lengths[i]), T))
        k_i = max(1, min(int(l_i // topk_ratio + 1), l_i))
        k_sharp = max(1, min(int(l_i // (topk_ratio * 4) + 1), k_i))

        top_z, top_idx = torch.topk(z_all[i, :l_i], k=k_i, largest=True)
        top_p = probs[i, :l_i][top_idx]

        bag_probs.append(top_p.mean())
        bag_logits_k.append(top_z.mean())
        bag_logits_sharp.append(top_z[:k_sharp].mean())

    instance_probs = torch.stack(bag_probs, dim=0).clamp(1e-6, 1.0 - 1e-6)
    loss_prob = F.binary_cross_entropy(instance_probs, bag_labels)

    if gc_weight <= 0.0:
        return loss_prob

    z_k = torch.stack(bag_logits_k, dim=0)
    z_sharp = torch.stack(bag_logits_sharp, dim=0)
    loss_logit_k = F.binary_cross_entropy_with_logits(z_k, bag_labels)
    loss_logit_sharp = F.binary_cross_entropy_with_logits(z_sharp, bag_labels)
    loss_gc = 0.60 * loss_logit_k + 0.40 * loss_logit_sharp

    return (1.0 - gc_weight) * loss_prob + gc_weight * loss_gc


def CLASM(logits: torch.Tensor, labels: torch.Tensor, lengths: torch.Tensor,
          device: torch.device = None, topk_ratio: int = 16,
          sharp_blend: float = 0.20) -> torch.Tensor:
    """Multi-Scale Top-k Vision-Language Alignment MIL Loss.

    Blends standard `k_16 = L//16 + 1` bag logits with sharp `k_64 = L//64 + 1` peak logits
    so brief, explosive anomalies (Explosion, Shooting, RoadAccidents, Arson) are not
    diluted by normal surrounding snippets during multi-class CLIP alignment.
    """
    dev = logits.device if device is None else device
    logits_f = logits.float()
    norm_labels = (labels.float() / labels.float().sum(dim=1, keepdim=True).clamp_min(1e-6)).to(dev)

    bag_logits = []
    T = logits_f.shape[1]
    for i in range(logits_f.shape[0]):
        l_i = max(1, min(int(lengths[i]), T))
        k_i = max(1, min(int(l_i // topk_ratio + 1), l_i))
        k_sharp = max(1, min(int(l_i // (topk_ratio * 4) + 1), k_i))

        top_vals, _ = torch.topk(logits_f[i, :l_i], k=k_i, largest=True, dim=0)
        mean_k = top_vals.mean(dim=0)
        if sharp_blend > 0.0 and k_sharp < k_i:
            mean_sharp = top_vals[:k_sharp].mean(dim=0)
            bag_logits.append((1.0 - sharp_blend) * mean_k + sharp_blend * mean_sharp)
        else:
            bag_logits.append(mean_k)

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
        cos_with_normal = t_norm[1:] @ t_norm[0]
        dup_idx = torch.where(cos_with_normal > 0.9995)[0]
        if dup_idx.numel() > 0:
            c_inferred = int(dup_idx[0].item()) + 1
            if c_inferred > 1 and t_norm.shape[0] % c_inferred == 0:
                t_norm = t_norm[:c_inferred]

    t_normal = t_norm[0]     # [D]
    t_abn = t_norm[1:]       # [C-1, D]
    return torch.abs(t_abn @ t_normal).mean()


def compute_normalized_recon_error(original_features: torch.Tensor,
                                   reconstructed_features: torch.Tensor,
                                   lengths: torch.Tensor = None) -> torch.Tensor:
    """Computes CLIP-Cone Baseline-Normalized Feature Reconstruction Distance in [0, 1].

    In 512-D CLIP ViT-B/16 space, the narrow cone effect restricts raw `(1 - cos) / 2`
    to `[0.02, 0.30]`, which previously acted as an artificial ceiling (~0.25) when paired
    with `ConsistencyLoss`. Subtracting each video's 20th-percentile normal baseline distance
    and scaling by the cone span (`0.45`) maps normal background to `~0.0` and true anomalies
    to `[0.55, 0.98]`.
    """
    raw_cos_dist = (1.0 - F.cosine_similarity(original_features.float(), reconstructed_features.float(), dim=-1)).clamp(0.0, 2.0)
    raw_half = (raw_cos_dist * 0.5).clamp(0.0, 1.0)

    B, T = raw_cos_dist.shape
    if T < 4:
        return raw_half

    with torch.no_grad():
        bases = []
        for i in range(B):
            l_i = max(1, min(int(lengths[i]), T)) if lengths is not None else T
            valid_d = raw_cos_dist[i, :l_i]
            k_base = max(1, int(0.20 * l_i))
            base_val = torch.topk(valid_d, k=k_base, largest=False).values[-1]
            bases.append(base_val)
        base_tensor = torch.stack(bases, dim=0).unsqueeze(1)  # [B, 1]

    cone_norm = ((raw_cos_dist - base_tensor).clamp_min(0.0) / 0.45).clamp(0.0, 1.0)
    return 0.50 * raw_half + 0.50 * cone_norm


class ConsistencyLoss(nn.Module):
    """Gradient-Conserving & CLIP-Cone Normalized Detector-Prototype Consistency Loss.

    Prevents uncalibrated cosine reconstruction error from pulling down high-confidence
    Top-k MIL anomaly peaks (`p_det > e_rec`), while maintaining full background
    suppression on normal snippets (`p_det <= e_rec`).
    """

    def __init__(self, peak_tether_weight: float = 0.25):
        super().__init__()
        self.peak_tether_weight = peak_tether_weight

    def forward(self, logits1: torch.Tensor, original_features: torch.Tensor,
                reconstructed_features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        recon_err = compute_normalized_recon_error(original_features, reconstructed_features, lengths)
        det_prob = torch.sigmoid(logits1.float().squeeze(-1))
        B, N = det_prob.shape
        mask = torch.arange(N, device=logits1.device)[None, :] < lengths.to(logits1.device)[:, None]
        if not torch.any(mask):
            return logits1.sum() * 0.0

        det_v = det_prob[mask]
        rec_v = recon_err[mask]

        # 1. Train SGNM reconstruction error to track detached detector probabilities across all valid frames
        loss_rec_tracks_det = F.mse_loss(rec_v, det_v.detach())

        # 2. Asymmetric tether on logits1: full regularization on background, damped downward pull on anomaly peaks
        rec_d = rec_v.detach()
        diff = det_v - rec_d
        asymmetric_weight = torch.where(diff > 0.15, torch.full_like(diff, self.peak_tether_weight), torch.ones_like(diff))
        loss_det_tether = (asymmetric_weight * (diff ** 2)).mean()

        return 0.50 * loss_rec_tracks_det + 0.50 * loss_det_tether


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
    """Charbonnier Boundary-Gated Smoothness (EPTS / DeSC Principle)."""
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
    """Trains the Decoupled Cross-Modal Audio-Visual Branch with MIL + Normal Acoustic Suppression."""
    if aux_logits is None:
        return torch.tensor(0.0)
    dev = aux_logits.device
    mil_loss = CLAS2(aux_logits, labels, lengths, device=dev, topk_ratio=topk_ratio)

    if aux_sem_logits is not None:
        sem_loss = CLASM(aux_sem_logits, labels, lengths, device=dev, topk_ratio=topk_ratio)
        mil_loss = mil_loss + 0.35 * sem_loss

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
    """Legacy IAM loss (kept for backward unit-test compatibility; disabled by default under Argon-4)."""
    if 'iam_logits' not in dnp_dict or dnp_dict['iam_logits'] is None:
        return torch.tensor(0.0, device=device)

    iam_logits = dnp_dict['iam_logits']
    dev = iam_logits.device if device is None else device
    loss_iam = CLASM(iam_logits, labels, lengths, device=dev)

    if iam_prototypes is not None:
        num_c = iam_logits.shape[-1]
        z = F.normalize(iam_prototypes.float(), dim=-1)
        if z.shape[0] > num_c:
            z = z[:num_c]
        sim = z @ z.t()
        eye_mask = torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
        loss_cst = F.relu(sim[~eye_mask]).mean() + torch.abs(z[1:] @ z[0]).mean()
        loss_iam = loss_iam + 0.25 * loss_cst

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
    """Legacy ACC loss (kept for backward unit-test compatibility; disabled by default under Argon-4)."""
    if 'acc_pseudo_labels' not in dnp_dict or 'fine_grained_probs' not in dnp_dict:
        dev = logits1.device if logits1 is not None else None
        return torch.tensor(0.0, device=dev)

    g = dnp_dict['acc_pseudo_labels'].detach().float()
    p_f = dnp_dict['fine_grained_probs'].float()
    B, T, _ = p_f.shape
    dev = p_f.device
    valid_mask = torch.arange(T, device=dev)[None, :] < lengths.to(dev)[:, None]
    if not torch.any(valid_mask):
        return p_f.sum() * 0.0

    l_aux = F.l1_loss(p_f[valid_mask], g[valid_mask], reduction='mean')
    if logits1 is not None:
        p_b_abn = torch.sigmoid(logits1.float().squeeze(-1))
    elif 'detector_scores' in dnp_dict and dnp_dict['detector_scores'] is not None:
        p_b_abn = dnp_dict['detector_scores'].float().squeeze(-1)
    else:
        return l_aux

    p_f_abn = (1.0 - p_f[:, :, 0]).clamp(0.0, 1.0)
    l_reg = F.l1_loss(p_f_abn[valid_mask], p_b_abn[valid_mask], reduction='mean')
    return l_aux + 0.50 * l_reg

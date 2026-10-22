"""Loss Functions and Curriculum Scheduling for MA-RUPA.

Contains:
1. Dynamic Top-K Multiple Instance Learning (MIL) BCE Loss with Snippet Dropout and Non-Saturating BCE.
2. Temporal Total Variation Smoothness & Snippet Sparsity Loss.
3. Multi-class Semantic Alignment Loss (CLASM).
4. Prototype Reconstruction Consistency Loss.
5. Background-Centric & Event-Centric Losses.
6. Prototype Gather & Orthogonality Diversity Losses.
7. Smooth Cosine Curriculum Ramp-up factor omega(e).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_curriculum_omega(epoch_idx: int, curriculum_epochs: float = 5.0) -> float:
    """Computes smooth S-curve Cosine Ramp-up factor omega(e) in [0.0, 1.0].
    
    Prevents sudden prototype shock at early epochs by gradually introducing
    prototype reconstruction and alignment pressure over the first curriculum_epochs.
    
    omega(e) = 0.5 * (1.0 - cos(pi * min(1.0, e / curriculum_epochs)))
    """
    progress = min(1.0, max(0.0, float(epoch_idx) / float(curriculum_epochs)))
    omega = 0.5 * (1.0 - math.cos(math.pi * progress))
    return float(omega)


def compute_topk_mil_loss(scores: torch.Tensor,
                          binary_labels: torch.Tensor,
                          lengths: torch.Tensor,
                          device: torch.device = None,
                          topk_ratio: int = 16,
                          label_smoothing: float = 0.0,
                          snippet_dropout: float = 0.0,
                          is_training: bool = True,
                          training: bool = None) -> torch.Tensor:
    """Computes Dynamic Top-K Multiple Instance Learning (MIL) Binary Cross-Entropy loss.
    
    Features:
    - Strict valid-length masking: zero-padded frames (t >= seq_len) are NEVER selected.
    - Dynamic k_i = max(1, floor(L_i / topk_ratio) + 1).
    - Snippet Dropout (p = 0.15 - 0.20): randomly masks a fraction of valid snippets
      during training, forcing high anomaly scores across the entire event duration
      and preventing narrow 1-frame needle spikes.
    - Standard BCE or non-saturating mild label smoothing (never freezes at 0.0564 floor).
    - Always returns a 0-D scalar tensor compatible with DataParallel.
    """
    if training is not None:
        is_training = training
    if device is None:
        device = scores.device
    binary_labels = binary_labels.to(device).float()
    scores_flat = scores.squeeze(-1)  # [B, T]
    B, T = scores_flat.shape
    instance_scores = []

    for i in range(B):
        l_i = int(lengths[i].item()) if lengths is not None else T
        l_i = max(1, min(l_i, T))
        valid_s = scores_flat[i, :l_i]

        # Snippet Dropout during training: drops random snippets to enforce temporal continuity
        if is_training and snippet_dropout > 0.0 and l_i > 2:
            keep_mask = (torch.rand(l_i, device=device) >= snippet_dropout)
            if not keep_mask.any():
                keep_mask[torch.randint(0, l_i, (1,))] = True
            valid_for_topk = valid_s[keep_mask]
        else:
            valid_for_topk = valid_s

        k_count = max(1, int(l_i / topk_ratio) + 1)
        k_eff = min(k_count, valid_for_topk.shape[0])
        topk_vals, _ = torch.topk(valid_for_topk, k=k_eff, largest=True)
        instance_scores.append(topk_vals.mean().unsqueeze(0))

    bag_scores = torch.cat(instance_scores, dim=0).clamp(1e-6, 1.0 - 1e-6)

    targets = binary_labels
    if label_smoothing > 0.0:
        targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing

    loss = F.binary_cross_entropy(bag_scores, targets)
    return loss


def compute_temporal_smoothness_loss(scores: torch.Tensor,
                                     lengths: torch.Tensor,
                                     lambda_sparse: float = 8e-5,
                                     lambda_smooth: float = 1e-3) -> torch.Tensor:
    """Computes first-order temporal total variation smoothness and snippet sparsity loss.
    
    Encourages adjacent snippet predictions in continuous events to change smoothly:
    L_smooth = (1 / (L_i - 1)) * sum_{t=1}^{L_i - 1} (S_t - S_{t-1})^2
    
    This eliminates narrow 1-frame needle spikes and restores continuous event boundaries,
    directly overcoming the mAP@IoU degradation.
    """
    B = scores.shape[0]
    scores_flat = scores.squeeze(-1)
    T = scores_flat.shape[1]
    device = scores.device

    smooth_acc = torch.tensor(0.0, device=device)
    sparse_acc = torch.tensor(0.0, device=device)
    valid_count = 0

    for i in range(B):
        l_i = int(lengths[i].item()) if lengths is not None else T
        l_i = max(1, min(l_i, T))
        if l_i > 1:
            diff = scores_flat[i, 1:l_i] - scores_flat[i, :l_i - 1]
            smooth_acc = smooth_acc + (diff ** 2).mean()
            sparse_acc = sparse_acc + scores_flat[i, :l_i].mean()
            valid_count += 1

    denom = max(1, valid_count)
    mean_smooth = smooth_acc / denom
    mean_sparse = sparse_acc / denom

    return lambda_smooth * mean_smooth + lambda_sparse * mean_sparse


def compute_clasm_loss(logits: torch.Tensor,
                       labels: torch.Tensor,
                       lengths: torch.Tensor,
                       device: torch.device = None,
                       k_ratio: int = 16,
                       epsilon: float = 0.02) -> torch.Tensor:
    """Snippet-level multi-class MIL cross-entropy loss against fine-grained categories."""
    if device is None:
        device = logits.device
    labels_dev = labels.to(device)
    labels_norm = labels_dev / labels_dev.sum(dim=1, keepdim=True).clamp_min(1e-6)
    num_classes = logits.shape[-1]
    labels_sm = (1.0 - epsilon) * labels_norm + (epsilon / num_classes)

    B, T, _ = logits.shape
    instance_logits = []
    for i in range(B):
        l_i = int(lengths[i].item()) if lengths is not None else T
        l_i = max(1, min(l_i, T))
        k_count = max(1, int(l_i / k_ratio) + 1)
        topk_vals, _ = torch.topk(logits[i, :l_i], k=min(k_count, l_i), largest=True, dim=0)
        instance_logits.append(topk_vals.mean(dim=0, keepdim=True))

    instance_logits = torch.cat(instance_logits, dim=0)
    loss = -torch.mean(torch.sum(labels_sm * F.log_softmax(instance_logits, dim=1), dim=1))
    return loss


def compute_residual_event_loss(residual_logits: torch.Tensor,
                                labels: torch.Tensor,
                                lengths: torch.Tensor,
                                device: torch.device = None,
                                k_ratio: int = 16) -> torch.Tensor:
    """Aligns anomaly residual representations with fine-grained anomaly text prompts."""
    if device is None:
        device = residual_logits.device
    # Class 0 is Normal; anomaly videos have labels[:, 1:].sum() > 0
    anomaly_mask = labels[:, 1:].sum(dim=1) > 0
    if not torch.any(anomaly_mask):
        return torch.tensor(0.0, device=device)

    sub_logits = residual_logits[anomaly_mask]
    sub_labels = labels[anomaly_mask].to(device)
    sub_lengths = lengths[anomaly_mask]

    labels_norm = sub_labels / sub_labels.sum(dim=1, keepdim=True).clamp_min(1e-6)

    instance_logits = []
    for i in range(sub_logits.shape[0]):
        l_i = int(sub_lengths[i].item())
        k_count = max(1, int(l_i / k_ratio) + 1)
        topk_vals, _ = torch.topk(sub_logits[i, :l_i], k=min(k_count, l_i), largest=True, dim=0)
        instance_logits.append(topk_vals.mean(dim=0, keepdim=True))

    instance_logits = torch.cat(instance_logits, dim=0)
    loss = -torch.mean(torch.sum(labels_norm * F.log_softmax(instance_logits, dim=-1), dim=-1))
    return loss


def compute_bkg_loss(logits4: torch.Tensor,
                     device: torch.device = None,
                     num_classes: int = 14,
                     epsilon: float = 0.02) -> torch.Tensor:
    """Background-centric loss forcing reconstructed prototypes to classify as class 0 (Normal)."""
    if device is None:
        device = logits4.device
    nor_logits = logits4.squeeze(1) if logits4.dim() == 3 else logits4  # [B, C]
    target = torch.full_like(nor_logits, epsilon / num_classes, device=device)
    target[:, 0] = 1.0 - epsilon + (epsilon / num_classes)
    loss = -torch.mean(torch.sum(target * F.log_softmax(nor_logits, dim=1), dim=1))
    return loss


def compute_event_loss(logits3: torch.Tensor,
                       labels: torch.Tensor,
                       device: torch.device = None,
                       epsilon: float = 0.02) -> torch.Tensor:
    """Event-centric loss matching pooled anomaly features against true crime/violence class."""
    if device is None:
        device = logits3.device
    anom_mask = labels[:, 1:].sum(dim=1) > 0
    if not torch.any(anom_mask):
        return torch.tensor(0.0, device=device)

    sub_logits = logits3[anom_mask].squeeze(1) if logits3.dim() == 3 else logits3[anom_mask]
    sub_labels = labels[anom_mask].to(device)
    num_classes = sub_logits.shape[-1]

    labels_norm = sub_labels / sub_labels.sum(dim=1, keepdim=True).clamp_min(1e-6)
    labels_sm = (1.0 - epsilon) * labels_norm + (epsilon / num_classes)
    loss = -torch.mean(torch.sum(labels_sm * F.log_softmax(sub_logits, dim=1), dim=1))
    return loss


def compute_consistency_loss(p_routing: torch.Tensor,
                             original_features: torch.Tensor,
                             reconstructed_features: torch.Tensor,
                             lengths: torch.Tensor,
                             device: torch.device = None) -> torch.Tensor:
    """Enforces consistency between detector routing anomaly score and prototype reconstruction error."""
    if device is None:
        device = p_routing.device
    cos_sim = F.cosine_similarity(original_features, reconstructed_features, dim=-1)  # [B, T]
    recon_error = ((1.0 - cos_sim) / 2.0).clamp(0.0, 1.0)
    pred_score = p_routing.squeeze(-1)  # [B, T]

    B, T = pred_score.shape
    mask = torch.arange(T, device=device)[None, :] < lengths.to(device)[:, None]
    if not torch.any(mask):
        return torch.tensor(0.0, device=device)

    return F.mse_loss(pred_score[mask], recon_error[mask])


def compute_normal_contrastive_loss(text_features: torch.Tensor,
                                    device: torch.device = None,
                                    num_classes: int = None) -> torch.Tensor:
    """Encourages text representations of normal vs. anomalies to remain distinct."""
    if device is None:
        device = text_features.device
    if num_classes is not None and text_features.shape[0] > num_classes:
        text_features = text_features[:num_classes]
    norm_text = F.normalize(text_features, dim=-1)
    normal_vec = norm_text[0]  # [D]
    anom_vecs = norm_text[1:]  # [C-1, D]
    sim = torch.abs(anom_vecs @ normal_vec)
    return sim.mean()


def compute_gather_loss(arg1, dynamic_normal_patterns: torch.Tensor = None, reliability: torch.Tensor = None) -> torch.Tensor:
    """Pulls dynamic prototypes towards selected normal snippet representations.
    
    Accepts either dnp_dict containing 'g_loss' / 'selected_normal' OR (selected_normal, dynamic_normal_patterns, reliability).
    Returns 0-D scalar tensor.
    """
    if isinstance(arg1, dict):
        if 'g_loss' in arg1 and isinstance(arg1['g_loss'], torch.Tensor):
            return arg1['g_loss'].mean()
        if 'selected_normal' in arg1 and 'dynamic_normal_patterns' in arg1:
            arg1, dynamic_normal_patterns = arg1['selected_normal'], arg1['dynamic_normal_patterns']
        else:
            return torch.tensor(0.0)

    selected_normal = arg1
    dist = 1.0 - F.cosine_similarity(selected_normal.unsqueeze(2), dynamic_normal_patterns.unsqueeze(1), dim=-1)
    min_dist, _ = torch.min(dist, dim=2)  # [B, K]
    if reliability is None:
        return min_dist.mean()
    rel_norm = reliability / reliability.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return (min_dist * rel_norm).sum(dim=1).mean()


def compute_orthogonality_loss(arg1) -> torch.Tensor:
    """Encourages diversity among the learned Dynamic Normal Prototypes.
    
    Accepts either dnp_dict containing 'orth_loss' OR prototypes [B, M, D].
    Returns 0-D scalar tensor.
    """
    if isinstance(arg1, dict):
        if 'orth_loss' in arg1 and isinstance(arg1['orth_loss'], torch.Tensor):
            return arg1['orth_loss'].mean()
        if 'dynamic_normal_patterns' in arg1:
            arg1 = arg1['dynamic_normal_patterns']
        else:
            return torch.tensor(0.0)

    prototypes = arg1
    B, M, D = prototypes.shape
    p_norm = F.normalize(prototypes, dim=-1)  # [B, M, D]
    sim_matrix = torch.bmm(p_norm, p_norm.transpose(1, 2))  # [B, M, M]
    identity = torch.eye(M, device=prototypes.device).unsqueeze(0).expand(B, -1, -1)
    off_diag = sim_matrix - identity
    loss_orth = (off_diag ** 2).sum(dim=(1, 2)) / max(1, M * (M - 1))
    return loss_orth.mean()

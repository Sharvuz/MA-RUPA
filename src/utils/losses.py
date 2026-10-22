"""Loss Functions and Curriculum Scheduling for MA-RUPA.

Contains:
1. Variance-Adaptive Top-K (VAT-K) selection strictly on valid unpadded sequence length.
2. Detached Adaptive-Temperature BCE (AT-BCE) + Online Hard Negative Mining (OHNM)
   + Soft Entropy Regularization + Snippet Dropout.
3. Feature-Guided Edge-Preserving TV Loss (FG-TV) & Regional Compactness Loss.
4. Confidence-Gated Asymmetric Consistency Loss.
5. Multi-class Semantic Alignment Loss (CLASM).
6. Background-Centric & Event-Centric Losses.
7. Prototype Gather & Orthogonality Diversity Losses.
8. Smooth Cosine Curriculum Ramp-up factor omega(e).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_curriculum_omega(epoch_idx: int, curriculum_epochs: float = 5.0, max_omega: float = 1.0) -> float:
    """Computes smooth S-curve Cosine Ramp-up factor omega(e) in [0.0, max_omega].
    
    Prevents sudden prototype shock at early epochs by gradually introducing
    prototype reconstruction and alignment pressure over the first curriculum_epochs.
    
    omega(e) = 0.5 * max_omega * (1.0 - cos(pi * min(1.0, e / curriculum_epochs)))
    """
    progress = min(1.0, max(0.0, float(epoch_idx) / float(curriculum_epochs)))
    omega = 0.5 * float(max_omega) * (1.0 - math.cos(math.pi * progress))
    return float(omega)


def compute_vat_k(scores_flat: torch.Tensor,
                  lengths: torch.Tensor,
                  ratio_base: float = 10.0,
                  alpha: float = 0.4,
                  beta: float = 0.6) -> list:
    """Computes Variance-Adaptive Top-K (VAT-K) count strictly on valid unpadded snippets [0:N_i].
    
    Correct Physical Direction for Ratio & k_i:
    - Short explosive events (high CV, low high_mass) -> larger ratio_i -> smaller k_i
      (avoids pulling normal frames into abnormal Top-K set).
    - Long sustained events (low CV, high high_mass) -> smaller ratio_i -> larger k_i
      (covers the full duration of the anomaly).
      
    ratio_i = ratio_base * (1 + alpha * log(1 + CV_i)) / (1 + beta * high_mass_i)
    k_i = clip(floor(N_i / ratio_i), k_min, k_max)
    """
    B, T = scores_flat.shape
    k_list = []
    for i in range(B):
        n_i = int(lengths[i].item()) if lengths is not None else T
        n_i = max(1, min(n_i, T))
        s_i = scores_flat[i, :n_i].detach()

        mu_i = s_i.mean()
        sigma_i = s_i.std(unbiased=False)
        cv_i = sigma_i / (mu_i + 1e-6)

        # High-response mass: fraction of frames exceeding mu + sigma
        high_mass_i = (s_i > (mu_i + sigma_i)).float().mean()

        ratio_i = ratio_base * (1.0 + alpha * torch.log1p(cv_i)) / (1.0 + beta * high_mass_i)
        ratio_val = float(ratio_i.item())

        k_min = 1
        k_max = max(2, int(n_i / 4.0))
        k_i = int(math.floor(n_i / max(1.0, ratio_val)))
        k_i = max(k_min, min(k_i, k_max, n_i))
        k_list.append(k_i)

    return k_list


def compute_at_bce_topk_mil_loss(scores: torch.Tensor,
                                 binary_labels: torch.Tensor,
                                 lengths: torch.Tensor,
                                 device: torch.device = None,
                                 topk_ratio: int = 10,
                                 vat_k: bool = True,
                                 snippet_dropout: float = 0.15,
                                 lambda_ent: float = 0.02,
                                 lambda_ohnm: float = 0.2,
                                 target_loss: float = 0.15,
                                 is_training: bool = True,
                                 training: bool = None) -> torch.Tensor:
    """Detached Adaptive-Temperature BCE (AT-BCE) + Online Hard Negative Mining (OHNM)
    + Soft Entropy Regularization + Snippet Dropout.
    
    Prevents loss from collapsing to 0.0038 while avoiding the 0.0560 label-smoothing floor.
    """
    if training is not None:
        is_training = training
    if device is None:
        device = scores.device
    binary_labels = binary_labels.to(device).float()
    scores_flat = scores.squeeze(-1)  # [B, T]
    B, T = scores_flat.shape

    if vat_k:
        k_list = compute_vat_k(scores_flat, lengths, ratio_base=float(topk_ratio))
    else:
        k_list = [max(1, int((int(lengths[i].item()) if lengths is not None else T) / float(topk_ratio)) + 1) for i in range(B)]

    instance_scores = []
    hard_negatives = []

    for i in range(B):
        l_i = int(lengths[i].item()) if lengths is not None else T
        l_i = max(1, min(l_i, T))
        valid_s = scores_flat[i, :l_i]
        k_i = min(k_list[i], l_i)

        # Snippet Dropout on training features to prevent 1-frame spike collapse
        if is_training and snippet_dropout > 0.0 and l_i > 2:
            keep_mask = (torch.rand(l_i, device=device) >= snippet_dropout)
            if not keep_mask.any():
                keep_mask[torch.randint(0, l_i, (1,))] = True
            valid_for_topk = valid_s[keep_mask]
        else:
            valid_for_topk = valid_s

        k_eff = min(k_i, valid_for_topk.shape[0])
        topk_vals, _ = torch.topk(valid_for_topk, k=k_eff, largest=True)
        instance_scores.append(topk_vals.mean().unsqueeze(0))

        # Online Hard Negative Mining (OHNM) on Normal bags (binary_labels[i] == 0)
        if binary_labels[i].item() < 0.5 and l_i > k_i:
            sorted_s, _ = torch.sort(valid_s, descending=True)
            k_end = min(2 * k_i, l_i)
            if k_end > k_i:
                hard_negatives.append(sorted_s[k_i:k_end])
            else:
                uncert_mask = (valid_s >= 0.25) & (valid_s <= 0.75)
                if uncert_mask.any():
                    hard_negatives.append(valid_s[uncert_mask])

    bag_scores = torch.cat(instance_scores, dim=0).clamp(1e-6, 1.0 - 1e-6)

    # Detached Adaptive Temperature Scaling: tau in [1.0, 3.0]
    with torch.no_grad():
        current_bce = F.binary_cross_entropy(bag_scores, binary_labels, reduction='mean')
        tau = torch.clamp(1.0 + 0.5 * (target_loss / (current_bce + 1e-6)), min=1.0, max=3.0)

    # Scale logits via detached tau
    logits_bag = torch.logit(bag_scores)
    scaled_bag_scores = torch.sigmoid(logits_bag / tau)
    loss_bce = F.binary_cross_entropy(scaled_bag_scores, binary_labels)

    # Soft Maximum Entropy Regularization: -lambda_ent * H(p)
    entropy = -(bag_scores * torch.log(bag_scores + 1e-6) + (1.0 - bag_scores) * torch.log(1.0 - bag_scores + 1e-6))
    loss_entropy = -lambda_ent * entropy.mean()

    # OHNM penalty on hard negatives in normal videos
    if hard_negatives and is_training and lambda_ohnm > 0.0:
        all_hn = torch.cat(hard_negatives, dim=0).clamp(1e-6, 1.0 - 1e-6)
        loss_ohnm = lambda_ohnm * F.binary_cross_entropy(all_hn, torch.zeros_like(all_hn))
    else:
        loss_ohnm = torch.tensor(0.0, device=device)

    total_mil_loss = loss_bce + loss_entropy + loss_ohnm
    return total_mil_loss


def compute_topk_mil_loss(scores: torch.Tensor,
                          binary_labels: torch.Tensor,
                          lengths: torch.Tensor,
                          device: torch.device = None,
                          topk_ratio: int = 10,
                          label_smoothing: float = 0.0,
                          snippet_dropout: float = 0.15,
                          is_training: bool = True,
                          training: bool = None,
                          use_at_bce: bool = True) -> torch.Tensor:
    """Wrapper for Dynamic Top-K MIL loss with AT-BCE + VAT-K support.
    
    If label_smoothing > 0.0, falls back to standard label-smoothed BCE for backward compatibility.
    Otherwise, applies AT-BCE with VAT-K, OHNM, and soft entropy regularization.
    """
    if use_at_bce and label_smoothing <= 0.0:
        return compute_at_bce_topk_mil_loss(
            scores=scores,
            binary_labels=binary_labels,
            lengths=lengths,
            device=device,
            topk_ratio=topk_ratio,
            vat_k=True,
            snippet_dropout=snippet_dropout,
            lambda_ent=0.02,
            lambda_ohnm=0.2,
            target_loss=0.15,
            is_training=is_training,
            training=training
        )

    # Fallback standard Top-K BCE
    if training is not None:
        is_training = training
    if device is None:
        device = scores.device
    binary_labels = binary_labels.to(device).float()
    scores_flat = scores.squeeze(-1)
    B, T = scores_flat.shape
    instance_scores = []

    for i in range(B):
        l_i = int(lengths[i].item()) if lengths is not None else T
        l_i = max(1, min(l_i, T))
        valid_s = scores_flat[i, :l_i]

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

    return F.binary_cross_entropy(bag_scores, targets)


def compute_fg_tv_smoothness_loss(scores: torch.Tensor,
                                  visual_features: torch.Tensor,
                                  lengths: torch.Tensor,
                                  gamma_edge: float = 5.0,
                                  lambda_compact: float = 0.1,
                                  binary_labels: torch.Tensor = None,
                                  device: torch.device = None) -> torch.Tensor:
    """Feature-Guided Edge-Preserving TV Loss (L_FG-TV) + Regional Compactness Loss.
    
    L_FG-TV = (1 / (N_i - 1)) * sum_{t=1}^{N_i - 1} exp(-gamma_edge * (1 - cos(X_t, X_{t-1}))) * |S_t - S_{t-1}|
    L_compact: For abnormal bags, penalizes |S_t - S_{t+-1}|^2 for neighbors with cosine sim > 0.85.
    """
    if device is None:
        device = scores.device
    scores_flat = scores.squeeze(-1)
    B, T = scores_flat.shape

    tv_losses = []
    compact_losses = []

    v_norm = F.normalize(visual_features.detach(), dim=-1)

    for i in range(B):
        l_i = int(lengths[i].item()) if lengths is not None else T
        l_i = max(1, min(l_i, T))
        if l_i < 2:
            continue

        s_seq = scores_flat[i, :l_i]
        v_seq = v_norm[i, :l_i]

        # Feature change between adjacent frames: 1 - cos(X_t, X_{t-1})
        cos_adj = (v_seq[1:] * v_seq[:-1]).sum(dim=-1).clamp(-1.0, 1.0)
        delta_f = (1.0 - cos_adj).clamp_min(0.0)

        edge_weight = torch.exp(-gamma_edge * delta_f)
        score_diff = torch.abs(s_seq[1:] - s_seq[:-1])

        tv_loss_i = (edge_weight * score_diff).mean()
        tv_losses.append(tv_loss_i)

        if binary_labels is not None and binary_labels[i].item() > 0.5 and l_i > 3:
            k_eff = max(1, min(l_i // 8, 16))
            _, topk_idx = torch.topk(s_seq.detach(), k=k_eff, largest=True)
            comp_diffs = []
            for idx in topk_idx:
                t = int(idx.item())
                if t > 0 and cos_adj[t - 1].item() > 0.85:
                    comp_diffs.append((s_seq[t] - s_seq[t - 1]) ** 2)
                if t < l_i - 1 and cos_adj[t].item() > 0.85:
                    comp_diffs.append((s_seq[t] - s_seq[t + 1]) ** 2)
            if comp_diffs:
                compact_losses.append(torch.stack(comp_diffs).mean())

    mean_tv = torch.stack(tv_losses).mean() if tv_losses else torch.tensor(0.0, device=device)
    mean_compact = torch.stack(compact_losses).mean() if compact_losses else torch.tensor(0.0, device=device)

    return mean_tv + lambda_compact * mean_compact


def compute_temporal_smoothness_loss(scores: torch.Tensor,
                                     lengths: torch.Tensor,
                                     lambda_sparse: float = 8e-5,
                                     lambda_smooth: float = 1e-3,
                                     visual_features: torch.Tensor = None,
                                     binary_labels: torch.Tensor = None) -> torch.Tensor:
    """Computes temporal smoothness loss.
    
    If visual_features is provided, uses Feature-Guided Edge-Preserving TV Loss (FG-TV);
    otherwise falls back to first-order temporal total variation.
    """
    if visual_features is not None:
        return compute_fg_tv_smoothness_loss(
            scores=scores,
            visual_features=visual_features,
            lengths=lengths,
            gamma_edge=5.0,
            lambda_compact=0.1,
            binary_labels=binary_labels,
            device=scores.device
        )

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

    return mean_smooth + lambda_sparse * mean_sparse


def compute_confidence_gated_consistency_loss(s_det: torch.Tensor,
                                              s_rupa: torch.Tensor,
                                              lengths: torch.Tensor,
                                              device: torch.device = None) -> torch.Tensor:
    """Confidence-Gated Asymmetric Consistency Loss.
    
    Weight consistency by point-wise certainty (1 - normalized binary entropy).
    The more certain stream acts as the detached teacher, while the less certain stream is the student.
    L_cons = w_cert * SmoothL1(S_student, S_teacher.detach()).
    """
    if device is None:
        device = s_det.device
    s_d = s_det.squeeze(-1).clamp(1e-6, 1.0 - 1e-6)
    s_r = s_rupa.squeeze(-1).clamp(1e-6, 1.0 - 1e-6)
    B, T = s_d.shape

    # Binary Shannon entropy normalized to [0, 1]
    log2 = math.log(2.0)
    h_d = -(s_d * torch.log(s_d) + (1.0 - s_d) * torch.log(1.0 - s_d)) / log2
    h_r = -(s_r * torch.log(s_r) + (1.0 - s_r) * torch.log(1.0 - s_r)) / log2

    cert_d = (1.0 - h_d).clamp(0.0, 1.0)
    cert_r = (1.0 - h_r).clamp(0.0, 1.0)

    # Detach the more confident stream as teacher
    d_is_teacher = (cert_d >= cert_r).float()
    teacher = d_is_teacher * s_d.detach() + (1.0 - d_is_teacher) * s_r.detach()
    student = d_is_teacher * s_r + (1.0 - d_is_teacher) * s_d
    w_cert = torch.max(cert_d, cert_r)

    mask = torch.arange(T, device=device)[None, :] < lengths.to(device)[:, None]
    if not torch.any(mask):
        return torch.tensor(0.0, device=device)

    loss_per_snippet = w_cert * F.smooth_l1_loss(student, teacher, reduction='none')
    return loss_per_snippet[mask].mean()


def compute_consistency_loss(p_routing: torch.Tensor,
                             original_features: torch.Tensor,
                             reconstructed_features: torch.Tensor,
                             lengths: torch.Tensor,
                             device: torch.device = None,
                             s_det: torch.Tensor = None) -> torch.Tensor:
    """Consistency loss between detector and reconstruction/RUPA streams.
    
    If s_det is supplied, computes Confidence-Gated Asymmetric Consistency Loss.
    """
    if device is None:
        device = p_routing.device

    cos_sim = F.cosine_similarity(original_features, reconstructed_features, dim=-1)  # [B, T]
    recon_error = ((1.0 - cos_sim) / 2.0).clamp(0.0, 1.0)

    if s_det is not None:
        return compute_confidence_gated_consistency_loss(
            s_det=s_det,
            s_rupa=recon_error,
            lengths=lengths,
            device=device
        )

    pred_score = p_routing.squeeze(-1)  # [B, T]
    B, T = pred_score.shape
    mask = torch.arange(T, device=device)[None, :] < lengths.to(device)[:, None]
    if not torch.any(mask):
        return torch.tensor(0.0, device=device)

    return F.mse_loss(pred_score[mask], recon_error[mask])


def compute_clasm_loss(logits: torch.Tensor,
                       labels: torch.Tensor,
                       lengths: torch.Tensor,
                       device: torch.device = None,
                       k_ratio: int = 10,
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
                                k_ratio: int = 10) -> torch.Tensor:
    """Aligns anomaly residual representations with fine-grained anomaly text prompts."""
    if device is None:
        device = residual_logits.device
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

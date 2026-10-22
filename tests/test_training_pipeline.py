import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import numpy as np
from utils.losses import (
    get_curriculum_omega,
    compute_topk_mil_loss,
    compute_temporal_smoothness_loss,
    compute_residual_event_loss,
    compute_clasm_loss,
    compute_bkg_loss,
    compute_consistency_loss,
    compute_event_loss,
    compute_gather_loss,
    compute_orthogonality_loss,
    compute_normal_contrastive_loss
)


def test_interleaved_dataparallel_distribution():
    print("Testing Interleaved DataParallel Batch Distribution...")
    B_half = 4  # 4 normal, 4 abnormal
    T, D = 16, 512

    # Construct normal (0) and abnormal (1) tensors
    norm_feat = torch.full((B_half, T, D), 10.0)
    anom_feat = torch.full((B_half, T, D), 20.0)
    norm_lens = torch.full((B_half,), 16, dtype=torch.long)
    anom_lens = torch.full((B_half,), 16, dtype=torch.long)

    # Interleave along dim=0
    visual_batch = torch.empty((2 * B_half, T, D), dtype=norm_feat.dtype)
    visual_batch[0::2] = norm_feat
    visual_batch[1::2] = anom_feat

    binary_labels = torch.empty(2 * B_half, dtype=torch.float32)
    binary_labels[0::2] = 0.0
    binary_labels[1::2] = 1.0

    # Simulate 2-GPU DataParallel split along dim=0
    gpu0_visual = visual_batch[:B_half]
    gpu0_labels = binary_labels[:B_half]

    gpu1_visual = visual_batch[B_half:]
    gpu1_labels = binary_labels[B_half:]

    # Check GPU 0 composition
    gpu0_normals = (gpu0_labels == 0.0).sum().item()
    gpu0_abnormals = (gpu0_labels == 1.0).sum().item()
    assert gpu0_normals == B_half // 2, f"GPU 0 should have {B_half // 2} normals, got {gpu0_normals}"
    assert gpu0_abnormals == B_half // 2, f"GPU 0 should have {B_half // 2} abnormals, got {gpu0_abnormals}"

    # Check GPU 1 composition
    gpu1_normals = (gpu1_labels == 0.0).sum().item()
    gpu1_abnormals = (gpu1_labels == 1.0).sum().item()
    assert gpu1_normals == B_half // 2, f"GPU 1 should have {B_half // 2} normals, got {gpu1_normals}"
    assert gpu1_abnormals == B_half // 2, f"GPU 1 should have {B_half // 2} abnormals, got {gpu1_abnormals}"

    print(f"  GPU 0 received: {gpu0_normals} Normal + {gpu0_abnormals} Abnormal (50/50 balanced)")
    print(f"  GPU 1 received: {gpu1_normals} Normal + {gpu1_abnormals} Abnormal (50/50 balanced)")
    print("  Interleaved DataParallel distribution passed with flying colors!")


def test_cosine_curriculum_omega():
    print("Testing S-curve Cosine Curriculum Omega...")
    omegas = [get_curriculum_omega(e, 5.0) for e in range(1, 16)]

    # Epoch 1 to 5 should smoothly monotonically increase from >0 to 1.0
    for i in range(len(omegas) - 1):
        assert omegas[i] <= omegas[i+1] + 1e-6, f"Omega should be non-decreasing: {omegas}"

    assert 0.0 < omegas[0] < 0.15, f"Epoch 1 omega should be small (~0.095), got {omegas[0]}"
    assert abs(omegas[4] - 1.0) < 1e-5, f"Epoch 5 omega should be exactly 1.0, got {omegas[4]}"
    for e in range(5, 15):
        assert abs(omegas[e] - 1.0) < 1e-5, f"Epoch {e+1} omega should be 1.0, got {omegas[e]}"

    print(f"  Cosine Ramp-up values (Epochs 1-6): {[round(o, 4) for o in omegas[:6]]}")
    print("  S-curve Cosine Curriculum Omega passed!")


def test_losses_end_to_end():
    print("Testing Loss Formulations and Rebalanced Weights...")
    B, T, D, C = 4, 32, 512, 14
    device = torch.device("cpu")

    scores = torch.sigmoid(torch.randn(B, T, 1, requires_grad=True))
    binary_labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
    lengths = torch.tensor([32, 28, 20, 30])
    one_hot = torch.zeros(B, C)
    one_hot[0, 0] = 1.0
    one_hot[1, 1] = 1.0
    one_hot[2, 0] = 1.0
    one_hot[3, 3] = 1.0

    # 1. Top-K MIL Loss
    l_mil = compute_topk_mil_loss(scores, binary_labels, lengths, topk_ratio=8, label_smoothing=0.0, snippet_dropout=0.15, is_training=True)
    assert l_mil.dim() == 0 and not torch.isnan(l_mil)

    # 2. Smoothness Loss
    l_smooth = compute_temporal_smoothness_loss(scores, lengths, lambda_smooth=1e-3)
    assert l_smooth.dim() == 0 and not torch.isnan(l_smooth)

    # 3. CLASM Loss
    logits2 = torch.randn(B, T, C)
    l_clasm = compute_clasm_loss(logits2, one_hot, lengths, k_ratio=8)
    assert l_clasm.dim() == 0 and not torch.isnan(l_clasm)

    # 4. Consistency Loss
    f_orig = torch.randn(B, T, D)
    f_rec = torch.randn(B, T, D)
    l_consist = compute_consistency_loss(scores, f_orig, f_rec, lengths)
    assert l_consist.dim() == 0 and not torch.isnan(l_consist)

    # 5. RUPA Losses
    dnp_dict = {
        'g_loss': torch.tensor([[0.04], [0.03], [0.05], [0.04]]),
        'orth_loss': torch.tensor([[0.01], [0.02], [0.01], [0.02]]),
        'residual_semantic_logits': torch.randn(B, T, C)
    }
    l_gather = compute_gather_loss(dnp_dict)
    l_orth = compute_orthogonality_loss(dnp_dict)
    l_res = compute_residual_event_loss(dnp_dict['residual_semantic_logits'], one_hot, lengths, k_ratio=8)

    assert l_gather.dim() == 0 and not torch.isnan(l_gather)
    assert l_orth.dim() == 0 and not torch.isnan(l_orth)
    assert l_res.dim() == 0 and not torch.isnan(l_res)

    # Total loss with Pareto optimal weights
    omega = 1.0
    total_loss = (
        l_mil
        + 1.0 * l_clasm
        + 0.25 * l_consist
        + 0.05 * l_smooth
        + omega * (
            0.05 * l_gather
            + 0.02 * l_orth
            + 0.05 * l_res
        )
    )
    assert total_loss.dim() == 0
    total_loss.backward()
    print("  Loss formulations and rebalanced backward pass passed successfully!")


if __name__ == '__main__':
    test_interleaved_dataparallel_distribution()
    test_cosine_curriculum_omega()
    test_losses_end_to_end()
    print("\n[SUCCESS] ALL TRAINING PIPELINE TESTS PASSED!")

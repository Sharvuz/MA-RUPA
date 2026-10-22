import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
from model import (
    MDTBackbone,
    ModalityAdaptiveBranch,
    GradientSafeRUPA,
    QuadSafeGate,
    MARUPA
)
from utils.descriptions import DESCRIPTIONS_ORI, DESCRIPTIONS_ORI_XD


def test_mdt_backbone():
    print("Testing MDTBackbone...")
    B, T, D = 2, 32, 512
    backbone = MDTBackbone(visual_length=T, embed_dim=D, attn_window=8, visual_layers=2)
    x = torch.randn(B, T, D)
    lengths = torch.tensor([32, 24])
    f_base, s_det, logits_det = backbone(x, lengths=lengths)

    assert f_base.shape == (B, T, D), f"Expected {(B, T, D)}, got {f_base.shape}"
    assert s_det.shape == (B, T, 1), f"Expected {(B, T, 1)}, got {s_det.shape}"
    assert (s_det >= 0.0).all() and (s_det <= 1.0).all(), "s_det out of [0, 1] range"
    print("  MDTBackbone passed!")


def test_modality_branch():
    print("Testing ModalityAdaptiveBranch...")
    B, T, D = 2, 32, 512
    branch = ModalityAdaptiveBranch(embed_dim=D, audio_dim=128)
    f_base = torch.randn(B, T, D)
    visual_raw = torch.randn(B, T, D)

    # 1. With audio (different length T_a=24)
    audio = torch.randn(B, 24, 128)
    f_fused, s_aux, logits_aux, g_av = branch(f_base, visual_raw, audio_features=audio)
    assert f_fused.shape == (B, T, D)
    assert s_aux.shape == (B, T, 1)
    assert (s_aux >= 0.0).all() and (s_aux <= 1.0).all()
    print("  ModalityAdaptiveBranch with audio passed!")

    # 2. Without audio (Motion Residual Mode)
    f_fused_m, s_aux_m, logits_aux_m, g_av_m = branch(f_base, visual_raw, audio_features=None)
    assert f_fused_m.shape == (B, T, D)
    assert s_aux_m.shape == (B, T, 1)
    print("  ModalityAdaptiveBranch motion residual mode passed!")

    # 3. With 512-D audio (e.g. CLAP / AST features)
    audio_512 = torch.randn(B, 32, 512)
    f_fused_512, s_aux_512, _, _ = branch(f_base, visual_raw, audio_features=audio_512)
    assert f_fused_512.shape == (B, T, D)
    assert s_aux_512.shape == (B, T, 1)
    print("  ModalityAdaptiveBranch with 512-D audio passed!")


def test_gradient_safe_rupa():
    print("Testing GradientSafeRUPA...")
    B, T, D, C = 2, 32, 512, 14
    f_fused = torch.randn(B, T, D, requires_grad=True)
    s_det = torch.sigmoid(torch.randn(B, T, 1))
    text_feat = torch.randn(C, D)
    lengths = torch.tensor([32, 28])

    # Top-K sampling
    rupa_topk = GradientSafeRUPA(feature_dim=D, num_prototypes=16, sampling='topk')
    f_rec, res, s_rec, s_sem, dnp_dict = rupa_topk(f_fused, s_det, text_feat, lengths=lengths)
    assert f_rec.shape == (B, T, D)
    assert res.shape == (B, T, D)
    assert s_rec.shape == (B, T, 1)
    assert s_sem.shape == (B, T, 1)
    assert 'g_loss' in dnp_dict
    assert 'orth_loss' in dnp_dict
    print("  GradientSafeRUPA top-k passed!")

    # Otsu sampling
    rupa_otsu = GradientSafeRUPA(feature_dim=D, num_prototypes=16, sampling='otsu')
    f_rec_o, res_o, s_rec_o, s_sem_o, dnp_dict_o = rupa_otsu(f_fused, s_det, text_feat, lengths=lengths)
    assert f_rec_o.shape == (B, T, D)
    print("  GradientSafeRUPA otsu passed!")


def test_quad_safegate():
    print("Testing QuadSafeGate...")
    B, T = 2, 32
    gate_module = QuadSafeGate(dataset_type='xd', hidden_dim=16)
    s_det = torch.full((B, T, 1), 0.5)
    s_rec = torch.full((B, T, 1), 0.8)
    s_sem = torch.full((B, T, 1), 0.3)
    s_aux = torch.full((B, T, 1), 0.6)

    p_routing, g, s_fixed = gate_module(s_det, s_rec, s_sem, s_aux)
    assert p_routing.shape == (B, T, 1)
    # Check that initial gate output g is small (~0.01) due to bias=-4.5
    assert g.mean().item() < 0.05, f"Initial gate should be small (~0.01), got {g.mean().item()}"
    print(f"  QuadSafeGate passed! Initial gate mean: {g.mean().item():.4f}")


def test_marupa_end_to_end():
    print("Testing MARUPA End-to-End...")
    device = "cpu"
    B, T, D, C = 2, 32, 512, 14

    class Args:
        no_mdtp = False
        no_audio = False
        no_rupa = False
        no_safegate = False
        sampling = 'topk'
        num_prototypes = 16
        decoder_depth = 2
        normal_selection_ratio = 0.8
        mlp_dropout = 0.1
        audio_dim = 128

    args = Args()
    model = MARUPA(
        num_class=C, embed_dim=D, visual_length=T, visual_width=D,
        visual_head=4, visual_layers=2, attn_window=8, args=args, device=device
    )

    visual = torch.randn(B, T, D, requires_grad=True)
    audio = torch.randn(B, T, 128)
    lengths = torch.tensor([32, 28])
    text = list(DESCRIPTIONS_ORI.keys())

    text_features_ori, logits1, logits2, logits3, logits4, dnp_dict = model(
        visual=visual, padding_mask=None, text=text, lengths=lengths, audio=audio, DNP_use=True
    )

    assert logits1.shape == (B, T, 1)
    assert logits2.shape == (B, T, C)
    assert logits3.shape == (B, 1, C)
    assert logits4.shape == (B, 1, C)
    assert 'routing_scores' in dnp_dict
    assert 'reconstructed_features' in dnp_dict
    assert 'g_loss' in dnp_dict
    assert 'orth_loss' in dnp_dict

    # Check backward pass
    routing_scores = dnp_dict['routing_scores']
    loss = routing_scores.mean() + dnp_dict['g_loss'].mean() + dnp_dict['orth_loss'].mean()
    loss.backward()

    # Check that gradients flowed to MDTBackbone, ModalityAdaptiveBranch, and RUPA
    assert visual.grad is not None, "visual.grad should not be None"
    assert not torch.isnan(visual.grad).any(), "NaN found in visual.grad"
    print("  MARUPA End-to-End forward and backward passed successfully!")


def test_ablations():
    print("Testing MARUPA Ablation configurations...")
    device = "cpu"
    B, T, D, C = 2, 32, 512, 7

    class Args:
        no_mdtp = True
        no_audio = True
        no_rupa = True
        no_safegate = True
        sampling = 'topk'
        num_prototypes = 16
        decoder_depth = 2
        normal_selection_ratio = 0.8
        mlp_dropout = 0.1
        audio_dim = 128

    args = Args()
    model = MARUPA(
        num_class=C, embed_dim=D, visual_length=T, visual_width=D,
        visual_head=4, visual_layers=1, attn_window=16, args=args, device=device
    )

    visual = torch.randn(B, T, D)
    lengths = torch.tensor([32, 30])
    text = list(DESCRIPTIONS_ORI_XD.keys())

    out_dnp = model(
        visual=visual, padding_mask=None, text=text, lengths=lengths, audio=None, DNP_use=True
    )
    assert len(out_dnp) == 6, f"Expected 6 outputs when DNP_use=True, got {len(out_dnp)}"

    out_no_dnp = model(
        visual=visual, padding_mask=None, text=text, lengths=lengths, audio=None, DNP_use=False
    )
    assert len(out_no_dnp) == 5, f"Expected 5 outputs when DNP_use=False, got {len(out_no_dnp)}"
    print("  Ablations (--no-mdtp, --no-audio, --no-rupa, --no-safegate) passed!")


def test_dataparallel_scalar_loss_reduction():
    print("Testing DataParallel multi-GPU loss reduction simulation...")
    # Simulate multi-GPU gathered losses (shape [2] for 2 GPUs)
    g_loss_gathered = torch.tensor([0.45, 0.42], requires_grad=True)
    orth_loss_gathered = torch.tensor([0.15, 0.18], requires_grad=True)
    mil_loss = torch.tensor(0.69, requires_grad=True)

    # Reduction via .mean()
    loss_gather = g_loss_gathered.mean()
    loss_orth = orth_loss_gathered.mean()

    total_loss = (mil_loss + 0.1 * loss_gather + 0.1 * loss_orth).mean()
    assert total_loss.dim() == 0, f"Expected 0-D scalar, got dim {total_loss.dim()}"

    # Verify backward works without error
    total_loss.backward()
    assert g_loss_gathered.grad is not None
    assert orth_loss_gathered.grad is not None

    # Verify .item() works on all terms
    val_total = total_loss.item()
    val_gather = loss_gather.item()
    val_orth = loss_orth.item()
    assert isinstance(val_total, float)
    assert isinstance(val_gather, float)
    assert isinstance(val_orth, float)
    print("  DataParallel multi-GPU loss reduction simulation passed!")


def test_padded_sequence_attention_nan_guard():
    print("Testing Padded Sequence Window Attention NaN Guard...")
    device = 'cpu'
    B, T, D, C = 2, 256, 512, 14

    class Args:
        no_mdtp = False
        no_audio = False
        no_rupa = False
        no_safegate = False
        sampling = 'topk'
        visual_length = 256
        visual_width = 512
        visual_head = 4
        visual_layers = 2
        attn_window = 8
        classes_num = 14
        embed_dim = 512
        temp = 5.0
        mlp_dropout = 0.1

    args = Args()
    model = MARUPA(
        num_class=C, embed_dim=D, visual_length=T, visual_width=D,
        visual_head=4, visual_layers=2, attn_window=8, args=args, device=device
    )
    model.eval()

    # Test short video (e.g., 50 frames padded to 256) where multiple windows are 100% padded
    visual = torch.randn(B, T, D)
    lengths = torch.tensor([50, 80])
    padding_mask = torch.arange(T)[None, :] >= lengths[:, None]
    text = list(DESCRIPTIONS_ORI.keys())

    with torch.no_grad():
        out = model(visual, padding_mask, text, lengths=lengths, audio=None, DNP_use=True)
        dnp = out[5]
        s_routing = dnp['routing_scores']
        s_det = dnp['detector_scores']
        s_rec = dnp['reconstruction_scores']
        s_sem = dnp['semantic_scores']
        s_aux = dnp['aux_scores']

        assert not torch.isnan(s_routing).any(), "s_routing contains NaN on padded sequence!"
        assert not torch.isnan(s_det).any(), "s_det contains NaN on padded sequence!"
        assert not torch.isnan(s_rec).any(), "s_rec contains NaN on padded sequence!"
        assert not torch.isnan(s_sem).any(), "s_sem contains NaN on padded sequence!"
        assert not torch.isnan(s_aux).any(), "s_aux contains NaN on padded sequence!"
    print("  Padded sequence window attention NaN guard passed!")


def test_xd_test_collate_and_split_mismatch():
    print("Testing xd_test_collate_fn & Split Count Mismatch...")
    from utils.tools import xd_test_collate_fn
    from torch.utils.data import DataLoader

    # 1. Test collate_fn with audio = None
    sample_none_audio = (
        torch.randn(256, 512),
        "Fighting",
        180,
        "/path/to/test.npy",
        "Fighting",
        None
    )
    col_out = xd_test_collate_fn([sample_none_audio])
    assert len(col_out) == 6
    assert col_out[0].shape == (1, 256, 512)
    assert col_out[5] is None
    print("  xd_test_collate_fn with audio=None passed!")

    # 2. Test collate_fn with audio = Tensor
    sample_with_audio = (
        torch.randn(256, 512),
        "Fighting",
        180,
        "/path/to/test.npy",
        "Fighting",
        torch.randn(256, 512)
    )
    col_out_a = xd_test_collate_fn([sample_with_audio])
    assert len(col_out_a) == 6
    assert col_out_a[5].shape == (1, 256, 512)
    print("  xd_test_collate_fn with audio=Tensor passed!")

    # 3. Test DataLoader with mixed dataset containing None
    class DummyXDTestDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 4
        def __getitem__(self, idx):
            if idx % 2 == 0:
                return torch.randn(256, 512), "Normal", 256, "p", "Normal", None
            return torch.randn(256, 512), "Fighting", 256, "p", "Fighting", torch.randn(256, 512)

    loader = DataLoader(DummyXDTestDataset(), batch_size=1, shuffle=False, collate_fn=xd_test_collate_fn)
    for batch_idx, item in enumerate(loader):
        assert item[0].shape == (1, 256, 512)
        if batch_idx % 2 == 0:
            assert item[5] is None
        else:
            assert item[5] is not None and item[5].shape == (1, 256, 512)
    print("  DataLoader iteration with xd_test_collate_fn passed!")

    # 4. Test ModalityAdaptiveBranch split count mismatch (visual has 3 chunks, audio has 2 chunks)
    branch = ModalityAdaptiveBranch(embed_dim=512, audio_dim=512)
    f_base = torch.randn(3, 256, 512)
    visual_raw = torch.randn(3, 256, 512)
    audio_mismatch = torch.randn(2, 256, 512)
    f_fused, s_aux, _, _ = branch(f_base, visual_raw, audio_features=audio_mismatch)
    assert f_fused.shape == (3, 256, 512)
    assert s_aux.shape == (3, 256, 1)
    print("  ModalityAdaptiveBranch split count mismatch recovery passed!")


def test_topk_ratio_and_label_smoothing():
    print("Testing Top-K Ratio and Label Smoothing...")
    from utils.losses import compute_topk_mil_loss

    B, T = 4, 128
    scores = torch.sigmoid(torch.randn(B, T, 1))
    binary_labels = torch.tensor([0, 1, 0, 1])
    lengths = torch.tensor([128, 100, 64, 120])

    # 1. Standard (topk_ratio=16, label_smoothing=0.0)
    loss_std = compute_topk_mil_loss(scores, binary_labels, lengths, topk_ratio=16, label_smoothing=0.0)
    assert loss_std > 0.0

    # 2. Optimized (topk_ratio=8, label_smoothing=0.02)
    loss_opt = compute_topk_mil_loss(scores, binary_labels, lengths, topk_ratio=8, label_smoothing=0.02)
    assert loss_opt > 0.0

    # 3. Label smoothing keeps gradient alive even on saturated predictions
    perfect_scores = torch.zeros(B, T, 1)
    perfect_scores[1] = 1.0
    perfect_scores[3] = 1.0
    loss_no_smooth = compute_topk_mil_loss(perfect_scores, binary_labels, lengths, topk_ratio=8, label_smoothing=0.0)
    loss_with_smooth = compute_topk_mil_loss(perfect_scores, binary_labels, lengths, topk_ratio=8, label_smoothing=0.02)
    assert loss_with_smooth > loss_no_smooth
    assert loss_with_smooth > 0.005, "Gradient should stay alive via label smoothing!"
    print("  Top-K ratio and Label Smoothing tests passed!")


def test_vat_k_and_at_bce():
    print("Testing VAT-K and AT-BCE with OHNM...")
    from utils.losses import compute_vat_k, compute_at_bce_topk_mil_loss

    B, T = 4, 128
    scores_flat = torch.sigmoid(torch.randn(B, T))
    lengths = torch.tensor([128, 96, 64, 110])
    k_list = compute_vat_k(scores_flat, lengths, ratio_base=10.0)
    assert len(k_list) == B
    for i, k in enumerate(k_list):
        assert 1 <= k <= max(2, int(lengths[i].item() / 4.0))

    binary_labels = torch.tensor([0, 1, 0, 1])
    loss = compute_at_bce_topk_mil_loss(
        scores_flat.unsqueeze(-1), binary_labels, lengths,
        topk_ratio=10, vat_k=True, snippet_dropout=0.15,
        lambda_ent=0.02, lambda_ohnm=0.2, target_loss=0.15, is_training=True
    )
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss > 0.0
    print("  VAT-K and AT-BCE with OHNM passed!")


def test_confidence_gated_consistency():
    print("Testing Confidence-Gated Asymmetric Consistency...")
    from utils.losses import compute_confidence_gated_consistency_loss

    B, T = 2, 64
    s_det = torch.sigmoid(torch.randn(B, T, 1))
    s_rupa = torch.sigmoid(torch.randn(B, T, 1))
    lengths = torch.tensor([64, 48])

    loss = compute_confidence_gated_consistency_loss(s_det, s_rupa, lengths)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss >= 0.0
    print("  Confidence-Gated Asymmetric Consistency passed!")


def test_fg_tv_smoothness():
    print("Testing Feature-Guided Edge-Preserving TV Loss & Compactness...")
    from utils.losses import compute_fg_tv_smoothness_loss

    B, T, D = 2, 64, 512
    scores = torch.sigmoid(torch.randn(B, T, 1))
    visual = torch.randn(B, T, D)
    lengths = torch.tensor([64, 50])
    labels = torch.tensor([0, 1])

    loss = compute_fg_tv_smoothness_loss(
        scores=scores, visual_features=visual, lengths=lengths,
        gamma_edge=5.0, lambda_compact=0.1, binary_labels=labels
    )
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss >= 0.0
    print("  Feature-Guided Edge-Preserving TV Loss passed!")


def test_gradient_projection_junction():
    print("Testing GradientProjectionJunction (Fast PCGrad)...")
    from model import GradientProjectionJunction

    x = torch.randn(2, 64, 512, requires_grad=True)
    x_main, x_rupa = GradientProjectionJunction.apply(x)

    # Force conflicting gradients: main = +1, rupa = -2 (dot product < 0)
    loss = (x_main * 1.0).sum() + (x_rupa * (-2.0)).sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    # Gradient of main should be preserved or orthogonalized without nan/inf
    print("  GradientProjectionJunction passed!")


def test_single_pass_uccf_and_dodo():
    print("Testing Single-Pass UCCF and DODO Proposal Refinement...")
    from utils.tools import compute_single_pass_uccf, apply_dodo_refinement

    T = 100
    p_routing = torch.sigmoid(torch.randn(T))
    p_det = torch.sigmoid(torch.randn(T))
    p_sem = torch.sigmoid(torch.randn(T))
    p_rupa = torch.sigmoid(torch.randn(T))

    prob_dict = {
        'routing': p_routing,
        'det': p_det,
        'sem_rupa': p_rupa,
        'sem': p_sem
    }
    priors = {'routing': 1.0, 'det': 0.9, 'sem_rupa': 0.9, 'sem': 0.6}

    fused = compute_single_pass_uccf(prob_dict, priors, delta_dis=0.35)
    assert fused.shape == (T,)
    assert (fused >= 0.0).all() and (fused <= 1.0).all()

    fused_np = fused.numpy()
    dodo_np = apply_dodo_refinement(fused_np, sigma=1.5)
    assert dodo_np.shape == (T,)
    assert (dodo_np >= 0.0).all() and (dodo_np <= 1.0).all()
    print("  Single-Pass UCCF and DODO Proposal Refinement passed!")


if __name__ == '__main__':
    test_mdt_backbone()
    test_modality_branch()
    test_gradient_safe_rupa()
    test_quad_safegate()
    test_marupa_end_to_end()
    test_ablations()
    test_dataparallel_scalar_loss_reduction()
    test_padded_sequence_attention_nan_guard()
    test_xd_test_collate_and_split_mismatch()
    test_topk_ratio_and_label_smoothing()
    test_vat_k_and_at_bce()
    test_confidence_gated_consistency()
    test_fg_tv_smoothness()
    test_gradient_projection_junction()
    test_single_pass_uccf_and_dodo()
    print("\n[SUCCESS] ALL TESTS PASSED SUCCESSFULLY!")

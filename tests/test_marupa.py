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
    loss = routing_scores.mean() + dnp_dict['g_loss'] + dnp_dict['orth_loss']
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


if __name__ == '__main__':
    test_mdt_backbone()
    test_modality_branch()
    test_gradient_safe_rupa()
    test_quad_safegate()
    test_marupa_end_to_end()
    test_ablations()
    print("\n[SUCCESS] ALL TESTS PASSED SUCCESSFULLY!")

import os
import sys
import tempfile
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import (
    TemporalWindowTransformer,
    MSTSensitivityAdapter,
    GatedCrossModalAudioAdapter,
    CLIPTextAdapter,
    IntentionAwarenessModule,
    AnomalyConnectedComponentModule,
    SGNM,
    MARUPA,
    DSANet,
    load_compatible_model_state,
)
from utils.descriptions import DESCRIPTIONS_ORI, DESCRIPTIONS_ORI_XD
from utils.dataset import load_multicrop_test_video, resolve_multicrop_paths
from utils.tools import xd_test_collate_fn, calibrate_stream_global


def test_mst_sensitivity_adapter_zero_init():
    print("Testing MSTSensitivityAdapter (Zero-Init Step 0 Identity Guarantee)...")
    B, T, D = 2, 32, 512
    adapter = MSTSensitivityAdapter(embed_dim=D, dilations=(1, 2, 4), dropout=0.0)
    x = torch.randn(B, T, D)
    valid_mask = torch.ones(B, T, 1)
    valid_mask[1, 24:] = 0.0

    out = adapter(x, valid_mask=valid_mask)
    assert torch.allclose(out, x, atol=1e-6), "MSTSensitivityAdapter must preserve exact identity at Step 0!"

    adapter.gamma.data.fill_(0.1)
    out2 = adapter(x, valid_mask=valid_mask)
    out2.mean().backward()
    assert adapter.gamma.grad is not None and torch.isfinite(adapter.gamma.grad).all()
    print("  MSTSensitivityAdapter zero-init & backward passed!")


def test_gated_cross_modal_audio_adapter():
    print("Testing GatedCrossModalAudioAdapter (VGGish 128-D & CLAP 512-D + Missing Audio Guard)...")
    B, T, D = 2, 32, 512
    adapter = GatedCrossModalAudioAdapter(visual_dim=D, audio_dim=128, num_heads=4, dropout=0.0)
    f_vis = torch.randn(B, T, D, requires_grad=True)
    valid_mask = torch.ones(B, T, 1)
    valid_mask[1, 24:] = 0.0

    # 1. Mismatched audio length (T_a = 20 -> auto interpolated to T = 32)
    audio_128 = torch.randn(B, 20, 128)
    f_mm, s_aux, logits_aux, g_av = adapter(f_vis, audio_128, valid_mask=valid_mask)
    assert f_mm.shape == (B, T, D)
    assert s_aux.shape == (B, T, 1)
    assert logits_aux.shape == (B, T, 1)
    assert g_av.shape == (B, T, 1)
    assert (s_aux >= 0.0).all() and (s_aux <= 1.0).all()

    # 2. Zero/missing audio sample in batch -> gate g_av must automatically shut to 0.0!
    zero_audio = torch.zeros(B, T, 128)
    f_mm_zero, _, _, g_zero = adapter(f_vis, zero_audio, valid_mask=valid_mask)
    assert torch.allclose(g_zero, torch.zeros_like(g_zero)), "Missing audio must force g_av == 0!"
    assert torch.allclose(f_mm_zero, f_vis * valid_mask, atol=1e-6)

    # 3. 512-D audio graceful adaptation
    audio_512 = torch.randn(B, T, 512)
    f_mm_512, s_aux_512, _, _ = adapter(f_vis, audio_512, valid_mask=valid_mask)
    assert f_mm_512.shape == (B, T, D)
    assert s_aux_512.shape == (B, T, 1)
    print("  GatedCrossModalAudioAdapter passed!")


def test_sgnm_prototype_refiner():
    print("Testing SGNM Dynamic Normal Pattern Refiner...")
    B, T, D = 2, 32, 512
    sgnm = SGNM(feature_dim=D, num_prototypes=16, num_heads=8, extractor_depth=1, decoder_depth=2)
    vis = torch.randn(B, T, D, requires_grad=True)
    logits1 = torch.randn(B, T, 1, requires_grad=True)
    lengths = torch.tensor([32, 20])

    rec, dnp, g_loss = sgnm(vis, logits1, lengths)
    assert rec.shape == (B, T, D)
    assert dnp.shape == (B, 16, D)
    assert g_loss.dim() == 0 and torch.isfinite(g_loss)
    (rec.mean() + g_loss).backward()
    assert vis.grad is not None and not torch.isnan(vis.grad).any()
    print("  SGNM prototype refiner passed!")


def test_marupa_v4_ucf_and_xd_visual_protection():
    print("Testing MARUPA v4 End-to-End (UCF-14, XD-7 Visual-Protection & DataParallel 1-D g_loss)...")
    assert DSANet is MARUPA
    device = "cpu"
    B, T, D = 2, 32, 512

    class Args:
        use_mdtp = False
        no_mdtp = True
        no_audio = False
        no_rupa = False
        no_safegate = False
        num_prototypes = 16
        decoder_depth = 2
        normal_selection_ratio = 0.8
        mlp_dropout = 0.1
        audio_dim = 128

    args = Args()

    # 1. UCF-Crime (14 classes, pure LGT-SGNM + legacy 5-arg call)
    model_ucf = MARUPA(
        num_class=14, embed_dim=D, visual_length=T, visual_width=D,
        visual_head=1, visual_layers=2, attn_window=8, args=args, device=device,
    )
    assert model_ucf.mst_adapter is None, "Pure LGT Dual-GCN must keep mst_adapter=None by default!"
    assert not hasattr(model_ucf.clip_adapter.clipmodel, 'visual'), "Unused 86M CLIP visual tower must be deleted!"

    visual = torch.randn(B, T, D, requires_grad=True)
    lengths = torch.tensor([32, 12])
    text_ucf = list(DESCRIPTIONS_ORI.keys())

    text_feat, logits1, logits2, logits3, logits4, dnp_dict = model_ucf(
        visual, None, text_ucf, lengths, True
    )
    assert text_feat.shape == (14, D)
    assert logits1.shape == (B, T, 1)
    assert logits2.shape == (B, T, 14)
    assert logits3.shape == (B, 1, 14)
    assert logits4.shape == (B, 1, 14)
    assert dnp_dict['g_loss'].dim() == 1 and dnp_dict['g_loss'].shape == (1,), "g_loss must be 1-D for DataParallel!"
    assert torch.allclose(dnp_dict['semantic_logits'], logits2), "semantic_logits must preserve supervised logits2!"
    assert not torch.isnan(logits1).any(), "NaN in logits1 on padded windows!"
    assert not torch.isnan(dnp_dict['routing_scores']).any(), "NaN in routing_scores!"

    # Verify detached SafeGate: backpropagating ONLY routing_logits must update rupa_gate WITHOUT touching visual!
    dnp_dict['routing_logits'].mean().backward(retain_graph=True)
    assert visual.grad is None, "Detached SafeGate must NOT inject duplicate MIL gradients into visual backbone!"
    assert any(p.grad is not None for p in model_ucf.rupa_gate.parameters()), "rupa_gate must receive calibration gradients!"
    model_ucf.zero_grad()

    loss_ucf = logits1.mean() + logits2.mean() + dnp_dict['g_loss'].mean()
    loss_ucf.backward()
    assert visual.grad is not None and torch.isfinite(visual.grad).all()

    # 2. XD-Violence (7 classes, Visual-Protected Decoupled VGGish Audio Branch)
    model_xd = MARUPA(
        num_class=7, embed_dim=D, visual_length=T, visual_width=D,
        visual_head=1, visual_layers=1, attn_window=16, args=args, device=device,
    )
    assert model_xd.audio_adapter is not None
    visual_xd = torch.randn(B, T, D, requires_grad=True)
    audio_xd = torch.randn(B, T, 128, requires_grad=True)
    text_xd = list(DESCRIPTIONS_ORI_XD.keys())

    tf_xd, l1_xd, l2_xd, l3_xd, l4_xd, dnp_xd = model_xd(
        visual_xd, None, text_xd, lengths, audio=audio_xd, DNP_use=True
    )
    assert tf_xd.shape == (7, D)
    assert l2_xd.shape == (B, T, 7)
    assert dnp_xd['aux_logits'] is not None and dnp_xd['aux_logits'].shape == (B, T, 1)

    # Verify 100% Visual Protection: backpropagating ONLY aux_logits updates audio_adapter, NEVER visual_xd!
    dnp_xd['aux_logits'].mean().backward(retain_graph=True)
    assert visual_xd.grad is None, "Decoupled audio branch must NEVER contaminate visual backbone!"
    assert audio_xd.grad is not None and torch.isfinite(audio_xd.grad).all()
    model_xd.zero_grad()
    audio_xd.grad = None

    loss_xd = l1_xd.mean() + l2_xd.mean() + dnp_xd['aux_logits'].mean() + dnp_xd['routing_scores'].mean()
    loss_xd.backward()
    assert visual_xd.grad is not None and torch.isfinite(visual_xd.grad).all()
    assert audio_xd.grad is not None and torch.isfinite(audio_xd.grad).all()
    print("  MARUPA v4 UCF-14 & XD-7 Visual-Protected forward/backward passed!")


def test_multicrop_discovery_and_ram_cache():
    print("Testing Multi-Crop Spatial Discovery & Float16 RAM Caching...")
    with tempfile.TemporaryDirectory() as tmpdir:
        for c in range(5):
            crop_p = os.path.join(tmpdir, f"fight_scene__+00-01-00_label_B1-0-0__{c}.npy")
            np.save(crop_p, np.random.randn(40, 512).astype(np.float32))
        aud_p = os.path.join(tmpdir, "fight_scene__+00-01-00_label_B1-0-0__vggish.npy")
        np.save(aud_p, np.random.randn(40, 128).astype(np.float32))

        csv_query = os.path.join(tmpdir, "fight_scene__#00-01-00_label_B1-0-0__0.npy")
        crops = resolve_multicrop_paths(csv_query, data_dir=tmpdir, max_crops=5)
        assert len(crops) == 5, f"Expected 5 spatial crops, found {len(crops)}"

        vis_crops, aud_crops, clip_len, num_crops = load_multicrop_test_video(
            raw_path=csv_query, maxlen=64, data_dir=tmpdir, audio_dir=tmpdir,
            no_audio=False, max_crops=5,
        )
        assert num_crops == 5
        assert clip_len == 40
        assert vis_crops.shape == (5, 1, 64, 512)
        assert aud_crops is not None and aud_crops.shape == (5, 1, 64, 128)
    print("  Multi-crop spatial discovery & audio pairing passed!")


def test_ablations_and_checkpoint_loading():
    print("Testing Ablation Flags & Compact Checkpoint Loading...")
    B, T, D = 2, 32, 512

    class Args:
        use_mdtp = True
        no_mdtp = False
        no_audio = True
        no_rupa = True
        no_safegate = True
        decoder_depth = 2

    model = MARUPA(num_class=7, visual_length=T, attn_window=16, args=Args(), device="cpu")
    assert model.mst_adapter is not None, "Optional --use-mdtp ablation must instantiate mst_adapter!"
    assert model.audio_adapter is None

    out = model(torch.randn(B, T, D), None, list(DESCRIPTIONS_ORI_XD.keys()), torch.tensor([32, 24]), DNP_use=False)
    assert len(out) == 6

    compact_sd = {
        f"module.{k}": v.clone()
        for k, v in model.state_dict().items()
        if not k.startswith('clip_adapter.clipmodel.')
    }
    incomp = load_compatible_model_state(model, compact_sd, strict=False)
    assert len(incomp.unexpected_keys) == 0

    cal = calibrate_stream_global(np.linspace(0.01, 0.99, 200))
    assert cal.shape == (200,) and np.all((cal >= 0.0) & (cal <= 1.0))
    col = xd_test_collate_fn([(torch.randn(64, 512), "A", 40, "x.npy", "A", None)])
    assert len(col) == 6 and col[5] is None
    print("  Ablations & compact checkpoint round-trip passed!")


def test_las_vad_iam_and_acc_modules():
    print("Testing LAS-VAD IntentionAwarenessModule (IAM) & AnomalyConnectedComponentModule (ACC)...")
    B, T, D, C = 2, 32, 512, 14
    valid_mask = torch.ones(B, T, 1)
    valid_mask[1, 20:] = 0.0
    lengths = torch.tensor([32, 20])

    # 1. IntentionAwarenessModule (Position X_p, Velocity X_v, Acceleration X_a + Prototypes Z)
    iam = IntentionAwarenessModule(embed_dim=D, num_classes=C, momentum=0.10, dropout=0.0)
    iam.train()
    f_vis = torch.randn(B, T, D, requires_grad=True)
    proto_before = iam.intention_prototypes.clone()
    iam_out = iam(f_vis, valid_mask=valid_mask, sem_temp=2.5)
    assert iam_out['iam_logits'].shape == (B, T, C)
    assert iam_out['iam_scores'].shape == (B, T, 1)
    assert iam_out['kin_continuity'].shape == (B, T, 1)
    assert not torch.allclose(iam.intention_prototypes, proto_before), "IAM prototypes must momentum-update during train!"
    iam_out['iam_logits'].mean().backward()
    assert f_vis.grad is not None and torch.isfinite(f_vis.grad).all()

    # 2. AnomalyConnectedComponentModule (Cross-Modal Affinity Rectification & Boolean Graph Closure)
    acc = AnomalyConnectedComponentModule(eta=0.5, tau=0.85, propagation_steps=4)
    logits2 = torch.randn(B, T, C)
    det_scores = torch.sigmoid(torch.randn(B, T, 1))
    acc_out = acc(f_vis.detach(), logits2, det_scores, lengths=lengths, sem_temp=2.5)
    assert acc_out['acc_pseudo_labels'].shape == (B, T, C)
    assert acc_out['acc_scores'].shape == (B, T, 1)
    assert acc_out['acc_adj'].shape == (B, T, T)
    assert (acc_out['acc_scores'] >= 0.0).all() and (acc_out['acc_scores'] <= 1.0).all()
    print("  LAS-VAD IAM & ACC modules passed!")


if __name__ == '__main__':
    test_mst_sensitivity_adapter_zero_init()
    test_gated_cross_modal_audio_adapter()
    test_sgnm_prototype_refiner()
    test_las_vad_iam_and_acc_modules()
    test_marupa_v4_ucf_and_xd_visual_protection()
    test_multicrop_discovery_and_ram_cache()
    test_ablations_and_checkpoint_loading()
    print("\n[SUCCESS] ALL MA-RUPA v5 (LAS-VAD CONVERGENCE) ARCHITECTURE TESTS PASSED!")

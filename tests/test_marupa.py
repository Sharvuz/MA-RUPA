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
    # Because gamma is initialized to 0.0, tanh(0.0) == 0.0 -> exact identity at Step 0!
    assert torch.allclose(out, x, atol=1e-6), "MSTSensitivityAdapter must preserve exact identity at Step 0!"

    # Verify gradient flows into gamma and conv branches once loss backpropagates
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


def test_marupa_ucf_and_xd_end_to_end():
    print("Testing MARUPA v3 End-to-End (UCF-14 & XD-7 Multimodal + Legacy 5-Arg Call)...")
    assert DSANet is MARUPA
    device = "cpu"
    B, T, D = 2, 32, 512

    class Args:
        no_mdtp = False
        no_audio = False
        no_rupa = False
        no_safegate = False
        num_prototypes = 16
        decoder_depth = 2
        normal_selection_ratio = 0.8
        mlp_dropout = 0.1
        audio_dim = 128

    args = Args()

    # 1. UCF-Crime (14 classes, visual-only + legacy 5-arg call)
    model_ucf = MARUPA(
        num_class=14, embed_dim=D, visual_length=T, visual_width=D,
        visual_head=1, visual_layers=2, attn_window=8, args=args, device=device,
    )
    assert not hasattr(model_ucf.clip_adapter.clipmodel, 'visual'), "Unused 86M CLIP visual tower must be deleted!"

    visual = torch.randn(B, T, D, requires_grad=True)
    lengths = torch.tensor([32, 12])  # includes heavily padded windows for NaN guard check
    text_ucf = list(DESCRIPTIONS_ORI.keys())

    # Legacy 5-arg call: model(visual, padding_mask, text, lengths, DNP_use)
    text_feat, logits1, logits2, logits3, logits4, dnp_dict = model_ucf(
        visual, None, text_ucf, lengths, True
    )
    assert text_feat.shape == (14, D)
    assert logits1.shape == (B, T, 1)
    assert logits2.shape == (B, T, 14)
    assert logits3.shape == (B, 1, 14)
    assert logits4.shape == (B, 1, 14)
    assert not torch.isnan(logits1).any(), "NaN in logits1 on padded windows!"
    assert not torch.isnan(dnp_dict['routing_scores']).any(), "NaN in routing_scores!"

    loss_ucf = logits1.mean() + logits2.mean() + dnp_dict['routing_scores'].mean() + dnp_dict['g_loss']
    loss_ucf.backward()
    assert visual.grad is not None and torch.isfinite(visual.grad).all()

    # 2. XD-Violence (7 classes, multimodal visual + 128-D VGGish audio)
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

    loss_xd = l1_xd.mean() + l2_xd.mean() + dnp_xd['aux_logits'].mean() + dnp_xd['routing_scores'].mean()
    loss_xd.backward()
    assert visual_xd.grad is not None and torch.isfinite(visual_xd.grad).all()
    assert audio_xd.grad is not None and torch.isfinite(audio_xd.grad).all()
    print("  MARUPA v3 UCF-14 & XD-7 forward/backward passed!")


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
        no_mdtp = True
        no_audio = True
        no_rupa = True
        no_safegate = True
        decoder_depth = 2

    model = MARUPA(num_class=7, visual_length=T, attn_window=16, args=Args(), device="cpu")
    assert model.mst_adapter is None
    assert model.audio_adapter is None

    out = model(torch.randn(B, T, D), None, list(DESCRIPTIONS_ORI_XD.keys()), torch.tensor([32, 24]), DNP_use=False)
    assert len(out) == 6

    # Compact state dict (stripped frozen CLIP weights) round-trip load
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


if __name__ == '__main__':
    test_mst_sensitivity_adapter_zero_init()
    test_gated_cross_modal_audio_adapter()
    test_sgnm_prototype_refiner()
    test_marupa_ucf_and_xd_end_to_end()
    test_multicrop_discovery_and_ram_cache()
    test_ablations_and_checkpoint_loading()
    print("\n[SUCCESS] ALL MA-RUPA v3 ARCHITECTURE TESTS PASSED!")

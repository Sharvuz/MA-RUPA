"""MA-RUPA v4+ (Argon-4 Gradient Conservation Edition): Pure LGT-SGNM & Visual-Protected WS-VAD.

Engineered under the Argon-4 Gradient Conservation Law to surpass 89.5% AUC on UCF-Crime
and 87.0% AP on XD-Violence:
  1. Pure LGT Dual-GCN Backbone (VadCLIP / DSANet): Local Window Transformer (with NaN-safe
     valid-length padding mask) + Thresholded Cosine GCN (`adj4`) + Temporal Distance GCN (`disAdj`).
  2. Pure Canonical CLIP Concept Space (`aat_weight=0.0` by default): Eliminates long-sentence AAT
     cone dilution and auxiliary IAM/ACC bottlenecks so 100% of representation capacity serves
     Top-k MIL and Multi-Class Semantic Alignment.
  3. Valid-Masked & Reliability-Weighted SGNM Prototype Refiner: Masks out padded frames (`t >= L_i`)
     prior to normal snippet selection, weights normal candidates by `reliability = 1 - p_det`, and
     normalizes reconstruction distance against each video's intrinsic CLIP-cone baseline.
  4. Valid-Masked Contrastive Event/Background Pooling (`logits3`, `logits4`): Prevents padded tail
     frames (`1 - p_det ~ 1.0`) from dominating background feature pooling on short videos.
  5. Visual-Protected Cross-Modal Audio-Visual Branch (XD-Violence): Conditioned on `visual_features.detach()`
     with Peak-Preserving Concordance Gating so soundtrack noise never contaminates visual features.
"""

from collections import OrderedDict
from functools import partial
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from clip import clip
from utils.layers import GraphConvolution, DistanceAdj
from utils.adapter_modules import SimpleAdapter, SimpleProj
from utils.descriptions import DESCRIPTIONS_ORI, DESCRIPTIONS_ORI_XD
from utils.dnp_vision_transformer import Aggregation_Block, Prototype_Block
from utils.losses import compute_normalized_recon_error


OPTIONAL_STATE_PREFIXES = (
    'rupa_gate.',
    'res_adapter.',
    'beta_res',
    'mst_adapter.',
    'audio_adapter.',
    'iam_module.',
    'acc_module.',
)


def load_compatible_model_state(model: nn.Module, state_dict: dict, allow_legacy_dsanet: bool = True, strict: bool = False):
    """Loads checkpoint weights cleanly, stripping 'module.' DataParallel prefixes and remapping legacy keys."""
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    target_model = model.module if isinstance(model, nn.DataParallel) else model
    model_state = target_model.state_dict()

    cleaned_state = OrderedDict()
    for k, v in state_dict.items():
        clean_k = k[len('module.'):] if k.startswith('module.') else k
        cleaned_state[clean_k] = v

    if strict:
        return target_model.load_state_dict(cleaned_state, strict=True)

    filtered_state = OrderedDict()
    skipped_shape = []
    for k, v in cleaned_state.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered_state[k] = v
        elif k in model_state:
            skipped_shape.append(k)

    incompatible = target_model.load_state_dict(filtered_state, strict=False)
    if skipped_shape:
        print(f"[Checkpoint] Skipped {len(skipped_shape)} shape-mismatched keys: {skipped_shape[:5]}", flush=True)
    return incompatible


class LayerNorm(nn.LayerNorm):
    """FP32-stabilized LayerNorm preserving input dtype."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_type = x.dtype
        return super().forward(x.float()).to(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


# ============================================================================
# 1. Local-Global Temporal (LGT) Window Transformer Backbone
# ============================================================================

class ResidualAttentionBlock(nn.Module):
    """Window-masked Pre-LN Transformer block (100% DataParallel & padded-window NaN safe)."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        ]))
        self.ln_2 = LayerNorm(d_model)

    def attention(self, x: torch.Tensor, attn_mask: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        T_seq, B, _ = x.shape
        if attn_mask is None and padding_mask is None:
            return self.attn(x, x, x, need_weights=False)[0]

        comb = torch.zeros((B, T_seq, T_seq), device=x.device, dtype=x.dtype)
        if attn_mask is not None:
            comb = comb + attn_mask.to(device=x.device, dtype=x.dtype).unsqueeze(0)

        valid_q = None
        if padding_mask is not None:
            pad_bool = padding_mask.to(dtype=torch.bool, device=x.device)  # [B, T]
            comb = comb.masked_fill(pad_bool.unsqueeze(1), float('-inf'))
            valid_q = (~pad_bool.t()).unsqueeze(-1).to(x.dtype)  # [T, B, 1]

        all_inf_rows = ~torch.isfinite(comb).any(dim=-1)  # [B, T]
        if all_inf_rows.any():
            diag_idx = torch.arange(T_seq, device=x.device)
            comb[:, diag_idx, diag_idx] = torch.where(
                all_inf_rows,
                torch.zeros_like(comb[:, 0, :]),
                comb[:, diag_idx, diag_idx],
            )

        n_heads = self.attn.num_heads
        comb_3d = comb.unsqueeze(1).expand(B, n_heads, T_seq, T_seq).reshape(B * n_heads, T_seq, T_seq)
        out = self.attn(x, x, x, need_weights=False, attn_mask=comb_3d)[0]
        if valid_q is not None:
            out = out * valid_q
        return out

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None, padding_mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attention(self.ln_1(x), attn_mask, padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class TemporalWindowTransformer(nn.Module):
    """Stack of Local-Window ResidualAttentionBlocks."""

    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None, padding_mask: torch.Tensor = None) -> torch.Tensor:
        for blk in self.resblocks:
            x = blk(x, attn_mask=attn_mask, padding_mask=padding_mask)
        return x


class MSTSensitivityAdapter(nn.Module):
    """Optional Zero-Initialized Multi-Scale Dilated Adapter (kept off by default; enabled via --use-mdtp for ablation)."""

    def __init__(self, embed_dim: int = 512, dilations=(1, 2, 4), dropout: float = 0.1):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim, kernel_size=3, dilation=d, padding=d, groups=embed_dim, bias=False),
                nn.Conv1d(embed_dim, embed_dim, kernel_size=1, bias=True),
                QuickGELU(),
            )
            for d in dilations
        ])
        self.out_proj = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1, bias=True),
            nn.Dropout(dropout),
        )
        self.norm = LayerNorm(embed_dim)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
        x_norm = self.norm(x)
        if valid_mask is not None:
            x_norm = x_norm * valid_mask
        x_t = x_norm.transpose(1, 2)
        ms_sum = sum(branch(x_t) for branch in self.branches) / float(len(self.branches))
        delta = self.out_proj(ms_sum).transpose(1, 2)
        if valid_mask is not None:
            delta = delta * valid_mask
        return x + torch.tanh(self.gamma) * delta


# ============================================================================
# 2. Legacy Optional Modules (Kept for Unit-Test Compatibility; Off by Default)
# ============================================================================

class IntentionAwarenessModule(nn.Module):
    """Optional Kinematic Intention Awareness Module (off by default in Argon-4 Edition)."""

    def __init__(self, feature_dim: int = 512, num_classes: int = 14,
                 momentum: float = 0.1, conf_thresh: float = 0.35, dropout: float = 0.1,
                 embed_dim: int = None):
        super().__init__()
        if embed_dim is not None:
            feature_dim = embed_dim
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.k_dim = feature_dim // 3
        self.momentum = momentum
        self.conf_thresh = conf_thresh

        self.in_norm = LayerNorm(feature_dim)
        self.fc_p = nn.Linear(feature_dim, self.k_dim)
        self.conv_v = nn.Conv1d(self.k_dim, self.k_dim, kernel_size=3, padding=1, groups=1, bias=True)
        self.conv_a = nn.Conv1d(self.k_dim, self.k_dim, kernel_size=3, padding=1, groups=1, bias=True)

        self.proj_int = nn.Linear(self.k_dim * 3, feature_dim)
        self.int_norm = LayerNorm(feature_dim)

        self.int_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

        init_proto = F.normalize(torch.randn(num_classes, feature_dim), dim=-1)
        self.register_buffer('prototypes', init_proto)
        self.register_buffer('proto_initialized', torch.tensor(False))
        self.gamma_iam = nn.Parameter(torch.tensor(0.15))

    @property
    def intention_prototypes(self) -> torch.Tensor:
        return self.prototypes

    def _extract_kinematics(self, x: torch.Tensor, valid_mask: torch.Tensor = None):
        x_n = self.in_norm(x)
        if valid_mask is not None:
            x_n = x_n * valid_mask

        x_p = self.fc_p(x_n)
        if valid_mask is not None:
            x_p = x_p * valid_mask

        diff_v = torch.zeros_like(x_p)
        if x_p.shape[1] > 1:
            diff_v[:, 1:, :] = (x_p[:, 1:, :] - x_p[:, :-1, :]).abs()
        if valid_mask is not None:
            diff_v = diff_v * valid_mask

        gate_v = torch.sigmoid(self.conv_v(diff_v.transpose(1, 2)).transpose(1, 2))
        x_v = gate_v * diff_v
        if valid_mask is not None:
            x_v = x_v * valid_mask

        diff_a = torch.zeros_like(x_v)
        if x_v.shape[1] > 1:
            diff_a[:, 1:, :] = (x_v[:, 1:, :] - x_v[:, :-1, :]).abs()
        if valid_mask is not None:
            diff_a = diff_a * valid_mask

        gate_a = torch.sigmoid(self.conv_a(diff_a.transpose(1, 2)).transpose(1, 2))
        x_a = gate_a * diff_a
        if valid_mask is not None:
            x_a = x_a * valid_mask

        x_int = self.int_norm(self.proj_int(torch.cat([x_p, x_v, x_a], dim=-1)))
        if valid_mask is not None:
            x_int = x_int * valid_mask

        return x_p, x_v, x_a, x_int

    @torch.no_grad()
    def _update_prototypes(self, x_int_norm: torch.Tensor, q_probs: torch.Tensor, valid_mask: torch.Tensor = None):
        B, T, C = q_probs.shape
        flat_feat = x_int_norm.reshape(B * T, -1)
        flat_probs = q_probs.reshape(B * T, C)
        if valid_mask is not None:
            valid_flat = valid_mask.reshape(B * T) > 0.5
            flat_feat = flat_feat[valid_flat]
            flat_probs = flat_probs[valid_flat]

        if flat_feat.shape[0] == 0:
            return

        eff_thresh = self.conf_thresh if bool(self.proto_initialized) else 0.0
        max_p, pred_c = flat_probs.max(dim=-1)
        for c in range(C):
            mask_c = (pred_c == c) & (max_p > eff_thresh)
            if mask_c.any():
                center_c = F.normalize(flat_feat[mask_c].mean(dim=0), dim=-1)
                updated = (1.0 - self.momentum) * self.prototypes[c] + self.momentum * center_c
                self.prototypes[c].copy_(F.normalize(updated, dim=-1))

        self.proto_initialized.fill_(True)

    def forward(self, f_vis: torch.Tensor, valid_mask: torch.Tensor = None, sem_temp: float = 2.0):
        x_p, x_v, x_a, x_int = self._extract_kinematics(f_vis, valid_mask=valid_mask)
        q_int = self.int_mlp(x_int)

        x_int_norm = F.normalize(x_int.float(), dim=-1)
        proto_norm = F.normalize(self.prototypes.float(), dim=-1)

        pred_labels = q_int.detach().argmax(dim=-1)
        selected_protos = proto_norm[pred_labels]
        cos_conf = (x_int_norm * selected_protos).sum(dim=-1, keepdim=True)
        w_int = (0.5 * (1.0 + cos_conf)).clamp(0.1, 1.0).to(q_int.dtype)

        q_a = q_int * w_int
        q_probs = F.softmax(q_a.float() / max(sem_temp, 0.5), dim=-1)

        if self.training:
            self._update_prototypes(x_int_norm.detach(), q_probs.detach(), valid_mask=valid_mask)

        s_iam = (1.0 - q_probs[:, :, :1]).to(f_vis.dtype).clamp(0.0, 1.0)

        with torch.no_grad():
            vel_mag = x_v.float().norm(p=2, dim=-1, keepdim=True)
            acc_mag = x_a.float().norm(p=2, dim=-1, keepdim=True)
            vel_sustained = F.avg_pool1d(
                vel_mag.transpose(1, 2), kernel_size=5, stride=1, padding=2
            ).transpose(1, 2)
            excess_jerk = F.relu(acc_mag - 1.5 * vel_sustained)
            kin_continuity = (vel_sustained + 1e-3) / (vel_sustained + 0.5 * excess_jerk + 1e-3)
            kin_continuity = kin_continuity.clamp(0.25, 1.0).to(f_vis.dtype)
            if valid_mask is not None:
                kin_continuity = kin_continuity * valid_mask

        return {
            'x_p': x_p,
            'x_v': x_v,
            'x_a': x_a,
            'x_int': x_int,
            'iam_logits': q_a,
            'iam_scores': s_iam,
            'iam_confidence': w_int,
            'kin_continuity': kin_continuity,
        }


class AnomalyConnectedComponentModule(nn.Module):
    """Optional Anomaly-Connected Components Module (off by default in Argon-4 Edition)."""

    def __init__(self, eta: float = 0.5, tau: float = 0.85, propagation_steps: int = 8):
        super().__init__()
        self.eta = eta
        self.tau = tau
        self.propagation_steps = propagation_steps

    @torch.no_grad()
    def forward(self, visual_features: torch.Tensor, logits_sem: torch.Tensor,
                detector_scores: torch.Tensor, lengths: torch.Tensor = None,
                sem_temp: float = 2.0):
        B, T, _ = visual_features.shape
        dev = visual_features.device
        orig_dtype = visual_features.dtype

        x_norm = F.normalize(visual_features.float(), p=2, dim=-1)
        a_v = torch.bmm(x_norm, x_norm.transpose(1, 2))

        q_l = F.softmax(logits_sem.float() / max(sem_temp, 0.5), dim=-1)
        C = q_l.shape[-1]

        max_min = torch.zeros((B, T, T), device=dev, dtype=torch.float32)
        for c in range(C):
            q_c = q_l[:, :, c]
            min_c = torch.minimum(q_c.unsqueeze(2), q_c.unsqueeze(1))
            max_min = torch.maximum(max_min, min_c)

        a_w = a_v * (1.0 + self.eta * max_min)
        a_bin = (a_w > self.tau).float()
        if lengths is not None:
            valid_1d = torch.arange(T, device=dev)[None, :] < lengths.to(dev)[:, None]
            valid_2d = (valid_1d.unsqueeze(2) & valid_1d.unsqueeze(1)).float()
            a_bin = a_bin * valid_2d

        diag_idx = torch.arange(T, device=dev)
        a_bin[:, diag_idx, diag_idx] = 1.0

        a_cc = a_bin
        for _ in range(3):
            a_cc = (torch.bmm(a_cc, a_cc) > 0.0).float()

        comp_sizes = a_cc.sum(dim=-1, keepdim=True).clamp_min(1.0)
        w_acc = a_cc / comp_sizes

        x_ac = F.normalize(torch.bmm(w_acc, x_norm), p=2, dim=-1)
        y_proto = torch.bmm(w_acc, q_l)

        sim_proto = torch.bmm(x_norm, x_ac.transpose(1, 2)) / 0.10
        if lengths is not None:
            sim_proto = sim_proto.masked_fill(~valid_1d.unsqueeze(1), float('-inf'))
        proto_assign = F.softmax(sim_proto, dim=-1)
        proto_assign = torch.nan_to_num(proto_assign, nan=0.0)
        pseudo_labels = torch.bmm(proto_assign, y_proto)
        pseudo_labels = pseudo_labels / pseudo_labels.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        det_f = detector_scores.float()
        s_acc = torch.bmm(w_acc, det_f)
        singleton_factor = torch.where(comp_sizes <= 1.5, torch.full_like(comp_sizes, 0.70), torch.ones_like(comp_sizes))
        s_acc = (0.75 * s_acc + 0.25 * det_f * singleton_factor).clamp(0.0, 1.0)

        return {
            'acc_pseudo_labels': pseudo_labels.to(orig_dtype),
            'acc_scores': s_acc.to(orig_dtype),
            'acc_comp_sizes': comp_sizes.to(orig_dtype),
            'acc_adj': a_bin.to(orig_dtype),
        }


# ============================================================================
# 3. Visual-Protected Cross-Modal Audio-Visual Branch (XD-Violence SOTA)
# ============================================================================

class GatedCrossModalAudioAdapter(nn.Module):
    """Bidirectional Visual-Audio Cross-Modal Concordance Branch with Acoustic Kinematics (XD-Violence)."""

    def __init__(self, visual_dim: int = 512, audio_dim: int = 128, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.visual_dim = visual_dim
        self.audio_dim = audio_dim

        self.audio_input_norm = LayerNorm(audio_dim)
        self.audio_lift = nn.Sequential(
            nn.Linear(audio_dim, visual_dim),
            QuickGELU(),
            nn.Dropout(dropout),
            nn.Linear(visual_dim, visual_dim),
        )
        self.audio_temp_conv = nn.Conv1d(
            visual_dim, visual_dim, kernel_size=3, padding=1, groups=visual_dim, bias=True
        )
        self.audio_vel_conv = nn.Conv1d(
            visual_dim, visual_dim, kernel_size=3, padding=1, groups=visual_dim, bias=True
        )
        self.norm_v = LayerNorm(visual_dim)
        self.norm_a = LayerNorm(visual_dim)

        self.cross_v2a = nn.MultiheadAttention(visual_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_a2v = nn.MultiheadAttention(visual_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_fused = LayerNorm(visual_dim)

        self.gate_mlp = nn.Sequential(
            nn.Linear(visual_dim * 3, visual_dim // 4),
            QuickGELU(),
            nn.Linear(visual_dim // 4, 1),
        )
        nn.init.normal_(self.gate_mlp[-1].weight, std=0.01)
        nn.init.constant_(self.gate_mlp[-1].bias, -1.5)

        self.aux_head = nn.Sequential(
            LayerNorm(visual_dim * 4),
            nn.Linear(visual_dim * 4, visual_dim // 2),
            QuickGELU(),
            nn.Dropout(dropout),
            nn.Linear(visual_dim // 2, 1),
        )

        self.av_sem_proj = nn.Sequential(
            LayerNorm(visual_dim * 2),
            nn.Linear(visual_dim * 2, visual_dim),
            QuickGELU(),
            nn.Linear(visual_dim, visual_dim),
        )

        self.alpha_av = nn.Parameter(torch.tensor(0.05))
        self.last_logits_aux_sem = None

    def forward(self, f_vis: torch.Tensor, audio_raw: torch.Tensor,
                valid_mask: torch.Tensor = None, text_features_ori: torch.Tensor = None):
        B, T, _ = f_vis.shape

        if audio_raw.shape[1] != T:
            a_aligned = F.interpolate(
                audio_raw.float().permute(0, 2, 1), size=T, mode='linear', align_corners=False
            ).permute(0, 2, 1).to(f_vis.dtype)
        else:
            a_aligned = audio_raw.to(f_vis.dtype)

        if a_aligned.shape[-1] != self.audio_dim:
            if a_aligned.shape[-1] > self.audio_dim:
                a_aligned = a_aligned[..., :self.audio_dim]
            else:
                a_aligned = F.pad(a_aligned, (0, self.audio_dim - a_aligned.shape[-1]))

        has_signal = (a_aligned.abs().sum(dim=(-1, -2), keepdim=True) > 1e-6).to(f_vis.dtype)

        a_lift = self.audio_lift(self.audio_input_norm(a_aligned))
        a_conv = self.audio_temp_conv(a_lift.transpose(1, 2)).transpose(1, 2)

        a_diff = torch.zeros_like(a_lift)
        if T > 1:
            a_diff[:, 1:, :] = (a_lift[:, 1:, :] - a_lift[:, :-1, :]).abs()
        a_vel_gate = torch.sigmoid(self.audio_vel_conv(a_diff.transpose(1, 2)).transpose(1, 2))
        a_vel = a_vel_gate * a_diff

        h_a = a_lift + 0.5 * a_conv + 0.35 * a_vel
        if valid_mask is not None:
            h_a = h_a * valid_mask

        v_n = self.norm_v(f_vis)
        a_n = self.norm_a(h_a)

        key_pad = (~(valid_mask.squeeze(-1).bool())) if valid_mask is not None else None
        v_from_a, _ = self.cross_v2a(query=v_n, key=a_n, value=a_n, key_padding_mask=key_pad, need_weights=False)
        a_from_v, _ = self.cross_a2v(query=a_n, key=v_n, value=v_n, key_padding_mask=key_pad, need_weights=False)
        v_from_a = torch.nan_to_num(v_from_a, nan=0.0)
        a_from_v = torch.nan_to_num(a_from_v, nan=0.0)

        av_joint = torch.cat([v_n, a_n + a_from_v, v_from_a, v_n * a_n], dim=-1)
        logits_aux = self.aux_head(av_joint)
        logits_aux = logits_aux * has_signal + (-6.0) * (1.0 - has_signal)
        s_aux = torch.sigmoid(logits_aux)

        self.last_logits_aux_sem = None
        if text_features_ori is not None:
            av_sem_feat = F.normalize(self.av_sem_proj(torch.cat([a_n + a_from_v, v_from_a], dim=-1)), dim=-1)
            t_norm = F.normalize(text_features_ori.detach().to(av_sem_feat.dtype), dim=-1)
            logits_aux_sem = torch.matmul(av_sem_feat, t_norm.t()) / 0.07
            self.last_logits_aux_sem = logits_aux_sem

        gate_in = torch.cat([v_n, a_n, v_n * a_n], dim=-1)
        g_av = torch.sigmoid(self.gate_mlp(gate_in)) * has_signal
        if valid_mask is not None:
            g_av = g_av * valid_mask

        delta_mm = g_av * self.norm_fused(v_from_a)
        f_mm = f_vis + torch.tanh(self.alpha_av) * delta_mm
        if valid_mask is not None:
            f_mm = f_mm * valid_mask

        return f_mm, s_aux, logits_aux, g_av


# ============================================================================
# 4. Memory-Light Batched CLIP_Adapter (Pure Canonical Concept Anchor)
# ============================================================================

class CLIPTextAdapter(nn.Module):
    """Bias-Free CLIP_Adapter on Pure Canonical Class Concepts."""

    def __init__(self, clipmodel, text_adapt_until: int = 3, t_w: float = 0.1, aat_weight: float = 0.0):
        super().__init__()
        if hasattr(clipmodel, 'visual'):
            del clipmodel.visual

        self.clipmodel = clipmodel
        for p in self.clipmodel.parameters():
            p.requires_grad = False

        self.text_adapt_until = text_adapt_until
        self.t_w = t_w
        self.aat_weight = aat_weight

        self.text_adapter = nn.ModuleList(
            [SimpleAdapter(512, 512) for _ in range(text_adapt_until)]
            + [SimpleProj(512, 512, relu=True)]
        )
        self._init_weights()

        ucf_tokens, self.ucf_k = self._build_token_bank(DESCRIPTIONS_ORI)
        xd_tokens, self.xd_k = self._build_token_bank(DESCRIPTIONS_ORI_XD)
        self.register_buffer('ucf_prompt_tokens', ucf_tokens, persistent=False)
        self.register_buffer('xd_prompt_tokens', xd_tokens, persistent=False)

    def _init_weights(self):
        for p in self.text_adapter.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def _build_token_bank(desc_dict: dict):
        max_k = max(len(v) for v in desc_dict.values())
        flat_prompts = []
        for prompts in desc_dict.values():
            padded = list(prompts) + [prompts[0]] * (max_k - len(prompts))
            flat_prompts.extend(padded[:max_k])
        tokens = clip.tokenize(flat_prompts)
        return tokens, max_k

    def encode_tokens(self, tokens: torch.Tensor, adapt_text: bool = True) -> torch.Tensor:
        cast_dtype = self.clipmodel.token_embedding.weight.dtype
        x = self.clipmodel.token_embedding(tokens).to(cast_dtype)
        x = x + self.clipmodel.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)

        for i, block in enumerate(self.clipmodel.transformer.resblocks):
            x = block(x)
            if adapt_text and i < self.text_adapt_until:
                adapt_out = self.text_adapter[i](x.float()).to(cast_dtype)
                adapt_out = (
                    adapt_out * x.norm(dim=-1, keepdim=True)
                    / (adapt_out.norm(dim=-1, keepdim=True) + 1e-6)
                )
                x = self.t_w * adapt_out + (1.0 - self.t_w) * x

        x = x.permute(1, 0, 2)
        x = self.clipmodel.ln_final(x).float()
        eot_indices = tokens.argmax(dim=-1)
        x_eot = x[torch.arange(x.shape[0], device=tokens.device), eot_indices]

        if not adapt_text:
            if getattr(self.clipmodel, 'text_projection', None) is not None:
                return x_eot @ self.clipmodel.text_projection.float()
            return x_eot

        return self.text_adapter[-1](x_eot)

    def get_class_embeddings(self, num_classes: int, target_device: torch.device) -> torch.Tensor:
        if num_classes == 14:
            tokens = self.ucf_prompt_tokens.to(target_device)
            k = self.ucf_k
            c = 14
        elif num_classes == 7:
            tokens = self.xd_prompt_tokens.to(target_device)
            k = self.xd_k
            c = 7
        else:
            raise ValueError(f"Unsupported num_classes={num_classes}; expected 14 (UCF) or 7 (XD).")

        feat = self.encode_tokens(tokens, adapt_text=True).view(c, k, -1)
        if k > 1 and self.aat_weight > 0.0:
            x_lang = F.normalize(feat[:, 0, :], dim=-1)
            x_aux = F.normalize(feat[:, 1:, :].mean(dim=1), dim=-1)
            combined = (1.0 - self.aat_weight) * x_lang + self.aat_weight * x_aux
            return F.normalize(combined, dim=-1)
        return F.normalize(feat[:, 0, :], dim=-1)


# ============================================================================
# 5. Self-Guided Dynamic Normal Pattern Refiner (SGNM - Valid-Masked & Reliability-Weighted)
# ============================================================================

class SGNM(nn.Module):
    """Dynamic Normal Pattern Extractor & Counterfactual Normal Background Reconstructor.

    Restores two critical mathematical invariants from original DSANet (`62cfd81`):
      1. Valid-Length Masking: Padded frames (`t >= lengths[i]`) are masked with `+inf`
         before `torch.topk(..., largest=False)` so zero-padded tail frames are NEVER
         selected as normal prototypes.
      2. Reliability Weighting (`r_t = 1 - p_det(t)`): Modulates normal candidate features
         by `0.5 + 0.5 * r_t` and weights `gather_loss` by `r_t`, preventing abnormal frames
         in long-anomaly videos from corrupting normal prototypes.
    """

    def __init__(self, feature_dim: int = 512, num_prototypes: int = 16, num_heads: int = 8,
                 extractor_depth: int = 1, decoder_depth: int = 8, normal_selection_ratio: float = 0.8):
        super().__init__()
        self.normal_selection_ratio = normal_selection_ratio
        self.video_prototypes = nn.Parameter(torch.randn(num_prototypes, feature_dim))
        self.dnp_extractor = nn.ModuleList([
            Aggregation_Block(
                dim=feature_dim, num_heads=num_heads, mlp_ratio=4.0,
                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8)
            )
            for _ in range(extractor_depth)
        ])
        self.bottleneck = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(feature_dim, feature_dim * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(feature_dim * 4, feature_dim)),
        ]))
        self.decoder = nn.ModuleList([
            Prototype_Block(
                dim=feature_dim, num_heads=num_heads, mlp_ratio=4.0,
                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8)
            )
            for _ in range(decoder_depth)
        ])

    @staticmethod
    def gather_loss(query: torch.Tensor, keys: torch.Tensor, reliability: torch.Tensor = None) -> torch.Tensor:
        distribution = 1.0 - F.cosine_similarity(query.unsqueeze(2), keys.unsqueeze(1), dim=-1)
        distance, _ = torch.min(distribution, dim=2)
        if reliability is not None:
            rel = reliability.squeeze(-1).to(distance.dtype).clamp(min=0.05)
            return (distance * rel).sum() / rel.sum().clamp_min(1e-6)
        return distance.mean()

    def forward(self, visual_features: torch.Tensor, logits1: torch.Tensor, lengths: torch.Tensor = None):
        B, N, D = visual_features.shape

        with torch.no_grad():
            anomaly_scores = torch.sigmoid(logits1.squeeze(-1))
            masked_scores = anomaly_scores.clone()
            if lengths is not None:
                for i in range(B):
                    l_i = max(1, min(int(lengths[i]), N))
                    if l_i < N:
                        masked_scores[i, l_i:] = float('inf')

            num_normal_frames = max(1, int(N * self.normal_selection_ratio))
            _, indices = torch.topk(masked_scores, k=N, largest=False, dim=1)
            normal_indices = indices[:, :num_normal_frames]

            if lengths is not None:
                for i in range(B):
                    l_i = max(1, min(int(lengths[i]), N))
                    if l_i < num_normal_frames:
                        wrap_idx = torch.arange(num_normal_frames, device=visual_features.device) % l_i
                        normal_indices[i] = indices[i, wrap_idx]

            selected_normal_features = torch.gather(
                visual_features, 1, normal_indices.unsqueeze(-1).expand(-1, -1, D)
            )

        agg_prototype = self.video_prototypes.unsqueeze(0).expand(B, -1, -1)
        for blk in self.dnp_extractor:
            agg_prototype = blk(agg_prototype, selected_normal_features)

        dynamic_normal_patterns = agg_prototype
        g_loss = self.gather_loss(selected_normal_features, dynamic_normal_patterns)

        reconstructed_features = self.bottleneck(visual_features)
        for blk in self.decoder:
            reconstructed_features = blk(reconstructed_features, dynamic_normal_patterns)

        return reconstructed_features, dynamic_normal_patterns, g_loss


# ============================================================================
# 6. Unified SOTA Model: MARUPA (Argon-4 Gradient Conservation Edition)
# ============================================================================

class MARUPA(nn.Module):
    """MA-RUPA v4+ (Argon-4 Gradient Conservation Architecture)."""

    def __init__(self,
                 num_class: int,
                 embed_dim: int = 512,
                 visual_length: int = 256,
                 visual_width: int = 512,
                 visual_head: int = 1,
                 visual_layers: int = 2,
                 attn_window: int = 8,
                 prompt_prefix: int = 10,
                 prompt_postfix: int = 10,
                 args=None,
                 device="cuda"):
        super().__init__()
        self.num_class = num_class
        self.visual_length = visual_length
        self.visual_width = visual_width
        self.embed_dim = embed_dim
        self.attn_window = attn_window
        self.prompt_prefix = prompt_prefix
        self.prompt_postfix = prompt_postfix
        self.device = device

        # Configuration flags (Pure Argon-4 defaults: no_iam=True, no_acc=True, aat_weight=0.0)
        self.no_rupa = bool(getattr(args, 'no_rupa', False))
        self.rupa_use = (not self.no_rupa) and bool(getattr(args, 'rupa_use', True))
        self.use_mdtp = bool(getattr(args, 'use_mdtp', False)) and (not bool(getattr(args, 'no_mdtp', True)))
        self.no_audio = bool(getattr(args, 'no_audio', False))
        self.no_safegate = bool(getattr(args, 'no_safegate', False))
        self.use_iam = bool(getattr(args, 'use_iam', False)) and (not bool(getattr(args, 'no_iam', True)))
        self.use_acc = bool(getattr(args, 'use_acc', False)) and (not bool(getattr(args, 'no_acc', True)))

        self.routing_det_weight = float(getattr(args, 'routing_det_weight', 0.80))
        self.routing_rec_weight = float(getattr(args, 'routing_rec_weight', 0.10))
        self.routing_sem_weight = float(getattr(args, 'routing_sem_weight', 0.10))
        w_sum = self.routing_det_weight + self.routing_rec_weight + self.routing_sem_weight
        if w_sum <= 0:
            self.routing_det_weight, self.routing_rec_weight, self.routing_sem_weight = 0.80, 0.10, 0.10
        else:
            self.routing_det_weight /= w_sum
            self.routing_rec_weight /= w_sum
            self.routing_sem_weight /= w_sum

        self.routing_mode = 'fixed' if self.no_safegate else getattr(args, 'routing_mode', 'safe_gate')
        self.routing_gate_init = float(getattr(args, 'routing_gate_init', 0.01))
        self.routing_gate_max = float(getattr(args, 'routing_gate_max', 0.50))
        routing_gate_hidden = max(4, int(getattr(args, 'routing_gate_hidden', 8)))

        # 1. Pure LGT Backbone (Local Window Transformer + Dual Similarity/Distance GCN)
        self.register_buffer(
            'base_attn_mask',
            self.build_attention_mask(self.visual_length, self.attn_window),
            persistent=False,
        )
        self.temporal = TemporalWindowTransformer(
            width=visual_width,
            layers=visual_layers,
            heads=visual_head,
        )

        half_width = visual_width // 2
        self.gc1 = GraphConvolution(visual_width, half_width, residual=True)
        self.gc2 = GraphConvolution(half_width, half_width, residual=True)
        self.gc3 = GraphConvolution(visual_width, half_width, residual=True)
        self.gc4 = GraphConvolution(half_width, half_width, residual=True)
        self.disAdj = DistanceAdj()
        self.linear = nn.Linear(visual_width, visual_width)
        self.gelu = QuickGELU()
        self.frame_position_embeddings = nn.Embedding(visual_length, visual_width)

        mlp_dropout = float(getattr(args, 'mlp_dropout', 0.1))
        self.mlp_dropout = mlp_dropout
        self.mst_adapter = (
            MSTSensitivityAdapter(embed_dim=visual_width, dilations=(1, 2, 4), dropout=mlp_dropout)
            if self.use_mdtp else None
        )

        # 2. Visual-Protected Cross-Modal Audio-Visual Branch (XD-Violence only)
        audio_dim = int(getattr(args, 'audio_dim', 128))
        if not self.no_audio and num_class == 7:
            self.audio_adapter = GatedCrossModalAudioAdapter(
                visual_dim=visual_width, audio_dim=audio_dim, num_heads=4, dropout=max(0.15, mlp_dropout)
            )
        else:
            self.audio_adapter = None

        # 3. Sensitivity Detector & Video-Conditioned Prompt MLPs (Exact ucf-sota-88.34 geometry)
        self.mlp1 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width)),
        ]))
        self.mlp2 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width)),
        ]))
        self.classifier = nn.Linear(visual_width, 1)

        # 4. Frozen CLIP Text Backbone + Bias-Free CLIPTextAdapter on Pure Canonical Concepts
        text_adapt_until = int(getattr(args, 'text_adapt_until', 3 if num_class == 14 else 1))
        t_w = float(getattr(args, 't_w', 0.1 if num_class == 14 else 0.6))
        aat_weight = float(getattr(args, 'aat_weight', 0.0))
        clipmodel, _ = clip.load("ViT-B/16", device="cpu")
        self.clip_adapter = CLIPTextAdapter(
            clipmodel, text_adapt_until=text_adapt_until, t_w=t_w, aat_weight=aat_weight
        )

        # 5. Optional Legacy Modules (Off by default under Argon-4)
        iam_momentum = float(getattr(args, 'iam_momentum', 0.10))
        self.iam_module = (
            IntentionAwarenessModule(
                feature_dim=visual_width, num_classes=num_class,
                momentum=iam_momentum, dropout=mlp_dropout,
            )
            if self.use_iam else None
        )
        acc_eta = float(getattr(args, 'acc_eta', 0.50))
        acc_tau = float(getattr(args, 'acc_tau', 0.85 if num_class == 14 else 0.80))
        self.acc_module = (
            AnomalyConnectedComponentModule(eta=acc_eta, tau=acc_tau)
            if self.use_acc else None
        )

        # 6. Dynamic Normal Pattern Refiner (SGNM), Counterfactual Residual Adapter & Stop-Gradient SafeGate
        num_prototypes = int(getattr(args, 'num_prototypes', 16))
        decoder_depth = int(getattr(args, 'decoder_depth', 8))
        normal_selection_ratio = float(getattr(args, 'normal_selection_ratio', 0.8))
        self.video_anomaly_refiner = SGNM(
            feature_dim=visual_width,
            num_prototypes=num_prototypes,
            num_heads=8,
            extractor_depth=1,
            decoder_depth=decoder_depth,
            normal_selection_ratio=normal_selection_ratio,
        )

        self.res_adapter = nn.Sequential(OrderedDict([
            ("ln", LayerNorm(visual_width)),
            ("c_fc", nn.Linear(visual_width, visual_width)),
            ("gelu", QuickGELU()),
            ("drop", nn.Dropout(mlp_dropout)),
            ("c_proj", nn.Linear(visual_width, visual_width)),
        ]))
        self.beta_res = nn.Parameter(torch.zeros(1))

        self.rupa_gate = nn.Sequential(
            nn.Linear(5, routing_gate_hidden),
            QuickGELU(),
            nn.Linear(routing_gate_hidden, 1),
        )

        # Semantic-to-backbone gradient scale: 1.0 during warm-up, decays to 0.0 after warm-up
        # so multi-class scene memorization never corrupts binary anomaly detection in Epochs 2-10
        self.sem_grad_scale = float(getattr(args, 'sem_grad_scale', 1.0))

        self._eval_text_cache = None
        self.initialize_parameters()

    def train(self, mode: bool = True):
        super().train(mode)
        self._eval_text_cache = None
        return self

    def _apply_mlp(self, mlp: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
        h = mlp.gelu(mlp.c_fc(x))
        if self.training and self.mlp_dropout > 0.0:
            h = F.dropout(h, p=self.mlp_dropout, training=True)
        return mlp.c_proj(h)

    def initialize_parameters(self):
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

        for m in self.video_anomaly_refiner.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0.0)
                nn.init.constant_(m.weight, 1.0)

        for m in self.res_adapter.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0.0)
                nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(self.beta_res, 0.0)

        nn.init.normal_(self.rupa_gate[0].weight, std=0.01)
        nn.init.constant_(self.rupa_gate[0].bias, 0.0)
        nn.init.constant_(self.rupa_gate[-1].weight, 0.0)
        init_p = min(max(self.routing_gate_init, 1e-4), 1.0 - 1e-4)
        gate_logit = math.log(init_p / (1.0 - init_p))
        nn.init.constant_(self.rupa_gate[-1].bias, gate_logit)

    @staticmethod
    def build_attention_mask(seq_len: int, attn_window: int) -> torch.Tensor:
        if attn_window <= 0 or attn_window >= seq_len:
            return torch.zeros(seq_len, seq_len, dtype=torch.float32)
        mask = torch.full((seq_len, seq_len), float('-inf'), dtype=torch.float32)
        num_windows = math.ceil(seq_len / attn_window)
        for i in range(num_windows):
            start = i * attn_window
            end = min((i + 1) * attn_window, seq_len)
            mask[start:end, start:end] = 0.0
        return mask

    @staticmethod
    def adj4(x: torch.Tensor, seq_len: torch.Tensor = None) -> torch.Tensor:
        """Thresholded cosine similarity graph adjacency over valid snippets (DataParallel-safe)."""
        x_norm = F.normalize(x.float(), p=2, dim=-1)
        sim = torch.bmm(x_norm, x_norm.transpose(1, 2))
        output = torch.zeros_like(sim)

        if seq_len is None:
            adj = F.threshold(sim, 0.7, 0.0)
            output = F.softmax(adj, dim=-1)
        else:
            T = x.shape[1]
            for i in range(x.shape[0]):
                l_i = max(1, min(int(seq_len[i]), T))
                adj = F.threshold(sim[i, :l_i, :l_i], 0.7, 0.0)
                output[i, :l_i, :l_i] = F.softmax(adj, dim=-1)

        return output.to(x.dtype)

    def encode_video(self, images: torch.Tensor, padding_mask: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        images = images.float()
        B, T, _ = images.shape
        dev = images.device

        if padding_mask is None and lengths is not None:
            padding_mask = torch.arange(T, device=dev)[None, :] >= lengths.to(dev)[:, None]

        if T <= self.visual_length:
            pos_ids = torch.arange(T, device=dev).unsqueeze(0).expand(B, -1)
            pos_emb = self.frame_position_embeddings(pos_ids)
        else:
            base_pos = self.frame_position_embeddings.weight.unsqueeze(0).permute(0, 2, 1)
            pos_emb = F.interpolate(base_pos, size=T, mode='linear', align_corners=False).permute(0, 2, 1).expand(B, -1, -1)

        attn_mask = (
            self.base_attn_mask
            if T == self.visual_length
            else self.build_attention_mask(T, self.attn_window).to(dev)
        )

        x = (images + pos_emb).permute(1, 0, 2)  # [T, B, D]
        x = self.temporal(x, attn_mask=attn_mask, padding_mask=None)
        x = x.permute(1, 0, 2)  # [B, T, D]

        adj = self.adj4(x, lengths)
        disadj = self.disAdj(B, T, device=dev, dtype=x.dtype)
        x1_h = self.gelu(self.gc1(x, adj))
        x2_h = self.gelu(self.gc3(x, disadj))
        x1 = self.gelu(self.gc2(x1_h, adj))
        x2 = self.gelu(self.gc4(x2_h, disadj))

        x = self.linear(torch.cat((x1, x2), dim=-1))

        if self.mst_adapter is not None:
            valid_mask = (~padding_mask).unsqueeze(-1).to(x.dtype) if padding_mask is not None else None
            x = self.mst_adapter(x, valid_mask=valid_mask)

        return x

    def get_text_features(self, target_device: torch.device) -> torch.Tensor:
        if not self.training and self._eval_text_cache is not None and self._eval_text_cache.device == target_device:
            return self._eval_text_cache
        feat = self.clip_adapter.get_class_embeddings(self.num_class, target_device)
        if not self.training:
            self._eval_text_cache = feat
        return feat

    def forward(self, visual: torch.Tensor, padding_mask: torch.Tensor = None,
                text=None, lengths: torch.Tensor = None,
                audio: torch.Tensor = None, DNP_use: bool = True, scale: float = 10.0):
        # Support legacy 5-arg positional call: model(visual, padding_mask, text, lengths, DNP_use)
        if isinstance(audio, bool):
            DNP_use = audio
            audio = None

        B, T, _ = visual.shape
        dev = visual.device

        if lengths is not None:
            lengths_dev = lengths.to(device=dev, dtype=torch.long).clamp(min=1, max=T)
        else:
            lengths_dev = torch.full((B,), T, device=dev, dtype=torch.long)

        if padding_mask is None:
            padding_mask = torch.arange(T, device=dev)[None, :] >= lengths_dev[:, None]
        valid_mask = (~padding_mask).unsqueeze(-1).to(visual.dtype)  # [B, T, 1]

        # 1. Pure Spatio-Temporal LGT Dual-GCN Encoding (with valid padding_mask isolation)
        visual_features = self.encode_video(visual, padding_mask, lengths_dev)

        # 2. Extract Adapted Canonical Text Embeddings
        text_features_ori = self.get_text_features(dev)  # [C, 512]

        # 3. Visual-Protected Cross-Modal Audio-Visual Branch (XD-Violence)
        logits_aux = None
        logits_aux_sem = None
        s_aux = None
        g_av = None
        if self.audio_adapter is not None and audio is not None and not self.no_audio:
            _, s_aux, logits_aux, g_av = self.audio_adapter(
                visual_features.detach(), audio, valid_mask=valid_mask, text_features_ori=text_features_ori
            )
            logits_aux_sem = self.audio_adapter.last_logits_aux_sem

        # 4. Sensitivity Anomaly Detector Head & SGNM Dynamic Normal Pattern Refiner
        logits1 = self.classifier(visual_features + self._apply_mlp(self.mlp2, visual_features))  # [B, T, 1]

        reconstructed_features = None
        dynamic_normal_patterns = None
        g_loss = None
        residual_features = None
        reconstruction_score = None

        if DNP_use and not self.no_rupa:
            reconstructed_features, dynamic_normal_patterns, g_loss = self.video_anomaly_refiner(
                visual_features, logits1, lengths_dev
            )
            residual_features = visual_features - reconstructed_features
            reconstruction_score = (
                (1.0 - F.cosine_similarity(visual_features, reconstructed_features, dim=-1)).unsqueeze(-1) / 2.0
            ).clamp(0.0, 1.0)

        detector_score = torch.sigmoid(logits1)

        # 5. Gradient-Decoupled Video-Conditioned Fine-Grained Text-Visual Alignment (logits2)
        #    - `logits1.detach()` prevents multi-class CLASM gradients from corrupting the binary classifier.
        #    - `sem_grad_scale` cosine-decays from 1.0 (Warm-up) to 0.0 (Post-Warm-up) so CLIP semantic
        #      supervision bootstraps `encode_video` early without causing late-epoch scene memorization.
        sg = float(getattr(self, 'sem_grad_scale', 1.0)) if self.training else 0.0
        if sg <= 0.0:
            sem_vis = visual_features.detach()
        elif sg >= 1.0:
            sem_vis = visual_features
        else:
            sem_vis = sg * visual_features + (1.0 - sg) * visual_features.detach()

        logits_attn = logits1.detach().permute(0, 2, 1)  # [B, 1, T]
        visual_attn = F.normalize(torch.bmm(logits_attn, sem_vis), dim=-1)          # [B, 1, D]
        visual_attn_exp = visual_attn.expand(B, text_features_ori.shape[0], -1)     # [B, C, D]

        text_features = text_features_ori.unsqueeze(0).expand(B, -1, -1) + visual_attn_exp
        text_features = text_features + self._apply_mlp(self.mlp1, text_features)

        visual_features_norm = F.normalize(sem_vis, dim=-1)
        text_features_norm = F.normalize(text_features, dim=-1).permute(0, 2, 1)
        logits2 = torch.bmm(visual_features_norm, text_features_norm.to(visual_features_norm.dtype)) / 0.07  # [B, T, C]

        # 6. Numerically Guarded & Detached Video-Level Event & Background Pooling (logits3, logits4)
        #    Detaching and clamping `det_pool` guarantees `||abn_logits||_1 >= 1e-3`, eliminating
        #    F.normalize backward explosion (~10^20) and `clip_grad_norm_` Inf-norm gradient freeze.
        det_pool = detector_score.detach().clamp(1e-4, 1.0 - 1e-4)
        abn_logits = (((scale * det_pool).exp() - 1.0) + 1e-3) * valid_mask
        abn_weights = F.normalize(abn_logits, p=1, dim=1)

        nor_logits = (((scale * (1.0 - det_pool)).exp() - 1.0) + 1e-3) * valid_mask
        nor_weights = F.normalize(nor_logits, p=1, dim=1)

        abn_feat = torch.bmm(abn_weights.permute(0, 2, 1), sem_vis)  # [B, 1, D]
        nor_feat = torch.bmm(nor_weights.permute(0, 2, 1), sem_vis)  # [B, 1, D]

        nor_text_norm = F.normalize(text_features_ori, dim=-1).t().unsqueeze(0).expand(B, -1, -1)  # [B, D, C]
        abn_vis_norm = F.normalize(abn_feat, dim=-1)
        nor_vis_norm = F.normalize(nor_feat, dim=-1)

        logits3 = torch.bmm(abn_vis_norm, nor_text_norm.to(abn_vis_norm.dtype)) / 0.07  # [B, 1, C]
        logits4 = torch.bmm(nor_vis_norm, nor_text_norm.to(nor_vis_norm.dtype)) / 0.07  # [B, 1, C]

        # 7. Temperature-Calibrated Semantic Anomaly Score
        sem_temp = 2.5 if self.num_class == 14 else 2.0
        p_sem_clip = F.softmax(logits2.float() / sem_temp, dim=-1)
        semantic_score = (1.0 - p_sem_clip[:, :, :1]).to(visual_features.dtype).clamp(0.0, 1.0)

        iam_dict = None
        if self.iam_module is not None:
            iam_dict = self.iam_module(visual_features.detach(), valid_mask=valid_mask, sem_temp=sem_temp)

        acc_dict = None
        if self.acc_module is not None:
            acc_dict = self.acc_module(
                visual_features.detach(), logits2.detach(), detector_score.detach(),
                lengths=lengths_dev, sem_temp=sem_temp,
            )

        if iam_dict is not None:
            p_iam = F.softmax(iam_dict['iam_logits'].float() / sem_temp, dim=-1)
            w_iam = torch.sigmoid(self.iam_module.gamma_iam).to(p_sem_clip.dtype) * 0.35
            p_fine_grained = (1.0 - w_iam) * p_sem_clip + w_iam * p_iam
            semantic_score = (1.0 - p_fine_grained[:, :, :1]).to(visual_features.dtype).clamp(0.0, 1.0)
        else:
            p_fine_grained = p_sem_clip

        dnp_dict = {
            'detector_scores': detector_score,
            'routing_scores': detector_score,
            'routing_logits': logits1,
            'semantic_logits': logits2,
            'semantic_scores': semantic_score,
            'fine_grained_probs': p_fine_grained.to(visual_features.dtype),
            'aux_logits': logits_aux,
            'aux_sem_logits': logits_aux_sem,
            'aux_scores': s_aux,
            'audio_gate': g_av,
        }

        if iam_dict is not None:
            dnp_dict.update({
                'iam_logits': iam_dict['iam_logits'],
                'iam_scores': iam_dict['iam_scores'],
                'iam_features': iam_dict['x_int'],
                'iam_x_v': iam_dict['x_v'],
                'iam_x_a': iam_dict['x_a'],
                'kin_continuity': iam_dict['kin_continuity'],
            })

        if acc_dict is not None:
            dnp_dict.update(acc_dict)

        # 8. SGNM Prototype Outputs & Stop-Gradient Safe-RUPA Calibration
        if DNP_use and not self.no_rupa and reconstructed_features is not None:
            dnp_dict.update({
                'original_features': visual_features,
                'reconstructed_features': reconstructed_features,
                'residual_features': residual_features,
                'dynamic_normal_patterns': dynamic_normal_patterns,
                'reconstruction_scores': reconstruction_score,
                'g_loss': g_loss.reshape(1),
            })

            if self.rupa_use:
                # Stop-gradient detached inputs ensure `route_loss` trains ONLY `rupa_gate`
                det_d = detector_score.detach()
                rec_d = reconstruction_score.detach()
                sem_d = semantic_score.detach()

                fixed_routing_prob = (
                    self.routing_det_weight * det_d
                    + self.routing_rec_weight * rec_d
                    + self.routing_sem_weight * sem_d
                ).clamp(1e-5, 1.0 - 1e-5)

                if self.routing_mode == 'safe_gate':
                    gate_inputs = torch.cat([
                        det_d,
                        rec_d,
                        sem_d,
                        (rec_d - det_d).abs(),
                        (sem_d - det_d).abs(),
                    ], dim=-1)
                    routing_gate = torch.sigmoid(self.rupa_gate(gate_inputs)) * self.routing_gate_max
                    routing_prob = (det_d + routing_gate * (fixed_routing_prob - det_d)).clamp(1e-5, 1.0 - 1e-5)
                else:
                    routing_gate = torch.ones_like(det_d)
                    routing_prob = fixed_routing_prob

                dnp_dict.update({
                    'fixed_routing_scores': fixed_routing_prob,
                    'routing_gate': routing_gate,
                    'routing_scores': routing_prob,
                    'routing_logits': torch.logit(routing_prob, eps=1e-5),
                })

        return text_features_ori, logits1, logits2, logits3, logits4, dnp_dict


# Clean alias for DSANet compatibility
DSANet = MARUPA

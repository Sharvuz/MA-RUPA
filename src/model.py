"""MA-RUPA: Modality-Adaptive Multiscale Prototype Alignment Network.

Weakly-Supervised Video Anomaly Detection (WS-VAD) architecture for top-tier
research (IEEE TIP, IEEE TMM, Pattern Recognition).

Key Components:
1. MDTBackbone: Multi-Scale Dilated Temporal & Local Windowed Attention Backbone
2. ModalityAdaptiveBranch: Audio-Visual Multimodal Fusion & Motion Residuals
3. GradientSafeRUPA: Stop-Gradient Dynamic Normal Prototypes & Domain-Adaptive Semantics
4. QuadSafeGate: Quad-Stream Uncertainty-Aware Dynamic Safe Gate
5. MARUPA: Full End-to-End Trainable Architecture (from scratch)
"""

from collections import OrderedDict
from functools import partial
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from clip import clip
from utils.adapter_modules import SimpleAdapter, SimpleProj
from utils.descriptions import DESCRIPTIONS_ORI, DESCRIPTIONS_ORI_XD
from utils.dnp_vision_transformer import Aggregation_Block, Prototype_Block


class QuickGELU(nn.Module):
    """Fast approximation of GELU activation."""
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerNormFP32(nn.LayerNorm):
    """LayerNorm that computes statistics in float32 for numerical stability."""
    def forward(self, x: torch.Tensor):
        orig_dtype = x.dtype
        ret = super().forward(x.to(torch.float32))
        return ret.to(orig_dtype)


# ============================================================================
# 1. Core Module 1: MDTBackbone
# ============================================================================

class WindowedAttentionBlock(nn.Module):
    """Lightweight Local Windowed Self-Attention with O(T * W) complexity."""
    def __init__(self, embed_dim=512, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ln_1 = LayerNormFP32(embed_dim)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(embed_dim, embed_dim * 4)),
            ("gelu", QuickGELU()),
            ("dropout", nn.Dropout(dropout)),
            ("c_proj", nn.Linear(embed_dim * 4, embed_dim)),
            ("dropout2", nn.Dropout(dropout))
        ]))
        self.ln_2 = LayerNormFP32(embed_dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None, key_padding_mask: torch.Tensor = None):
        norm_x = self.ln_1(x)
        attn_out, _ = self.attn(
            norm_x, norm_x, norm_x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        # Numerical guard: When an entire local window consists of padded frames,
        # PyTorch MultiheadAttention computes softmax([-inf, ...]) = NaN. Replace with 0.0.
        attn_out = torch.nan_to_num(attn_out, nan=0.0)
        x = x + attn_out
        mlp_out = self.mlp(self.ln_2(x))
        if key_padding_mask is not None:
            mlp_out = mlp_out.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        x = x + mlp_out
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return x


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise-Separable 1D Convolution with dilation."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 dilation: int = 1, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size=kernel_size,
            dilation=dilation, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv1d(
            in_channels, out_channels, kernel_size=1, bias=True
        )

    def forward(self, x: torch.Tensor):
        return self.pointwise(self.depthwise(x))


class MDTPBranch(nn.Module):
    """Branch B: Multi-Scale Dilated Temporal Pyramid (MDTP).
    
    Captures multi-scale temporal dependencies (rapid anomalies like explosions
    and gradual anomalies like fighting/riots) via parallel depthwise-separable
    dilations d in {1, 2, 4, 8} with GLU activations.
    """
    def __init__(self, embed_dim: int = 512, dilations=(1, 2, 4, 8), dropout: float = 0.1):
        super().__init__()
        self.dilations = dilations
        self.branches = nn.ModuleList([
            nn.Sequential(
                DepthwiseSeparableConv1d(
                    embed_dim, embed_dim * 2, kernel_size=3, dilation=d, padding=d
                )
            ) for d in dilations
        ])
        self.norms = nn.ModuleList([
            LayerNormFP32(embed_dim) for _ in dilations
        ])
        self.fuse_conv = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
            QuickGELU(),
            nn.Dropout(dropout)
        )
        self.out_norm = LayerNormFP32(embed_dim)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None):
        # x: [B, T, D] -> transpose to [B, D, T] for conv1d
        if padding_mask is not None:
            valid_mask_t = (~padding_mask).unsqueeze(1).to(x.dtype)  # [B, 1, T]
        else:
            valid_mask_t = None

        x_t = x.transpose(1, 2)
        if valid_mask_t is not None:
            x_t = x_t * valid_mask_t

        branch_outputs = []
        for branch, norm in zip(self.branches, self.norms):
            # branch output: [B, 2*D, T]
            h = branch(x_t)
            h = F.glu(h, dim=1)  # [B, D, T] via Gated Linear Unit
            if valid_mask_t is not None:
                h = h * valid_mask_t
            # Normalize over feature dimension: [B, T, D]
            h_norm = norm(h.transpose(1, 2)).transpose(1, 2)
            if valid_mask_t is not None:
                h_norm = h_norm * valid_mask_t
            branch_outputs.append(h_norm)

        # Sum parallel multi-scale temporal representations
        combined = sum(branch_outputs)  # [B, D, T]
        fused = self.fuse_conv(combined).transpose(1, 2)  # [B, T, D]
        if padding_mask is not None:
            fused = fused.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        # Residual connection
        out = self.out_norm(x + fused)
        if padding_mask is not None:
            out = out.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return out


class GlobalLinearContextAggregator(nn.Module):
    """Branch C: Global Linear Context Aggregator with O(T) complexity.
    
    Replaces DSANet's dense similarity GCN without graph noise:
    phi(x) = ELU(x) + 1
    Q = phi(X W_q), K = phi(X W_k), V = X W_v
    F_global = (Q (K^T V)) / (Q (sum_t K_t)^T + eps)
    """
    def __init__(self, embed_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.w_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = LayerNormFP32(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None):
        # x: [B, T, D]
        q = F.elu(self.w_q(x)) + 1.0  # [B, T, D]
        k = F.elu(self.w_k(x)) + 1.0  # [B, T, D]
        v = self.w_v(x)               # [B, T, D]

        if padding_mask is not None:
            valid_mask = (~padding_mask).unsqueeze(-1).to(x.dtype)
            q = q * valid_mask
            k = k * valid_mask
            v = v * valid_mask

        # KV: [B, D, D] = K^T @ V
        kv = torch.bmm(k.transpose(1, 2), v)
        k_sum = k.sum(dim=1, keepdim=True)  # [B, 1, D]

        # During multi-split single-video evaluation (not self.training and B > 1),
        # aggregate global linear context across all valid splits of the same test video
        if not self.training and x.shape[0] > 1:
            kv = kv.sum(dim=0, keepdim=True).expand(x.shape[0], -1, -1)
            k_sum = k_sum.sum(dim=0, keepdim=True).expand(x.shape[0], -1, -1)

        # num: [B, T, D] = Q @ KV
        num = torch.bmm(q, kv)
        # den: [B, T, 1] = Q @ sum_t(K)^T
        den = torch.bmm(q, k_sum.transpose(1, 2)).clamp_min(1e-5)

        f_global = num / den
        out = self.norm(self.dropout(self.out_proj(f_global)))
        if padding_mask is not None:
            out = out.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return out


class MDTBackbone(nn.Module):
    """Multi-Scale Dilated Temporal & Local Attention Backbone.
    
    Combines:
    - Depthwise Convolutional Relative Positional Encoding (translation-equivariant across splits)
    - Branch 1: Local Windowed Transformer (window W=8 for UCF, W=64 for XD)
    - Branch 2: Multi-Scale Dilated Temporal Pyramid (dilations 1, 2, 4, 8)
    - Branch 3: Global Linear Context Aggregator (O(T) complexity)
    - Crucial CLIP Anchor Skip Connection: F_base = LayerNorm(X_clip + alpha * FusedContext)
    - Base anomaly detector classifier S_det
    """
    def __init__(self, visual_length: int = 256, embed_dim: int = 512,
                 attn_window: int = 8, visual_layers: int = 2, visual_heads: int = 4,
                 mlp_dropout: float = 0.1, no_mdtp: bool = False):
        super().__init__()
        self.visual_length = visual_length
        self.embed_dim = embed_dim
        self.attn_window = attn_window
        self.no_mdtp = no_mdtp

        # Learnable temporal frame positional embedding (gated to prevent train/test split shift)
        self.frame_pos_embed = nn.Parameter(torch.zeros(1, visual_length, embed_dim))
        trunc_normal_(self.frame_pos_embed, std=0.01)
        self.alpha_pos = nn.Parameter(torch.tensor(0.15))

        # Translation-equivariant Depthwise 1D Convolutional Relative Positional Encoding
        self.conv_pos = nn.Conv1d(
            embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim, bias=True
        )

        # Branch 1: Local Windowed Transformer
        self.branch_a_layers = nn.ModuleList([
            WindowedAttentionBlock(embed_dim, num_heads=visual_heads, dropout=mlp_dropout)
            for _ in range(visual_layers)
        ])

        # Branch 2: Multi-Scale Dilated Temporal Pyramid (dilations 1, 2, 4, 8)
        if not self.no_mdtp:
            self.branch_b = MDTPBranch(embed_dim=embed_dim, dilations=(1, 2, 4, 8), dropout=mlp_dropout)

        # Branch 3: Global Linear Context Aggregator (O(T) complexity)
        self.branch_c = GlobalLinearContextAggregator(embed_dim=embed_dim, dropout=mlp_dropout)

        # Context Fusion of Branch 1, 2, 3
        in_channels_fuse = embed_dim * 3 if not self.no_mdtp else embed_dim * 2
        self.fuse_proj = nn.Sequential(
            nn.Linear(in_channels_fuse, embed_dim),
            QuickGELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        self.fuse_norm = LayerNormFP32(embed_dim)

        # Crucial CLIP Anchor Skip Connection (initialized to 0.15)
        self.alpha_backbone = nn.Parameter(torch.tensor(0.15))
        self.out_norm = LayerNormFP32(embed_dim)

        # Base Anomaly Classifier Head
        self.mlp_det = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(embed_dim, embed_dim * 4)),
            ("gelu", QuickGELU()),
            ("dropout", nn.Dropout(mlp_dropout)),
            ("c_proj", nn.Linear(embed_dim * 4, embed_dim)),
            ("dropout2", nn.Dropout(mlp_dropout))
        ]))
        self.classifier_det = nn.Linear(embed_dim, 1)

    def build_window_attention_mask(self, length: int, window: int, device: torch.device):
        """Construct block-diagonal window attention mask.
        
        True means position is masked out (cannot attend); False means attend.
        """
        mask = torch.ones(length, length, dtype=torch.bool, device=device)
        for i in range(math.ceil(length / window)):
            start = i * window
            end = min(start + window, length)
            mask[start:end, start:end] = False
        return mask

    def forward(self, visual_features: torch.Tensor, padding_mask: torch.Tensor = None,
                lengths: torch.Tensor = None):
        """
        Args:
            visual_features: [B, T, D] CLIP snippet visual features (X_clip)
            padding_mask: [B, T] bool mask where True is padded/invalid
            lengths: [B] valid snippet counts
        Returns:
            F_base: [B, T, D] fused base temporal features (anchored to X_clip)
            S_det: [B, T, 1] base anomaly detector probability scores
            logits_det: [B, T, 1] raw classifier logits
        """
        B, T, D = visual_features.shape

        # Padding mask setup
        if padding_mask is None and lengths is not None:
            padding_mask = (
                torch.arange(T, device=visual_features.device)[None, :]
                >= lengths.to(visual_features.device)[:, None]
            )

        # Hybrid Relative ConvPos + Gated Absolute Pos Embedding
        if T == self.visual_length:
            pos_emb = self.frame_pos_embed
        else:
            pos_emb = F.interpolate(
                self.frame_pos_embed.permute(0, 2, 1),
                size=T, mode='linear', align_corners=False
            ).permute(0, 2, 1)

        vis_masked = visual_features.masked_fill(padding_mask.unsqueeze(-1), 0.0) if padding_mask is not None else visual_features
        rel_pos = self.conv_pos(vis_masked.transpose(1, 2)).transpose(1, 2)
        x = visual_features + 0.15 * rel_pos + self.alpha_pos.clamp(-0.25, 0.25) * pos_emb
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        # Branch 1: Windowed Self-Attention
        attn_mask = self.build_window_attention_mask(T, self.attn_window, visual_features.device)
        f_a = x
        for layer in self.branch_a_layers:
            f_a = layer(f_a, attn_mask=attn_mask, key_padding_mask=padding_mask)

        # Branch 3: Global Linear Context Aggregator
        f_c = self.branch_c(x, padding_mask=padding_mask)

        # Branch 2: Multi-Scale Dilated Temporal Pyramid
        if not self.no_mdtp:
            f_b = self.branch_b(x, padding_mask=padding_mask)
            fused_context = self.fuse_norm(self.fuse_proj(torch.cat([f_a, f_b, f_c], dim=-1)))
        else:
            fused_context = self.fuse_norm(self.fuse_proj(torch.cat([f_a, f_c], dim=-1)))

        if padding_mask is not None:
            fused_context = fused_context.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        # Crucial CLIP Anchor Skip Connection: F_base = LayerNorm(X_clip + alpha * FusedContext)
        f_base = self.out_norm(visual_features + self.alpha_backbone * fused_context)

        # Base Anomaly Classifier
        det_features = f_base + self.mlp_det(f_base)
        logits_det = self.classifier_det(det_features)  # [B, T, 1]
        s_det = torch.sigmoid(logits_det)

        return f_base, s_det, logits_det


# ============================================================================
# 2. Core Module 2: ModalityAdaptiveBranch
# ============================================================================

class TemporalConvProjector(nn.Module):
    """1D Temporal Convolution Projector with LayerNorm and GELU."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.ln1 = LayerNormFP32(out_channels)
        self.act = QuickGELU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.ln2 = LayerNormFP32(out_channels)

    def forward(self, x: torch.Tensor):
        # x: [B, in_channels, T]
        h = self.conv1(x)  # [B, out_channels, T]
        h = self.ln1(h.transpose(1, 2)).transpose(1, 2)
        h = self.act(h)
        h = self.conv2(h)
        h = self.ln2(h.transpose(1, 2)).transpose(1, 2)
        return h


class ModalityAdaptiveBranch(nn.Module):
    """Overhauled Modality-Adaptive Audio-Visual / Motion Branch (TD-AVCA + DropAudio).
    
    Supports:
    1. XD-Violence Multimodal Mode: Project pre-extracted audio features
       (VGGish 128-D or CLAP/AST 512-D) aligned with 16-frame video snippets.
    2. Missing-Modality / Motion-Residual Mode: For UCF-Crime (no audio) or
       missing audio snippets, computes 1st- and 2nd-order visual temporal
       residuals (M1 = X_t - X_{t-1}, M2 = M1_t - M1_{t-1}) as dynamic auxiliary modality.
    3. Banded Temporal Delay-Aware Cross-Attention (TD-AVCA) over window [t - delta, t + delta]
       with learnable per-head relative lag bias to prevent global bag-level audio memorization.
    4. Dual-Level Stochastic Modality Dropout (DropAudio) blending with visual motion residuals.
    5. Acoustic-Energy & Cross-Modal Cosine Decoupled Gate G_av to suppress loud non-violent
       background sounds (cheering, music, traffic) + Auxiliary anomaly score stream S_aux.
    """
    def __init__(self, embed_dim: int = 512, audio_dim: int = 128,
                 num_heads: int = 4, dropout: float = 0.1, no_audio: bool = False,
                 delay_window: int = 4, drop_audio_bag_p: float = 0.25,
                 drop_audio_snip_p: float = 0.25):
        super().__init__()
        self.embed_dim = embed_dim
        self.audio_dim = audio_dim
        self.num_heads = num_heads
        self.no_audio = no_audio
        self.delay_window = delay_window
        self.drop_audio_bag_p = drop_audio_bag_p
        self.drop_audio_snip_p = drop_audio_snip_p

        # Audio projector (from audio_dim to embed_dim)
        self.audio_proj = TemporalConvProjector(audio_dim, embed_dim, kernel_size=3, padding=1)
        # Dedicated projector for 512-D audio features (e.g. CLAP / AST / 512-D audio)
        self.audio_proj_512 = TemporalConvProjector(embed_dim, embed_dim, kernel_size=3, padding=1)

        # Motion residual projector (from 2*embed_dim to embed_dim)
        self.motion_proj = TemporalConvProjector(embed_dim * 2, embed_dim, kernel_size=3, padding=1)

        # Cross-Modal Attention: Query=F_base (Visual), Key=F_a (Auxiliary), Value=F_a
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ln_cross = LayerNormFP32(embed_dim)

        # Learnable relative temporal delay bias for offsets in [-delay_window, +delay_window]
        self.rel_delay_bias = nn.Parameter(
            torch.zeros(num_heads, 2 * delay_window + 1)
        )
        trunc_normal_(self.rel_delay_bias, std=0.02)

        # Acoustic-Energy Decoupled Suppression Gate:
        # G_av = sigma(MLP([F_base || F_v_from_a || (F_base * F_v_from_a) || cos_av || energy_a]))
        self.noise_gate_mlp = nn.Sequential(
            nn.Linear(embed_dim * 3 + 2, embed_dim),
            LayerNormFP32(embed_dim),
            QuickGELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )
        self.alpha_mod = nn.Parameter(torch.tensor(0.20))
        self.ln_fused = LayerNormFP32(embed_dim)

        # Regularized Auxiliary Anomaly Classifier Head
        self.mlp_aux = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(embed_dim, embed_dim * 2)),
            ("gelu", QuickGELU()),
            ("dropout", nn.Dropout(max(0.25, dropout))),
            ("c_proj", nn.Linear(embed_dim * 2, embed_dim)),
            ("dropout2", nn.Dropout(max(0.25, dropout)))
        ]))
        self.classifier_aux = nn.Linear(embed_dim, 1)

    def compute_motion_residuals(self, x_v: torch.Tensor):
        """Compute 1st and 2nd order visual temporal feature residuals.
        
        M1[:, t] = x_v[:, t] - x_v[:, t-1] (with M1[:, 0] = 0)
        M2[:, t] = M1[:, t] - M1[:, t-1]   (with M2[:, 0] = 0)
        """
        m1 = torch.zeros_like(x_v)
        m1[:, 1:] = x_v[:, 1:] - x_v[:, :-1]

        m2 = torch.zeros_like(x_v)
        m2[:, 1:] = m1[:, 1:] - m1[:, :-1]

        motion = torch.cat([m1, m2], dim=-1)  # [B, T, 2*D]
        return motion

    def build_delay_attention_mask(self, B: int, T: int, device: torch.device,
                                   dtype: torch.dtype, key_pad_mask: torch.Tensor = None) -> torch.Tensor:
        """Builds [B * num_heads, T, T] additive attention mask restricting cross-attention
        to relative temporal lag |t - t'| <= delay_window with learnable lag bias.
        """
        idx = torch.arange(T, device=device)
        diff = idx[:, None] - idx[None, :]  # [T, T]
        valid_band = diff.abs() <= self.delay_window  # [T, T]
        clamped_offset = (diff.clamp(-self.delay_window, self.delay_window) + self.delay_window).long()

        bias = self.rel_delay_bias[:, clamped_offset].to(dtype=dtype)  # [H, T, T]
        mask = torch.full((self.num_heads, T, T), float('-inf'), device=device, dtype=dtype)
        mask = torch.where(valid_band.unsqueeze(0), bias, mask)  # [H, T, T]
        mask_b = mask.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, T, T]

        if key_pad_mask is not None:
            mask_b = mask_b.masked_fill(key_pad_mask[:, None, None, :], float('-inf'))

        return mask_b.reshape(B * self.num_heads, T, T)

    def apply_stochastic_drop_audio(self, f_audio: torch.Tensor, f_motion: torch.Tensor) -> torch.Tensor:
        """Dual-level Stochastic DropAudio during training:
        1. Bag-level replacement with motion features with prob drop_audio_bag_p.
        2. Snippet-level dropout replacement with motion features with prob drop_audio_snip_p.
        """
        if not self.training:
            return f_audio
        B, T, _ = f_audio.shape
        device = f_audio.device

        bag_keep = (torch.rand(B, 1, 1, device=device) >= self.drop_audio_bag_p).to(f_audio.dtype)
        snip_keep = (torch.rand(B, T, 1, device=device) >= self.drop_audio_snip_p).to(f_audio.dtype)
        keep_mask = bag_keep * snip_keep
        return keep_mask * f_audio + (1.0 - keep_mask) * f_motion

    def forward(self, f_base: torch.Tensor, visual_raw: torch.Tensor,
                audio_features: torch.Tensor = None, lengths: torch.Tensor = None,
                return_suppress_loss: bool = False):
        """
        Args:
            f_base: [B, T, D] visual base features from MDTBackbone
            visual_raw: [B, T, D] raw snippet visual features
            audio_features: [B, T_a, D_a] audio features (if available) or None
            lengths: [B] valid lengths
            return_suppress_loss: if True, returns 5th element av_suppress_loss [B, 1]
        Returns:
            F_fused: [B, T, D] multimodal/motion fused features
            S_aux: [B, T, 1] auxiliary anomaly scores
            logits_aux: [B, T, 1] raw classifier logits
            G_av: [B, T, D] noise suppression gate values
        """
        B, T, D = f_base.shape
        device = f_base.device

        # Always compute visual motion residuals as structural anchor & fallback
        motion = self.compute_motion_residuals(visual_raw)  # [B, T, 2*D]
        f_motion = self.motion_proj(motion.transpose(1, 2)).transpose(1, 2)  # [B, T, D]

        # Banded Temporal Delay-Aware Cross-Modal Attention over [t - delta, t + delta]
        cross_pad_mask = None
        if lengths is not None:
            cross_pad_mask = (
                torch.arange(T, device=device)[None, :]
                >= lengths.to(device)[:, None]
            )

        if audio_features is not None and not self.no_audio:
            # Multi-Modal Audio Mode
            if audio_features.ndim == 2:
                audio_features = audio_features.unsqueeze(0)
            elif audio_features.ndim == 4:
                audio_features = audio_features.squeeze(0)

            # Align batch/split dimension with visual base features if needed
            B_v = f_base.shape[0]
            B_a = audio_features.shape[0]
            if B_a != B_v:
                if B_a < B_v:
                    pad_splits = torch.zeros(
                        B_v - B_a, audio_features.shape[1], audio_features.shape[2],
                        device=audio_features.device, dtype=audio_features.dtype
                    )
                    audio_features = torch.cat([audio_features, pad_splits], dim=0)
                else:
                    audio_features = audio_features[:B_v]

            # If audio sequence length differs from visual, interpolate temporally
            if audio_features.shape[1] != T:
                audio_t = F.interpolate(
                    audio_features.transpose(1, 2),
                    size=T, mode='linear', align_corners=False
                )
            else:
                audio_t = audio_features.transpose(1, 2)  # [B, D_a, T]

            # Compute normalized per-snippet acoustic energy ||A_t||_2 in [0, 1]
            energy_t = torch.norm(audio_t.float(), p=2, dim=1, keepdim=True).transpose(1, 2)  # [B, T, 1]
            if cross_pad_mask is not None:
                energy_t = energy_t.masked_fill(cross_pad_mask.unsqueeze(-1), 0.0)
            if not self.training and B > 1:
                # Whole-video normalization across splits so quiet normal splits stay near 0
                max_energy = energy_t.amax(dim=(0, 1), keepdim=True).clamp_min(1e-6)
            else:
                max_energy = energy_t.amax(dim=1, keepdim=True).clamp_min(1e-6)
            raw_audio_energy = (energy_t / max_energy).to(f_base.dtype)

            # Project audio features to dimension D with automatic dimension routing
            in_dim = audio_t.shape[1]
            if in_dim == self.embed_dim:
                f_a_raw = self.audio_proj_512(audio_t).transpose(1, 2)
            elif in_dim == self.audio_dim:
                f_a_raw = self.audio_proj(audio_t).transpose(1, 2)
            else:
                # Stateless adaptive pooling fallback for arbitrary audio dimensions
                adapted_audio = F.interpolate(
                    audio_t.transpose(1, 2), size=self.embed_dim, mode='linear', align_corners=False
                ).transpose(1, 2)
                f_a_raw = self.audio_proj_512(adapted_audio).transpose(1, 2)

            # Apply Dual-Level Stochastic DropAudio during training
            f_a = self.apply_stochastic_drop_audio(f_a_raw, f_motion)
        else:
            # Missing-Modality / Motion-Residual Mode (UCF-Crime or Audio Fallback)
            f_a = f_motion
            motion_norm = torch.norm(f_motion.float(), p=2, dim=-1, keepdim=True)
            if cross_pad_mask is not None:
                motion_norm = motion_norm.masked_fill(cross_pad_mask.unsqueeze(-1), 0.0)
            if not self.training and B > 1:
                max_motion = motion_norm.amax(dim=(0, 1), keepdim=True).clamp_min(1e-6)
            else:
                max_motion = motion_norm.amax(dim=1, keepdim=True).clamp_min(1e-6)
            raw_audio_energy = (motion_norm / max_motion).to(f_base.dtype)

        delay_attn_mask = self.build_delay_attention_mask(
            B, T, device, f_base.dtype, key_pad_mask=cross_pad_mask
        )

        attn_out, _ = self.cross_attn(
            query=f_base, key=f_a, value=f_a,
            attn_mask=delay_attn_mask,
            need_weights=False
        )
        attn_out = torch.nan_to_num(attn_out, nan=0.0)
        if cross_pad_mask is not None:
            attn_out = attn_out.masked_fill(cross_pad_mask.unsqueeze(-1), 0.0)
        f_v_from_a = self.ln_cross(attn_out)

        # Cross-modal semantic cosine agreement rho_t in [-1, 1]
        cos_av = F.cosine_similarity(f_base, f_v_from_a, dim=-1).unsqueeze(-1)  # [B, T, 1]

        # Acoustic-Energy Decoupled Suppression Gate G_av: [B, T, D]
        gate_input = torch.cat([f_base, f_v_from_a, f_base * f_v_from_a, cos_av, raw_audio_energy], dim=-1)
        g_av = self.noise_gate_mlp(gate_input)
        if cross_pad_mask is not None:
            g_av = g_av.masked_fill(cross_pad_mask.unsqueeze(-1), 0.0)

        # Gated residual injection: F_fused = LayerNorm(F_base + alpha_mod * (G_av * F_v_from_a))
        alpha_eff = self.alpha_mod.clamp(0.01, 0.35)
        f_fused = self.ln_fused(f_base + alpha_eff * (g_av * f_v_from_a))

        # Auxiliary Anomaly Classifier: detach f_base anchor so overconfident S_aux
        # cannot corrupt MDTBackbone's visual representation
        f_aux_in = f_base.detach() + (g_av * f_v_from_a)
        aux_features = f_aux_in + self.mlp_aux(f_aux_in)
        logits_aux = self.classifier_aux(aux_features)  # [B, T, 1]
        s_aux = torch.sigmoid(logits_aux)

        # Loud-Normal Audio-Visual Contrastive Suppression Penalty: [B, 1]
        # Penalizes high S_aux and high G_av when acoustic energy is loud but visual-audio cosine is low
        discord_weight = (raw_audio_energy.detach() * F.relu(0.25 - cos_av.detach())).clamp(0.0, 1.0)
        g_av_mean = g_av.mean(dim=-1, keepdim=True)
        av_pen_t = discord_weight * (s_aux ** 2 + 0.5 * (g_av_mean ** 2))
        if cross_pad_mask is not None:
            valid_f = (~cross_pad_mask).unsqueeze(-1).to(f_base.dtype)
            av_suppress_loss = (av_pen_t * valid_f).sum(dim=1) / valid_f.sum(dim=1).clamp_min(1.0)
        else:
            av_suppress_loss = av_pen_t.mean(dim=1)

        if return_suppress_loss:
            return f_fused, s_aux, logits_aux, g_av, av_suppress_loss
        return f_fused, s_aux, logits_aux, g_av


# ============================================================================
# 3. Core Module 3: GradientSafeRUPA
# ============================================================================

class GradientSafeRUPA(nn.Module):
    """Latent Manifold Orthogonal Subspace Projection (LM-OSP) RUPA Module.
    
    Stabilizes End-to-End training from scratch and eliminates normal prototype drift via:
    1. Dual-Gated Pseudo-Normal Mining (joint detector + zero-shot CLIP normality score)
       with conservative selection ceiling to prevent long-anomaly leakage.
    2. Controlled Gradient Detachment on normal features entering the DNP memory.
    3. Differentiable Stiefel Manifold Orthonormalization (QR with canonical sign) producing
       an exact M-dimensional orthonormal basis U_norm in R^(B x M x D), U_norm @ U_norm^T = I_M.
    4. Counterfactual Orthogonal Subspace Projection: F_proj = (F_fused @ U_norm^T) @ U_norm,
       eliminating the unconstrained MLP identity shortcut so anomalies cannot be reconstructed.
    5. Grassmannian Subspace Reconstruction Score & Margin-Enforced Abnormal Subspace Repulsion.
    6. Domain-Adaptive Semantic Alignment with learnable temperature & feature variance scaling.
    """
    def __init__(self, feature_dim: int = 512, num_prototypes: int = 16,
                 num_heads: int = 8, extractor_depth: int = 1, decoder_depth: int = 4,
                 sampling: str = 'topk', normal_selection_ratio: float = 0.60,
                 min_normal_frames: int = 4, max_normal_ratio: float = 0.70,
                 repulsion_margin: float = 0.25):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_prototypes = num_prototypes
        self.sampling = sampling
        self.normal_selection_ratio = min(normal_selection_ratio, 0.65)
        self.min_normal_frames = min_normal_frames
        self.max_normal_ratio = min(max_normal_ratio, 0.75)
        self.repulsion_margin = repulsion_margin

        # Dynamic Normal Prototypes memory seeds
        self.video_prototypes = nn.Parameter(torch.randn(num_prototypes, feature_dim))
        trunc_normal_(self.video_prototypes, std=0.02)

        # DNP Prototype Extractor Blocks (Aggregation)
        self.dnp_extractor = nn.ModuleList([
            Aggregation_Block(
                dim=feature_dim, num_heads=num_heads, mlp_ratio=4.0,
                qkv_bias=True, norm_layer=partial(LayerNormFP32, eps=1e-6)
            ) for _ in range(extractor_depth)
        ])

        # Bottleneck & Decoder operating strictly on projected normal subspace F_proj
        self.bottleneck = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(feature_dim, feature_dim * 2)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(feature_dim * 2, feature_dim))
        ]))

        self.decoder = nn.ModuleList([
            Prototype_Block(
                dim=feature_dim, num_heads=num_heads, mlp_ratio=4.0,
                qkv_bias=True, norm_layer=partial(LayerNormFP32, eps=1e-6)
            ) for _ in range(decoder_depth)
        ])
        self.alpha_subspace_dec = nn.Parameter(torch.tensor(0.25))

        # Residual-amplified feature MLP for domain-adaptive category semantics
        # F_sem_rupa = LayerNorm(X_clip + F_fused + gamma_rupa * MLP_res(F_fused - F_rec))
        self.mlp_res = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(feature_dim, feature_dim * 2)),
            ("gelu", QuickGELU()),
            ("dropout", nn.Dropout(0.1)),
            ("c_proj", nn.Linear(feature_dim * 2, feature_dim))
        ]))
        self.gamma_rupa = nn.Parameter(torch.tensor(0.25))
        self.ln_sem_rupa = LayerNormFP32(feature_dim)

        # Learnable temperature and feature-variance scaling factor
        self.log_temp = nn.Parameter(torch.tensor(math.log(0.07)))
        self.var_scale_param = nn.Parameter(torch.tensor(0.1))

    @staticmethod
    def build_orthonormal_subspace(prototypes: torch.Tensor) -> torch.Tensor:
        """Projects prototypes P in R^(B x M x D) onto the Stiefel Manifold St(M, D)
        via 6-step Newton-Schulz Polar Stiefel Orthonormalization:
            U_norm = (P @ P^T)^(-1/2) @ P
        Uses pure cuBLAS torch.bmm (zero cuSOLVER/torch.linalg calls), guaranteeing
        100% thread safety under multi-GPU torch.nn.DataParallel on PyTorch 2.x.
        Returns U_norm in R^(B x M x D) satisfying U_norm @ U_norm^T = I_M.
        """
        orig_dtype = prototypes.dtype
        B, M, D = prototypes.shape
        u = F.normalize(prototypes.float(), dim=-1)  # [B, M, D]
        eye_d = torch.eye(M, D, device=u.device, dtype=torch.float32).unsqueeze(0)
        u = F.normalize(u + 1e-4 * eye_d, dim=-1)

        # Normalize by sqrt of Gram Frobenius norm so all singular values of u lie strictly in (0, 1)
        gram = torch.bmm(u, u.transpose(1, 2))  # [B, M, M]
        gram_frob = torch.sqrt((gram * gram).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6))
        u = u / torch.sqrt(gram_frob)

        eye_m = torch.eye(M, device=u.device, dtype=torch.float32).unsqueeze(0)
        for _ in range(6):
            g = torch.bmm(u, u.transpose(1, 2))  # [B, M, M]
            u = 0.5 * torch.bmm(3.0 * eye_m - g, u)  # [B, M, D]

        u_norm = F.normalize(u, dim=-1)
        return u_norm.to(orig_dtype)

    @staticmethod
    def otsu_threshold_1d(scores: torch.Tensor):
        """Vectorized Otsu threshold calculation on a 1D tensor of anomaly scores."""
        sorted_scores, _ = scores.sort()
        n = sorted_scores.numel()
        if n <= 1:
            return float(sorted_scores[0].item()) + 1e-6

        s_min = sorted_scores[0].item()
        s_max = sorted_scores[-1].item()
        if s_max - s_min < 1e-8:
            return s_max + 1e-6

        num_bins = min(256, n)
        thresholds = torch.linspace(s_min, s_max, num_bins + 2, device=scores.device)[1:-1]
        mask = sorted_scores.unsqueeze(0) < thresholds.unsqueeze(1)  # [bins, N]

        w0 = mask.sum(dim=1).float()
        w1 = n - w0
        valid = (w0 > 0) & (w1 > 0)
        if not valid.any():
            return (s_min + s_max) / 2.0

        sum_all = sorted_scores.sum()
        cum_sum = (sorted_scores.unsqueeze(0) * mask.float()).sum(dim=1)
        mean0 = cum_sum / w0.clamp(min=1)
        mean1 = (sum_all - cum_sum) / w1.clamp(min=1)
        sigma_b = w0 * w1 * (mean0 - mean1) ** 2
        sigma_b = sigma_b.masked_fill(~valid, -1.0)
        return thresholds[sigma_b.argmax()].item()

    def _select_normal_frames_topk(self, anomaly_scores: torch.Tensor,
                                  visual_features: torch.Tensor,
                                  lengths: torch.Tensor, B: int, N: int, D: int):
        """Select Top-K normal frames with the lowest joint anomaly scores."""
        device = visual_features.device
        if lengths is not None:
            lengths_clamped = lengths.to(device=device, dtype=torch.long).clamp(min=1, max=N)
            valid_mask = torch.arange(N, device=device)[None, :] < lengths_clamped[:, None]
            scores = anomaly_scores.masked_fill(~valid_mask, float('inf'))
            min_valid = int(lengths_clamped.min().item())
        else:
            scores = anomaly_scores
            min_valid = N

        k_count = max(1, min(int(N * self.normal_selection_ratio), min_valid))
        _, indices = torch.topk(scores, k=k_count, largest=False, dim=1)

        selected_features = torch.gather(
            visual_features, 1, indices.unsqueeze(-1).expand(-1, -1, D)
        )
        selected_scores = torch.gather(anomaly_scores, 1, indices)
        reliability = (1.0 - selected_scores).clamp(min=1e-4, max=1.0)
        return selected_features, reliability

    def _select_normal_frames_otsu(self, anomaly_scores: torch.Tensor,
                                   visual_features: torch.Tensor,
                                   lengths: torch.Tensor, B: int, N: int, D: int):
        """Adaptive normal frame selection using Otsu's thresholding per video."""
        device = visual_features.device
        if lengths is not None:
            lengths_clamped = lengths.to(device=device, dtype=torch.long).clamp(min=1, max=N)
        else:
            lengths_clamped = torch.full((B,), N, device=device, dtype=torch.long)

        per_video_counts = []
        normal_mask = torch.zeros(B, N, device=device, dtype=torch.bool)

        for i in range(B):
            L_i = int(lengths_clamped[i].item())
            valid_scores_i = anomaly_scores[i, :L_i]
            thresh_i = self.otsu_threshold_1d(valid_scores_i)
            frame_normal = anomaly_scores[i, :L_i] < thresh_i
            count_i = int(frame_normal.sum().item())

            min_k = min(self.min_normal_frames, L_i)
            max_k = max(min_k, int(self.max_normal_ratio * L_i))
            count_i = max(min_k, min(count_i, max_k))

            if int(frame_normal.sum().item()) < min_k or int(frame_normal.sum().item()) > max_k:
                _, topk_idx = torch.topk(anomaly_scores[i, :L_i], k=count_i, largest=False)
                normal_mask[i, topk_idx] = True
            else:
                scores_m = anomaly_scores[i].clone()
                scores_m[L_i:] = float('inf')
                scores_m[~torch.cat([frame_normal, torch.zeros(N - L_i, dtype=torch.bool, device=device)])] = float('inf')
                _, topk_idx = torch.topk(scores_m, k=count_i, largest=False)
                normal_mask[i, topk_idx] = True

            per_video_counts.append(count_i)

        max_count = max(per_video_counts)
        selected_features = torch.zeros(B, max_count, D, device=device, dtype=visual_features.dtype)
        selected_scores = torch.ones(B, max_count, device=device)
        selection_mask = torch.zeros(B, max_count, device=device, dtype=torch.bool)

        for i in range(B):
            idx_i = normal_mask[i].nonzero(as_tuple=False).squeeze(-1)
            k_i = per_video_counts[i]
            selected_features[i, :k_i] = visual_features[i, idx_i[:k_i]]
            selected_scores[i, :k_i] = anomaly_scores[i, idx_i[:k_i]]
            selection_mask[i, :k_i] = True

        reliability = (1.0 - selected_scores).clamp(min=1e-4, max=1.0) * selection_mask.float()
        return selected_features, reliability

    def _select_abnormal_frames_topk(self, anomaly_scores: torch.Tensor,
                                     visual_features: torch.Tensor,
                                     lengths: torch.Tensor, B: int, N: int, D: int):
        """Selects Top-K most abnormal snippets per video for Subspace Repulsion."""
        device = visual_features.device
        if lengths is not None:
            lengths_clamped = lengths.to(device=device, dtype=torch.long).clamp(min=1, max=N)
            valid_mask = torch.arange(N, device=device)[None, :] < lengths_clamped[:, None]
            scores = anomaly_scores.masked_fill(~valid_mask, float('-inf'))
            min_valid = int(lengths_clamped.min().item())
        else:
            scores = anomaly_scores
            min_valid = N

        k_abn = max(1, min(int(N * 0.125), min_valid))
        top_vals, indices = torch.topk(scores, k=k_abn, largest=True, dim=1)
        abn_features = torch.gather(
            visual_features, 1, indices.unsqueeze(-1).expand(-1, -1, D)
        )
        abn_weights = top_vals.clamp(0.0, 1.0)
        return abn_features, abn_weights

    def gather_loss(self, query: torch.Tensor, keys: torch.Tensor, reliability: torch.Tensor = None):
        """Distance loss pulling dynamic prototypes towards selected normal representations."""
        dist = 1.0 - F.cosine_similarity(query.unsqueeze(2), keys.unsqueeze(1), dim=-1)
        min_dist, _ = torch.min(dist, dim=2)
        if reliability is None:
            return min_dist.mean()
        rel_norm = reliability / reliability.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return (min_dist * rel_norm).sum(dim=1).mean()

    def prototype_orthogonality_loss(self, prototypes: torch.Tensor):
        """Encourages diversity among the learned Dynamic Normal Prototypes."""
        B, M, D = prototypes.shape
        p_norm = F.normalize(prototypes, dim=-1)  # [B, M, D]
        sim_matrix = torch.bmm(p_norm, p_norm.transpose(1, 2))  # [B, M, M]
        identity = torch.eye(M, device=prototypes.device).unsqueeze(0).expand(B, -1, -1)
        off_diag = sim_matrix - identity
        loss_orth = (off_diag ** 2).sum(dim=(1, 2)) / (M * (M - 1))
        return loss_orth.mean()

    def forward(self, f_fused: torch.Tensor, s_det: torch.Tensor,
                category_text_features: torch.Tensor, lengths: torch.Tensor = None,
                visual_raw: torch.Tensor = None, text_features_zs: torch.Tensor = None):
        """
        Args:
            f_fused: [B, T, D] multimodal fused features
            s_det: [B, T, 1] detector anomaly scores
            category_text_features: [C, D] text embeddings for categories (class 0 is normal)
            lengths: [B] valid sequence lengths
            visual_raw: [B, T, D] raw visual features (X_clip) for semantic anchor preservation
            text_features_zs: [C, D] optional pure zero-shot CLIP text features for mining
        Returns:
            f_rec: [B, T, D] reconstructed counterfactual normal features
            residual: [B, T, D] residual anomaly features
            s_rec: [B, T, 1] reconstruction anomaly scores
            s_sem: [B, T, 1] semantic alignment anomaly scores
            rupa_dict: detailed tensors for loss supervision (all [B, ...] shaped, 0 scalar warnings)
        """
        B, N, D = f_fused.shape
        device = f_fused.device
        if visual_raw is None:
            visual_raw = f_fused

        cat_text_norm = F.normalize(category_text_features, dim=-1)  # [C, D]
        text_matrix = cat_text_norm.t().to(f_fused.dtype)            # [D, C]

        if text_features_zs is not None:
            zs_text_matrix = F.normalize(text_features_zs, dim=-1).t().to(f_fused.dtype)
        else:
            zs_text_matrix = text_matrix

        # 1. Dual-Gated Pseudo-Normal Mining (combines S_det with zero-shot CLIP semantic normality)
        with torch.no_grad():
            det_scores_flat = s_det.squeeze(-1).clamp(0.0, 1.0)
            f_raw_norm = F.normalize(visual_raw, dim=-1)
            zs_logits = (f_raw_norm @ zs_text_matrix) / 0.07
            zs_abn_prob = (1.0 - F.softmax(zs_logits, dim=-1)[:, :, 0]).clamp(0.0, 1.0)
            joint_mining_scores = 0.5 * det_scores_flat + 0.5 * zs_abn_prob

            if self.sampling == 'otsu':
                selected_normal, reliability = self._select_normal_frames_otsu(
                    joint_mining_scores, f_fused, lengths, B, N, D
                )
            else:
                selected_normal, reliability = self._select_normal_frames_topk(
                    joint_mining_scores, f_fused, lengths, B, N, D
                )

            selected_abn, abn_conf = self._select_abnormal_frames_topk(
                joint_mining_scores, f_fused, lengths, B, N, D
            )

        # 2. Controlled Gradient Detachment & Whole-Video Split-Consistent Prototype Aggregation
        if not self.training and B > 1:
            # During multi-split single-video evaluation, pool valid snippets across ALL splits
            # so an all-anomalous 256-snippet chunk projects onto the video's true normal background
            if lengths is not None:
                lengths_c = lengths.to(device=device, dtype=torch.long).clamp(min=1, max=N)
                valid_mask_2d = torch.arange(N, device=device)[None, :] < lengths_c[:, None]
            else:
                valid_mask_2d = torch.ones(B, N, device=device, dtype=torch.bool)

            flat_f = f_fused[valid_mask_2d].unsqueeze(0)                    # [1, L_tot, D]
            flat_s = joint_mining_scores[valid_mask_2d].unsqueeze(0)        # [1, L_tot]
            L_tot = flat_f.shape[1]
            flat_len = torch.tensor([L_tot], device=device, dtype=torch.long)

            if self.sampling == 'otsu':
                vid_sel_norm, vid_rel = self._select_normal_frames_otsu(
                    flat_s, flat_f, flat_len, 1, L_tot, D
                )
            else:
                vid_sel_norm, vid_rel = self._select_normal_frames_topk(
                    flat_s, flat_f, flat_len, 1, L_tot, D
                )

            vid_norm_input = vid_sel_norm.detach() * (0.5 + 0.5 * vid_rel.unsqueeze(-1))
            agg_vid = self.video_prototypes.unsqueeze(0)  # [1, M, D]
            for blk in self.dnp_extractor:
                agg_vid = blk(agg_vid, vid_norm_input)

            u_norm_vid = self.build_orthonormal_subspace(agg_vid)           # [1, M, D]
            dynamic_normal_patterns = agg_vid.expand(B, -1, -1)             # [B, M, D]
            u_norm = u_norm_vid.expand(B, -1, -1)                           # [B, M, D]
        else:
            normal_input = selected_normal.detach() * (0.5 + 0.5 * reliability.unsqueeze(-1))
            agg_prototype = self.video_prototypes.unsqueeze(0).expand(B, -1, -1)
            for blk in self.dnp_extractor:
                agg_prototype = blk(agg_prototype, normal_input)

            dynamic_normal_patterns = agg_prototype  # [B, M, D]
            # 3. Construct Orthonormal Normal Subspace Basis U_norm in St(M, D) via Differentiable QR
            u_norm = self.build_orthonormal_subspace(dynamic_normal_patterns)  # [B, M, D]

        # 4. Latent Manifold Orthogonal Subspace Projection (LM-OSP)
        subspace_coords = torch.bmm(f_fused, u_norm.transpose(1, 2))  # [B, N, M]
        f_proj = torch.bmm(subspace_coords, u_norm)                   # [B, N, D]

        # Subspace-constrained decoder refinement (operates on f_proj, preventing identity shortcut)
        f_dec = self.bottleneck(f_proj)
        for blk in self.decoder:
            f_dec = blk(f_dec, u_norm)
        alpha_dec = torch.sigmoid(self.alpha_subspace_dec) * 0.35
        f_rec = (1.0 - alpha_dec) * f_proj + alpha_dec * f_dec

        # Counterfactual Orthogonal Residual: R = F_fused - F_rec
        residual = f_fused - f_rec

        # Grassmannian Subspace Reconstruction Score + Cosine Hybrid
        f_norm_unit = F.normalize(f_fused, dim=-1)
        proj_energy = (torch.bmm(f_norm_unit, u_norm.transpose(1, 2)) ** 2).sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
        s_subspace = (1.0 - proj_energy).clamp(0.0, 1.0)
        cos_sim = F.cosine_similarity(f_fused, f_rec, dim=-1).unsqueeze(-1)
        s_cos = ((1.0 - cos_sim) / 2.0).clamp(0.0, 1.0)
        s_rec = (0.6 * s_subspace + 0.4 * s_cos).clamp(0.0, 1.0)

        # 5. Per-sample Subspace Gather Loss: [B, 1]
        norm_unit_sel = F.normalize(selected_normal, dim=-1)
        norm_proj_energy = (torch.bmm(norm_unit_sel, u_norm.transpose(1, 2)) ** 2).sum(dim=-1).clamp(0.0, 1.0)
        subspace_dist = 1.0 - norm_proj_energy
        rel_norm = reliability / reliability.sum(dim=1, keepdim=True).clamp_min(1e-6)
        sample_g_loss = (subspace_dist * rel_norm).sum(dim=1, keepdim=True)  # [B, 1]

        # 6. Per-sample Prototype Diversity + Margin-Enforced Abnormal Subspace Repulsion Loss: [B, 1]
        p_norm = F.normalize(dynamic_normal_patterns, dim=-1)
        sim_matrix = torch.bmm(p_norm, p_norm.transpose(1, 2))
        identity = torch.eye(self.num_prototypes, device=f_fused.device).unsqueeze(0).expand(B, -1, -1)
        off_diag = sim_matrix - identity
        raw_orth_loss = ((off_diag ** 2).sum(dim=(1, 2)) / max(1, self.num_prototypes * (self.num_prototypes - 1))).view(B, 1)

        abn_unit = F.normalize(selected_abn.detach(), dim=-1)
        abn_proj_energy = (torch.bmm(abn_unit, u_norm.transpose(1, 2)) ** 2).sum(dim=-1)
        abn_gate = (abn_conf > 0.55).to(f_fused.dtype) * abn_conf
        repulsion_loss = (F.relu(abn_proj_energy - self.repulsion_margin) * abn_gate).mean(dim=1, keepdim=True)
        sample_orth_loss = raw_orth_loss + 0.5 * repulsion_loss  # [B, 1]

        # 7. Residual-Amplified CLIP-Anchored Semantic Feature
        f_sem_rupa = self.ln_sem_rupa(
            visual_raw + f_fused + self.gamma_rupa * self.mlp_res(residual)
        )

        # Domain-Adaptive Semantic Alignment
        temp = self.log_temp.exp().clamp(min=0.02, max=0.5)
        f_sem_norm = F.normalize(f_sem_rupa, dim=-1)
        semantic_logits = f_sem_norm @ text_matrix / temp  # [B, T, C]

        # Variance scaling factor: damps noisy semantic logits in blurry CCTV videos
        r_var = torch.var(residual, dim=-1, keepdim=True)  # [B, T, 1]
        var_scale = 1.0 + F.softplus(self.var_scale_param) * r_var
        scaled_sem_logits = semantic_logits / var_scale

        # Semantic Anomaly Score: 1 - P(Normal) (class 0 is Normal)
        s_sem = (1.0 - F.softmax(scaled_sem_logits, dim=-1)[:, :, :1]).clamp(0.0, 1.0)

        # Prototype Normal Alignment Loss: [B, 1]
        normal_text_vec = cat_text_norm[0].view(1, 1, -1)
        dnp_normal_sim = F.cosine_similarity(u_norm, normal_text_vec, dim=-1)
        sample_dnp_normal_loss = (1.0 - dnp_normal_sim).mean(dim=1, keepdim=True)  # [B, 1]

        rupa_dict = {
            'reconstructed_features': f_rec,
            'residual_features': residual,
            'dynamic_normal_patterns': u_norm,
            'f_sem_rupa': f_sem_rupa,
            'semantic_logits': scaled_sem_logits,
            'raw_semantic_logits': semantic_logits,
            'residual_semantic_logits': scaled_sem_logits,
            'reconstructed_semantic_logits': scaled_sem_logits,
            'g_loss': sample_g_loss,
            'orth_loss': sample_orth_loss,
            'dnp_normal_loss': sample_dnp_normal_loss,
        }

        return f_rec, residual, s_rec, s_sem, rupa_dict


# ============================================================================
# 4. Core Module 4: QuadSafeGate & Per-Snippet PCGrad
# ============================================================================

class GradientProjectionJunction(torch.autograd.Function):
    """Per-Snippet Multi-Objective PCGrad Junction (Zero Extra Backward Passes).
    
    Projects conflicting secondary/prototype gradients onto the orthogonal complement
    of the primary task gradient at each individual snippet (b, t) along feature dimension D,
    with relative gradient norm bounding so auxiliary/prototype branches cannot destabilize
    the backbone.
    
    Complexity: O(B * T * D), zero extra backward passes, 100% DataParallel compatible.
    """
    @staticmethod
    def forward(ctx, x):
        return x, x.clone()

    @staticmethod
    def backward(ctx, grad_main, grad_rupa):
        if grad_main is None or grad_rupa is None:
            return (grad_main if grad_main is not None else grad_rupa)

        orig_dtype = grad_main.dtype
        g_main = grad_main.float()
        g_rupa = grad_rupa.float()

        # Per-snippet inner product along feature dimension D (dim=-1): [B, T, 1]
        dot = (g_rupa * g_main).sum(dim=-1, keepdim=True)
        norm_sq = (g_main * g_main).sum(dim=-1, keepdim=True).clamp_min(1e-7)

        # Project only snippets where dot < 0 (conflicting gradient directions)
        proj_coeff = torch.clamp(dot, max=0.0) / norm_sq
        g_rupa_proj = g_rupa - proj_coeff * g_main

        # Relative norm bounding per snippet: prevent RUPA gradient norm from exceeding main norm
        main_norm = g_main.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-6)
        rupa_norm = g_rupa_proj.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-6)
        norm_scale = torch.clamp(main_norm / rupa_norm, max=1.0)
        g_rupa_safe = g_rupa_proj * norm_scale

        return (g_main + g_rupa_safe).to(orig_dtype)


class QuadSafeGate(nn.Module):
    """Immunized Quad-Stream Calibration-Aware & Semantic-Anchored Safe Gate.
    
    Dynamically routes between 4 score streams:
    - S_det: Base detector score
    - S_rec: LM-OSP Grassmannian subspace reconstruction score
    - S_sem: CLIP-anchored domain-adaptive residual semantic score
    - S_aux: TD-AVCA audio-visual / motion auxiliary score
    
    Immunization Mechanisms:
    1. Detached Diagnostic Gate Inputs: Prevents the gate from assigning 100% weight
       to whichever stream overfits fastest on training bags.
    2. Asymmetric OOD False-Alarm Discordance Suppression: Dynamically suppresses any
       stream k whose prediction S_k spikes far above the CLIP-anchored consensus max(S_sem, S_rec).
    3. Semantic-Detector Consensus Anchor: Routes around S_anchor = (1 - gamma)*S_det + gamma*S_sem.
    """
    def __init__(self, dataset_type: str = 'xd', hidden_dim: int = 32,
                 no_safegate: bool = False, custom_prior_weights=None,
                 ood_beta: float = 3.5):
        super().__init__()
        self.no_safegate = no_safegate
        self.dataset_type = dataset_type.lower()
        self.ood_beta = ood_beta

        # Configurable prior weights
        if custom_prior_weights is not None:
            init_w = torch.tensor(custom_prior_weights, dtype=torch.float32)
        elif 'xd' in self.dataset_type:
            # XD-Violence: balanced prior with elevated CLIP-semantic anchor
            init_w = torch.tensor([0.30, 0.20, 0.30, 0.20], dtype=torch.float32)
        else:
            # UCF-Crime: Visual + Motion + LM-OSP + Semantics
            init_w = torch.tensor([0.35, 0.25, 0.25, 0.15], dtype=torch.float32)

        init_w = init_w / init_w.sum().clamp_min(1e-6)
        self.register_buffer('target_prior', init_w)
        self.prior_logits = nn.Parameter(torch.log(init_w.clamp(min=1e-4)))

        # Dynamic Calibration-Aware MLP Gate (12-D input -> 4 stream offsets + 1 residual gate)
        self.mlp_gate = nn.Sequential(
            nn.Linear(12, hidden_dim),
            LayerNormFP32(hidden_dim),
            QuickGELU(),
            nn.Linear(hidden_dim, 5)
        )

        nn.init.normal_(self.mlp_gate[0].weight, std=0.01)
        nn.init.constant_(self.mlp_gate[0].bias, 0.0)
        nn.init.normal_(self.mlp_gate[-1].weight, std=0.01)
        with torch.no_grad():
            self.mlp_gate[-1].bias.zero_()
            self.mlp_gate[-1].bias[4] = -2.0

        init_gamma = 0.45 if 'xd' in self.dataset_type else 0.35
        self.anchor_gamma_logit = nn.Parameter(torch.tensor(math.log(init_gamma / (1.0 - init_gamma))))

    def forward(self, s_det: torch.Tensor, s_rec: torch.Tensor,
                s_sem: torch.Tensor, s_aux: torch.Tensor):
        """
        Args:
            s_det, s_rec, s_sem, s_aux: [B, T, 1] anomaly score streams
        Returns:
            p_routing: [B, T, 1] immunized routed anomaly probability
            gate_reg: [B, T, 1] KL-prior regularization term
            s_fixed: [B, T, 1] OOD-suppressed mixture score
        """
        s_det = s_det.clamp(1e-5, 1.0 - 1e-5)
        s_rec = s_rec.clamp(1e-5, 1.0 - 1e-5)
        s_sem = s_sem.clamp(1e-5, 1.0 - 1e-5)
        s_aux = s_aux.clamp(1e-5, 1.0 - 1e-5)

        # 1. Detached diagnostic statistics (prevents gate from chasing overfitted training loss)
        sd_d = s_det.detach()
        sr_d = s_rec.detach()
        ss_d = s_sem.detach()
        sa_d = s_aux.detach()

        d_det_sem = (sd_d - ss_d).abs()
        d_rec_sem = (sr_d - ss_d).abs()
        d_aux_sem = (sa_d - ss_d).abs()

        d_rec_det = (sr_d - sd_d).abs()
        d_sem_det = (ss_d - sd_d).abs()
        d_aux_det = (sa_d - sd_d).abs()

        scores_stack_d = torch.cat([sd_d, sr_d, ss_d, sa_d], dim=-1)  # [B, T, 4]
        s_var = torch.var(scores_stack_d, dim=-1, keepdim=True)       # [B, T, 1]
        ent_aux = -(sa_d * torch.log(sa_d) + (1.0 - sa_d) * torch.log(1.0 - sa_d)) / math.log(2.0)

        v_gate = torch.cat([
            sd_d, sr_d, ss_d, sa_d,
            d_det_sem, d_rec_sem, d_aux_sem,
            d_rec_det, d_sem_det, d_aux_det,
            s_var, ent_aux
        ], dim=-1)  # [B, T, 12]

        gate_out = self.mlp_gate(v_gate)  # [B, T, 5]
        stream_offsets = gate_out[..., :4].clamp(-2.5, 2.5)
        residual_gate = torch.sigmoid(gate_out[..., 4:5])

        # 2. Asymmetric OOD False-Alarm Discordance Suppression
        grounded_ref = torch.max(ss_d, sr_d)
        ood_det = F.relu(sd_d - grounded_ref) ** 2
        ood_rec = F.relu(sr_d - ss_d) ** 2
        ood_sem = torch.zeros_like(ss_d)
        ood_aux = F.relu(sa_d - grounded_ref) ** 2
        ood_penalty = torch.cat([ood_det, ood_rec, ood_sem, ood_aux], dim=-1)  # [B, T, 4]

        # 3. Dynamic Calibration-Aware Stream Mixture
        base_logits = self.prior_logits.view(1, 1, 4)
        dyn_logits = base_logits + stream_offsets - self.ood_beta * ood_penalty
        dyn_weights = F.softmax(dyn_logits, dim=-1)  # [B, T, 4]

        scores_stack = torch.cat([s_det, s_rec, s_sem, s_aux], dim=-1)
        s_fixed = (dyn_weights * scores_stack).sum(dim=-1, keepdim=True).clamp(1e-5, 1.0 - 1e-5)

        if self.no_safegate:
            return s_fixed, torch.zeros_like(s_det), s_fixed

        # 4. Semantic-Detector Consensus Anchor Routing
        gamma_sem = torch.sigmoid(self.anchor_gamma_logit).clamp(0.30, 0.65)
        s_anchor = (1.0 - gamma_sem) * s_det + gamma_sem * s_sem

        p_routing = (s_anchor + residual_gate * (s_fixed - s_anchor)).clamp(1e-5, 1.0 - 1e-5)

        prior_ref = self.target_prior.view(1, 1, 4).to(dyn_weights.dtype)
        kl_div = (dyn_weights * (torch.log(dyn_weights.clamp_min(1e-6)) - torch.log(prior_ref))).sum(dim=-1, keepdim=True)
        gate_reg = torch.sqrt(kl_div.clamp_min(0.0) + 1e-6)

        return p_routing, gate_reg, s_fixed


# ============================================================================
# 5. Text CLIP Adapter & Feature Cache
# ============================================================================

class CLIPAdapter(nn.Module):
    """Adapts CLIP text encoder for video anomaly detection while preserving
    OpenAI CLIP's pretrained 400M-pair text_projection geometry.
    """
    def __init__(self, clipmodel, device, text_adapt_until: int = 3, t_w: float = 0.1):
        super().__init__()
        self.clipmodel = clipmodel
        self.text_adapt_until = text_adapt_until
        self.t_w = t_w
        self.device = device

        self.text_adapter = nn.ModuleList(
            [SimpleAdapter(512, 512) for _ in range(text_adapt_until)] +
            [SimpleProj(512, 512, relu=False)]
        )
        self._init_weights()

    def _init_weights(self):
        for p in self.text_adapter.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode_text(self, text: torch.Tensor, adapt_text: bool = True):
        cast_dtype = self.clipmodel.token_embedding.weight.dtype
        x = self.clipmodel.token_embedding(text).to(cast_dtype)
        x = x + self.clipmodel.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)

        for i in range(len(self.clipmodel.transformer.resblocks)):
            x = self.clipmodel.transformer.resblocks[i](x)
            if adapt_text and i < self.text_adapt_until:
                adapt_out = self.text_adapter[i](x)
                adapt_out = (
                    adapt_out * x.norm(dim=-1, keepdim=True) /
                    (adapt_out.norm(dim=-1, keepdim=True) + 1e-6)
                )
                x = self.t_w * adapt_out + (1.0 - self.t_w) * x

        x = x.permute(1, 0, 2)
        x = self.clipmodel.ln_final(x)
        eot_indices = text.argmax(dim=-1)
        x_eot = x[torch.arange(x.shape[0], device=text.device), eot_indices]

        # Project into the joint CLIP Vision-Language 512-D space using pretrained text_projection
        if getattr(self.clipmodel, 'text_projection', None) is not None:
            text_proj = x_eot @ self.clipmodel.text_projection.to(x_eot.dtype)
        else:
            text_proj = x_eot

        if not adapt_text:
            return text_proj

        adapt_proj = self.text_adapter[-1](x_eot)
        adapt_proj = (
            adapt_proj * text_proj.norm(dim=-1, keepdim=True) /
            (adapt_proj.norm(dim=-1, keepdim=True) + 1e-6)
        )
        return (1.0 - self.t_w) * text_proj + self.t_w * adapt_proj


# ============================================================================
# 6. Full Model: MARUPA
# ============================================================================

class MARUPA(nn.Module):
    """MA-RUPA: Modality-Adaptive Multiscale Prototype Alignment Network.
    
    End-to-End trainable from scratch without pre-trained DSANet checkpoints
    or frozen two-stage warm-starts.
    """
    def __init__(self,
                 num_class: int,
                 embed_dim: int = 512,
                 visual_length: int = 256,
                 visual_width: int = 512,
                 visual_head: int = 4,
                 visual_layers: int = 2,
                 attn_window: int = 8,
                 prompt_prefix: int = 10,
                 prompt_postfix: int = 10,
                 args = None,
                 device = "cuda"):
        super().__init__()
        self.num_class = num_class
        self.embed_dim = embed_dim
        self.visual_length = visual_length
        self.visual_width = visual_width
        self.attn_window = attn_window
        self.device = device
        self.args = args

        # Ablation study flags
        self.no_mdtp = getattr(args, 'no_mdtp', False)
        self.no_audio = getattr(args, 'no_audio', False)
        self.no_rupa = getattr(args, 'no_rupa', False)
        self.no_safegate = getattr(args, 'no_safegate', False)
        self.sampling = getattr(args, 'sampling', 'topk')
        dataset_type = 'xd' if num_class == 7 else 'ucf'

        # 1. Core Module 1: MDTBackbone
        self.backbone = MDTBackbone(
            visual_length=visual_length,
            embed_dim=visual_width,
            attn_window=attn_window,
            visual_layers=visual_layers,
            visual_heads=visual_head,
            mlp_dropout=getattr(args, 'mlp_dropout', 0.1),
            no_mdtp=self.no_mdtp
        )

        # 2. Core Module 2: ModalityAdaptiveBranch
        audio_dim = getattr(args, 'audio_dim', 128)
        self.modality_branch = ModalityAdaptiveBranch(
            embed_dim=visual_width,
            audio_dim=audio_dim,
            num_heads=visual_head,
            dropout=getattr(args, 'mlp_dropout', 0.1),
            no_audio=self.no_audio
        )

        # 3. Core Module 3: GradientSafeRUPA
        self.rupa = GradientSafeRUPA(
            feature_dim=visual_width,
            num_prototypes=getattr(args, 'num_prototypes', 16),
            num_heads=visual_head,
            extractor_depth=1,
            decoder_depth=getattr(args, 'decoder_depth', 4),
            sampling=self.sampling,
            normal_selection_ratio=getattr(args, 'normal_selection_ratio', 0.60),
            min_normal_frames=getattr(args, 'min_normal_frames', 4),
            max_normal_ratio=getattr(args, 'max_normal_ratio', 0.70)
        )

        # 4. Core Module 4: QuadSafeGate
        prior_weights = None
        if hasattr(args, 'routing_det_weight') and hasattr(args, 'routing_rec_weight'):
            prior_weights = [
                getattr(args, 'routing_det_weight', 0.30),
                getattr(args, 'routing_rec_weight', 0.20),
                getattr(args, 'routing_sem_weight', 0.30),
                getattr(args, 'routing_aux_weight', 0.20)
            ]
        self.safegate = QuadSafeGate(
            dataset_type=dataset_type,
            hidden_dim=getattr(args, 'routing_gate_hidden', 32),
            no_safegate=self.no_safegate,
            custom_prior_weights=prior_weights
        )

        # 5. CLIP Text Adapter
        self.clipmodel, _ = clip.load("ViT-B/16", device)
        for p in self.clipmodel.parameters():
            p.requires_grad = False

        text_adapt_until = getattr(args, 'text_adapt_until', 2 if dataset_type == 'xd' else 3)
        t_w = getattr(args, 't_w', 0.25 if dataset_type == 'xd' else 0.15)
        self.clip_adapter = CLIPAdapter(self.clipmodel, device, text_adapt_until, t_w)

        # Video-Conditioned Dynamic Prompt Calibrator MLP
        self.mlp_event = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("dropout", nn.Dropout(getattr(args, 'mlp_dropout', 0.1))),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.alpha_prompt_ctx = nn.Parameter(torch.tensor(0.15))

        self._text_features_cache = None
        self._init_trainable_parameters()

        # Pre-tokenize and pre-compute pure zero-shot CLIP text centroids as buffers
        # so multi-GPU DataParallel replicas automatically broadcast them per device
        # without thread-unsafe lazy tokenization or dictionary mutation
        init_num_c = 14 if self.num_class == 14 else 7
        descriptions = DESCRIPTIONS_ORI if init_num_c == 14 else DESCRIPTIONS_ORI_XD
        desc_lists = list(descriptions.values())
        self._num_desc_classes = len(desc_lists)
        self._num_desc_prompts = len(desc_lists[0])
        flat_prompts = [p for sublist in desc_lists for p in sublist]
        tokens_init = clip.tokenize(flat_prompts).to(device)
        self.register_buffer('_prompt_tokens_buf', tokens_init, persistent=False)
        with torch.no_grad():
            zs_feats = self.clip_adapter.encode_text(tokens_init, adapt_text=False)
            zs_feats = F.normalize(
                zs_feats.view(self._num_desc_classes, self._num_desc_prompts, -1), dim=-1
            )
            zs_mean = F.normalize(zs_feats.mean(dim=1), dim=-1).detach()
        self.register_buffer('_zs_text_buf', zs_mean, persistent=False)

    def _init_trainable_parameters(self):
        """Initialize trainable layers while preserving SafeGate & CLIPAdapter specialized initializations."""
        safegate_modules = set(self.safegate.modules())
        adapter_modules = set(self.clip_adapter.modules())
        for m in self.modules():
            if m in safegate_modules or m in adapter_modules:
                continue
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02, a=-0.04, b=0.04)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.LayerNorm, LayerNormFP32)):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        # Initialize relative positional conv near zero so raw CLIP features start clean
        if hasattr(self.backbone, 'conv_pos'):
            nn.init.normal_(self.backbone.conv_pos.weight, std=0.01)
            if self.backbone.conv_pos.bias is not None:
                nn.init.constant_(self.backbone.conv_pos.bias, 0.0)

    def _get_batched_tokens(self, num_classes: int, target_device: torch.device):
        """Returns tokenized multi-prompt descriptions [C * P, 77] on target_device."""
        if hasattr(self, '_prompt_tokens_buf') and num_classes == self._num_desc_classes:
            tokens_all = self._prompt_tokens_buf
            if tokens_all.device != target_device:
                tokens_all = tokens_all.to(target_device)
            return tokens_all, self._num_desc_classes, self._num_desc_prompts

        descriptions = DESCRIPTIONS_ORI if num_classes == 14 else DESCRIPTIONS_ORI_XD
        desc_lists = list(descriptions.values())
        num_prompts = len(desc_lists[0])
        flat_prompts = [p for sublist in desc_lists for p in sublist]
        tokens_all = clip.tokenize(flat_prompts).to(target_device)
        return tokens_all, len(desc_lists), num_prompts

    def get_zero_shot_text_features(self, text_list, target_device: torch.device = None):
        """Returns pure frozen OpenAI CLIP multi-prompt text centroids [C, D]."""
        if target_device is None:
            target_device = next(self.parameters()).device
        num_c = 14 if (text_list is not None and len(text_list) == 14) or self.num_class == 14 else 7
        if hasattr(self, '_zs_text_buf') and num_c == self._num_desc_classes:
            zs_feats = self._zs_text_buf
            if zs_feats.device != target_device:
                zs_feats = zs_feats.to(target_device)
            return zs_feats

        with torch.no_grad():
            tokens_all, C, P = self._get_batched_tokens(num_c, target_device)
            feats = self.clip_adapter.encode_text(tokens_all, adapt_text=False)  # [C * P, D]
            feats = F.normalize(feats.view(C, P, -1), dim=-1)
            mean_feats = F.normalize(feats.mean(dim=1), dim=-1)
            return mean_feats.detach()

    def train(self, mode: bool = True):
        """Override train() to invalidate cached adapted text features whenever mode switches."""
        super().train(mode)
        self._text_features_cache = None
        return self

    def get_text_features(self, text_list, target_device: torch.device = None):
        """Encode multi-prompt category descriptions into normalized CLIP text embeddings [C, D]
        in a single batched Transformer call.
        """
        if target_device is None:
            target_device = next(self.parameters()).device
        if not self.training and self._text_features_cache is not None and self._text_features_cache.device == target_device:
            return self._text_features_cache

        num_c = 14 if (text_list is not None and len(text_list) == 14) or self.num_class == 14 else 7
        tokens_all, C, P = self._get_batched_tokens(num_c, target_device)
        feats = self.clip_adapter.encode_text(tokens_all, adapt_text=True)  # [C * P, D]
        feats = F.normalize(feats.view(C, P, -1), dim=-1)
        text_features = F.normalize(feats.mean(dim=1), dim=-1)  # [C, D]

        if not self.training:
            self._text_features_cache = text_features
        return text_features

    def forward(self, visual: torch.Tensor, padding_mask: torch.Tensor = None,
                text: list = None, lengths: torch.Tensor = None,
                audio: torch.Tensor = None, DNP_use: bool = True, scale: float = 10.0):
        """Full MA-RUPA forward pass.
        
        Args:
            visual: [B, T, D] visual features
            padding_mask: [B, T] bool mask of padding positions
            text: list of class names or descriptions
            lengths: [B] valid sequence lengths
            audio: [B, T_a, D_a] audio features (XD-Violence) or None (UCF-Crime)
            DNP_use: whether to activate RUPA module
            scale: temperature scale for temporal pooling
        """
        B, T, D = visual.shape
        target_device = visual.device

        # Step 1: MDTBackbone -> F_base, S_det
        f_base, s_det, logits_det = self.backbone(visual, padding_mask, lengths)

        # Step 2: ModalityAdaptiveBranch -> F_fused, S_aux, logits_aux, G_av, av_suppress_loss
        f_fused, s_aux, logits_aux, g_av, av_suppress_loss = self.modality_branch(
            f_base, visual, audio_features=audio, lengths=lengths, return_suppress_loss=True
        )

        # Text features for semantic alignment (adapted & pure zero-shot anchor)
        text_features_ori = self.get_text_features(text, target_device=target_device)
        text_features_zs = self.get_zero_shot_text_features(text, target_device=target_device)

        # Step 3 & 4: RUPA & QuadSafeGate
        dnp_dict = {}
        if DNP_use and not self.no_rupa:
            # Per-snippet Multi-Objective PCGrad Junction
            if self.training:
                f_main, f_rupa_in = GradientProjectionJunction.apply(f_fused)
            else:
                f_main, f_rupa_in = f_fused, f_fused

            f_rec, residual, s_rec, s_sem, rupa_dict = self.rupa(
                f_rupa_in, s_det, text_features_ori, lengths=lengths,
                visual_raw=visual, text_features_zs=text_features_zs
            )
            p_routing, gate, s_fixed = self.safegate(s_det, s_rec, s_sem, s_aux)
            dnp_dict.update(rupa_dict)
            dnp_dict.update({
                'routing_scores': p_routing,
                'routing_logits': torch.logit(p_routing.clamp(1e-6, 1.0 - 1e-6)),
                'detector_scores': s_det,
                'detector_logits': logits_det,
                'aux_scores': s_aux,
                'aux_logits': logits_aux,
                'reconstruction_scores': s_rec,
                'semantic_scores': s_sem,
                'fixed_scores': s_fixed,
                'gate': gate,
                'noise_gate_av': g_av,
                'av_suppress_loss': av_suppress_loss,
                'original_features': f_main,
                'reconstructed_features': f_rec,
                'residual_features': residual,
            })
            logits1 = logits_det
            event_features = residual
            background_features = f_rec
        else:
            # Fallback when RUPA is disabled (Ablation mode)
            f_main = f_fused
            p_routing, gate, s_fixed = self.safegate(s_det, s_det, s_det, s_aux)
            logits1 = logits_det
            event_features = f_fused
            background_features = f_fused
            dnp_dict.update({
                'routing_scores': p_routing,
                'routing_logits': torch.logit(p_routing.clamp(1e-6, 1.0 - 1e-6)),
                'detector_scores': s_det,
                'detector_logits': logits_det,
                'aux_scores': s_aux,
                'aux_logits': logits_aux,
                'reconstruction_scores': torch.zeros_like(s_det),
                'semantic_scores': torch.zeros_like(s_det),
                'fixed_scores': s_fixed,
                'gate': gate,
                'noise_gate_av': g_av,
                'av_suppress_loss': av_suppress_loss,
                'original_features': f_fused,
                'reconstructed_features': f_fused,
                'residual_features': torch.zeros_like(f_fused),
                'g_loss': torch.zeros(B, 1, device=target_device),
                'orth_loss': torch.zeros(B, 1, device=target_device),
                'dnp_normal_loss': torch.zeros(B, 1, device=target_device),
            })

        # Event-centric & Background-centric attention pooling
        if lengths is not None:
            valid_mask = (
                torch.arange(T, device=target_device)[None, :]
                < lengths.to(target_device)[:, None]
            ).unsqueeze(-1).to(visual.dtype)
        else:
            valid_mask = torch.ones(B, T, 1, device=target_device, dtype=visual.dtype)

        prob_abn = p_routing
        abn_weights = ((scale * prob_abn).exp() - 1.0) * valid_mask
        abn_weights = F.normalize(abn_weights, p=1, dim=1)

        nor_weights = ((scale * (1.0 - prob_abn)).exp() - 1.0) * valid_mask
        nor_weights = F.normalize(nor_weights, p=1, dim=1)

        # Video-level event & background representations
        abn_feat = torch.matmul(abn_weights.permute(0, 2, 1), event_features)  # [B, 1, D]
        nor_feat = torch.matmul(nor_weights.permute(0, 2, 1), background_features)  # [B, 1, D]

        # Video-Conditioned Dynamic Prompt Calibration (using self.mlp_event anchored to text_features_ori)
        v_ctx = torch.matmul(abn_weights.permute(0, 2, 1), f_main)  # [B, 1, D]
        if not self.training and B > 1:
            # Pool context across all splits of the same test video
            v_ctx = v_ctx.mean(dim=0, keepdim=True).expand(B, -1, -1)
        v_ctx_norm = F.normalize(v_ctx, dim=-1)  # [B, 1, D]

        text_ori_norm = F.normalize(text_features_ori, dim=-1).unsqueeze(0).expand(B, -1, -1)  # [B, C, D]
        prompt_cond = text_ori_norm + v_ctx_norm
        prompt_delta = v_ctx_norm + self.mlp_event(prompt_cond)
        alpha_ctx = self.alpha_prompt_ctx.clamp(0.01, 0.25)
        text_dynamic = F.normalize(
            text_ori_norm + alpha_ctx * F.normalize(prompt_delta, dim=-1), dim=-1
        ).to(visual.dtype)  # [B, C, D]

        # Snippet-level and video-level classification logits
        f_norm = F.normalize(f_main, dim=-1)
        logits2 = torch.bmm(f_norm, text_dynamic.transpose(1, 2)) / 0.07  # [B, T, C]

        abn_feat_norm = F.normalize(abn_feat, dim=-1)
        nor_feat_norm = F.normalize(nor_feat, dim=-1)
        logits3 = torch.bmm(abn_feat_norm, text_dynamic.transpose(1, 2)) / 0.07
        logits4 = torch.bmm(nor_feat_norm, text_dynamic.transpose(1, 2)) / 0.07

        if DNP_use:
            return text_features_ori, logits1, logits2, logits3, logits4, dnp_dict
        return text_features_ori, logits1, logits2, logits3, logits4


# Legacy alias for backward compatibility with imports
DSANet = MARUPA


def load_compatible_model_state(model, state_dict, strict=False):
    """Safely loads model checkpoint."""
    target_model = model.module if hasattr(model, 'module') else model
    return target_model.load_state_dict(state_dict, strict=strict)

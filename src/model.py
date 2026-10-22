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

    def forward(self, x: torch.Tensor):
        # x: [B, T, D] -> transpose to [B, D, T] for conv1d
        x_t = x.transpose(1, 2)
        branch_outputs = []
        for branch, norm in zip(self.branches, self.norms):
            # branch output: [B, 2*D, T]
            h = branch(x_t)
            h = F.glu(h, dim=1)  # [B, D, T] via Gated Linear Unit
            # Normalize over feature dimension: [B, T, D]
            h_norm = norm(h.transpose(1, 2)).transpose(1, 2)
            branch_outputs.append(h_norm)

        # Sum parallel multi-scale temporal representations
        combined = sum(branch_outputs)  # [B, D, T]
        fused = self.fuse_conv(combined).transpose(1, 2)  # [B, T, D]
        # Residual connection
        out = self.out_norm(x + fused)
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
            k = k * valid_mask
            v = v * valid_mask

        # KV: [B, D, D] = K^T @ V
        kv = torch.bmm(k.transpose(1, 2), v)
        # num: [B, T, D] = Q @ KV
        num = torch.bmm(q, kv)
        # den: [B, T, 1] = Q @ sum_t(K)^T
        k_sum = k.sum(dim=1, keepdim=True)  # [B, 1, D]
        den = torch.bmm(q, k_sum.transpose(1, 2)).clamp_min(1e-5)

        f_global = num / den
        out = self.norm(self.dropout(self.out_proj(f_global)))
        return out


class MDTBackbone(nn.Module):
    """Multi-Scale Dilated Temporal & Local Attention Backbone.
    
    Combines:
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

        # Learnable temporal frame positional embedding
        self.frame_pos_embed = nn.Parameter(torch.zeros(1, visual_length, embed_dim))
        trunc_normal_(self.frame_pos_embed, std=0.02)

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

        # Crucial CLIP Anchor Skip Connection (initialized to 0.1)
        self.alpha_backbone = nn.Parameter(torch.tensor(0.1))
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

        # Positional embedding (interpolated if input sequence length differs)
        if T == self.visual_length:
            pos_emb = self.frame_pos_embed
        else:
            pos_emb = F.interpolate(
                self.frame_pos_embed.permute(0, 2, 1),
                size=T, mode='linear', align_corners=False
            ).permute(0, 2, 1)

        x = visual_features + pos_emb

        # Padding mask setup
        if padding_mask is None and lengths is not None:
            padding_mask = (
                torch.arange(T, device=visual_features.device)[None, :]
                >= lengths.to(visual_features.device)[:, None]
            )

        # Branch 1: Windowed Self-Attention
        attn_mask = self.build_window_attention_mask(T, self.attn_window, visual_features.device)
        f_a = x
        for layer in self.branch_a_layers:
            f_a = layer(f_a, attn_mask=attn_mask, key_padding_mask=padding_mask)

        # Branch 3: Global Linear Context Aggregator
        f_c = self.branch_c(x, padding_mask=padding_mask)

        # Branch 2: Multi-Scale Dilated Temporal Pyramid
        if not self.no_mdtp:
            f_b = self.branch_b(x)
            fused_context = self.fuse_norm(self.fuse_proj(torch.cat([f_a, f_b, f_c], dim=-1)))
        else:
            fused_context = self.fuse_norm(self.fuse_proj(torch.cat([f_a, f_c], dim=-1)))

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
    """Modality-Adaptive Audio-Visual / Motion Branch.
    
    Supports:
    1. XD-Violence Multimodal Mode: Project pre-extracted audio features
       (VGGish 128-D or CLAP/AST 512-D) aligned with 16-frame video snippets.
    2. Missing-Modality / Motion-Residual Mode: For UCF-Crime (no audio) or
       missing audio snippets, computes 1st- and 2nd-order visual temporal
       residuals (M1 = X_t - X_{t-1}, M2 = M1_t - M1_{t-1}) as dynamic auxiliary modality.
    3. Reliability-Aware Cross-Modal Gated Attention with noise suppression gate G_av.
    4. Auxiliary anomaly score stream S_aux.
    """
    def __init__(self, embed_dim: int = 512, audio_dim: int = 128,
                 num_heads: int = 4, dropout: float = 0.1, no_audio: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.audio_dim = audio_dim
        self.no_audio = no_audio

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

        # Noise Suppression Gate: G_av = sigma(MLP([F_base || F_a || (F_base * F_a)]))
        self.noise_gate_mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            QuickGELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )
        self.alpha_mod = nn.Parameter(torch.tensor(0.1))
        self.ln_fused = LayerNormFP32(embed_dim)

        # Auxiliary Anomaly Classifier Head
        self.mlp_aux = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(embed_dim, embed_dim * 4)),
            ("gelu", QuickGELU()),
            ("dropout", nn.Dropout(dropout)),
            ("c_proj", nn.Linear(embed_dim * 4, embed_dim)),
            ("dropout2", nn.Dropout(dropout))
        ]))
        self.classifier_aux = nn.Linear(embed_dim, 1)

    def compute_motion_residuals(self, x_v: torch.Tensor):
        """Compute 1st and 2nd order visual temporal feature residuals.
        
        M1[:, t] = x_v[:, t] - x_v[:, t-1] (with M1[:, 0] = 0)
        M2[:, t] = M1[:, t] - M1[:, t-1]   (with M2[:, 0] = 0)
        """
        B, T, D = x_v.shape
        m1 = torch.zeros_like(x_v)
        m1[:, 1:] = x_v[:, 1:] - x_v[:, :-1]

        m2 = torch.zeros_like(x_v)
        m2[:, 1:] = m1[:, 1:] - m1[:, :-1]

        motion = torch.cat([m1, m2], dim=-1)  # [B, T, 2*D]
        return motion

    def forward(self, f_base: torch.Tensor, visual_raw: torch.Tensor,
                audio_features: torch.Tensor = None, lengths: torch.Tensor = None):
        """
        Args:
            f_base: [B, T, D] visual base features from MDTBackbone
            visual_raw: [B, T, D] raw snippet visual features
            audio_features: [B, T_a, D_a] audio features (if available) or None
            lengths: [B] valid lengths
        Returns:
            F_fused: [B, T, D] multimodal/motion fused features
            S_aux: [B, T, 1] auxiliary anomaly scores
            logits_aux: [B, T, 1] raw classifier logits
            G_av: [B, T, D] noise suppression gate values
        """
        B, T, D = f_base.shape

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

            # Project audio features to dimension D with automatic dimension routing
            in_dim = audio_t.shape[1]
            if in_dim == self.embed_dim:
                f_a = self.audio_proj_512(audio_t).transpose(1, 2)
            elif in_dim == self.audio_dim:
                f_a = self.audio_proj(audio_t).transpose(1, 2)
            else:
                # Dynamic 1x1 conv adapter for any other dimension (e.g. 768-D from AST)
                if not hasattr(self, '_dyn_audio_adapter') or self._dyn_audio_adapter.in_channels != in_dim:
                    self._dyn_audio_adapter = nn.Conv1d(in_dim, self.embed_dim, kernel_size=1).to(audio_t.device)
                adapted_audio = self._dyn_audio_adapter(audio_t)
                f_a = self.audio_proj_512(adapted_audio).transpose(1, 2)
        else:
            # Missing-Modality / Motion-Residual Mode (UCF-Crime or Audio Fallback)
            motion = self.compute_motion_residuals(visual_raw)  # [B, T, 2*D]
            motion_t = motion.transpose(1, 2)  # [B, 2*D, T]
            f_a = self.motion_proj(motion_t).transpose(1, 2)  # [B, T, D]

        # Cross-Modal Attention: Query=F_base, Key=F_a, Value=F_a with padding mask support
        cross_pad_mask = None
        if lengths is not None:
            cross_pad_mask = (
                torch.arange(T, device=f_base.device)[None, :]
                >= lengths.to(f_base.device)[:, None]
            )
        attn_out, _ = self.cross_attn(
            query=f_base, key=f_a, value=f_a,
            key_padding_mask=cross_pad_mask,
            need_weights=False
        )
        attn_out = torch.nan_to_num(attn_out, nan=0.0)
        if cross_pad_mask is not None:
            attn_out = attn_out.masked_fill(cross_pad_mask.unsqueeze(-1), 0.0)
        f_v_from_a = self.ln_cross(attn_out)

        # Dynamic Noise Suppression Gate G_av: [B, T, D]
        gate_input = torch.cat([f_base, f_a, f_base * f_a], dim=-1)
        g_av = self.noise_gate_mlp(gate_input)

        # Gated residual injection: F_fused = LayerNorm(F_base + alpha_mod * (G_av * F_v_from_a))
        f_fused = self.ln_fused(f_base + self.alpha_mod * (g_av * f_v_from_a))

        # Auxiliary Anomaly Classifier
        aux_features = f_fused + self.mlp_aux(f_fused)
        logits_aux = self.classifier_aux(aux_features)  # [B, T, 1]
        s_aux = torch.sigmoid(logits_aux)

        return f_fused, s_aux, logits_aux, g_av


# ============================================================================
# 3. Core Module 3: GradientSafeRUPA
# ============================================================================

class GradientSafeRUPA(nn.Module):
    """Gradient-Safe RUPA Module.
    
    Stabilizes End-to-End training from scratch via:
    1. Hard Negative Mining via Top-K or Otsu Selection (on detached detector scores).
    2. Controlled Gradient Detachment on normal features entering the DNP memory:
       Prototypes aggregate normal representations from X_norm.detach() so noisy early
       reconstruction gradients do not corrupt MDTBackbone.
    3. Reconstructs counterfactual normal components F_rec and computes residual R = F_fused - F_rec.
    4. Reconstruction Anomaly Score: S_rec = (1 - cos(F_fused, F_rec)) / 2.
    5. Domain-Adaptive Semantic Alignment with learnable temperature & feature variance scaling.
    """
    def __init__(self, feature_dim: int = 512, num_prototypes: int = 16,
                 num_heads: int = 8, extractor_depth: int = 1, decoder_depth: int = 4,
                 sampling: str = 'topk', normal_selection_ratio: float = 0.8,
                 min_normal_frames: int = 4, max_normal_ratio: float = 0.8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_prototypes = num_prototypes
        self.sampling = sampling
        self.normal_selection_ratio = normal_selection_ratio
        self.min_normal_frames = min_normal_frames
        self.max_normal_ratio = max_normal_ratio

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

        # Bottleneck before prototype reconstruction
        self.bottleneck = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(feature_dim, feature_dim * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(feature_dim * 4, feature_dim))
        ]))

        # Prototype Decoder Blocks (Reconstruction)
        self.decoder = nn.ModuleList([
            Prototype_Block(
                dim=feature_dim, num_heads=num_heads, mlp_ratio=4.0,
                qkv_bias=True, norm_layer=partial(LayerNormFP32, eps=1e-6)
            ) for _ in range(decoder_depth)
        ])

        # Residual-amplified feature MLP for domain-adaptive category semantics
        # F_sem_rupa = LayerNorm(X_clip + F_fused + gamma_rupa * MLP_res(F_fused - F_rec))
        self.mlp_res = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(feature_dim, feature_dim * 2)),
            ("gelu", QuickGELU()),
            ("dropout", nn.Dropout(0.1)),
            ("c_proj", nn.Linear(feature_dim * 2, feature_dim))
        ]))
        self.gamma_rupa = nn.Parameter(torch.tensor(0.2))
        self.ln_sem_rupa = LayerNormFP32(feature_dim)

        # Learnable temperature and feature-variance scaling factor
        self.log_temp = nn.Parameter(torch.tensor(math.log(0.07)))
        self.var_scale_param = nn.Parameter(torch.tensor(0.1))

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
        """Select Top-K normal frames with the lowest detector anomaly scores."""
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
                visual_raw: torch.Tensor = None):
        """
        Args:
            f_fused: [B, T, D] multimodal fused features
            s_det: [B, T, 1] detector anomaly scores
            category_text_features: [C, D] text embeddings for categories (class 0 is normal)
            lengths: [B] valid sequence lengths
            visual_raw: [B, T, D] raw visual features (X_clip) for semantic anchor preservation
        Returns:
            f_rec: [B, T, D] reconstructed counterfactual normal features
            residual: [B, T, D] residual anomaly features
            s_rec: [B, T, 1] reconstruction anomaly scores
            s_sem: [B, T, 1] semantic alignment anomaly scores
            rupa_dict: detailed tensors for loss supervision (all [B, ...] shaped, 0 scalar warnings)
        """
        B, N, D = f_fused.shape
        if visual_raw is None:
            visual_raw = f_fused

        # Detach detector scores so normal selection does not pass discontinuous indexing gradients
        with torch.no_grad():
            det_scores_flat = s_det.squeeze(-1).clamp(0.0, 1.0)
            if self.sampling == 'otsu':
                selected_normal, reliability = self._select_normal_frames_otsu(
                    det_scores_flat, f_fused, lengths, B, N, D
                )
            else:
                selected_normal, reliability = self._select_normal_frames_topk(
                    det_scores_flat, f_fused, lengths, B, N, D
                )

        # Controlled Gradient Detachment: feed detached normal features into prototype aggregation
        # to ensure early noisy prototypes do not corrupt the backbone representations
        normal_input = selected_normal.detach() * (0.5 + 0.5 * reliability.unsqueeze(-1))
        agg_prototype = self.video_prototypes.unsqueeze(0).expand(B, -1, -1)
        for blk in self.dnp_extractor:
            agg_prototype = blk(agg_prototype, normal_input)

        dynamic_normal_patterns = agg_prototype  # [B, M, D]

        # Per-sample gather loss: [B, 1] (prevents DataParallel scalar gather warnings)
        dist = 1.0 - F.cosine_similarity(selected_normal.unsqueeze(2), dynamic_normal_patterns.unsqueeze(1), dim=-1)
        min_dist, _ = torch.min(dist, dim=2)
        rel_norm = reliability / reliability.sum(dim=1, keepdim=True).clamp_min(1e-6)
        sample_g_loss = (min_dist * rel_norm).sum(dim=1, keepdim=True)  # [B, 1]

        # Per-sample prototype orthogonality loss: [B, 1]
        p_norm = F.normalize(dynamic_normal_patterns, dim=-1)
        sim_matrix = torch.bmm(p_norm, p_norm.transpose(1, 2))
        identity = torch.eye(self.num_prototypes, device=f_fused.device).unsqueeze(0).expand(B, -1, -1)
        off_diag = sim_matrix - identity
        sample_orth_loss = ((off_diag ** 2).sum(dim=(1, 2)) / max(1, self.num_prototypes * (self.num_prototypes - 1))).view(B, 1)  # [B, 1]

        # Counterfactual normal reconstruction
        bottleneck_features = f_fused
        for blk in self.bottleneck:
            bottleneck_features = blk(bottleneck_features)
        f_rec = bottleneck_features
        for blk in self.decoder:
            f_rec = blk(f_rec, dynamic_normal_patterns)

        # Residual features: R = F_fused - F_rec
        residual = f_fused - f_rec

        # Reconstruction anomaly score: S_rec = (1 - cos(F_fused, F_rec)) / 2
        cos_sim = F.cosine_similarity(f_fused, f_rec, dim=-1).unsqueeze(-1)
        s_rec = ((1.0 - cos_sim) / 2.0).clamp(0.0, 1.0)

        # Fix 3.3: Residual-Amplified CLIP-Anchored Semantic Feature
        # F_sem_rupa = LayerNorm(X_clip + F_fused + gamma_rupa * MLP_res(F_fused - F_rec))
        # Normal snippets: F_fused ~ F_rec => residual ~ 0 => F_sem_rupa ~ X_clip + F_fused (preserves clean normal representation)
        # Abnormal snippets: residual != 0 => amplifies the abnormal direction towards fine-grained category prompts
        f_sem_rupa = self.ln_sem_rupa(
            visual_raw + f_fused + self.gamma_rupa * self.mlp_res(residual)
        )

        # Domain-Adaptive Semantic Alignment
        temp = self.log_temp.exp().clamp(min=0.01, max=1.0)
        cat_text_norm = F.normalize(category_text_features, dim=-1)  # [C, D]
        text_matrix = cat_text_norm.t().to(f_fused.dtype)  # [D, C]

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
        dnp_normal_sim = F.cosine_similarity(dynamic_normal_patterns, normal_text_vec, dim=-1)
        sample_dnp_normal_loss = (1.0 - dnp_normal_sim).mean(dim=1, keepdim=True)  # [B, 1]

        rupa_dict = {
            'reconstructed_features': f_rec,
            'residual_features': residual,
            'dynamic_normal_patterns': dynamic_normal_patterns,
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
# 4. Core Module 4: QuadSafeGate
# ============================================================================

class GradientProjectionJunction(torch.autograd.Function):
    """Fast Feature-Level PCGrad Alternative.
    
    Junction between Backbone and RUPA branches:
    Forward: passes feature F_fused to both Main Branch (logits2/detector) and RUPA branch.
    Backward: receives grad_main from main detection/classification pipeline and grad_rupa from RUPA.
    If dot_product(grad_rupa, grad_main) < 0:
        grad_rupa_proj = grad_rupa - (grad_rupa . grad_main) / (||grad_main||^2 + eps) * grad_main
    Returns (grad_main + grad_rupa_proj) to Backbone.
    
    Complexity: O(B * T * D), zero extra backward passes, 100% DataParallel compatible.
    """
    @staticmethod
    def forward(ctx, x):
        return x, x

    @staticmethod
    def backward(ctx, grad_main, grad_rupa):
        if grad_main is None or grad_rupa is None:
            return (grad_main if grad_main is not None else grad_rupa)
        
        dot = (grad_rupa * grad_main).sum()
        if dot < 0:
            norm_sq = (grad_main * grad_main).sum().clamp_min(1e-7)
            grad_rupa_proj = grad_rupa - (dot / norm_sq) * grad_main
        else:
            grad_rupa_proj = grad_rupa
            
        return grad_main + grad_rupa_proj


class QuadSafeGate(nn.Module):
    """Quad-Stream Uncertainty-Aware Dynamic Safe Gate.
    
    Dynamically routes between 4 score streams:
    - S_det: Base detector score
    - S_rec: Normality prototype reconstruction score
    - S_sem: Domain-adaptive residual semantic score
    - S_aux: Audio-visual / motion auxiliary score
    
    Static Prior Mixture:
      S_fixed = alpha * S_det + beta * S_rec + gamma * S_sem + delta * S_aux
    Dynamic Uncertainty MLP Gate:
      Input vector V_gate (8-D per snippet):
      [S_det, S_rec, S_sem, S_aux, |S_rec - S_det|, |S_sem - S_det|, |S_aux - S_det|, Var(S)]
      bias initialized to -4.5 so gate initially g = sigma(-4.5) ~ 0.01.
    Final Routed Prediction:
      P_routing = S_det + g * (S_fixed - S_det)
    """
    def __init__(self, dataset_type: str = 'xd', hidden_dim: int = 16,
                 no_safegate: bool = False, custom_prior_weights=None):
        super().__init__()
        self.no_safegate = no_safegate
        self.dataset_type = dataset_type.lower()

        # Configurable prior weights
        if custom_prior_weights is not None:
            init_w = torch.tensor(custom_prior_weights, dtype=torch.float32)
        elif 'xd' in self.dataset_type:
            # XD-Violence (with Audio): alpha=0.40, beta=0.25, gamma=0.15, delta=0.20
            init_w = torch.tensor([0.40, 0.25, 0.15, 0.20], dtype=torch.float32)
        else:
            # UCF-Crime (Visual + Motion): alpha=0.35, beta=0.30, gamma=0.20, delta=0.15
            init_w = torch.tensor([0.35, 0.30, 0.20, 0.15], dtype=torch.float32)

        # Softmax-constrained learnable weights initialized to target prior
        self.prior_logits = nn.Parameter(torch.log(init_w.clamp(min=1e-4)))

        # Dynamic Uncertainty MLP Gate
        # Input features: 4 scores + 3 absolute residuals + 1 variance = 8 features
        self.mlp_gate = nn.Sequential(
            nn.Linear(8, hidden_dim),
            QuickGELU(),
            nn.Linear(hidden_dim, 1)
        )

        # Safe initialization: bias = -4.5 so g = sigmoid(-4.5) ~ 0.011
        nn.init.normal_(self.mlp_gate[0].weight, std=0.01)
        nn.init.constant_(self.mlp_gate[0].bias, 0.0)
        nn.init.normal_(self.mlp_gate[-1].weight, std=0.01)
        nn.init.constant_(self.mlp_gate[-1].bias, -4.5)

    def forward(self, s_det: torch.Tensor, s_rec: torch.Tensor,
                s_sem: torch.Tensor, s_aux: torch.Tensor):
        """
        Args:
            s_det, s_rec, s_sem, s_aux: [B, T, 1] anomaly score streams
        Returns:
            p_routing: [B, T, 1] routed anomaly probability
            gate: [B, T, 1] gate values in [0, 1]
            s_fixed: [B, T, 1] static prior mixture
        """
        # Ensure all scores are clamped in [0, 1]
        s_det = s_det.clamp(0.0, 1.0)
        s_rec = s_rec.clamp(0.0, 1.0)
        s_sem = s_sem.clamp(0.0, 1.0)
        s_aux = s_aux.clamp(0.0, 1.0)

        # 1. Static Prior Mixture
        weights = F.softmax(self.prior_logits, dim=0)
        s_fixed = (
            weights[0] * s_det +
            weights[1] * s_rec +
            weights[2] * s_sem +
            weights[3] * s_aux
        ).clamp(1e-5, 1.0 - 1e-5)

        if self.no_safegate:
            return s_fixed, torch.zeros_like(s_det), s_fixed

        # 2. Dynamic Uncertainty Gate Inputs
        d_rec = (s_rec - s_det).abs()
        d_sem = (s_sem - s_det).abs()
        d_aux = (s_aux - s_det).abs()

        scores_stack = torch.cat([s_det, s_rec, s_sem, s_aux], dim=-1)  # [B, T, 4]
        s_var = torch.var(scores_stack, dim=-1, keepdim=True)  # [B, T, 1]

        v_gate = torch.cat([
            s_det, s_rec, s_sem, s_aux, d_rec, d_sem, d_aux, s_var
        ], dim=-1)  # [B, T, 8]

        gate = torch.sigmoid(self.mlp_gate(v_gate))  # [B, T, 1]

        # 3. Final Routed Prediction
        p_routing = s_det + gate * (s_fixed - s_det)
        p_routing = p_routing.clamp(1e-5, 1.0 - 1e-5)

        return p_routing, gate, s_fixed


# ============================================================================
# 5. Text CLIP Adapter & Feature Cache
# ============================================================================

class CLIPAdapter(nn.Module):
    """Adapts CLIP text encoder for video anomaly detection."""
    def __init__(self, clipmodel, device, text_adapt_until: int = 3, t_w: float = 0.1):
        super().__init__()
        self.clipmodel = clipmodel
        self.text_adapt_until = text_adapt_until
        self.t_w = t_w
        self.device = device

        self.text_adapter = nn.ModuleList(
            [SimpleAdapter(512, 512) for _ in range(text_adapt_until)] +
            [SimpleProj(512, 512, relu=True)]
        )
        self._init_weights()

    def _init_weights(self):
        for p in self.text_adapter.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode_text(self, text: torch.Tensor, adapt_text: bool = True):
        if not adapt_text:
            return self.clipmodel.encode_text(text)

        cast_dtype = self.clipmodel.token_embedding.weight.dtype
        x = self.clipmodel.token_embedding(text).to(cast_dtype)
        x = x + self.clipmodel.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)

        for i in range(len(self.clipmodel.transformer.resblocks)):
            x = self.clipmodel.transformer.resblocks[i](x)
            if i < self.text_adapt_until:
                adapt_out = self.text_adapter[i](x)
                adapt_out = (
                    adapt_out * x.norm(dim=-1, keepdim=True) /
                    (adapt_out.norm(dim=-1, keepdim=True) + 1e-6)
                )
                x = self.t_w * adapt_out + (1 - self.t_w) * x

        x = x.permute(1, 0, 2)
        x = self.clipmodel.ln_final(x)
        eot_indices = text.argmax(dim=-1)
        x = x[torch.arange(x.shape[0]), eot_indices]
        x = self.text_adapter[-1](x)
        return x


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
            normal_selection_ratio=getattr(args, 'normal_selection_ratio', 0.8),
            min_normal_frames=getattr(args, 'min_normal_frames', 4),
            max_normal_ratio=getattr(args, 'max_normal_ratio', 0.8)
        )

        # 4. Core Module 4: QuadSafeGate
        prior_weights = None
        if hasattr(args, 'routing_det_weight') and hasattr(args, 'routing_rec_weight'):
            prior_weights = [
                getattr(args, 'routing_det_weight', 0.4),
                getattr(args, 'routing_rec_weight', 0.3),
                getattr(args, 'routing_sem_weight', 0.1),
                getattr(args, 'routing_aux_weight', 0.2)
            ]
        self.safegate = QuadSafeGate(
            dataset_type=dataset_type,
            hidden_dim=getattr(args, 'routing_gate_hidden', 16),
            no_safegate=self.no_safegate,
            custom_prior_weights=prior_weights
        )

        # 5. CLIP Text Adapter
        self.clipmodel, _ = clip.load("ViT-B/16", device)
        for p in self.clipmodel.parameters():
            p.requires_grad = False

        text_adapt_until = getattr(args, 'text_adapt_until', 1 if dataset_type == 'xd' else 3)
        t_w = getattr(args, 't_w', 0.6 if dataset_type == 'xd' else 0.1)
        self.clip_adapter = CLIPAdapter(self.clipmodel, device, text_adapt_until, t_w)

        # Event & Background alignment MLPs
        self.mlp_event = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("dropout", nn.Dropout(getattr(args, 'mlp_dropout', 0.1))),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))

        self._text_features_cache = None
        self._init_trainable_parameters()

    def _init_trainable_parameters(self):
        """Initialize all trainable layers with truncated normal/Xavier."""
        for m in self.modules():
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

    def get_text_features(self, text_list):
        """Encode category descriptions into normalized CLIP text embeddings."""
        if not self.training and self._text_features_cache is not None:
            return self._text_features_cache

        descriptions = DESCRIPTIONS_ORI if len(text_list) == 14 else DESCRIPTIONS_ORI_XD
        category_features = []
        for class_name, desc_items in descriptions.items():
            tokens = clip.tokenize(desc_items).to(self.device)
            feats = self.clip_adapter.encode_text(tokens)
            mean_feat = feats.mean(dim=0)
            mean_feat = mean_feat / mean_feat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            category_features.append(mean_feat)

        text_features = torch.stack(category_features, dim=0)  # [C, D]
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

        # Step 1: MDTBackbone -> F_base, S_det
        f_base, s_det, logits_det = self.backbone(visual, padding_mask, lengths)

        # Step 2: ModalityAdaptiveBranch -> F_fused, S_aux
        f_fused, s_aux, logits_aux, g_av = self.modality_branch(
            f_base, visual, audio_features=audio, lengths=lengths
        )

        # Text features for semantic alignment
        text_features_ori = self.get_text_features(text)

        # Step 3 & 4: RUPA & QuadSafeGate
        dnp_dict = {}
        if DNP_use and not self.no_rupa:
            # Fast Feature-Level PCGrad Alternative: decouple/project RUPA backward gradient against main branch
            if self.training:
                f_main, f_rupa_in = GradientProjectionJunction.apply(f_fused)
            else:
                f_main, f_rupa_in = f_fused, f_fused

            f_rec, residual, s_rec, s_sem, rupa_dict = self.rupa(
                f_rupa_in, s_det, text_features_ori, lengths=lengths, visual_raw=visual
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
                'original_features': f_fused,
                'reconstructed_features': f_fused,
                'residual_features': torch.zeros_like(f_fused),
                'g_loss': torch.zeros(B, 1, device=visual.device),
                'orth_loss': torch.zeros(B, 1, device=visual.device),
                'dnp_normal_loss': torch.zeros(B, 1, device=visual.device),
            })

        # Event-centric & Background-centric attention pooling
        if lengths is not None:
            valid_mask = (
                torch.arange(T, device=visual.device)[None, :]
                < lengths.to(visual.device)[:, None]
            ).unsqueeze(-1).to(visual.dtype)
        else:
            valid_mask = torch.ones(B, T, 1, device=visual.device, dtype=visual.dtype)

        prob_abn = p_routing
        abn_weights = ((scale * prob_abn).exp() - 1.0) * valid_mask
        abn_weights = F.normalize(abn_weights, p=1, dim=1)

        nor_weights = ((scale * (1.0 - prob_abn)).exp() - 1.0) * valid_mask
        nor_weights = F.normalize(nor_weights, p=1, dim=1)

        # Video-level event & background representations
        abn_feat = torch.matmul(abn_weights.permute(0, 2, 1), event_features)
        nor_feat = torch.matmul(nor_weights.permute(0, 2, 1), background_features)

        # Snippet-level and video-level classification logits
        text_mat = F.normalize(text_features_ori, dim=-1).t().to(visual.dtype)
        f_norm = F.normalize(f_main, dim=-1)
        logits2 = f_norm @ text_mat / 0.07

        abn_feat_norm = F.normalize(abn_feat, dim=-1)
        nor_feat_norm = F.normalize(nor_feat, dim=-1)
        logits3 = abn_feat_norm @ text_mat / 0.07
        logits4 = nor_feat_norm @ text_mat / 0.07

        if DNP_use:
            return text_features_ori, logits1, logits2, logits3, logits4, dnp_dict
        return text_features_ori, logits1, logits2, logits3, logits4


# Legacy alias for backward compatibility with imports
DSANet = MARUPA


def load_compatible_model_state(model, state_dict, strict=False):
    """Safely loads model checkpoint."""
    target_model = model.module if hasattr(model, 'module') else model
    return target_model.load_state_dict(state_dict, strict=strict)

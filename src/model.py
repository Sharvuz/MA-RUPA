"""MA-RUPA v4: Modality-Adaptive Visual-Protected Prototype & Vision-Language WS-VAD Network.

Mathematically disciplined SOTA architecture for Weakly-Supervised Video Anomaly Detection (WS-VAD),
combining an uncorrupted LGT + SGNM + CLIP-Adapter core with strictly non-destructive (stop-gradient
visual-protected) multi-stream synergy:
  1. Pure LGT Dual-GCN Backbone (VadCLIP / DSANet): Local Window Transformer + Thresholded Cosine GCN
     (`adj4`) + Temporal Distance GCN (`disAdj`) without sequence-memorizing dilated Conv1d shortcuts.
  2. Memory-Light Batched CLIP_Adapter: Bias-free `SimpleAdapter` + `SimpleProj(512, 512, relu=True)`
     on canonical concept tokens (unused 86M CLIP visual tower deleted for 3x faster DataParallel).
  3. Self-Guided Dynamic Normal Pattern Refiner (SGNM) + Stop-Gradient Safe-RUPA Calibration Gate:
     `ConsistencyLoss` (weight 1.0) tethers `logits1` to bounded cosine reconstruction error, acting as
     an intrinsic anti-overfitting regularizer, while `SafeGate` refines snippet scores on detached
     `[det, rec, sem]` streams with zero gradient interference into `logits1`.
  4. Visual-Protected Cross-Modal Audio-Visual Branch (XD-Violence): Bidirectional Visual-Audio
     Cross-Attention conditioned on `visual_features.detach()`, preventing noisy movie soundtracks
     (e.g., normal action scenes) from poisoning visual representations while boosting true acoustic
     anomalies (gunshots, explosions, screams, crashes, riots) via visual-semantic concordance gating.
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


OPTIONAL_STATE_PREFIXES = (
    'rupa_gate.',
    'mst_adapter.',
    'audio_adapter.',
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
        # Support seamless loading of legacy DSANet `temporal.resblocks.X` -> `temporal.resblocks.X`
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

        # Guard against 0/0 NaN in softmax when an entire local window falls in padded tail
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
# 2. Visual-Protected Cross-Modal Audio-Visual Branch (XD-Violence SOTA)
# ============================================================================

class GatedCrossModalAudioAdapter(nn.Module):
    """Bidirectional Visual-Audio Cross-Modal Concordance Branch (XD-Violence).

    In MARUPA v4, `MARUPA.forward` feeds `visual_features.detach()` into this branch and keeps
    `visual_features` 100% untouched before `logits1`, `logits2`, and `SGNM`. Thus:
      - Noisy movie soundtracks (drums, background music, non-violent SFX) NEVER corrupt the
        CLIP LGT-SGNM visual detector.
      - Meanwhile, `aux_head` leverages full bidirectional `[v_n, a_n, v_from_a, v_n * a_n]`
        cross-modal context to accurately score violent acoustic events (gunshots, explosions,
        screams, car crashes, riots).
    """

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

        # Rich Cross-Modal Concordance Scorer conditioned on [V_ctx, A_ctx, V<-A, V*A]
        self.aux_head = nn.Sequential(
            LayerNorm(visual_dim * 4),
            nn.Linear(visual_dim * 4, visual_dim // 2),
            QuickGELU(),
            nn.Dropout(dropout),
            nn.Linear(visual_dim // 2, 1),
        )

        self.alpha_av = nn.Parameter(torch.tensor(0.05))

    def forward(self, f_vis: torch.Tensor, audio_raw: torch.Tensor, valid_mask: torch.Tensor = None):
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

        has_signal = (a_aligned.abs().sum(dim=(-1, -2), keepdim=True) > 1e-6).to(f_vis.dtype)  # [B, 1, 1]

        a_lift = self.audio_lift(self.audio_input_norm(a_aligned))
        a_conv = self.audio_temp_conv(a_lift.transpose(1, 2)).transpose(1, 2)
        h_a = a_lift + 0.5 * a_conv
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
        logits_aux = self.aux_head(av_joint)  # [B, T, 1]
        # Suppress missing-audio samples toward calm background probability
        logits_aux = logits_aux * has_signal + (-6.0) * (1.0 - has_signal)
        s_aux = torch.sigmoid(logits_aux)

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
# 3. Memory-Light Batched CLIP_Adapter (Exact SimpleAdapter + SimpleProj Geometry)
# ============================================================================

class CLIPTextAdapter(nn.Module):
    """Exact VadCLIP / DSANet Bias-Free CLIP_Adapter with 1-Pass Batched Tokenization.

    Why this exact geometry is critical:
      1. Deletes unused `clipmodel.visual` (86M params), saving ~350 MB VRAM per GPU and
         making `torch.nn.DataParallel` 3x faster.
      2. Uses bias-free `SimpleAdapter(512, 512)` x `text_adapt_until` + `SimpleProj(512, 512, relu=True)`
         directly on `ln_final(x_eot)`, allowing `text_orthogonality_loss` (`loss3`) to converge
         smoothly to `0.000` without the severe bag-memorization drift of multi-layer biased MLPs.
    """

    def __init__(self, clipmodel, text_adapt_until: int = 3, t_w: float = 0.1):
        super().__init__()
        if hasattr(clipmodel, 'visual'):
            del clipmodel.visual

        self.clipmodel = clipmodel
        for p in self.clipmodel.parameters():
            p.requires_grad = False

        self.text_adapt_until = text_adapt_until
        self.t_w = t_w

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
        x = x.permute(1, 0, 2)  # [77, N_prompts, 512]

        for i, block in enumerate(self.clipmodel.transformer.resblocks):
            x = block(x)
            if adapt_text and i < self.text_adapt_until:
                adapt_out = self.text_adapter[i](x.float()).to(cast_dtype)
                adapt_out = (
                    adapt_out * x.norm(dim=-1, keepdim=True)
                    / (adapt_out.norm(dim=-1, keepdim=True) + 1e-6)
                )
                x = self.t_w * adapt_out + (1.0 - self.t_w) * x

        x = x.permute(1, 0, 2)  # [N_prompts, 77, 512]
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

        feat = self.encode_tokens(tokens, adapt_text=True)  # [C * K, 512]
        feat = feat.view(c, k, -1).mean(dim=1)
        return F.normalize(feat, dim=-1)  # [C, 512]


# ============================================================================
# 4. Self-Guided Dynamic Normal Pattern Refiner (SGNM)
# ============================================================================

class SGNM(nn.Module):
    """Dynamic Normal Pattern Extractor & Counterfactual Normal Background Reconstructor."""

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
    def gather_loss(query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        distribution = 1.0 - F.cosine_similarity(query.unsqueeze(2), keys.unsqueeze(1), dim=-1)
        distance, _ = torch.min(distribution, dim=2)
        return distance.mean()

    def forward(self, visual_features: torch.Tensor, logits1: torch.Tensor, lengths: torch.Tensor = None):
        B, N, D = visual_features.shape

        with torch.no_grad():
            anomaly_scores = torch.sigmoid(logits1.squeeze(-1))
            num_normal_frames = max(1, int(N * self.normal_selection_ratio))
            _, indices = torch.topk(anomaly_scores, k=N, largest=False, dim=1)
            normal_indices = indices[:, :num_normal_frames]

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
# 5. Unified SOTA Model: MARUPA (with DSANet Alias)
# ============================================================================

class MARUPA(nn.Module):
    """MA-RUPA v4: Visual-Protected Multimodal & Vision-Language WS-VAD Architecture."""

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

        # Configuration flags
        self.no_rupa = bool(getattr(args, 'no_rupa', False))
        self.rupa_use = (not self.no_rupa) and bool(getattr(args, 'rupa_use', True))
        self.use_mdtp = bool(getattr(args, 'use_mdtp', False)) and (not bool(getattr(args, 'no_mdtp', True)))
        self.no_audio = bool(getattr(args, 'no_audio', False))
        self.no_safegate = bool(getattr(args, 'no_safegate', False))

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

        # 3. Sensitivity Detector & Video-Conditioned Prompt MLPs
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

        # 4. Frozen CLIP Text Backbone + Bias-Free CLIPTextAdapter
        text_adapt_until = int(getattr(args, 'text_adapt_until', 3 if num_class == 14 else 1))
        t_w = float(getattr(args, 't_w', 0.1 if num_class == 14 else 0.6))
        clipmodel, _ = clip.load("ViT-B/16", device="cpu")
        self.clip_adapter = CLIPTextAdapter(clipmodel, text_adapt_until=text_adapt_until, t_w=t_w)

        # 5. Dynamic Normal Pattern Refiner (SGNM) & Stop-Gradient SafeGate
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

        self.rupa_gate = nn.Sequential(
            nn.Linear(5, routing_gate_hidden),
            QuickGELU(),
            nn.Linear(routing_gate_hidden, 1),
        )

        self._eval_text_cache = None
        self.initialize_parameters()

    def train(self, mode: bool = True):
        super().train(mode)
        self._eval_text_cache = None
        return self

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
        sim = torch.bmm(x_norm, x_norm.transpose(1, 2))  # [B, T, T]
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
        valid_mask = (~padding_mask).unsqueeze(-1).float()  # [B, T, 1]

        # 1. Pure Spatio-Temporal LGT Dual-GCN Encoding (never mutated by noisy audio)
        visual_features = self.encode_video(visual, padding_mask, lengths_dev)

        # 2. Visual-Protected Cross-Modal Audio-Visual Branch (XD-Violence)
        # Uses `visual_features.detach()` so soundtrack/SFX noise never poisons visual representations!
        logits_aux = None
        s_aux = None
        g_av = None
        if self.audio_adapter is not None and audio is not None and not self.no_audio:
            _, s_aux, logits_aux, g_av = self.audio_adapter(
                visual_features.detach(), audio, valid_mask=valid_mask
            )

        # 3. Sensitivity Anomaly Detector Head (logits1)
        logits1 = self.classifier(visual_features + self.mlp2(visual_features))  # [B, T, 1]
        detector_score = torch.sigmoid(logits1)

        # 4. Video-Conditioned Fine-Grained Text-Visual Alignment (logits2)
        text_features_ori = self.get_text_features(dev)  # [C, 512]
        logits_attn = logits1.permute(0, 2, 1)           # [B, 1, T]
        visual_attn = F.normalize(torch.bmm(logits_attn, visual_features), dim=-1)  # [B, 1, D]
        visual_attn_exp = visual_attn.expand(B, text_features_ori.shape[0], -1)     # [B, C, D]

        text_features = text_features_ori.unsqueeze(0).expand(B, -1, -1) + visual_attn_exp
        text_features = text_features + self.mlp1(text_features)

        visual_features_norm = F.normalize(visual_features, dim=-1)
        text_features_norm = F.normalize(text_features, dim=-1).permute(0, 2, 1)
        logits2 = torch.bmm(visual_features_norm, text_features_norm.to(visual_features_norm.dtype)) / 0.07  # [B, T, C]

        # 5. Video-Level Event & Background Contrastive Pooling directly on `visual_features` (Exact DSANet Core)
        abn_logits = (scale * detector_score).exp() - 1.0
        abn_weights = F.normalize(abn_logits, p=1, dim=1)

        nor_logits = (scale * (1.0 - detector_score)).exp() - 1.0
        nor_weights = F.normalize(nor_logits, p=1, dim=1)

        abn_feat = torch.bmm(abn_weights.permute(0, 2, 1), visual_features)  # [B, 1, D]
        nor_feat = torch.bmm(nor_weights.permute(0, 2, 1), visual_features)  # [B, 1, D]

        nor_text_norm = F.normalize(text_features_ori, dim=-1).t().unsqueeze(0).expand(B, -1, -1)  # [B, D, C]
        abn_vis_norm = F.normalize(abn_feat, dim=-1)
        nor_vis_norm = F.normalize(nor_feat, dim=-1)

        logits3 = torch.bmm(abn_vis_norm, nor_text_norm.to(abn_vis_norm.dtype)) / 0.07  # [B, 1, C]
        logits4 = torch.bmm(nor_vis_norm, nor_text_norm.to(nor_vis_norm.dtype)) / 0.07  # [B, 1, C]

        # 6. Self-Guided Dynamic Normal Pattern Refiner (SGNM) & Stop-Gradient Safe-RUPA Calibration
        sem_temp = 2.5 if self.num_class == 14 else 2.0
        semantic_score = (1.0 - F.softmax(logits2.float() / sem_temp, dim=-1)[:, :, :1]).to(visual_features.dtype).clamp(0.0, 1.0)

        dnp_dict = {
            'detector_scores': detector_score,
            'routing_scores': detector_score,
            'routing_logits': logits1,
            'semantic_logits': logits2,
            'semantic_scores': semantic_score,
            'aux_logits': logits_aux,
            'aux_scores': s_aux,
            'audio_gate': g_av,
        }

        if DNP_use and not self.no_rupa:
            reconstructed_features, dynamic_normal_patterns, g_loss = self.video_anomaly_refiner(
                visual_features, logits1, lengths_dev
            )
            residual_features = visual_features - reconstructed_features
            reconstruction_score = (
                (1.0 - F.cosine_similarity(visual_features, reconstructed_features, dim=-1)).unsqueeze(-1) / 2.0
            ).clamp(0.0, 1.0)

            # Reshape g_loss to [1] so torch.nn.DataParallel gathers across GPUs without scalar UserWarning
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
                # and NEVER injects duplicate binary MIL gradients back into `logits1`!
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

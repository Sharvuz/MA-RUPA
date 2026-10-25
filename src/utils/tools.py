import math
import torch
import numpy as np

def get_batch_label(texts, prompt_text, label_map: dict):
    label_vectors = torch.zeros(0)
    if len(label_map) != 7:
        if len(label_map) == 2:
            for text in texts:
                label_vector = torch.zeros(2)
                if text == 'Normal':
                    label_vector[0] = 1
                else:
                    label_vector[1] = 1
                label_vector = label_vector.unsqueeze(0)
                label_vectors = torch.cat([label_vectors, label_vector], dim=0)
        else:
            for text in texts:
                label_vector = torch.zeros(len(prompt_text))
                if text in label_map:
                    label_text = label_map[text]
                    label_vector[prompt_text.index(label_text)] = 1

                label_vector = label_vector.unsqueeze(0)
                label_vectors = torch.cat([label_vectors, label_vector], dim=0)
    else:
        for text in texts:
            label_vector = torch.zeros(len(prompt_text))
            labels = text.split('-')
            for label in labels:
                if label in label_map:
                    label_text = label_map[label]
                    label_vector[prompt_text.index(label_text)] = 1

            label_vector = label_vector.unsqueeze(0)
            label_vectors = torch.cat([label_vectors, label_vector], dim=0)

    return label_vectors

def get_prompt_text(label_map: dict):
    prompt_text = []
    for v in label_map.values():
        prompt_text.append(v)

    return prompt_text

def get_batch_mask(lengths, maxlen):
    batch_size = lengths.shape[0]
    mask = torch.empty(batch_size, maxlen)
    mask.fill_(0)
    for i in range(batch_size):
        if lengths[i] < maxlen:
            mask[i, lengths[i]:maxlen] = 1

    return mask.bool()

def random_extract(feat, t_max):
   r = np.random.randint(feat.shape[0] - t_max)
   return feat[r : r+t_max, :]

def uniform_extract(feat, t_max, avg: bool = True):
    new_feat = np.zeros((t_max, feat.shape[1])).astype(np.float32)
    r = np.linspace(0, len(feat), t_max+1, dtype=np.int32)
    if avg == True:
        for i in range(t_max):
            if r[i]!=r[i+1]:
                new_feat[i,:] = np.mean(feat[r[i]:r[i+1],:], 0)
            else:
                new_feat[i,:] = feat[r[i],:]
    else:
        r = np.linspace(0, feat.shape[0]-1, t_max, dtype=np.uint16)
        new_feat = feat[r, :]

    return new_feat

def pad(feat, min_len):
    clip_length = feat.shape[0]
    if clip_length <= min_len:
       return np.pad(feat, ((0, min_len - clip_length), (0, 0)), mode='constant', constant_values=0)
    else:
       return feat



def process_feat(feat, length, is_random=False):
    clip_length = feat.shape[0] #chia segment
    #TH: video dài hơn 256 segment thì lấy N segment / 256 và lấy TB từng khoảng trải đều
    if feat.shape[0] > length:
        if is_random:
            return random_extract(feat, length), length
        else:
            return uniform_extract(feat, length), length
    else:
        return pad(feat, length), clip_length

def process_split(feat, length):
    clip_length = feat.shape[0]

    #TH: video ngắn hơn 256 segment thì padding thêm vecto 0 vào sau cho đủ 256 segment
    if clip_length < length:
        return pad(feat, length), clip_length
    else:
        split_num = int(clip_length / length) + 1
        for i in range(split_num):
            if i == 0:
                split_feat = feat[i*length:i*length+length, :].reshape(1, length, feat.shape[1])
            elif i < split_num - 1:
                split_feat = np.concatenate([split_feat, feat[i*length:i*length+length, :].reshape(1, length, feat.shape[1])], axis=0)
            else:
                split_feat = np.concatenate([split_feat, pad(feat[i*length:i*length+length, :], length).reshape(1, length, feat.shape[1])], axis=0)

        return split_feat, clip_length


def xd_collate_fn(batch):
    """Custom collate function for XDDataset supporting optional/missing audio."""
    visual_batch = torch.stack([item[0] for item in batch], dim=0)
    labels = [item[1] for item in batch]
    lengths = torch.tensor([item[2] for item in batch], dtype=torch.long)
    audios = [item[3] for item in batch]

    has_audio = any(a is not None for a in audios)
    if not has_audio:
        audio_batch = None
    else:
        first_audio = next(a for a in audios if a is not None)
        target_len, audio_dim = first_audio.shape
        padded_audios = []
        for a in audios:
            if a is None:
                padded_audios.append(torch.zeros(target_len, audio_dim, dtype=torch.float32))
            else:
                padded_audios.append(a)
        audio_batch = torch.stack(padded_audios, dim=0)

    return visual_batch, labels, lengths, audio_batch


def xd_test_collate_fn(batch):
    """Custom collate function for XDDataset test dataloader (batch_size=1).
    
    Safely handles test samples where audio is None or missing, preventing
    PyTorch's default_collate from raising:
    TypeError: default_collate: batch must contain tensors... found <class 'NoneType'>.
    """
    if len(batch) == 1:
        item = batch[0]
        # In test mode, item is:
        # (clip_tensor, label, clip_length, path, label2, audio_feature)
        clip_tensor = item[0]
        if isinstance(clip_tensor, torch.Tensor):
            visual = clip_tensor.unsqueeze(0)
        else:
            visual = torch.tensor(clip_tensor, dtype=torch.float32).unsqueeze(0)

        label = [item[1]]
        length = item[2] if isinstance(item[2], torch.Tensor) else torch.tensor([item[2]], dtype=torch.long)
        path = [item[3]] if len(item) > 3 else []
        label2 = [item[4]] if len(item) > 4 else []

        audio = None
        if len(item) >= 6 and item[5] is not None:
            if isinstance(item[5], torch.Tensor):
                audio = item[5].unsqueeze(0)
            else:
                audio = torch.tensor(item[5], dtype=torch.float32).unsqueeze(0)

        return visual, label, length, path, label2, audio

    # Fallback for len(batch) > 1
    visuals = torch.stack([
        it[0] if isinstance(it[0], torch.Tensor) else torch.tensor(it[0], dtype=torch.float32)
        for it in batch
    ], dim=0)
    labels = [it[1] for it in batch]
    lengths = torch.tensor([it[2] for it in batch], dtype=torch.long)
    paths = [it[3] for it in batch] if len(batch[0]) > 3 else []
    labels2 = [it[4] for it in batch] if len(batch[0]) > 4 else []
    audios = [it[5] for it in batch] if len(batch[0]) >= 6 else []

    if not any(a is not None for a in audios):
        audio_batch = None
    else:
        first_a = next(a for a in audios if a is not None)
        target_shape = first_a.shape
        audio_list = []
        for a in audios:
            if a is None:
                audio_list.append(torch.zeros(target_shape, dtype=torch.float32))
            else:
                if not isinstance(a, torch.Tensor):
                    a = torch.tensor(a, dtype=torch.float32)
                audio_list.append(a)
        audio_batch = torch.stack(audio_list, dim=0)

    return visuals, labels, lengths, paths, labels2, audio_batch


def smooth_scores(scores: np.ndarray, method: str = 'gaussian', sigma: float = 1.5, window_length: int = 5):
    """Smooth 1D anomaly score curve using Gaussian or Savitzky-Golay filter."""
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    if len(scores) < 3:
        return scores
    if method == 'gaussian':
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(scores, sigma=sigma)
        return np.nan_to_num(smoothed, nan=0.0, posinf=1.0, neginf=0.0)
    elif method == 'savgol':
        from scipy.signal import savgol_filter
        wl = min(window_length, len(scores))
        if wl % 2 == 0:
            wl -= 1
        if wl <= 2:
            return scores
        smoothed = savgol_filter(scores, window_length=wl, polyorder=1)
        return np.nan_to_num(smoothed, nan=0.0, posinf=1.0, neginf=0.0)
    return scores


def safe_load_checkpoint(path: str, map_location=None):
    """Safely load a PyTorch checkpoint supporting PyTorch 2.6+ weights_only policy."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def calibrate_stream_global(scores_np: np.ndarray,
                            target_q15: float = 0.04,
                            target_q90: float = 0.78) -> np.ndarray:
    """Unsupervised Monotonic Logit-Affine Calibration across a test prediction stream.

    Aligns global dynamic ranges across heterogeneous streams (e.g., CLIP semantic
    probabilities vs binary detector/audio sigmoid probabilities) using ONLY unlabeled
    predicted quantiles (zero ground-truth labels used). Preserves single-stream ROC-AUC
    and AP identically while preventing scale-dominant streams from drowning out gentler
    well-ranked semantic streams during multi-stream fusion.
    """
    s = np.nan_to_num(np.asarray(scores_np, dtype=np.float64), nan=0.0, posinf=1.0, neginf=0.0)
    if s.size < 16:
        return s.astype(np.float32)

    s_clamped = np.clip(s, 1e-5, 1.0 - 1e-5)
    logits = np.log(s_clamped / (1.0 - s_clamped))

    q15, q90 = np.percentile(logits, [15.0, 90.0])
    span = float(q90 - q15)
    if span < 1e-3:
        # Flat/degenerate stream: leave untouched so noise is never stretched
        return s.astype(np.float32)

    target_lo = math.log(target_q15 / (1.0 - target_q15))
    target_hi = math.log(target_q90 / (1.0 - target_q90))
    scale = float(np.clip((target_hi - target_lo) / span, 0.40, 2.75))
    bias = target_lo - scale * float(q15)

    cal_logits = np.clip(scale * logits + bias, -14.0, 14.0)
    cal_probs = 1.0 / (1.0 + np.exp(-cal_logits))
    return np.clip(cal_probs, 0.0, 1.0).astype(np.float32)


def compute_single_pass_uccf(prob_dict: dict, priors: dict, delta_dis: float = 0.35) -> torch.Tensor:
    """Contrast-Aware Calibrated Fusion (CA-UCCF v2).

    Replaces inverse-entropy weighting (which assigned near-100% weight to flat/missed
    near-zero streams where binary entropy -> 0) with:
    1. Peak-to-Background Discriminative SNR Weighting:
       SNR_j = (Q_0.95(s_j) - Q_0.50(s_j))_+ / (IQR(s_j) + 0.05)
       Dead/flat streams get SNR_j -> 0 and fall back smoothly without poisoning active streams.
    2. Local Consensus Outlier Suppression:
       Damps isolated single-stream false spikes or single-stream dropouts that deviate
       strongly from the multi-stream consensus anchor.
    """
    stream_keys = list(prob_dict.keys())
    probs = [prob_dict[k].float().clamp(1e-6, 1.0 - 1e-6) for k in stream_keys]
    stacked = torch.stack(probs, dim=0)  # [S, T]
    S, T = stacked.shape

    pi = torch.tensor(
        [priors.get(k, 1.0) for k in stream_keys],
        device=stacked.device, dtype=torch.float32
    ).unsqueeze(-1)  # [S, 1]

    if T < 4:
        w = pi / pi.sum(dim=0, keepdim=True).clamp_min(1e-6)
        return (w * stacked).sum(dim=0).clamp(0.0, 1.0)

    q_levels = torch.tensor([0.25, 0.50, 0.75, 0.95], device=stacked.device, dtype=torch.float32)
    quants = torch.quantile(stacked, q_levels, dim=-1)  # [4, S]
    q25, q50, q75, q95 = quants[0], quants[1], quants[2], quants[3]

    iqr = (q75 - q25).clamp_min(0.04)
    snr = ((q95 - q50).clamp_min(0.0) / (iqr + 0.05)).clamp(0.0, 4.0).unsqueeze(-1)  # [S, 1]

    # Active contrast-aware stream weights (smoothly reduces to prior pi on flat normal videos)
    stream_w = pi * (1.0 + 1.20 * snr)
    stream_w = stream_w / stream_w.sum(dim=0, keepdim=True).clamp_min(1e-6)  # [S, 1]

    # Multi-stream consensus anchor at each snippet t
    anchor_t = (stream_w * stacked).sum(dim=0, keepdim=True)  # [1, T]

    # Suppress isolated single-stream dropouts or uncorroborated noise spikes
    dev_sq = (stacked - anchor_t) ** 2  # [S, T]
    disagreement = stacked.max(dim=0, keepdim=True).values - stacked.min(dim=0, keepdim=True).values
    gate_strength = torch.where(disagreement > delta_dis, 3.0, 1.5)
    local_gate = torch.exp(-gate_strength * dev_sq)  # [S, T]

    eff_w = stream_w * (0.45 + 0.55 * local_gate)
    eff_w = eff_w / eff_w.sum(dim=0, keepdim=True).clamp_min(1e-6)

    fused_prob = (eff_w * stacked).sum(dim=0)
    return fused_prob.clamp(0.0, 1.0)


def apply_dodo_refinement(prob_np: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    """Edge-Preserving Peak-Safe Morphological / Gaussian Refinement (DODO v2).

    Preserves sharp 1-2s explosive anomaly peaks (Shooting, Explosion, RoadAccidents)
    via gradient-gated blending between a light sharp filter (sigma=0.45) at event
    boundaries and morphological closing on sustained interiors for mAP@IoU.
    """
    prob_np = np.nan_to_num(prob_np, nan=0.0, posinf=1.0, neginf=0.0)
    if len(prob_np) < 3:
        return prob_np

    from scipy.ndimage import gaussian_filter1d, grey_closing
    s_sharp = gaussian_filter1d(prob_np, sigma=0.45)
    s_smooth = gaussian_filter1d(prob_np, sigma=max(0.5, float(sigma)))
    try:
        s_closed = grey_closing(s_smooth, size=5)
    except Exception:
        s_closed = s_smooth

    # Edge gate: keep s_sharp near sharp transitions/peaks so peak amplitude is never clipped
    grad = np.zeros_like(prob_np, dtype=np.float64)
    grad[1:] = np.maximum(grad[1:], np.abs(prob_np[1:] - prob_np[:-1]))
    grad[:-1] = np.maximum(grad[:-1], np.abs(prob_np[1:] - prob_np[:-1]))
    edge_alpha = np.exp(-6.0 * (grad ** 2))  # -> 0 at sharp jumps (100% s_sharp), -> 1 on plateaus

    blend_w = 0.30 * edge_alpha
    s_dodo = (1.0 - blend_w) * s_sharp + blend_w * np.maximum(s_sharp, 0.85 * s_closed)
    return np.clip(s_dodo, 0.0, 1.0).astype(np.float32)



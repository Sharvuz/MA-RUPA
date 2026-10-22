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


def compute_single_pass_uccf(prob_dict: dict, priors: dict, delta_dis: float = 0.35) -> torch.Tensor:
    """Fast Single-Pass Uncertainty-Calibrated Fusion (UCCF).
    
    1. Computes normalized binary Shannon entropy u_{j,t} as instantaneous uncertainty:
       u_{j,t} = -[p log2(p) + (1-p) log2(1-p)] + eps.
    2. Computes inverse-uncertainty prior-aware weights:
       w_{j,t} = (pi_j / u_{j,t}^2) / sum_k (pi_k / u_{k,t}^2).
    3. Disagreement-Aware Gating:
       When stream disagreement max_j p_j - min_j p_j > delta_dis, sharpens exponent
       to u_{j,t}^-4 so the most confident stream dominates without dilution.
    """
    stream_keys = list(prob_dict.keys())
    probs = [prob_dict[k].clamp(1e-6, 1.0 - 1e-6) for k in stream_keys]
    stacked = torch.stack(probs, dim=0)  # [S, T]

    log2 = math.log(2.0)
    entropy = -(stacked * torch.log(stacked) + (1.0 - stacked) * torch.log(1.0 - stacked)) / log2
    u = entropy + 1e-4  # [S, T]

    disagreement = stacked.max(dim=0).values - stacked.min(dim=0).values
    is_disagreed = (disagreement > delta_dis)

    pi = torch.tensor(
        [priors.get(k, 1.0) for k in stream_keys],
        device=stacked.device, dtype=stacked.dtype
    ).unsqueeze(-1)  # [S, 1]

    u_sq = u ** 2
    u_sharp = u ** 4

    weights_normal = pi / (u_sq + 1e-5)
    weights_normal = weights_normal / weights_normal.sum(dim=0, keepdim=True).clamp_min(1e-6)

    weights_sharp = pi / (u_sharp + 1e-5)
    weights_sharp = weights_sharp / weights_sharp.sum(dim=0, keepdim=True).clamp_min(1e-6)

    weights = torch.where(is_disagreed.unsqueeze(0), weights_sharp, weights_normal)
    fused_prob = (weights * stacked).sum(dim=0)
    return fused_prob.clamp(0.0, 1.0)


def apply_dodo_refinement(prob_np: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Peak-Preserving Morphological / Gaussian Proposal Refinement (DODO).
    
    Combines lightly-smoothed score (preserving sharp local peaks for ROC-AUC)
    with connected-segment smoothed score (Gaussian + morphological closing to bridge
    short dips within continuous anomalies for mAP@IoU).
    
    S_DODO = 0.65 * S_AUC + 0.35 * S_mAP.
    """
    prob_np = np.nan_to_num(prob_np, nan=0.0, posinf=1.0, neginf=0.0)
    if len(prob_np) < 3:
        return prob_np

    from scipy.ndimage import gaussian_filter1d, grey_closing
    s_auc = gaussian_filter1d(prob_np, sigma=0.5)
    s_smooth = gaussian_filter1d(prob_np, sigma=sigma)
    try:
        s_closed = grey_closing(s_smooth, size=5)
    except Exception:
        s_closed = s_smooth

    s_dodo = 0.65 * s_auc + 0.35 * s_closed
    return np.clip(s_dodo, 0.0, 1.0)



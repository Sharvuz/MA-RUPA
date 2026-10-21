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

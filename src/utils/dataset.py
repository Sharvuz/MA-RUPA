import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import torch.nn.functional as F

import utils.tools as tools


_KAGGLE_FILE_MAP = None
_KAGGLE_AUDIO_MAP = None


def get_kaggle_file_maps():
    """Scans and indexes all .npy files in /kaggle/input and /kaggle/working once.
    Maintains a general file map and an audio-prioritized file map.
    """
    global _KAGGLE_FILE_MAP, _KAGGLE_AUDIO_MAP
    if _KAGGLE_FILE_MAP is not None and _KAGGLE_AUDIO_MAP is not None:
        return _KAGGLE_FILE_MAP, _KAGGLE_AUDIO_MAP

    _KAGGLE_FILE_MAP = {}
    _KAGGLE_AUDIO_MAP = {}
    search_dirs = []
    for d in ['/kaggle/input', '/kaggle/working', 'data', 'dataset']:
        if os.path.exists(d):
            search_dirs.append(d)

    for root_dir in search_dirs:
        for dirpath, _, filenames in os.walk(root_dir):
            dir_lower = dirpath.lower()
            is_audio_dir = any(kw in dir_lower for kw in ['audio', 'vgg', 'clap', 'ast', 'sound', 'wav'])
            for fname in filenames:
                if fname.endswith('.npy'):
                    full_p = os.path.join(dirpath, fname)
                    _KAGGLE_FILE_MAP[fname] = full_p
                    if is_audio_dir:
                        _KAGGLE_AUDIO_MAP[fname] = full_p

                    # Map '#' and '+' equivalents for Kaggle filename safety
                    variants = [fname]
                    if '+' in fname:
                        variants.append(fname.replace('+', '#'))
                    if '#' in fname:
                        variants.append(fname.replace('#', '+'))
                    if '_' in fname and ('+' in fname or '#' in fname):
                        variants.append(fname.replace('+', '_').replace('#', '_'))

                    for v in variants:
                        if v not in _KAGGLE_FILE_MAP:
                            _KAGGLE_FILE_MAP[v] = full_p
                        if is_audio_dir and v not in _KAGGLE_AUDIO_MAP:
                            _KAGGLE_AUDIO_MAP[v] = full_p

    return _KAGGLE_FILE_MAP, _KAGGLE_AUDIO_MAP


def get_kaggle_file_map():
    file_map, _ = get_kaggle_file_maps()
    return file_map


def get_kaggle_audio_map():
    _, audio_map = get_kaggle_file_maps()
    return audio_map


def get_path_variants(path_or_name: str) -> list[str]:
    """Generate filename/path variants accounting for Kaggle '# -> +' renaming."""
    variants = [path_or_name]
    if '#' in path_or_name:
        variants.append(path_or_name.replace('#', '+'))
        variants.append(path_or_name.replace('#', '_'))
        variants.append(path_or_name.replace('#', ''))
    if '+' in path_or_name:
        variants.append(path_or_name.replace('+', '#'))
        variants.append(path_or_name.replace('+', '_'))
        variants.append(path_or_name.replace('+', ''))
    return list(dict.fromkeys(variants))


_RESOLVE_CACHE = {}
_INFER_AUDIO_CACHE = {}
_AUDIO_PATH_BY_BASEKEY = {}


def resolve_feature_path(path: str, data_dir: str = None) -> str:
    """Resolve file path locally, in data_dir, or in Kaggle directories with '# <-> +' support."""
    cache_key = (path, data_dir)
    if cache_key in _RESOLVE_CACHE:
        return _RESOLVE_CACHE[cache_key]

    res = _resolve_feature_path_uncached(path, data_dir)
    _RESOLVE_CACHE[cache_key] = res
    return res


def _resolve_feature_path_uncached(path: str, data_dir: str = None) -> str:
    # 1. Direct path check with variants
    for v in get_path_variants(path):
        if os.path.exists(v):
            return v

    base_name = os.path.basename(path)
    base_variants = get_path_variants(base_name)

    # 2. Check in data_dir if provided
    if data_dir and os.path.exists(data_dir):
        for v in base_variants:
            cand = os.path.join(data_dir, v)
            if os.path.exists(cand):
                return cand
        # Also try relative path
        rel = path.lstrip('/\\')
        for v in get_path_variants(rel):
            cand = os.path.join(data_dir, v)
            if os.path.exists(cand):
                return cand

    # 3. Check in indexed Kaggle filesystem
    k_map = get_kaggle_file_map()
    for v in base_variants:
        if v in k_map and os.path.exists(k_map[v]):
            return k_map[v]

    return path


def infer_audio_path(visual_path: str, audio_dir: str = None) -> str:
    """Infer the audio feature file path corresponding to a visual snippet feature.
    
    Handles 5-crop/10-crop suffix stripping and Kaggle '# <-> +' character replacements.
    """
    cache_key = (visual_path, audio_dir)
    if cache_key in _INFER_AUDIO_CACHE:
        return _INFER_AUDIO_CACHE[cache_key]

    res = _infer_audio_path_uncached(visual_path, audio_dir)
    _INFER_AUDIO_CACHE[cache_key] = res
    return res


def _infer_audio_path_uncached(visual_path: str, audio_dir: str = None) -> str:
    base_name = os.path.basename(visual_path)
    base_no_crop = base_name
    for crop_idx in range(10):
        if f"__{crop_idx}.npy" in base_no_crop:
            base_no_crop = base_no_crop.replace(f"__{crop_idx}.npy", ".npy")
            break

    target_names = get_path_variants(base_name) + get_path_variants(base_no_crop)
    target_names = list(dict.fromkeys(target_names))

    # 1. Check in explicit audio_dir
    if audio_dir and os.path.exists(audio_dir):
        for name in target_names:
            candidate = os.path.join(audio_dir, name)
            if os.path.exists(candidate):
                return candidate

    # 2. Check Kaggle audio-prioritized file map
    audio_map = get_kaggle_audio_map()
    for name in target_names:
        if name in audio_map and os.path.exists(audio_map[name]):
            return audio_map[name]

    # 3. Common directory substitutions
    substitutions = [
        ('ClipFeatures', 'AudioFeatures'),
        ('clip_features', 'audio_features'),
        ('rgb', 'audio'),
        ('RGB', 'Audio'),
        ('XDTrainClipFeatures', 'XDTrainAudioFeatures'),
        ('XDTestClipFeatures', 'XDTestAudioFeatures'),
        ('XDTrainClipFeatures', 'XDTrainVGGishFeatures'),
        ('XDTestClipFeatures', 'XDTestVGGishFeatures'),
        ('clip', 'audio'),
        ('CLIP', 'AUDIO'),
    ]
    for old_sub, new_sub in substitutions:
        if old_sub in visual_path:
            cand_base = visual_path.replace(old_sub, new_sub)
            for v in get_path_variants(cand_base):
                if os.path.exists(v):
                    return v
            # Strip crop
            cand_no_crop = cand_base
            for crop_idx in range(10):
                if f"__{crop_idx}.npy" in cand_no_crop:
                    cand_no_crop = cand_no_crop.replace(f"__{crop_idx}.npy", ".npy")
                    break
            for v in get_path_variants(cand_no_crop):
                if os.path.exists(v):
                    return v

    # 4. Check general Kaggle file map with audio keywords
    k_map = get_kaggle_file_map()
    for name in target_names:
        if name in k_map and os.path.exists(k_map[name]):
            cand = k_map[name]
            cand_lower = cand.lower()
            if any(kw in cand_lower for kw in ['audio', 'vgg', 'clap', 'ast', 'sound', 'wav']):
                return cand

    return None


class UCFDataset(data.Dataset):
    """UCF-Crime Dataset Loader.
    
    Loads pre-extracted 512-D CLIP visual features. UCF-Crime does not have audio,
    so the architecture automatically operates in Missing-Modality / Motion-Residual Mode.
    Automatically resolves file paths with '# <-> +' support on Kaggle.
    Supports random single-crop selection per video per epoch to prevent overfitting/memorization.
    """
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool, label_map: dict,
                 normal: bool = False, data_dir: str = None, sample_one_crop_per_video: bool = False):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
        self.normal = normal
        self.data_dir = data_dir
        self.sample_one_crop_per_video = sample_one_crop_per_video

        if not test_mode:
            if normal:
                self.df = self.df.loc[self.df['label'] == 'Normal'].reset_index(drop=True)
            else:
                self.df = self.df.loc[self.df['label'] != 'Normal'].reset_index(drop=True)

        self.raw_paths = self.df['path'].tolist()
        self.labels = self.df['label'].tolist()

        # Group crops by base video ID for healthy epoch sampling without bag memorization
        self.video_groups = []
        if self.sample_one_crop_per_video and not self.test_mode:
            from collections import defaultdict
            grouped = defaultdict(list)
            for p, l in zip(self.raw_paths, self.labels):
                base_id = os.path.basename(p).split('__')[0]
                grouped[base_id].append((p, l))
            self.video_groups = list(grouped.values())

    def __len__(self):
        if self.sample_one_crop_per_video and not self.test_mode and self.video_groups:
            return len(self.video_groups)
        return len(self.raw_paths)

    def __getitem__(self, index):
        if self.sample_one_crop_per_video and not self.test_mode and self.video_groups:
            crops = self.video_groups[index]
            raw_path, label = crops[np.random.randint(len(crops))]
        else:
            raw_path = self.raw_paths[index]
            label = self.labels[index]

        path = resolve_feature_path(raw_path, self.data_dir)

        if os.path.exists(path):
            clip_feature = np.load(path)
        else:
            clip_feature = np.zeros((self.clip_dim, 512), dtype=np.float32)

        if not self.test_mode:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
            return torch.tensor(clip_feature, dtype=torch.float32), label, clip_length
        else:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)
            return torch.tensor(clip_feature, dtype=torch.float32), label, clip_length, path, label


class XDDataset(data.Dataset):
    """XD-Violence Multimodal Dataset Loader.
    
    Loads paired (visual_feature, audio_feature, label) tensors.
    Supports pre-extracted CLIP visual features (512-D) and VGGish/CLAP/AST audio features.
    
    Fixes Applied:
    1. Synchronized Audio-Visual Sampling: audio features are temporally aligned to visual length
       BEFORE cropping/sampling/padding so valid frames [0 : seq_len] and zero-padding [seq_len : T_max]
       are 100% in lockstep!
    2. Balanced Normal/Abnormal Bag filtering (normal=True / normal=False).
    3. Crop-Grouped Epoch Sampling (sample_one_crop_per_video) to prevent 10-crop memorization.
    4. Startup audio-visual pairing audit log.
    """
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool, label_map: dict,
                 audio_file_path: str = None, audio_dir: str = None,
                 data_dir: str = None, no_audio: bool = False,
                 normal: bool = None, sample_one_crop_per_video: bool = False):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
        self.audio_dir = audio_dir
        self.data_dir = data_dir
        self.no_audio = no_audio
        self.normal = normal
        self.sample_one_crop_per_video = sample_one_crop_per_video

        # Filter normal / anomaly bags if requested
        if not test_mode and normal is not None:
            if normal:
                # Class 'A' is normal in XD-Violence
                self.df = self.df.loc[self.df['label'].astype(str).str.startswith('A')].reset_index(drop=True)
            else:
                self.df = self.df.loc[~self.df['label'].astype(str).str.startswith('A')].reset_index(drop=True)

        self.raw_paths = self.df['path'].tolist()
        self.labels = self.df['label'].tolist()

        # Group crops by base video ID for healthy epoch sampling without bag memorization
        self.video_groups = []
        if self.sample_one_crop_per_video and not self.test_mode:
            from collections import defaultdict
            grouped = defaultdict(list)
            for p, l in zip(self.raw_paths, self.labels):
                base_id = os.path.basename(p).split('__')[0]
                grouped[base_id].append((p, l))
            self.video_groups = list(grouped.values())

        # Audio mapping
        self.audio_dict = {}
        if audio_file_path and os.path.exists(audio_file_path):
            audio_df = pd.read_csv(audio_file_path)
            for _, row in audio_df.iterrows():
                key = os.path.basename(row['path']).split('__')[0]
                self.audio_dict[key] = row['audio_path']
                if '+' in key:
                    self.audio_dict[key.replace('+', '#')] = row['audio_path']
                if '#' in key:
                    self.audio_dict[key.replace('#', '+')] = row['audio_path']

        self._audio_path_cache = {}

        # Audio-Visual Synchronization Startup Audit Log
        if not test_mode and not no_audio and len(self.raw_paths) > 0:
            sample_n = min(len(self.raw_paths), 500)
            matched = sum(1 for p in self.raw_paths[:sample_n] if self.get_audio_path_quick(p) is not None)
            pct = (matched / max(1, sample_n)) * 100.0
            print(f"[XDDataset] Audio-Visual synchronization audit: sampled {matched}/{sample_n} ({pct:.1f}%) videos successfully paired with audio features.", flush=True)

    def get_audio_path_quick(self, visual_path: str):
        base_key = os.path.basename(visual_path).split('__')[0]
        if base_key in self._audio_path_cache:
            return self._audio_path_cache[base_key]
        audio_path = None
        if self.audio_dict:
            audio_path = self.audio_dict.get(base_key)
            if not audio_path:
                for k in get_path_variants(base_key):
                    if k in self.audio_dict:
                        audio_path = self.audio_dict[k]
                        break
        if not audio_path or not os.path.exists(audio_path):
            audio_path = infer_audio_path(visual_path, self.audio_dir)
        self._audio_path_cache[base_key] = audio_path
        return audio_path

    def __len__(self):
        if self.sample_one_crop_per_video and not self.test_mode and self.video_groups:
            return len(self.video_groups)
        return len(self.raw_paths)

    def __getitem__(self, index):
        if self.sample_one_crop_per_video and not self.test_mode and self.video_groups:
            crops = self.video_groups[index]
            raw_path, label = crops[np.random.randint(len(crops))]
        else:
            raw_path = self.raw_paths[index]
            label = self.labels[index]

        path = resolve_feature_path(raw_path, self.data_dir)

        if os.path.exists(path):
            v_raw = np.load(path)
        else:
            v_raw = np.zeros((self.clip_dim, 512), dtype=np.float32)

        v_len = v_raw.shape[0]

        # Resolve and load raw audio
        a_raw = None
        if not self.no_audio:
            audio_path = self.get_audio_path_quick(path)
            if audio_path and os.path.exists(audio_path):
                try:
                    a_raw = np.load(audio_path)
                    if a_raw.ndim == 1:
                        a_raw = a_raw.reshape(-1, 128)
                except Exception:
                    a_raw = None

        # Fix 2: Step A - Temporally interpolate audio to match raw visual length Tv BEFORE sampling/padding
        if a_raw is not None and a_raw.shape[0] != v_len:
            a_t = torch.from_numpy(a_raw).float().transpose(0, 1).unsqueeze(0)  # [1, D_a, T_a]
            a_interp = F.interpolate(a_t, size=v_len, mode='linear', align_corners=False).squeeze(0).transpose(0, 1).numpy()
            a_raw = a_interp

        # Fix 2: Step B - Apply the EXACT SAME sampling/cropping/padding to BOTH visual and audio
        if not self.test_mode:
            if v_len > self.clip_dim:
                # Uniform extraction with identical segment intervals
                r = np.linspace(0, v_len, self.clip_dim + 1, dtype=np.int32)
                v_feat = np.zeros((self.clip_dim, v_raw.shape[1]), dtype=np.float32)
                if a_raw is not None:
                    a_feat = np.zeros((self.clip_dim, a_raw.shape[1]), dtype=np.float32)
                else:
                    a_feat = None

                for i in range(self.clip_dim):
                    if r[i] != r[i + 1]:
                        v_feat[i, :] = np.mean(v_raw[r[i]:r[i + 1], :], axis=0)
                        if a_raw is not None:
                            a_feat[i, :] = np.mean(a_raw[r[i]:r[i + 1], :], axis=0)
                    else:
                        v_feat[i, :] = v_raw[r[i], :]
                        if a_raw is not None:
                            a_feat[i, :] = a_raw[r[i], :]
                clip_length = self.clip_dim
            else:
                v_feat = tools.pad(v_raw, self.clip_dim)
                if a_raw is not None:
                    a_feat = tools.pad(a_raw, self.clip_dim)
                else:
                    a_feat = None
                clip_length = v_len

            v_tensor = torch.tensor(v_feat, dtype=torch.float32)
            a_tensor = torch.tensor(a_feat, dtype=torch.float32) if a_feat is not None else None
            return v_tensor, label, clip_length, a_tensor
        else:
            v_feat, clip_length = tools.process_split(v_raw, self.clip_dim)
            if a_raw is not None:
                a_feat, _ = tools.process_split(a_raw, self.clip_dim)
                a_tensor = torch.tensor(a_feat, dtype=torch.float32)
            else:
                a_tensor = None
            v_tensor = torch.tensor(v_feat, dtype=torch.float32)
            return v_tensor, label, clip_length, path, label, a_tensor

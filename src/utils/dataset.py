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


def resolve_feature_path(path: str, data_dir: str = None) -> str:
    """Resolve file path locally, in data_dir, or in Kaggle directories with '# <-> +' support."""
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
    """
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool, label_map: dict,
                 normal: bool = False, data_dir: str = None):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
        self.normal = normal
        self.data_dir = data_dir

        if not test_mode:
            if normal:
                self.df = self.df.loc[self.df['label'] == 'Normal'].reset_index(drop=True)
            else:
                self.df = self.df.loc[self.df['label'] != 'Normal'].reset_index(drop=True)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        raw_path = self.df.loc[index]['path']
        path = resolve_feature_path(raw_path, self.data_dir)
        label = self.df.loc[index]['label']

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
    Automatically handles Kaggle '# <-> +' filename replacement.
    If audio features are not found or disabled, returns None for audio, triggering
    the Motion-Residual Mode.
    """
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool, label_map: dict,
                 audio_file_path: str = None, audio_dir: str = None,
                 data_dir: str = None, no_audio: bool = False):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
        self.audio_dir = audio_dir
        self.data_dir = data_dir
        self.no_audio = no_audio

        self.audio_dict = {}
        if audio_file_path and os.path.exists(audio_file_path):
            audio_df = pd.read_csv(audio_file_path)
            for _, row in audio_df.iterrows():
                key = os.path.basename(row['path']).split('__')[0]
                self.audio_dict[key] = row['audio_path']
                # Map both '+' and '#' keys
                if '+' in key:
                    self.audio_dict[key.replace('+', '#')] = row['audio_path']
                if '#' in key:
                    self.audio_dict[key.replace('#', '+')] = row['audio_path']

    def __len__(self):
        return self.df.shape[0]

    def load_audio(self, visual_path: str, target_length: int):
        """Loads and temporally aligns audio features."""
        if self.no_audio:
            return None

        # 1. Lookup in explicit audio dictionary
        base_key = os.path.basename(visual_path).split('__')[0]
        audio_path = self.audio_dict.get(base_key)
        if not audio_path:
            for k in get_path_variants(base_key):
                if k in self.audio_dict:
                    audio_path = self.audio_dict[k]
                    break

        # 2. If not found, infer path
        if not audio_path or not os.path.exists(audio_path):
            audio_path = infer_audio_path(visual_path, self.audio_dir)

        if not audio_path or not os.path.exists(audio_path):
            return None

        try:
            audio_feat = np.load(audio_path)
            if audio_feat.ndim == 1:
                audio_feat = audio_feat.reshape(-1, 128)
            if not self.test_mode:
                audio_feat, _ = tools.process_feat(audio_feat, target_length)
            else:
                audio_feat, _ = tools.process_split(audio_feat, target_length)
            return torch.tensor(audio_feat, dtype=torch.float32)
        except Exception:
            return None

    def __getitem__(self, index):
        raw_path = self.df.loc[index]['path']
        path = resolve_feature_path(raw_path, self.data_dir)
        label = self.df.loc[index]['label']

        if os.path.exists(path):
            clip_feature = np.load(path)
        else:
            clip_feature = np.zeros((self.clip_dim, 512), dtype=np.float32)

        if not self.test_mode:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
            audio_feature = self.load_audio(path, self.clip_dim)
            clip_tensor = torch.tensor(clip_feature, dtype=torch.float32)
            return clip_tensor, label, clip_length, audio_feature
        else:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)
            audio_feature = self.load_audio(path, self.clip_dim)
            clip_tensor = torch.tensor(clip_feature, dtype=torch.float32)
            return clip_tensor, label, clip_length, path, label, audio_feature

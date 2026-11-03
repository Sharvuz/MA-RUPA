import os
from collections import defaultdict
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
        for dirpath, dirs, filenames in os.walk(root_dir):
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.ipynb_checkpoints', '.cache'}]
            dir_lower = dirpath.lower()
            is_audio_dir = any(kw in dir_lower for kw in ['audio', 'vgg', 'clap', 'ast', 'sound', 'wav'])
            for fname in filenames:
                if fname.endswith('.npy'):
                    full_p = os.path.join(dirpath, fname)
                    _KAGGLE_FILE_MAP[fname] = full_p
                    is_audio = is_audio_dir or fname.endswith('__vggish.npy') or fname.endswith('_vggish.npy')
                    if is_audio:
                        _KAGGLE_AUDIO_MAP[fname] = full_p

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
                        if is_audio and v not in _KAGGLE_AUDIO_MAP:
                            _KAGGLE_AUDIO_MAP[v] = full_p

                    clean_stem = fname.replace('__vggish.npy', '.npy').replace('_vggish.npy', '.npy')
                    if clean_stem != fname:
                        extra_variants = [clean_stem, clean_stem.replace('+', '#'), clean_stem.replace('#', '+')]
                        for ev in extra_variants:
                            if ev not in _KAGGLE_FILE_MAP:
                                _KAGGLE_FILE_MAP[ev] = full_p
                            if is_audio and ev not in _KAGGLE_AUDIO_MAP:
                                _KAGGLE_AUDIO_MAP[ev] = full_p

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
_MULTICROP_PATH_CACHE = {}
_TEST_FEATURE_RAM_CACHE = {}
_TEST_FEATURE_RAM_BYTES = 0
_MAX_TEST_RAM_CACHE_BYTES = 2200 * 1024 * 1024  # 2.2 GB safe ceiling on 30 GB Kaggle RAM


def resolve_feature_path(path: str, data_dir: str = None) -> str:
    """Resolve file path locally, in data_dir, or in Kaggle directories with '# <-> +' support."""
    cache_key = (path, data_dir)
    if cache_key in _RESOLVE_CACHE:
        return _RESOLVE_CACHE[cache_key]

    res = _resolve_feature_path_uncached(path, data_dir)
    _RESOLVE_CACHE[cache_key] = res
    return res


def _resolve_feature_path_uncached(path: str, data_dir: str = None) -> str:
    for v in get_path_variants(path):
        if os.path.exists(v):
            return v

    base_name = os.path.basename(path)
    base_variants = get_path_variants(base_name)

    if data_dir and os.path.exists(data_dir):
        for v in base_variants:
            cand = os.path.join(data_dir, v)
            if os.path.exists(cand):
                return cand
        rel = path.lstrip('/\\')
        for v in get_path_variants(rel):
            cand = os.path.join(data_dir, v)
            if os.path.exists(cand):
                return cand

    k_map = get_kaggle_file_map()
    for v in base_variants:
        if v in k_map and os.path.exists(k_map[v]):
            return k_map[v]

    return path


def strip_crop_suffix(path_or_name: str) -> str:
    """Strips __[0-9].npy crop suffix from visual path, preserving segment timestamps."""
    name = os.path.basename(path_or_name)
    for crop_idx in range(10):
        if f"__{crop_idx}.npy" in name:
            return name.replace(f"__{crop_idx}.npy", ".npy")
    return name


def infer_audio_path(visual_path: str, audio_dir: str = None) -> str:
    """Infer the audio feature file path corresponding to a visual snippet feature."""
    cache_key = (visual_path, audio_dir)
    if cache_key in _INFER_AUDIO_CACHE:
        return _INFER_AUDIO_CACHE[cache_key]

    res = _infer_audio_path_uncached(visual_path, audio_dir)
    _INFER_AUDIO_CACHE[cache_key] = res
    return res


def _infer_audio_path_uncached(visual_path: str, audio_dir: str = None) -> str:
    vis_abs = os.path.abspath(visual_path) if os.path.exists(visual_path) else None
    base_name = os.path.basename(visual_path)
    base_no_crop = strip_crop_suffix(base_name)

    # Prioritize explicit __vggish / _vggish and crop-stripped names first so visual crops never self-match
    raw_names = get_path_variants(base_no_crop) + get_path_variants(base_name)
    explicit_vgg_names = []
    plain_names = []
    for n in raw_names:
        if n.endswith('.npy'):
            explicit_vgg_names.append(n.replace('.npy', '__vggish.npy'))
            explicit_vgg_names.append(n.replace('.npy', '_vggish.npy'))
        plain_names.append(n)
    target_names = list(dict.fromkeys(explicit_vgg_names + plain_names))

    def _is_valid_audio_cand(cand_p: str) -> bool:
        if not cand_p or not os.path.exists(cand_p):
            return False
        if vis_abs is not None and os.path.abspath(cand_p) == vis_abs:
            return False
        return True

    if audio_dir and os.path.exists(audio_dir):
        for name in target_names:
            candidate = os.path.join(audio_dir, name)
            if _is_valid_audio_cand(candidate):
                return candidate
            for subdir in ['train', 'test', 'XDTrainAudioFeatures', 'XDTestAudioFeatures']:
                sub_candidate = os.path.join(audio_dir, subdir, name)
                if _is_valid_audio_cand(sub_candidate):
                    return sub_candidate

    audio_map = get_kaggle_audio_map()
    for name in target_names:
        if name in audio_map and _is_valid_audio_cand(audio_map[name]):
            return audio_map[name]

    substitutions = [
        ('ClipFeatures', 'AudioFeatures'),
        ('clip_features', 'audio_features'),
        ('rgb', 'audio'),
        ('RGB', 'Audio'),
        ('XDTrainClipFeatures', 'XDTrainAudioFeatures'),
        ('XDTestClipFeatures', 'XDTestAudioFeatures'),
        ('XDTrainClipFeatures', 'XDTrainVGGishFeatures'),
        ('XDTestClipFeatures', 'XDTestVGGishFeatures'),
        ('XDTrainClipFeatures', 'train'),
        ('XDTestClipFeatures', 'test'),
        ('clip', 'audio'),
        ('CLIP', 'AUDIO'),
    ]
    for old_sub, new_sub in substitutions:
        if old_sub in visual_path:
            cand_base = visual_path.replace(old_sub, new_sub)
            cand_no_crop = strip_crop_suffix(cand_base)
            cand_dir = os.path.dirname(cand_base)

            candidates_to_check = [os.path.join(cand_dir, name) for name in target_names]
            for v in get_path_variants(cand_base) + get_path_variants(cand_no_crop):
                candidates_to_check.append(v)
                if v.endswith('.npy'):
                    candidates_to_check.append(v.replace('.npy', '__vggish.npy'))
                    candidates_to_check.append(v.replace('.npy', '_vggish.npy'))

            for cand_p in dict.fromkeys(candidates_to_check):
                if os.path.exists(cand_p):
                    return cand_p

    k_map = get_kaggle_file_map()
    for name in target_names:
        if name in k_map and os.path.exists(k_map[name]):
            cand = k_map[name]
            if any(kw in cand.lower() for kw in ['audio', 'vgg', 'clap', 'ast', 'sound', 'wav']):
                return cand

    return None


def _load_npy_cached_fp32(path: str):
    """Loads a test .npy file with bounded float16 RAM caching for zero-I/O repeated evaluations."""
    global _TEST_FEATURE_RAM_BYTES
    if path in _TEST_FEATURE_RAM_CACHE:
        return _TEST_FEATURE_RAM_CACHE[path].astype(np.float32)
    if not path or not os.path.exists(path):
        return None
    arr = np.load(path)
    arr_fp16 = np.asarray(arr, dtype=np.float16)
    if _TEST_FEATURE_RAM_BYTES + arr_fp16.nbytes <= _MAX_TEST_RAM_CACHE_BYTES:
        _TEST_FEATURE_RAM_CACHE[path] = arr_fp16
        _TEST_FEATURE_RAM_BYTES += arr_fp16.nbytes
    return np.asarray(arr, dtype=np.float32)


def resolve_multicrop_paths(primary_path: str, data_dir: str = None, max_crops: int = 10) -> list[str]:
    """Discovers all spatial crop .npy files (__0.npy .. __{max_crops-1}.npy) for a test video."""
    cache_key = (primary_path, data_dir, max_crops)
    if cache_key in _MULTICROP_PATH_CACHE:
        return _MULTICROP_PATH_CACHE[cache_key]

    resolved_primary = resolve_feature_path(primary_path, data_dir)
    if max_crops <= 1:
        res = [resolved_primary]
        _MULTICROP_PATH_CACHE[cache_key] = res
        return res

    base_name = os.path.basename(resolved_primary)
    primary_dir = os.path.dirname(resolved_primary)

    stem_prefix = None
    for k in range(10):
        suffix = f"__{k}.npy"
        if base_name.endswith(suffix):
            stem_prefix = base_name[:-len(suffix)]
            break

    if stem_prefix is None:
        res = [resolved_primary]
        _MULTICROP_PATH_CACHE[cache_key] = res
        return res

    found_crops = []
    seen = set()
    for c_idx in range(max_crops):
        cand_name = f"{stem_prefix}__{c_idx}.npy"
        cand_full = None
        if primary_dir:
            for v_name in get_path_variants(cand_name):
                direct_p = os.path.join(primary_dir, v_name)
                if os.path.exists(direct_p):
                    cand_full = direct_p
                    break
        if cand_full is None:
            resolved_c = resolve_feature_path(cand_name, data_dir)
            if os.path.exists(resolved_c):
                cand_full = resolved_c
        if cand_full is not None and cand_full not in seen:
            seen.add(cand_full)
            found_crops.append(cand_full)

    if not found_crops:
        found_crops = [resolved_primary]

    _MULTICROP_PATH_CACHE[cache_key] = found_crops
    return found_crops


def load_multicrop_test_video(raw_path: str, maxlen: int, data_dir: str = None,
                              audio_dir: str = None, no_audio: bool = True,
                              max_crops: int = 10, fallback_visual: torch.Tensor = None,
                              fallback_length: int = None, fallback_audio: torch.Tensor = None):
    """Loads and batches all available spatial crops of a single test video for fast parallel GPU eval.

    Returns:
        visual_crops: Tensor [num_crops, num_splits, maxlen, D]
        audio_crops: Tensor [num_crops, num_splits, maxlen, D_a] or None
        clip_length: int (exact snippet length of primary crop)
        num_crops: int
    """
    crop_paths = resolve_multicrop_paths(raw_path, data_dir=data_dir, max_crops=max_crops)
    valid_feats = []
    for cp in crop_paths:
        arr = _load_npy_cached_fp32(cp)
        if arr is not None and arr.ndim == 2 and arr.shape[0] > 0:
            valid_feats.append(arr)

    if not valid_feats:
        if fallback_visual is not None:
            v = fallback_visual.float()
            if v.ndim == 2:
                v = v.unsqueeze(0)
            a = None
            if fallback_audio is not None and not no_audio:
                a = fallback_audio.float()
                if a.ndim == 2:
                    a = a.unsqueeze(0)
                a = a.unsqueeze(0)
            return v.unsqueeze(0), a, int(fallback_length), 1
        dummy = torch.zeros((1, 1, maxlen, 512), dtype=torch.float32)
        return dummy, None, maxlen, 1

    primary_len = int(fallback_length) if fallback_length is not None else int(valid_feats[0].shape[0])
    split_crops = []
    for arr in valid_feats:
        if arr.shape[0] != primary_len:
            t_arr = torch.from_numpy(arr).float().transpose(0, 1).unsqueeze(0)
            arr = F.interpolate(t_arr, size=primary_len, mode='linear', align_corners=False).squeeze(0).transpose(0, 1).numpy()
        split_v, _ = tools.process_split(arr, maxlen)
        if split_v.ndim == 2:
            split_v = split_v[np.newaxis, :, :]
        split_crops.append(torch.from_numpy(split_v).float())

    visual_crops = torch.stack(split_crops, dim=0)  # [C, num_splits, maxlen, 512]
    num_crops = visual_crops.shape[0]

    audio_crops = None
    if not no_audio:
        if fallback_audio is not None:
            a = fallback_audio.float()
            if a.ndim == 2:
                a = a.unsqueeze(0)
            if a.shape[0] != visual_crops.shape[1]:
                a_flat = a.reshape(-1, a.shape[-1]).transpose(0, 1).unsqueeze(0)
                a_interp = F.interpolate(
                    a_flat, size=visual_crops.shape[1] * maxlen, mode='linear', align_corners=False
                ).squeeze(0).transpose(0, 1)
                a = a_interp.view(visual_crops.shape[1], maxlen, -1)
            audio_crops = a.unsqueeze(0).expand(num_crops, -1, -1, -1).contiguous()
        else:
            a_path = infer_audio_path(crop_paths[0], audio_dir)
            a_raw = _load_npy_cached_fp32(a_path) if a_path else None
            if a_raw is not None:
                if a_raw.ndim == 1:
                    a_raw = a_raw.reshape(-1, 128)
                if a_raw.shape[0] != primary_len:
                    a_t = torch.from_numpy(a_raw).float().transpose(0, 1).unsqueeze(0)
                    a_raw = F.interpolate(a_t, size=primary_len, mode='linear', align_corners=False).squeeze(0).transpose(0, 1).numpy()
                split_a, _ = tools.process_split(a_raw, maxlen)
                if split_a.ndim == 2:
                    split_a = split_a[np.newaxis, :, :]
                a_tensor = torch.from_numpy(split_a).float()
                audio_crops = a_tensor.unsqueeze(0).expand(num_crops, -1, -1, -1).contiguous()

    return visual_crops, audio_crops, primary_len, num_crops


class UCFDataset(data.Dataset):
    """UCF-Crime Dataset Loader (512-D CLIP ViT-B/16 snippet features)."""

    def __init__(self, clip_dim: int, file_path: str, test_mode: bool, label_map: dict,
                 normal: bool = False, data_dir: str = None, sample_one_crop_per_video: bool = False,
                 crops_per_video: int = 1):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
        self.normal = normal
        self.data_dir = data_dir
        self.sample_one_crop_per_video = sample_one_crop_per_video
        self.crops_per_video = max(1, crops_per_video)

        if not test_mode:
            if normal:
                self.df = self.df.loc[self.df['label'] == 'Normal'].reset_index(drop=True)
            else:
                self.df = self.df.loc[self.df['label'] != 'Normal'].reset_index(drop=True)

        self.raw_paths = self.df['path'].tolist()
        self.labels = self.df['label'].tolist()

        self.video_groups = []
        if self.sample_one_crop_per_video and not self.test_mode:
            grouped = defaultdict(list)
            for p, l in zip(self.raw_paths, self.labels):
                base_id = os.path.basename(p).split('__')[0]
                grouped[base_id].append((p, l))
            self.video_groups = list(grouped.values())

    def __len__(self):
        if self.sample_one_crop_per_video and not self.test_mode and self.video_groups:
            return len(self.video_groups) * self.crops_per_video
        return len(self.raw_paths)

    def __getitem__(self, index):
        if self.sample_one_crop_per_video and not self.test_mode and self.video_groups:
            vid_idx = index % len(self.video_groups)
            crops = self.video_groups[vid_idx]
            raw_path, label = crops[np.random.randint(len(crops))]
        else:
            raw_path = self.raw_paths[index]
            label = self.labels[index]

        path = resolve_feature_path(raw_path, self.data_dir)
        if os.path.exists(path):
            clip_feature = _load_npy_cached_fp32(path) if self.test_mode else np.load(path)
        else:
            clip_feature = np.zeros((self.clip_dim, 512), dtype=np.float32)

        if not self.test_mode:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
            return torch.tensor(clip_feature, dtype=torch.float32), label, clip_length
        else:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)
            return torch.tensor(clip_feature, dtype=torch.float32), label, clip_length, path, label


class XDDataset(data.Dataset):
    """XD-Violence Multimodal Dataset Loader (512-D CLIP Visual + Synchronized 128-D VGGish Audio)."""

    def __init__(self, clip_dim: int, file_path: str, test_mode: bool, label_map: dict,
                 audio_file_path: str = None, audio_dir: str = None,
                 data_dir: str = None, no_audio: bool = False,
                 normal: bool = None, sample_one_crop_per_video: bool = False,
                 crops_per_video: int = 1):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
        self.audio_dir = audio_dir
        self.data_dir = data_dir
        self.no_audio = no_audio
        self.normal = normal
        self.sample_one_crop_per_video = sample_one_crop_per_video
        self.crops_per_video = max(1, crops_per_video)

        if not test_mode and normal is not None:
            if normal:
                self.df = self.df.loc[self.df['label'].astype(str).str.startswith('A')].reset_index(drop=True)
            else:
                self.df = self.df.loc[~self.df['label'].astype(str).str.startswith('A')].reset_index(drop=True)

        self.raw_paths = self.df['path'].tolist()
        self.labels = self.df['label'].tolist()

        self.video_groups = []
        if self.sample_one_crop_per_video and not self.test_mode:
            grouped = defaultdict(list)
            for p, l in zip(self.raw_paths, self.labels):
                base_id = strip_crop_suffix(p)
                grouped[base_id].append((p, l))
            self.video_groups = list(grouped.values())

        self.audio_dict = {}
        if audio_file_path and os.path.exists(audio_file_path):
            audio_df = pd.read_csv(audio_file_path)
            for _, row in audio_df.iterrows():
                key = strip_crop_suffix(row['path'])
                self.audio_dict[key] = row['audio_path']
                for v in get_path_variants(key):
                    self.audio_dict[v] = row['audio_path']
                    if v.endswith('.npy'):
                        self.audio_dict[v.replace('.npy', '__vggish.npy')] = row['audio_path']
                        self.audio_dict[v.replace('.npy', '_vggish.npy')] = row['audio_path']

        self._audio_path_cache = {}

        if not test_mode and not no_audio and len(self.raw_paths) > 0:
            sample_n = min(len(self.raw_paths), 500)
            matched = sum(1 for p in self.raw_paths[:sample_n] if self.get_audio_path_quick(p) is not None)
            pct = (matched / max(1, sample_n)) * 100.0
            print(
                f"[XDDataset] Audio-Visual synchronization audit: sampled {matched}/{sample_n} "
                f"({pct:.1f}%) videos successfully paired with audio features.",
                flush=True,
            )

    def get_audio_path_quick(self, visual_path: str):
        base_key = strip_crop_suffix(visual_path)
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
            return len(self.video_groups) * self.crops_per_video
        return len(self.raw_paths)

    def __getitem__(self, index):
        if self.sample_one_crop_per_video and not self.test_mode and self.video_groups:
            vid_idx = index % len(self.video_groups)
            crops = self.video_groups[vid_idx]
            raw_path, label = crops[np.random.randint(len(crops))]
        else:
            raw_path = self.raw_paths[index]
            label = self.labels[index]

        path = resolve_feature_path(raw_path, self.data_dir)
        if os.path.exists(path):
            v_raw = _load_npy_cached_fp32(path) if self.test_mode else np.load(path)
        else:
            v_raw = np.zeros((self.clip_dim, 512), dtype=np.float32)

        v_len = v_raw.shape[0]

        a_raw = None
        if not self.no_audio:
            audio_path = self.get_audio_path_quick(path)
            if audio_path and os.path.exists(audio_path):
                try:
                    a_raw = _load_npy_cached_fp32(audio_path) if self.test_mode else np.load(audio_path)
                    if a_raw is not None and a_raw.ndim == 1:
                        a_raw = a_raw.reshape(-1, 128)
                except Exception:
                    a_raw = None

        if a_raw is not None and a_raw.shape[0] != v_len:
            a_t = torch.from_numpy(a_raw).float().transpose(0, 1).unsqueeze(0)
            a_raw = F.interpolate(a_t, size=v_len, mode='linear', align_corners=False).squeeze(0).transpose(0, 1).numpy()

        if not self.test_mode:
            if v_len > self.clip_dim:
                r = np.linspace(0, v_len, self.clip_dim + 1, dtype=np.int32)
                v_feat = np.zeros((self.clip_dim, v_raw.shape[1]), dtype=np.float32)
                a_feat = np.zeros((self.clip_dim, a_raw.shape[1]), dtype=np.float32) if a_raw is not None else None

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
                a_feat = tools.pad(a_raw, self.clip_dim) if a_raw is not None else None
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

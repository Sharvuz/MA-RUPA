import os
import sys
import tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils.dataset import resolve_feature_path, get_path_variants, infer_audio_path


def test_hash_plus_resolution():
    print("Testing Kaggle '#' -> '+' filename resolution...")
    with tempfile.TemporaryDirectory() as temp_dir:
        # File on disk has '+' because Kaggle renamed it
        plus_name = "Bad.Boys.1995__+00-26-51_00-27-53_label_B2-0-0__0.npy"
        disk_path = os.path.join(temp_dir, plus_name)
        np.save(disk_path, np.zeros((32, 512), dtype=np.float32))

        # 1. Query path contains '#' (from CSV original list)
        csv_path = f"/data/VAD/XD/XDTrainClipFeatures/Bad.Boys.1995__#00-26-51_00-27-53_label_B2-0-0__0.npy"
        resolved = resolve_feature_path(csv_path, data_dir=temp_dir)
        assert os.path.exists(resolved), f"Failed to resolve {csv_path} to {disk_path}, got {resolved}"
        assert resolved == disk_path, f"Expected {disk_path}, got {resolved}"
        print("  Resolved '#' CSV path to '+' disk file successfully!")

        # 2. Reverse: CSV path has '+', file on disk has '#'
        hash_name = "Abuse001_x264__#test.npy"
        disk_hash = os.path.join(temp_dir, hash_name)
        np.save(disk_hash, np.zeros((32, 512), dtype=np.float32))

        csv_plus = f"/data/VAD/UCF/Abuse001_x264__+test.npy"
        resolved_reverse = resolve_feature_path(csv_plus, data_dir=temp_dir)
        assert os.path.exists(resolved_reverse)
        assert resolved_reverse == disk_hash
        print("  Resolved '+' CSV path to '#' disk file successfully!")


if __name__ == '__main__':
    test_hash_plus_resolution()
    print("\n[SUCCESS] Kaggle Hash-to-Plus resolution test passed!")

import os
import sys
import tempfile
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import ucf_train
import xd_train
import ucf_option
import xd_option


def create_dummy_data(temp_dir):
    features_dir = os.path.join(temp_dir, 'features')
    audio_dir = os.path.join(temp_dir, 'audio')
    lists_dir = os.path.join(temp_dir, 'lists')
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(lists_dir, exist_ok=True)

    # UCF dummy files
    ucf_records = []
    for i in range(4):
        lbl = 'Normal' if i % 2 == 0 else 'Fighting'
        p = os.path.join(features_dir, f'ucf_{i}.npy')
        np.save(p, np.random.randn(64, 512).astype(np.float32))
        ucf_records.append({'path': p, 'label': lbl})

    ucf_train_csv = os.path.join(lists_dir, 'ucf_train.csv')
    ucf_test_csv = os.path.join(lists_dir, 'ucf_test.csv')
    pd.DataFrame(ucf_records).to_csv(ucf_train_csv, index=False)
    pd.DataFrame(ucf_records).to_csv(ucf_test_csv, index=False)

    # XD dummy files
    xd_records = []
    for i in range(4):
        lbl = 'A' if i % 2 == 0 else 'B1'
        p = os.path.join(features_dir, f'xd_{i}.npy')
        p_aud = os.path.join(audio_dir, f'xd_{i}.npy')
        np.save(p, np.random.randn(64, 512).astype(np.float32))
        np.save(p_aud, np.random.randn(64, 128).astype(np.float32))
        xd_records.append({'path': p, 'label': lbl})

    xd_train_csv = os.path.join(lists_dir, 'xd_train.csv')
    xd_test_csv = os.path.join(lists_dir, 'xd_test.csv')
    pd.DataFrame(xd_records).to_csv(xd_train_csv, index=False)
    pd.DataFrame(xd_records).to_csv(xd_test_csv, index=False)

    # Ground truth
    total_frames = 4 * 64 * 16
    gt = np.zeros(total_frames, dtype=np.float32)
    gt[100:300] = 1.0
    gt_path = os.path.join(lists_dir, 'gt.npy')
    np.save(gt_path, gt)

    gt_segments = np.array([[100, 300]], dtype=object)
    gt_segment_path = os.path.join(lists_dir, 'gt_segment.npy')
    np.save(gt_segment_path, gt_segments)

    gt_labels = np.array([['Fighting']], dtype=object)
    gt_label_path = os.path.join(lists_dir, 'gt_label.npy')
    np.save(gt_label_path, gt_labels)

    return {
        'temp_dir': temp_dir,
        'features_dir': features_dir,
        'audio_dir': audio_dir,
        'ucf_train_csv': ucf_train_csv,
        'ucf_test_csv': ucf_test_csv,
        'xd_train_csv': xd_train_csv,
        'xd_test_csv': xd_test_csv,
        'gt_path': gt_path,
        'gt_segment_path': gt_segment_path,
        'gt_label_path': gt_label_path
    }


def test_ucf_train_step():
    print("Testing UCF-Crime Training Step (End-to-End)...")
    with tempfile.TemporaryDirectory() as temp_dir:
        paths = create_dummy_data(temp_dir)
        args = ucf_option.parser.parse_args([])
        args.visual_length = 64
        args.batch_size = 2
        args.max_epoch = 1
        args.num_workers = 0
        args.train_list = paths['ucf_train_csv']
        args.test_list = paths['ucf_test_csv']
        args.gt_path = paths['gt_path']
        args.gt_segment_path = paths['gt_segment_path']
        args.gt_label_path = paths['gt_label_path']
        args.model_path = os.path.join(temp_dir, 'model_ucf.pth')
        args.checkpoint_path = os.path.join(temp_dir, 'checkpoint_ucf.pth')
        args.eval_interval = 1000

        ucf_train.train(args)
        assert os.path.exists(args.checkpoint_path), "Checkpoint was not saved!"
        print("  UCF-Crime Training Step completed successfully!")


def test_xd_train_step():
    print("Testing XD-Violence Multimodal Training Step (End-to-End)...")
    with tempfile.TemporaryDirectory() as temp_dir:
        paths = create_dummy_data(temp_dir)
        args = xd_option.parser.parse_args([])
        args.visual_length = 64
        args.batch_size = 2
        args.max_epoch = 1
        args.num_workers = 0
        args.audio_dir = paths['audio_dir']
        args.train_list = paths['xd_train_csv']
        args.test_list = paths['xd_test_csv']
        args.gt_path = paths['gt_path']
        args.gt_segment_path = paths['gt_segment_path']
        args.gt_label_path = paths['gt_label_path']
        args.model_path = os.path.join(temp_dir, 'model_xd.pth')
        args.checkpoint_path = os.path.join(temp_dir, 'checkpoint_xd.pth')
        args.eval_interval = 1000

        xd_train.train(args)
        assert os.path.exists(args.checkpoint_path), "Checkpoint was not saved!"
        print("  XD-Violence Training Step completed successfully!")


if __name__ == '__main__':
    test_ucf_train_step()
    test_xd_train_step()
    print("\n[SUCCESS] ALL TRAINING INTEGRATION TESTS PASSED!")

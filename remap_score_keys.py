"""
Remap score pkl keys from feeder-style ('test_0', 'test_1', ...) to
frame_dir-style ('game_start_end_tracklet') expected by sequence_modelling.py.

Replays feeder_hockey filtering logic to build the index->frame_dir mapping.
Usage: python remap_score_keys.py <score_pkl> <annotation_pkl> <output_pkl>
"""
import argparse
import pickle
import numpy as np


def build_frame_dir_mapping(annotation_pkl_path, split='test'):
    with open(annotation_pkl_path, 'rb') as f:
        content = pickle.load(f)
    annotations = content['annotations'] if isinstance(content, dict) and 'annotations' in content else content

    frame_dirs = []
    for i, sample in enumerate(annotations):
        lbl = sample.get('label', -1)
        if lbl > 10 or lbl < 0:
            continue
        if 'keypoint' not in sample or sample['keypoint'] is None:
            continue
        frame_dirs.append(sample.get('frame_dir', f'{split}_{i}'))
    return frame_dirs


def remap(score_pkl_path, annotation_pkl_path, output_pkl_path, split='test'):
    with open(score_pkl_path, 'rb') as f:
        scores = pickle.load(f)

    frame_dirs = build_frame_dir_mapping(annotation_pkl_path, split)

    assert len(frame_dirs) == len(scores), (
        f"Mismatch: {len(frame_dirs)} frame_dirs vs {len(scores)} score entries"
    )

    remapped = {}
    for i, frame_dir in enumerate(frame_dirs):
        old_key = f'{split}_{i}'
        remapped[frame_dir] = scores[old_key]

    with open(output_pkl_path, 'wb') as f:
        pickle.dump(remapped, f)
    print(f"Saved {len(remapped)} remapped scores to {output_pkl_path}")
    print(f"Sample keys: {list(remapped.keys())[:3]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('score_pkl')
    parser.add_argument('annotation_pkl')
    parser.add_argument('output_pkl')
    parser.add_argument('--split', default='test')
    args = parser.parse_args()
    remap(args.score_pkl, args.annotation_pkl, args.output_pkl, args.split)

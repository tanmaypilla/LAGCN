import numpy as np
import pickle
from torch.utils.data import Dataset
from feeders import tools


class Feeder(Dataset):
    def __init__(self, data_path, split='train', p_interval=1, window_size=64,
                 random_rot=False, bone=False, vel=False, debug=False):
        """
        Hockey Skating Actions dataset feeder for LAGCN.

        Data format in pkl:
          sample['keypoint'] -> (1, T, 20, 2)  i.e. (M, T, V, C) in pyskl format
          sample['label']    -> int in [0, 10]

        Output per __getitem__:
          data  -> (C, window_size, V, M) = (2, 64, 20, 1)
          label -> int
          index -> int (dataset index)
        """
        self.data_path = data_path
        self.split = split
        self.p_interval = p_interval
        self.window_size = window_size
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.debug = debug
        self.load_data()

    def load_data(self):
        with open(self.data_path, 'rb') as f:
            content = pickle.load(f)

        # Support both raw list and pyskl dict with 'annotations' key
        if isinstance(content, dict) and 'annotations' in content:
            annotations = content['annotations']
        else:
            annotations = content

        self.data = []
        self.label = []
        self.sample_name = []

        for i, sample in enumerate(annotations):
            lbl = sample.get('label', -1)
            # Filter out RAPID_DECELERATION (label=11) from old 12-class pkl files
            if lbl > 10 or lbl < 0:
                continue
            if 'keypoint' not in sample or sample['keypoint'] is None:
                continue
            kp = np.array(sample['keypoint'], dtype=np.float32)  # (1, T, 20, 2)
            # Transpose from (M, T, V, C) -> (C, T, V, M)
            kp = kp.transpose(3, 1, 2, 0)  # (2, T, 20, 1)
            self.data.append(kp)
            self.label.append(int(lbl))
            self.sample_name.append(f'{self.split}_{i}')

        if self.debug:
            self.data = self.data[:100]
            self.label = self.label[:100]
            self.sample_name = self.sample_name[:100]

    def _normalize(self, data):
        # data: (C, T, V, M)
        # Step 1: subtract mean hip position (joints 5 & 6) across all frames
        hip = data[:, :, [5, 6], :].mean(axis=(1, 2, 3), keepdims=False).reshape(-1, 1, 1, 1)
        data = data - hip
        # Step 2: scale to [-1, 1] by dividing by max absolute value
        scale = np.abs(data).max()
        if scale > 1e-6:
            data = data / scale
        return data

    def _temporal_crop_resize(self, data, valid_frame_num, p_interval, window_size):
        import torch
        import torch.nn.functional as F
        C, T, V, M = data.shape

        if len(p_interval) == 1:
            # Test: centre crop of p * valid_frame_num frames
            p = p_interval[0]
            cropped_length = max(1, int(p * valid_frame_num))
            bias = (valid_frame_num - cropped_length) // 2
        else:
            # Train: random crop of [p_min, p_max] * valid_frame_num frames
            # Fix Issue 1: no 64-frame lower bound — use valid_frame_num directly
            p = np.random.uniform(p_interval[0], p_interval[1])
            cropped_length = max(1, int(p * valid_frame_num))
            bias = np.random.randint(0, max(1, valid_frame_num - cropped_length + 1))

        data = data[:, bias:bias + cropped_length, :, :]  # (C, cropped_length, V, M)

        # Resize to window_size using bilinear interpolation
        data_t = torch.tensor(data, dtype=torch.float)
        data_t = data_t.permute(0, 2, 3, 1).contiguous().view(C * V * M, cropped_length)
        data_t = data_t[None, None, :, :]
        data_t = F.interpolate(data_t, size=(C * V * M, window_size),
                               mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        data = data_t.contiguous().view(C, V, M, window_size).permute(0, 3, 1, 2).contiguous().numpy()
        return data

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = np.array(self.data[index])  # (2, T, 20, 1)
        label = self.label[index]

        # Count non-zero frames (zero frames are padding)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        if valid_frame_num == 0:
            valid_frame_num = data_numpy.shape[1]

        # Fix 2: normalize before any temporal processing
        data_numpy = self._normalize(data_numpy)

        # Fix 1 & 3: temporal crop + resize without 64-frame lower bound
        data_numpy = self._temporal_crop_resize(
            data_numpy, valid_frame_num, self.p_interval, self.window_size)

        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)

        if self.bone:
            from feeders import bone_pairs
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in bone_pairs.hockey_pairs:
                bone_data_numpy[:, :, v1] = data_numpy[:, :, v1] - data_numpy[:, :, v2]
            data_numpy = bone_data_numpy

        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

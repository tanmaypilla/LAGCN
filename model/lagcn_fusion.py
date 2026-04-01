"""
Joint + Motion dual-stream fusion model for LAGCN hockey action recognition.

Two independent towers (joint stream and motion stream) share the same
TCN-GCN architecture as model.lagcn.Model.  A lightweight StreamFusion
module combines the two tower outputs before the shared classification head.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from model.lagcn import import_class, TCN_GCN_unit, bn_init, conv_branch_init


# ---------------------------------------------------------------------------
# Fusion module
# ---------------------------------------------------------------------------

class StreamFusion(nn.Module):
    """
    Fuse joint and motion stream feature maps.

    Args:
        channels (int): channel dimension of each stream (both must match).
        mode (str): one of 'gate', 'concat', 'add'.
    """

    def __init__(self, channels: int, mode: str = 'gate'):
        super().__init__()
        self.mode = mode

        if mode == 'gate':
            self.bn = nn.BatchNorm2d(channels * 2)
            self.relu = nn.ReLU(inplace=True)
            self.gate_conv = nn.Conv2d(channels * 2, channels, kernel_size=1)
            self.sigmoid = nn.Sigmoid()
        elif mode == 'concat':
            self.conv = nn.Conv2d(channels * 2, channels, kernel_size=1)
            self.bn = nn.BatchNorm2d(channels)
            self.relu = nn.ReLU(inplace=True)
        elif mode == 'add':
            pass  # no learnable parameters needed
        else:
            raise ValueError(f'Unknown StreamFusion mode: {mode}')

    def forward(self, x_joint: torch.Tensor, x_motion: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_joint:  (N*M, C, T', V)
            x_motion: (N*M, C, T', V)
        Returns:
            fused:    (N*M, C, T', V)
        """
        if self.mode == 'gate':
            cat = torch.cat([x_joint, x_motion], dim=1)   # (N*M, 2C, T', V)
            cat = self.relu(self.bn(cat))
            gate = self.sigmoid(self.gate_conv(cat))       # (N*M, C, T', V)
            return gate * x_joint + (1 - gate) * x_motion
        elif self.mode == 'concat':
            cat = torch.cat([x_joint, x_motion], dim=1)   # (N*M, 2C, T', V)
            return self.relu(self.bn(self.conv(cat)))      # (N*M, C, T', V)
        else:  # 'add'
            return x_joint + x_motion


# ---------------------------------------------------------------------------
# Helper: build one stream tower as a nn.ModuleDict
# ---------------------------------------------------------------------------

def _build_tower(in_channels: int, num_point: int, num_person: int, A, adaptive: bool):
    """Return a dict of modules that form one stream tower."""
    base_channel = 64
    tower = nn.ModuleDict({
        'to_joint_embedding': nn.Linear(in_channels, base_channel),
        'data_bn': nn.BatchNorm1d(num_person * base_channel * num_point),
        'l1': TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive),
        'l2': TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive, loop_times=0),
        'l3': TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive, loop_times=0),
        'l4': TCN_GCN_unit(base_channel, base_channel * 2, A, stride=2, adaptive=adaptive, loop_times=0),
        'l5': TCN_GCN_unit(base_channel * 2, base_channel * 2, A, adaptive=adaptive, loop_times=0),
        'l6': TCN_GCN_unit(base_channel * 2, base_channel * 2, A, adaptive=adaptive, loop_times=0),
        'l7': TCN_GCN_unit(base_channel * 2, base_channel * 4, A, stride=2, adaptive=adaptive, loop_times=0),
        'l8': TCN_GCN_unit(base_channel * 4, base_channel * 4, A, adaptive=adaptive, loop_times=0),
        'l9': TCN_GCN_unit(base_channel * 4, base_channel * 4, A, adaptive=adaptive, loop_times=0),
    })
    return tower


def _run_tower(tower: nn.ModuleDict, x: torch.Tensor, num_point: int, M: int,
               pos_embedding: nn.Parameter) -> torch.Tensor:
    """
    Run a single stream through embedding + BN + 9 TCN-GCN layers.

    Args:
        tower: ModuleDict built by _build_tower.
        x:     (N, C, T, V, M) raw stream tensor.
        num_point: V.
        M:     number of persons.
        pos_embedding: (1, V, base_channel) learnable parameter.

    Returns:
        out: (N*M, 256, T', V) after l9.
    """
    N, C, T, V, _M = x.size()

    x = rearrange(x, 'n c t v m -> (n m t) v c', m=M, v=V).contiguous()
    x = tower['to_joint_embedding'](x)                     # (N*M*T, V, base_channel)
    x += pos_embedding[:, :num_point]
    x = rearrange(x, '(n m t) v c -> n (m v c) t', n=N, m=M, t=T).contiguous()
    x = tower['data_bn'](x)
    x = rearrange(x, 'n (m v c) t -> (n m) c t v', m=M, v=V).contiguous()

    x = tower['l1'](x)
    x = tower['l2'](x)
    x = tower['l3'](x)
    x = tower['l4'](x)
    x = tower['l5'](x)
    x = tower['l6'](x)
    x = tower['l7'](x)
    x = tower['l8'](x)
    x = tower['l9'](x)
    return x  # (N*M, 256, T', V)


# ---------------------------------------------------------------------------
# Main fusion model
# ---------------------------------------------------------------------------

class FusionModel(nn.Module):
    """
    Dual-stream (joint + motion) fusion model.

    Both towers replicate the architecture of model.lagcn.Model.
    The CPR auxiliary branch runs on the joint tower output only.
    StreamFusion combines the two towers before the shared FC head.
    """

    def __init__(
        self,
        num_class: int = 60,
        num_point: int = 25,
        num_person: int = 2,
        graph=None,
        graph_args: dict = dict(),
        examplar=None,
        examplar_args: dict = dict(),
        in_channels: int = 3,
        drop_out: float = 0,
        adaptive: bool = True,
        temporal_cpr: bool = False,
        fusion_mode: str = 'gate',
    ):
        super().__init__()

        if graph is None:
            raise ValueError('graph must be specified')

        Graph = import_class(graph)
        graph_obj = Graph(**graph_args)
        Examplar = import_class(examplar)
        examplar_obj = Examplar(**examplar_args)

        A = graph_obj.A  # (num_subset, V, V)

        self.num_class = num_class
        self.num_point = num_point
        self.num_person = num_person
        self.temporal_cpr = temporal_cpr

        base_channel = 64

        # Shared exemplar (CPR prior), shared FC heads
        self.examplar = nn.Parameter(
            torch.Tensor(examplar_obj.A), requires_grad=False
        )  # (num_class, V, V)

        self.fc = nn.Linear(base_channel * 4, num_class)
        self.aux_fc = nn.Conv2d(base_channel * 4, 1, 1, 1)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2.0 / num_class))

        # Joint tower
        self.joint_tower = _build_tower(in_channels, num_point, num_person, A, adaptive)
        self.joint_pos_embedding = nn.Parameter(
            torch.randn(1, num_point, base_channel)
        )
        bn_init(self.joint_tower['data_bn'], 1)

        # Motion tower
        self.motion_tower = _build_tower(in_channels, num_point, num_person, A, adaptive)
        self.motion_pos_embedding = nn.Parameter(
            torch.randn(1, num_point, base_channel)
        )
        bn_init(self.motion_tower['data_bn'], 1)

        # Fusion module
        self.fusion = StreamFusion(channels=base_channel * 4, mode=fusion_mode)

        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    # ------------------------------------------------------------------

    def forward(self, x_joint: torch.Tensor, x_motion: torch.Tensor):
        """
        Args:
            x_joint:  (N, C, T, V, M)  joint stream
            x_motion: (N, C, T, V, M)  motion stream

        Returns:
            logits:     (N, num_class)
            aux_logits: (N, num_class)  — CPR branch on joint tower only
        """
        N, C, T, V, M = x_joint.size()

        # --- Joint tower ---
        joint_feat = _run_tower(
            self.joint_tower, x_joint, self.num_point, M, self.joint_pos_embedding
        )  # (N*M, 256, T', V)

        # --- CPR auxiliary branch (joint only) ---
        if self.temporal_cpr:
            # Apply CPR at every timestep, then mean-pool over T
            aux_x = torch.einsum('bdtv,cvu->bdtcu', joint_feat, self.examplar).mean(2)
        else:
            # Mean-pool over T first, then CPR
            aux_x = joint_feat.mean(2)                               # (N*M, C, V)
            aux_x = torch.einsum('nmv,cvu->nmcu', aux_x, self.examplar)  # (N*M, C, ncls, V)

        aux_x = self.aux_fc(aux_x)   # (N*M, 1, ncls, V)
        aux_x = aux_x.squeeze(1)     # (N*M, ncls, V)
        aux_x = aux_x.mean(2)        # (N*M, ncls)
        aux_x = aux_x.reshape(N, M, self.num_class).mean(dim=1)  # (N, num_class)

        # --- Motion tower ---
        motion_feat = _run_tower(
            self.motion_tower, x_motion, self.num_point, M, self.motion_pos_embedding
        )  # (N*M, 256, T', V)

        # --- Fusion ---
        fused = self.fusion(joint_feat, motion_feat)  # (N*M, 256, T', V)

        # --- Classification head ---
        c_new = fused.size(1)
        fused = fused.view(N, M, c_new, -1).mean(3).mean(1)  # (N, 256)
        fused = self.drop_out(fused)
        logits = self.fc(fused)                               # (N, num_class)

        return logits, aux_x

    # ------------------------------------------------------------------

    def load_pretrained_tower(self, weights_path: str, prefix: str):
        """
        Load a single-stream checkpoint (model.lagcn.Model) into one tower.

        Mapping:
            l1 .. l9        -> {prefix}_tower / l1 .. l9
            data_bn         -> {prefix}_tower / data_bn
            to_joint_embedding -> {prefix}_tower / to_joint_embedding
            pos_embedding   -> {prefix}_pos_embedding
            fc              -> fc              (shared)
            aux_fc          -> aux_fc          (shared)
            examplar        -> examplar        (shared)

        Args:
            weights_path (str): path to .pt checkpoint saved by main.py.
            prefix (str): 'joint' or 'motion'.
        """
        assert prefix in ('joint', 'motion'), "prefix must be 'joint' or 'motion'"
        ckpt = torch.load(weights_path, map_location='cpu')
        # Strip possible 'module.' prefix from DataParallel checkpoints
        ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}

        tower_attr = f'{prefix}_tower'
        pos_attr   = f'{prefix}_pos_embedding'
        tower: nn.ModuleDict = getattr(self, tower_attr)

        tower_map = {
            'l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8', 'l9',
            'data_bn', 'to_joint_embedding',
        }

        own_state = self.state_dict()
        mapped = {}

        for src_key, val in ckpt.items():
            root = src_key.split('.')[0]
            suffix = src_key[len(root):]  # includes leading '.'

            if root == 'pos_embedding':
                dst_key = pos_attr
                mapped[dst_key] = val
            elif root in tower_map:
                dst_key = f'{tower_attr}.{root}{suffix}'
                mapped[dst_key] = val
            elif root in ('fc', 'aux_fc', 'examplar'):
                # shared weights — load from joint checkpoint only
                if prefix == 'joint':
                    mapped[src_key] = val

        missing = []
        for dst_key, val in mapped.items():
            if dst_key in own_state:
                own_state[dst_key].copy_(val)
            else:
                missing.append(dst_key)

        if missing:
            print(f'[load_pretrained_tower] Keys not found in FusionModel: {missing}')

        self.load_state_dict(own_state)
        print(f'[load_pretrained_tower] Loaded {len(mapped)} tensors into {prefix} tower.')

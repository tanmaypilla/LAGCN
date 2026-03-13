import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

num_node = 20
self_link = [(i, i) for i in range(num_node)]

# Hockey 20-joint skeleton
# Joints:
#   0: right_ear,  1: left_ear,   2: nose
#   3: right_shoulder, 4: left_shoulder
#   5: right_hip,  6: left_hip
#   7: right_elbow, 8: left_wrist, 9: left_elbow, 10: right_wrist
#   11: right_knee, 12: left_knee
#   13: left_ankle, 14: right_ankle
#   15: right_foot_tip, 16: left_foot_tip
#   17: stick_top, 18: stick_middle, 19: stick_tip

inward = [
    # Head
    (2, 1),   # nose -> left_ear
    (2, 0),   # nose -> right_ear
    (0, 1),   # right_ear -> left_ear
    # Torso / Shoulders
    (3, 4),   # right_shoulder -> left_shoulder
    (3, 5),   # right_shoulder -> right_hip
    (4, 6),   # left_shoulder -> left_hip
    (5, 6),   # right_hip -> left_hip
    # Arms
    (3, 7),   # right_shoulder -> right_elbow
    (7, 10),  # right_elbow -> right_wrist
    (4, 9),   # left_shoulder -> left_elbow
    (9, 8),   # left_elbow -> left_wrist
    # Legs
    (5, 11),  # right_hip -> right_knee
    (11, 14), # right_knee -> right_ankle
    (14, 15), # right_ankle -> right_foot_tip
    (6, 12),  # left_hip -> left_knee
    (12, 13), # left_knee -> left_ankle
    (13, 16), # left_ankle -> left_foot_tip
    # Stick (connected to both wrists)
    (8, 17),  # left_wrist -> stick_top
    (10, 17), # right_wrist -> stick_top
    (17, 18), # stick_top -> stick_middle
    (18, 19), # stick_middle -> stick_tip
]

outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError(f'Labeling mode "{labeling_mode}" not supported.')
        return A

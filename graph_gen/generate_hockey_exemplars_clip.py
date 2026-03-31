"""
Generate class-specific CPR exemplar matrices for hockey using CLIP (ViT-B/32).

Drop-in replacement for generate_hockey_exemplars.py — same output format (11, 20, 20),
same directory, different topo_str names so BERT matrices are never overwritten.

Three templates:
  Template A (PASTA-grounded):  "[class]. [body_part_pasta_desc]. [joint]."
  Template B (standard):        "when [class] [joint] of hockey player."
  Template C (CLIP-native):     "A hockey player in [class]. [body_part_pasta_desc]. [joint] position."

Usage:
    cd /home/tanmay-ura/LAGCN
    conda run -n gap_env python graph_gen/generate_hockey_exemplars_clip.py
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm

# Reuse the local CLIP implementation from GAP
sys.path.insert(0, '/home/tanmay-ura/GAP')
import importlib
clip_spec = importlib.util.spec_from_file_location(
    "clip", "/home/tanmay-ura/GAP/clip/__init__.py",
    submodule_search_locations=["/home/tanmay-ura/GAP/clip"]
)
clip_module = importlib.util.module_from_spec(clip_spec)
sys.modules["clip"] = clip_module
clip_spec.loader.exec_module(clip_module)

# ── Constants (identical to BERT version) ─────────────────────────────────────

JOINT_NAMES = [
    "right ear",           # 0
    "left ear",            # 1
    "nose",                # 2
    "right shoulder",      # 3
    "left shoulder",       # 4
    "right hip",           # 5
    "left hip",            # 6
    "right elbow",         # 7
    "left wrist",          # 8
    "left elbow",          # 9
    "right wrist",         # 10
    "right knee",          # 11
    "left knee",           # 12
    "left ankle",          # 13
    "right ankle",         # 14
    "right foot",          # 15
    "left foot",           # 16
    "hockey stick top",    # 17
    "hockey stick middle", # 18
    "hockey stick tip",    # 19
]

CLASS_NAMES = [
    "forward gliding",
    "forward acceleration",
    "backward gliding",
    "backward acceleration",
    "forward to backward transition",
    "backward to forward transition",
    "post whistle gliding",
    "faceoff body position",
    "maintaining position",
    "prone on ice",
    "kneeling on one knee",
]

HEAD_JOINTS = {0, 1, 2}
HAND_JOINTS = {3, 4, 7, 8, 9, 10, 17, 18, 19}
HIP_JOINTS  = {5, 6}
LEG_JOINTS  = {11, 12, 13, 14, 15, 16}

PASTA_FILE = os.path.join(
    os.path.dirname(__file__),
    "../../GAP/text/hockey_pasta_gemini_t01.txt"
)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../graph/cls_matrix_hockey")
TEMPERATURE = 200
NUM_CLASSES = 11
NUM_JOINTS  = 20


# ── Helpers (identical to BERT version) ───────────────────────────────────────

def load_pasta_descriptions(path):
    descriptions = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip().rstrip('.').strip() for p in line.split(';')]
            while len(parts) < 4:
                parts.append('')
            descriptions.append({
                'head':  parts[0],
                'hands': parts[1],
                'hips':  parts[2],
                'legs':  parts[3],
            })
    assert len(descriptions) == NUM_CLASSES
    return descriptions


def get_pasta_segment_for_joint(joint_idx, pasta_entry):
    if joint_idx in HEAD_JOINTS:
        return pasta_entry['head']
    elif joint_idx in HAND_JOINTS:
        return pasta_entry['hands']
    elif joint_idx in HIP_JOINTS:
        return pasta_entry['hips']
    elif joint_idx in LEG_JOINTS:
        return pasta_entry['legs']
    return ''


def compute_similarity_matrix(feats, temperature=200):
    """(V, D) normalised features → (V, V) similarity matrix."""
    feats = feats / feats.norm(dim=1, keepdim=True)
    mat   = (feats @ feats.T) * temperature
    return mat.cpu().numpy()


# ── CLIP feature extraction (replaces build_bert_features) ────────────────────

def build_clip_features(model, device, prompts_per_joint):
    """
    Encode 20 joint prompts with CLIP text encoder.
    Returns (20, 512) normalised float tensor.
    """
    tokens = clip_module.tokenize(prompts_per_joint, truncate=True).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens).float()   # (20, 512)
    return feats


# ── Generation (same structure as BERT version) ───────────────────────────────

def generate_and_save(model, device, prompts_fn, topo_str):
    all_mats  = []
    all_feats = []

    for c_idx in tqdm(range(NUM_CLASSES), desc=f"Generating {topo_str}"):
        prompts = [prompts_fn(c_idx, j_idx) for j_idx in range(NUM_JOINTS)]
        feats   = build_clip_features(model, device, prompts)       # (20, 512)
        mat     = compute_similarity_matrix(feats, TEMPERATURE)     # (20, 20)
        all_mats.append(mat)
        all_feats.append(feats.cpu().numpy())

    mats_arr  = np.stack(all_mats)                                  # (11, 20, 20)
    feats_arr = np.stack(all_feats).reshape(NUM_CLASSES, -1)        # (11, 20*512)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, topo_str + '.npy'),      mats_arr)
    np.save(os.path.join(OUTPUT_DIR, topo_str + '_feat.npy'), feats_arr)
    print(f"Saved {topo_str}.npy  shape={mats_arr.shape}")
    print(f"Saved {topo_str}_feat.npy  shape={feats_arr.shape}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    print("Loading CLIP ViT-B/32...")
    model, _ = clip_module.load("ViT-B/32", device=device)
    model     = model.float().eval()
    print("CLIP loaded.")

    pasta = load_pasta_descriptions(PASTA_FILE)
    print(f"Loaded {len(pasta)} PASTA entries.")

    # Template A: PASTA-grounded (same structure as BERT Template A)
    def template_a(c_idx, j_idx):
        cls  = CLASS_NAMES[c_idx]
        jnt  = JOINT_NAMES[j_idx]
        desc = get_pasta_segment_for_joint(j_idx, pasta[c_idx])
        return f"{cls}. {desc}. {jnt}." if desc else f"{cls}. {jnt}."

    generate_and_save(model, device, template_a,
                      "clip_pasta_grounded_[C]_[J]-with-punctuation")

    # Template B: Standard (same structure as BERT Template B)
    def template_b(c_idx, j_idx):
        return f"when {CLASS_NAMES[c_idx]} {JOINT_NAMES[j_idx]} of hockey player."

    generate_and_save(model, device, template_b,
                      "clip_when_[C]_[J]_of_hockey_player")

    # Template C: CLIP-native conversational style
    def template_c(c_idx, j_idx):
        cls  = CLASS_NAMES[c_idx]
        jnt  = JOINT_NAMES[j_idx]
        desc = get_pasta_segment_for_joint(j_idx, pasta[c_idx])
        return (f"A hockey player in {cls}. {desc}. {jnt} position."
                if desc else f"A hockey player in {cls}. {jnt} position.")

    generate_and_save(model, device, template_c,
                      "clip_natural_[C]_[J]")

    print("\nDone. New matrices saved to:", OUTPUT_DIR)
    print("BERT matrices are unchanged.")


if __name__ == '__main__':
    main()

"""
Generate class-specific CPR (Class Prior Relationship) exemplar matrices for hockey.

Two template variants are generated:
  Template A (PASTA-grounded): "[class_name]. [pasta_body_part_desc]. [joint_name]."
  Template B (standard):       "when [class_name] [joint_name] of hockey player."

Output: (11, 20, 20) npy files in ../graph/cls_matrix_hockey/

Usage:
    cd /home/tanmay-ura/LAGCN/graph_gen
    python generate_hockey_exemplars.py
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# ── Constants ─────────────────────────────────────────────────────────────────

JOINT_NAMES = [
    "right ear",       # 0
    "left ear",        # 1
    "nose",            # 2
    "right shoulder",  # 3
    "left shoulder",   # 4
    "right hip",       # 5
    "left hip",        # 6
    "right elbow",     # 7
    "left wrist",      # 8
    "left elbow",      # 9
    "right wrist",     # 10
    "right knee",      # 11
    "left knee",       # 12
    "left ankle",      # 13
    "right ankle",     # 14
    "right foot",      # 15
    "left foot",       # 16
    "hockey stick top",    # 17
    "hockey stick middle", # 18
    "hockey stick tip",    # 19
]

CLASS_NAMES = [
    "forward gliding",               # 0 GLID_FORW
    "forward acceleration",          # 1 ACCEL_FORW
    "backward gliding",              # 2 GLID_BACK
    "backward acceleration",         # 3 ACCEL_BACK
    "forward to backward transition",  # 4 TRANS_FORW_TO_BACK
    "backward to forward transition",  # 5 TRANS_BACK_TO_FORW
    "post whistle gliding",          # 6 POST_WHISTLE_GLIDING
    "faceoff body position",         # 7 FACEOFF_BODY_POSITION
    "maintaining position",          # 8 MAINTAIN_POSITION
    "prone on ice",                  # 9 PRONE
    "kneeling on one knee",          # 10 ON_A_KNEE
]

# Joint-to-body-part mapping for selecting which PASTA segment to use
HEAD_JOINTS  = {0, 1, 2}
HAND_JOINTS  = {3, 4, 7, 8, 9, 10, 17, 18, 19}
HIP_JOINTS   = {5, 6}
LEG_JOINTS   = {11, 12, 13, 14, 15, 16}

# PASTA file (11 lines, 4 semicolon-delimited segments per line)
PASTA_FILE = os.path.join(
    os.path.dirname(__file__),
    "../../GAP/text/hockey_pasta_gemini_t01.txt"
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../graph/cls_matrix_hockey")
TEMPERATURE = 200
NUM_CLASSES = 11
NUM_JOINTS = 20


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_pasta_descriptions(path):
    """Load and parse hockey PASTA text into per-class, per-part descriptions.

    Returns list of 11 dicts: {'head': str, 'hands': str, 'hips': str, 'legs': str}
    """
    descriptions = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip().rstrip('.').strip() for p in line.split(';')]
            if len(parts) < 4:
                # Pad if fewer than 4 segments
                while len(parts) < 4:
                    parts.append('')
            descriptions.append({
                'head':  parts[0],
                'hands': parts[1],
                'hips':  parts[2],
                'legs':  parts[3],
            })
    assert len(descriptions) == NUM_CLASSES, \
        f"Expected {NUM_CLASSES} PASTA lines, got {len(descriptions)}"
    return descriptions


def get_pasta_segment_for_joint(joint_idx, pasta_entry):
    """Return the PASTA description segment appropriate for the given joint."""
    if joint_idx in HEAD_JOINTS:
        return pasta_entry['head']
    elif joint_idx in HAND_JOINTS:
        return pasta_entry['hands']
    elif joint_idx in HIP_JOINTS:
        return pasta_entry['hips']
    elif joint_idx in LEG_JOINTS:
        return pasta_entry['legs']
    else:
        return ''


def build_bert_features(tokenizer, model, device, prompts_per_joint):
    """
    Extract BERT pooler_output for each joint's prompt.

    prompts_per_joint: list of 20 strings (one per joint)
    Returns: (20, 768) tensor of pooler outputs
    """
    feats = []
    with torch.no_grad():
        for prompt in prompts_per_joint:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
            outputs = model(**inputs)
            feats.append(outputs.pooler_output)  # (1, 768)
    feats = torch.cat(feats, dim=0)  # (20, 768)
    return feats


def compute_similarity_matrix(feats, temperature=200):
    """Compute normalized dot-product similarity matrix.

    feats: (V, 768)
    Returns: (V, V) numpy array
    """
    feats = feats / feats.norm(dim=1, keepdim=True)
    mat = (feats @ feats.T) * temperature  # (V, V)
    return mat.cpu().numpy()


def generate_and_save(tokenizer, model, device, prompts_fn, topo_str):
    """Generate (num_classes, V, V) matrix using prompts_fn(class_idx, joint_idx) -> str."""
    all_mats = []
    all_feats = []

    for c_idx in tqdm(range(NUM_CLASSES), desc=f"Generating {topo_str}"):
        prompts = [prompts_fn(c_idx, j_idx) for j_idx in range(NUM_JOINTS)]
        feats = build_bert_features(tokenizer, model, device, prompts)  # (20, 768)
        mat = compute_similarity_matrix(feats, temperature=TEMPERATURE)  # (20, 20)
        all_mats.append(mat)
        all_feats.append(feats.cpu().numpy())

    mats_arr = np.stack(all_mats)    # (11, 20, 20)
    feats_arr = np.stack(all_feats)  # (11, 20, 768) → reshape to (11, 20*768)
    feats_arr = feats_arr.reshape(NUM_CLASSES, -1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mat_path  = os.path.join(OUTPUT_DIR, topo_str + '.npy')
    feat_path = os.path.join(OUTPUT_DIR, topo_str + '_feat.npy')
    np.save(mat_path, mats_arr)
    np.save(feat_path, feats_arr)
    print(f"Saved {mat_path}  shape={mats_arr.shape}")
    print(f"Saved {feat_path} shape={feats_arr.shape}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load BERT
    print("Loading BERT (bert-base-uncased)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
        model = AutoModel.from_pretrained("bert-base-uncased", local_files_only=True)
    except Exception:
        print("Local BERT not found, downloading from HuggingFace Hub...")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModel.from_pretrained("bert-base-uncased")
    model = model.to(device).eval()
    print("BERT loaded.")

    # Load PASTA descriptions
    print(f"Loading PASTA descriptions from: {PASTA_FILE}")
    pasta = load_pasta_descriptions(PASTA_FILE)
    print(f"Loaded {len(pasta)} PASTA entries.")

    # ── Template A: PASTA-grounded ─────────────────────────────────────────────
    def template_a(c_idx, j_idx):
        desc = get_pasta_segment_for_joint(j_idx, pasta[c_idx])
        cls = CLASS_NAMES[c_idx]
        jnt = JOINT_NAMES[j_idx]
        if desc:
            return f"{cls}. {desc}. {jnt}."
        else:
            return f"{cls}. {jnt}."

    topo_str_a = "hockey_pasta_grounded_[C]_[J]-with-punctuation"
    generate_and_save(tokenizer, model, device, template_a, topo_str_a)

    # ── Template B: Standard (mirrors NTU "when [C] [J] of human body" approach) ─
    def template_b(c_idx, j_idx):
        cls = CLASS_NAMES[c_idx]
        jnt = JOINT_NAMES[j_idx]
        return f"when {cls} {jnt} of hockey player."

    topo_str_b = "when_[C]_[J]_of_hockey_player-with-punctuation"
    generate_and_save(tokenizer, model, device, template_b, topo_str_b)

    print("\nDone! All exemplar matrices saved to:", OUTPUT_DIR)
    print("Use in config:")
    print(f"  topo_str: \"{topo_str_b}\"  # standard (default)")
    print(f"  topo_str: \"{topo_str_a}\"  # PASTA-grounded (richer)")


if __name__ == '__main__':
    main()

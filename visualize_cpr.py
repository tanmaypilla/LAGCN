"""
Visualize CPR (Class Prior Relationship) exemplar matrices learned from text.

Generates 4 figures:
  1. Exemplar heatmaps for all classes (one subplot per class)
  2. Faceoff vs. most-confused class (GLID_FORW) side-by-side
  3. Inter-class cosine similarity in text embedding space (class-level)
  4. PCA of per-joint text embeddings (how text distinguishes classes per joint)

Usage:
    cd /home/tanmay-ura/LAGCN
    python visualize_cpr.py [--topo TOPO_STR] [--out OUT_DIR]

Defaults to PASTA-grounded BERT exemplars.
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

# ── Constants ─────────────────────────────────────────────────────────────────

CLASS_NAMES = [
    "GLID_FORW",
    "ACCEL_FORW",
    "GLID_BACK",
    "ACCEL_BACK",
    "TRANS_F2B",
    "TRANS_B2F",
    "POST_WHISTLE",
    "FACEOFF",
    "MAINTAIN",
    "PRONE",
    "ON_A_KNEE",
]

JOINT_NAMES = [
    "R ear", "L ear", "nose",
    "R shoulder", "L shoulder",
    "R hip", "L hip",
    "R elbow", "L wrist", "L elbow", "R wrist",
    "R knee", "L knee",
    "L ankle", "R ankle", "R foot", "L foot",
    "stick top", "stick mid", "stick tip",
]

FACEOFF_IDX = 7
CONFUSED_IDX = 0   # GLID_FORW (most common faceoff misclassification)

DEFAULT_TOPO = "hockey_pasta_grounded_[C]_[J]-with-punctuation"
MATRIX_DIR = "graph/cls_matrix_hockey"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_matrices(topo_str):
    mat_path  = os.path.join(MATRIX_DIR, topo_str + ".npy")
    feat_path = os.path.join(MATRIX_DIR, topo_str + "_feat.npy")
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Matrix not found: {mat_path}")
    mats = np.load(mat_path)   # (11, 20, 20)
    feats = None
    if os.path.exists(feat_path):
        feats = np.load(feat_path)  # (11, 20*D)
    return mats, feats


def plot_heatmap(ax, mat, title, joint_names, vmin=None, vmax=None, cmap='hot'):
    n = len(joint_names)
    im = ax.imshow(mat, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(joint_names, rotation=90, fontsize=5)
    ax.set_yticklabels(joint_names, fontsize=5)
    ax.set_title(title, fontsize=7, pad=2)
    return im


# ── Figure 1: All-class exemplar heatmaps ─────────────────────────────────────

def fig_all_classes(mats, out_dir, topo_str):
    num_cls = mats.shape[0]
    ncols = 4
    nrows = (num_cls + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.5))
    axes = axes.flatten()

    vmin, vmax = mats.min(), mats.max()
    for i in range(num_cls):
        ax = axes[i]
        title = f"[{i}] {CLASS_NAMES[i]}"
        if i == FACEOFF_IDX:
            title = f"★ {title}"
        im = plot_heatmap(ax, mats[i], title, JOINT_NAMES, vmin=vmin, vmax=vmax)
        if i == FACEOFF_IDX:
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(2.5)

    # hide unused axes
    for j in range(num_cls, len(axes)):
        axes[j].axis('off')

    fig.colorbar(im, ax=axes[:num_cls], shrink=0.4, label='Similarity × 200')
    fig.suptitle(f"CPR Exemplar Matrices — {topo_str}\n(★ = FACEOFF, outlined in red)", fontsize=10)
    plt.tight_layout()
    path = os.path.join(out_dir, "cpr_all_classes.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


# ── Figure 2: Faceoff vs. GLID_FORW comparison + difference ──────────────────

def fig_faceoff_vs_confused(mats, out_dir, topo_str):
    face = mats[FACEOFF_IDX]
    glid = mats[CONFUSED_IDX]
    diff = face - glid

    vmin = min(face.min(), glid.min())
    vmax = max(face.max(), glid.max())
    dmax = max(abs(diff.min()), abs(diff.max()))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    im0 = plot_heatmap(axes[0], face, f"FACEOFF [{FACEOFF_IDX}]", JOINT_NAMES, vmin=vmin, vmax=vmax)
    im1 = plot_heatmap(axes[1], glid, f"GLID_FORW [{CONFUSED_IDX}]\n(most confused with FACEOFF)", JOINT_NAMES, vmin=vmin, vmax=vmax)
    im2 = plot_heatmap(axes[2], diff, "FACEOFF − GLID_FORW\n(red = faceoff stronger, blue = glid stronger)",
                       JOINT_NAMES, vmin=-dmax, vmax=dmax, cmap='RdBu_r')

    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    fig.suptitle(f"Faceoff vs. Most-Confused Class — {topo_str}", fontsize=11)
    plt.tight_layout()
    path = os.path.join(out_dir, "cpr_faceoff_vs_glid.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


# ── Figure 3: Inter-class cosine similarity in text space ─────────────────────

def fig_interclass_similarity(feats, out_dir, topo_str):
    if feats is None:
        print("Skipping inter-class similarity (no _feat.npy found)")
        return

    # feats: (11, 20*D) — mean-pool across joints to get class-level embedding
    num_cls = feats.shape[0]
    num_joints = 20
    D = feats.shape[1] // num_joints
    feats_3d = feats.reshape(num_cls, num_joints, D)  # (11, 20, D)
    cls_emb = feats_3d.mean(axis=1)                   # (11, D) — class prototype

    cls_emb_norm = normalize(cls_emb, axis=1)
    sim = cls_emb_norm @ cls_emb_norm.T                # (11, 11) cosine sim

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(sim, vmin=-1, vmax=1, cmap='RdYlGn')
    ax.set_xticks(range(num_cls))
    ax.set_yticks(range(num_cls))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax.set_title(f"Inter-class Cosine Similarity (text space)\n{topo_str}\n"
                 f"(high = text model thinks classes are similar → harder to distinguish via CPR)", fontsize=9)

    for i in range(num_cls):
        for j in range(num_cls):
            ax.text(j, i, f"{sim[i,j]:.2f}", ha='center', va='center',
                    fontsize=5.5, color='black' if abs(sim[i,j]) < 0.7 else 'white')

    # highlight faceoff row/col
    ax.axhline(FACEOFF_IDX - 0.5, color='red', lw=1.5)
    ax.axhline(FACEOFF_IDX + 0.5, color='red', lw=1.5)
    ax.axvline(FACEOFF_IDX - 0.5, color='red', lw=1.5)
    ax.axvline(FACEOFF_IDX + 0.5, color='red', lw=1.5)

    fig.colorbar(im, ax=ax, fraction=0.046, label='Cosine Similarity')
    plt.tight_layout()
    path = os.path.join(out_dir, "cpr_interclass_sim.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


# ── Figure 4: PCA of per-joint embeddings (text space) ───────────────────────

def fig_pca_joint_embeddings(feats, out_dir, topo_str):
    if feats is None:
        print("Skipping PCA (no _feat.npy found)")
        return

    num_cls = feats.shape[0]
    num_joints = 20
    D = feats.shape[1] // num_joints
    feats_3d = feats.reshape(num_cls, num_joints, D)  # (11, 20, D)

    # Stack all (class, joint) embeddings for PCA
    all_emb = feats_3d.reshape(num_cls * num_joints, D)  # (220, D)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(all_emb)  # (220, 2)
    coords = coords.reshape(num_cls, num_joints, 2)

    # Color palette for classes
    colors = plt.cm.tab20(np.linspace(0, 1, num_cls))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ── Left: all joints, colored by class ──────────────────────────────────
    ax = axes[0]
    for c in range(num_cls):
        pts = coords[c]  # (20, 2)
        is_faceoff = (c == FACEOFF_IDX)
        marker = '*' if is_faceoff else 'o'
        size = 120 if is_faceoff else 40
        edge = 'red' if is_faceoff else 'none'
        ax.scatter(pts[:, 0], pts[:, 1], c=[colors[c]], marker=marker, s=size,
                   edgecolors=edge, linewidths=1.2, label=CLASS_NAMES[c], zorder=3 if is_faceoff else 2)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("PCA of per-joint text embeddings\n(colored by class — stars = FACEOFF)", fontsize=9)
    ax.legend(fontsize=6, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)

    # ── Right: class means + joint labels for faceoff vs glid ───────────────
    ax = axes[1]
    highlight = [FACEOFF_IDX, CONFUSED_IDX]
    for c in highlight:
        pts = coords[c]   # (20, 2)
        cname = CLASS_NAMES[c]
        col = 'red' if c == FACEOFF_IDX else 'steelblue'
        for j in range(num_joints):
            ax.scatter(pts[j, 0], pts[j, 1], c=col, s=60, zorder=3)
            ax.annotate(JOINT_NAMES[j], (pts[j, 0], pts[j, 1]),
                        textcoords="offset points", xytext=(4, 2), fontsize=5.5, color=col)
        # draw class prototype
        mean_pt = pts.mean(axis=0)
        ax.scatter(mean_pt[0], mean_pt[1], c=col, s=200, marker='D', zorder=5,
                   edgecolors='black', linewidths=1, label=f"{cname} (mean)")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("PCA: FACEOFF (red) vs GLID_FORW (blue)\nper-joint positions — diamonds = class mean", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Text Embedding Space — {topo_str}\n"
                 f"Var explained: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, PC2={pca.explained_variance_ratio_[1]*100:.1f}%",
                 fontsize=10)
    plt.tight_layout()
    path = os.path.join(out_dir, "cpr_pca_embeddings.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


# ── Figure 5: Faceoff joint-importance (row/col marginals of exemplar) ────────

def fig_faceoff_joint_importance(mats, out_dir, topo_str):
    """Which joints does the text model consider most important for each class?
    Marginal sum of exemplar matrix = effective joint weight."""
    num_cls = mats.shape[0]
    # Row sum (how much a joint attends to others)
    marginals = mats.sum(axis=2)  # (11, 20)

    # Normalize per class so they're comparable
    marginals_norm = marginals / marginals.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(marginals_norm, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(len(JOINT_NAMES)))
    ax.set_xticklabels(JOINT_NAMES, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(num_cls))
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax.set_title(f"Joint Importance per Class (normalized row-sum of CPR)\n{topo_str}\n"
                 f"(brighter = this joint is more central in the text-derived graph for this class)", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.02, label='Normalized importance')

    # highlight faceoff row
    ax.axhline(FACEOFF_IDX - 0.5, color='red', lw=2)
    ax.axhline(FACEOFF_IDX + 0.5, color='red', lw=2)
    ax.text(-0.5, FACEOFF_IDX, '► ', ha='right', va='center', color='red', fontsize=10, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(out_dir, "cpr_joint_importance.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topo', default=DEFAULT_TOPO,
                        help='topo_str name (without .npy), e.g. hockey_pasta_grounded_[C]_[J]-with-punctuation')
    parser.add_argument('--out', default='viz_cpr',
                        help='Output directory for figures')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Loading exemplar matrices: {args.topo}")
    mats, feats = load_matrices(args.topo)
    print(f"  mats  shape: {mats.shape}")
    if feats is not None:
        print(f"  feats shape: {feats.shape}")

    print("\nFaceoff exemplar summary:")
    face_mat = mats[FACEOFF_IDX]
    glid_mat = mats[CONFUSED_IDX]
    diff = face_mat - glid_mat
    print(f"  Faceoff  — mean: {face_mat.mean():.2f}, std: {face_mat.std():.2f}, range: [{face_mat.min():.1f}, {face_mat.max():.1f}]")
    print(f"  GlidForw — mean: {glid_mat.mean():.2f}, std: {glid_mat.std():.2f}, range: [{glid_mat.min():.1f}, {glid_mat.max():.1f}]")
    print(f"  Diff abs max: {abs(diff).max():.2f}  (larger = text sees them as more distinct)")

    print("\nGenerating figures...")
    fig_all_classes(mats, args.out, args.topo)
    fig_faceoff_vs_confused(mats, args.out, args.topo)
    fig_interclass_similarity(feats, args.out, args.topo)
    fig_pca_joint_embeddings(feats, args.out, args.topo)
    fig_faceoff_joint_importance(mats, args.out, args.topo)

    print(f"\nAll figures saved to: {args.out}/")


if __name__ == '__main__':
    main()

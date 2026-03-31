"""
Feature-space diagnostic for LAGCN hockey model.

Extracts the 256-dim pre-FC features for all test samples, then:
  1. t-SNE plot colored by true class
  2. t-SNE plot colored by correct/wrong prediction
  3. Per-class pairwise cosine similarity heatmap (are FACEOFF features
     overlapping with other classes in 256-d space?)
  4. Confusion matrix
  5. Per-class: nearest-neighbour purity (what fraction of a class's
     k-nearest neighbours in feature space share the same label?)

Usage:
    cd /home/tanmay-ura/LAGCN
    python diagnose_features.py \
        --config configs/hockey/joint.yaml \
        --checkpoint work_dir/hockey/joint_CUDNN/runs-65-20800.pt \
        --out viz_diagnostic
"""

import argparse, os, sys, yaml
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import normalize

# ── constants ─────────────────────────────────────────────────────────────────

CLASS_NAMES = [
    "GLID_FORW", "ACCEL_FORW", "GLID_BACK", "ACCEL_BACK",
    "TRANS_F2B", "TRANS_B2F", "POST_WHISTLE", "FACEOFF",
    "MAINTAIN", "PRONE", "ON_A_KNEE",
]
NUM_CLASSES = len(CLASS_NAMES)

# Classes we care about diagnosing
PROBLEM_CLASSES = {6: "POST_WHISTLE", 7: "FACEOFF"}

# ── helpers ───────────────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg):
    import importlib
    model_cls = importlib.import_module('.'.join(cfg['model'].split('.')[:-1]))
    model_cls = getattr(model_cls, cfg['model'].split('.')[-1])

    # Build exemplar
    ex_mod = importlib.import_module('.'.join(cfg['examplar'].split('.')[:-1]))
    ex_cls = getattr(ex_mod, cfg['examplar'].split('.')[-1])
    examplar = ex_cls(**cfg['examplar_args'])

    # Build graph
    g_mod = importlib.import_module('.'.join(cfg['graph'].split('.')[:-1]))
    g_cls = getattr(g_mod, cfg['graph'].split('.')[-1])
    graph = g_cls(**cfg['graph_args'])

    model = model_cls(
        **cfg['model_args'],
        graph=graph,
        examplar=examplar,
    )
    return model


def load_model(cfg_path, ckpt_path, device):
    cfg_full = load_config(cfg_path)

    # model_args already contains graph/graph_args/examplar/examplar_args as strings
    # exactly as main.py passes them
    import importlib
    model_cls_str = cfg_full['model']
    m_parts = model_cls_str.rsplit('.', 1)
    model_cls = getattr(importlib.import_module(m_parts[0]), m_parts[1])

    model = model_cls(**cfg_full['model_args'])

    state = torch.load(ckpt_path, map_location=device)
    # strip DataParallel prefix if present
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, cfg_full


def build_loader(cfg_full):
    import importlib
    feeder_str = cfg_full['feeder']
    f_parts = feeder_str.rsplit('.', 1)
    feeder_cls = getattr(importlib.import_module(f_parts[0]), f_parts[1])
    test_args = cfg_full['test_feeder_args']
    from torch.utils.data import DataLoader
    dataset = feeder_cls(**test_args)
    loader  = DataLoader(dataset, batch_size=64, shuffle=False,
                         num_workers=4, pin_memory=True, drop_last=False)
    return loader


# ── feature extraction ────────────────────────────────────────────────────────

def extract_features(model, loader, device):
    """
    Hook into the model just before self.fc to capture 256-dim embeddings.
    Returns:
        feats   (N, 256)  float32 numpy
        labels  (N,)      int numpy
        preds   (N,)      int numpy  (argmax of main logits)
    """
    feats_list, labels_list, preds_list = [], [], []

    captured = {}
    def hook_fn(module, input, output):
        # input[0] is the (N, 256) tensor going into the FC layer
        captured['feat'] = input[0].detach().cpu()

    handle = model.fc.register_forward_hook(hook_fn)

    with torch.no_grad():
        for data, label, _ in loader:
            data  = data.float().to(device)
            label = label.long()
            main_out, _ = model(data)
            preds = main_out.argmax(dim=1).cpu()
            feats_list.append(captured['feat'].numpy())
            labels_list.append(label.numpy())
            preds_list.append(preds.numpy())

    handle.remove()

    feats  = np.concatenate(feats_list,  axis=0)
    labels = np.concatenate(labels_list, axis=0)
    preds  = np.concatenate(preds_list,  axis=0)
    return feats, labels, preds


# ── Figure 1: t-SNE colored by true class ────────────────────────────────────

def fig_tsne_true(coords, labels, out_dir):
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, NUM_CLASSES))
    for c in range(NUM_CLASSES):
        mask = labels == c
        if mask.sum() == 0:
            continue
        is_problem = c in PROBLEM_CLASSES
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[colors[c]], s=30 if is_problem else 15,
                   marker='*' if is_problem else 'o',
                   edgecolors='black' if is_problem else 'none',
                   linewidths=0.5,
                   label=CLASS_NAMES[c], zorder=3 if is_problem else 2,
                   alpha=0.9 if is_problem else 0.6)

    ax.legend(fontsize=7, ncol=2, loc='best')
    ax.set_title("t-SNE of 256-d GCN features — colored by true class\n"
                 "(★ = FACEOFF / POST_WHISTLE)", fontsize=10)
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "diag_tsne_true.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


# ── Figure 2: t-SNE colored by correct / wrong ───────────────────────────────

def fig_tsne_correct(coords, labels, preds, out_dir):
    correct = (labels == preds)
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(coords[correct, 0],  coords[correct, 1],
               c='steelblue', s=12, alpha=0.5, label='Correct', zorder=2)
    ax.scatter(coords[~correct, 0], coords[~correct, 1],
               c='red', s=25, alpha=0.8, marker='x', label='Wrong', zorder=3)

    # label each wrong point with its true class
    for i in np.where(~correct)[0]:
        ax.annotate(CLASS_NAMES[labels[i]][:6],
                    (coords[i, 0], coords[i, 1]),
                    fontsize=4, color='darkred', alpha=0.7)

    ax.legend(fontsize=9)
    ax.set_title("t-SNE — correct (blue) vs. misclassified (red ×)\n"
                 "annotations show true class of misclassifications", fontsize=10)
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "diag_tsne_errors.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


# ── Figure 3: inter-class cosine sim in feature space ────────────────────────

def fig_feature_similarity(feats, labels, out_dir):
    # class prototypes = mean of all samples per class
    prototypes = np.zeros((NUM_CLASSES, feats.shape[1]))
    for c in range(NUM_CLASSES):
        mask = labels == c
        if mask.sum() > 0:
            prototypes[c] = feats[mask].mean(0)

    proto_norm = normalize(prototypes, axis=1)
    sim = proto_norm @ proto_norm.T

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim, vmin=-1, vmax=1, cmap='RdYlGn')
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax.set_title("Inter-class cosine similarity in 256-d feature space\n"
                 "(high = GCN produces similar features for these classes → hard to separate)", fontsize=9)

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f"{sim[i,j]:.2f}", ha='center', va='center',
                    fontsize=5.5, color='black' if abs(sim[i,j]) < 0.7 else 'white')

    # highlight problem classes
    for pc in PROBLEM_CLASSES:
        ax.axhline(pc - 0.5, color='red', lw=1.5)
        ax.axhline(pc + 0.5, color='red', lw=1.5)
        ax.axvline(pc - 0.5, color='red', lw=1.5)
        ax.axvline(pc + 0.5, color='red', lw=1.5)

    fig.colorbar(im, ax=ax, fraction=0.046, label='Cosine Similarity')
    plt.tight_layout()
    path = os.path.join(out_dir, "diag_feature_sim.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


# ── Figure 4: Confusion matrix ────────────────────────────────────────────────

def fig_confusion(labels, preds, out_dir):
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)), normalize='true')
    fig, ax = plt.subplots(figsize=(10, 9))
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, colorbar=True, xticks_rotation=45, values_format='.2f', cmap='Blues')
    ax.set_title("Normalised Confusion Matrix (row = true class)", fontsize=10)
    plt.tight_layout()
    path = os.path.join(out_dir, "diag_confusion.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


# ── Figure 5: kNN purity per class ───────────────────────────────────────────

def fig_knn_purity(feats, labels, out_dir, k=10):
    """
    For each sample, find its k nearest neighbours in feature space.
    Purity = fraction of those neighbours sharing the same label.
    Report mean purity per class.
    """
    feats_norm = normalize(feats, axis=1)
    sim_matrix = feats_norm @ feats_norm.T  # (N, N)
    np.fill_diagonal(sim_matrix, -1)        # exclude self

    purity_per_class = {}
    for c in range(NUM_CLASSES):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            purity_per_class[c] = 0.0
            continue
        purities = []
        for i in idx:
            top_k = np.argsort(sim_matrix[i])[-k:]
            same  = np.sum(labels[top_k] == c)
            purities.append(same / k)
        purity_per_class[c] = np.mean(purities)

    classes = list(range(NUM_CLASSES))
    purities = [purity_per_class[c] for c in classes]
    colors = ['red' if c in PROBLEM_CLASSES else 'steelblue' for c in classes]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(NUM_CLASSES), purities, color=colors)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel(f"{k}-NN purity")
    ax.set_ylim(0, 1.05)
    ax.axhline(1/NUM_CLASSES, color='gray', linestyle='--', label=f'Random baseline (1/{NUM_CLASSES})')
    ax.set_title(f"{k}-NN purity in 256-d feature space per class\n"
                 "(red = problem classes; higher = feature cluster is pure/separable)", fontsize=10)
    ax.legend(fontsize=8)

    for bar, p in zip(bars, purities):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{p:.2f}", ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    path = os.path.join(out_dir, "diag_knn_purity.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")

    # also print a summary
    print("\n--- kNN Purity Summary ---")
    for c in range(NUM_CLASSES):
        marker = " ← PROBLEM" if c in PROBLEM_CLASSES else ""
        print(f"  [{c:2d}] {CLASS_NAMES[c]:<25s}  purity={purity_per_class[c]:.3f}{marker}")


# ── Figure 6: per-class accuracy bar chart ───────────────────────────────────

def fig_per_class_acc(labels, preds, out_dir):
    accs = []
    counts = []
    for c in range(NUM_CLASSES):
        mask = labels == c
        n = mask.sum()
        counts.append(n)
        accs.append((preds[mask] == c).sum() / n if n > 0 else 0.0)

    colors = ['red' if c in PROBLEM_CLASSES else 'steelblue' for c in range(NUM_CLASSES)]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(NUM_CLASSES), accs, color=colors)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels([f"{CLASS_NAMES[c]}\n(n={counts[c]})" for c in range(NUM_CLASSES)],
                       rotation=45, ha='right', fontsize=7)
    ax.set_ylabel("Per-class accuracy")
    ax.set_ylim(0, 1.1)
    ax.set_title("Per-class accuracy (red = FACEOFF / POST_WHISTLE)", fontsize=10)

    for bar, a in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{a:.2f}", ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    path = os.path.join(out_dir, "diag_per_class_acc.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")

    print("\n--- Per-class Accuracy ---")
    for c in range(NUM_CLASSES):
        marker = " ← PROBLEM" if c in PROBLEM_CLASSES else ""
        print(f"  [{c:2d}] {CLASS_NAMES[c]:<25s}  acc={accs[c]:.3f}  (n={counts[c]}){marker}")
    print(f"\n  Overall accuracy: {(labels == preds).mean():.4f}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     default='configs/hockey/joint.yaml')
    parser.add_argument('--checkpoint', default='work_dir/hockey/joint_CUDNN/runs-65-20800.pt')
    parser.add_argument('--out',        default='viz_diagnostic')
    parser.add_argument('--tsne_perp',  type=int, default=30)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print(f"Loading model from {args.checkpoint}")
    model, cfg_full = load_model(args.config, args.checkpoint, device)

    print("Building test loader...")
    loader = build_loader(cfg_full)

    print("Extracting features...")
    feats, labels, preds = extract_features(model, loader, device)
    print(f"  feats shape: {feats.shape}   labels: {labels.shape}")

    print("\nRunning t-SNE (this takes ~30s)...")
    tsne = TSNE(n_components=2, perplexity=args.tsne_perp,
                random_state=42, n_iter=1000, verbose=1)
    coords = tsne.fit_transform(feats)

    print("\nGenerating figures...")
    fig_tsne_true(coords, labels, args.out)
    fig_tsne_correct(coords, labels, preds, args.out)
    fig_feature_similarity(feats, labels, args.out)
    fig_confusion(labels, preds, args.out)
    fig_knn_purity(feats, labels, args.out, k=10)
    fig_per_class_acc(labels, preds, args.out)

    print(f"\nAll figures saved to: {args.out}/")


if __name__ == '__main__':
    main()

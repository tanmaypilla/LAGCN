"""
Generate confusion matrix PNG from epoch CSV files saved by LAGCN main.py.

Usage:
  python plot_confusion_matrix.py \
      --csv work_dir/hockey/joint_CUDNN/epoch38_test_each_class_acc.csv \
      --title "joint_CUDNN — Test Confusion Matrix (Epoch 38, Best)" \
      --out work_dir/hockey/joint_CUDNN/confusion_matrix_epoch38.png

  # Or compare multiple runs at their best epoch:
  python plot_confusion_matrix.py --compare
"""

import argparse
import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


CLASS_SHORT = ["GF", "AF", "GB", "AB", "TFB", "TBF", "PWG", "FO", "MP", "P", "OK"]
CLASS_FULL  = [
    "GLID_FORW", "ACCEL_FORW", "GLID_BACK", "ACCEL_BACK",
    "TRANS_FORW_TO_BACK", "TRANS_BACK_TO_FORW", "POST_WHISTLE_GLIDING",
    "FACEOFF_BODY_POSITION", "MAINTAIN_POSITION", "PRONE", "ON_A_KNEE",
]


def load_csv(path):
    """Return (class_names, per_class_acc, confusion_matrix_counts)."""
    with open(path) as f:
        rows = list(csv.reader(f))
    class_names = rows[0]
    per_class_acc = [float(x) for x in rows[1]]
    counts = np.array([[int(x) for x in row] for row in rows[2:]], dtype=float)
    return class_names, per_class_acc, counts


def row_normalise(counts):
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return counts / row_sums * 100.0


def plot_confusion_matrix(csv_path, title, out_path, short_labels=True):
    class_names, per_class_acc, counts = load_csv(csv_path)
    n = len(class_names)
    labels = CLASS_SHORT[:n] if short_labels else CLASS_FULL[:n]

    pct = row_normalise(counts)
    top1 = counts.diagonal().sum() / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(12, 10))

    cmap = plt.cm.Blues
    im = ax.imshow(pct, vmin=0, vmax=100, cmap=cmap, aspect='auto')

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Row %', fontsize=11)
    cbar.set_ticks([0, 20, 40, 60, 80, 100])

    # Cell text
    for i in range(n):
        for j in range(n):
            val = pct[i, j]
            text_color = 'white' if val > 60 else 'black'
            weight = 'bold' if i == j else 'normal'
            ax.text(j, i, f"{val:.1f}%", ha='center', va='center',
                    fontsize=7.5, color=text_color, fontweight=weight)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Ground truth', fontsize=12)
    ax.set_title(f"{title}\nTop-1: {top1:.2f}%  |  Mean class acc: {np.mean(per_class_acc)*100:.2f}%",
                 fontsize=12, pad=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def compare_runs():
    """Side-by-side comparison of best epochs across runs."""
    runs = [
        ("joint_CUDNN",            38, "joint_CUDNN\n(CE loss, standard)"),
        ("joint_focal_CUDNN",      37, "joint_focal_CUDNN\n(Focal γ=2, standard)"),
        ("joint_weighted_sampler_CUDNN", 63, "joint_weighted_sampler_CUDNN\n(CE loss, weighted sampler)"),
    ]
    base = "work_dir/hockey"

    fig, axes = plt.subplots(1, len(runs), figsize=(36, 11))
    fig.suptitle("LAGCN Hockey — Best Epoch Confusion Matrices", fontsize=14, y=1.01)

    cmap = plt.cm.Blues
    norm = mcolors.Normalize(vmin=0, vmax=100)

    for ax, (run_dir, epoch, label) in zip(axes, runs):
        csv_path = f"{base}/{run_dir}/epoch{epoch}_test_each_class_acc.csv"
        if not os.path.exists(csv_path):
            ax.set_title(f"{label}\n(no data)", fontsize=9)
            ax.axis('off')
            continue

        class_names, per_class_acc, counts = load_csv(csv_path)
        n = len(class_names)
        labels = CLASS_SHORT[:n]
        pct = row_normalise(counts)
        top1 = counts.diagonal().sum() / counts.sum() * 100

        im = ax.imshow(pct, vmin=0, vmax=100, cmap=cmap, aspect='auto')
        for i in range(n):
            for j in range(n):
                val = pct[i, j]
                tc = 'white' if val > 60 else 'black'
                fw = 'bold' if i == j else 'normal'
                ax.text(j, i, f"{val:.1f}%", ha='center', va='center',
                        fontsize=6.5, color=tc, fontweight=fw)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('Ground truth', fontsize=10)
        ax.set_title(f"{label}\nTop-1: {top1:.2f}%  |  Mean: {np.mean(per_class_acc)*100:.2f}%",
                     fontsize=9, pad=8)

    # Shared colorbar
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=axes, fraction=0.01, pad=0.02)
    cbar.set_label('Row %', fontsize=11)
    cbar.set_ticks([0, 20, 40, 60, 80, 100])

    out_path = f"{base}/compare_confusion_matrices.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',     type=str, help='Path to epoch CSV file')
    parser.add_argument('--title',   type=str, default='LAGCN — Confusion Matrix')
    parser.add_argument('--out',     type=str, help='Output PNG path')
    parser.add_argument('--compare', action='store_true',
                        help='Generate side-by-side comparison of all runs')
    args = parser.parse_args()

    if args.compare:
        compare_runs()
    else:
        if not args.csv:
            parser.error('--csv required unless --compare is set')
        out = args.out or args.csv.replace('.csv', '.png')
        plot_confusion_matrix(args.csv, args.title, out)

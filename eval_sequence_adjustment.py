"""
Compare original vs Viterbi-adjusted predictions on the test set.
Usage: python eval_sequence_adjustment.py
"""
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix

ANNOTATION_PKL = "/home/tanmay-ura/LAGCN/data/annotations/test.pkl"
ORIGINAL_PKL   = "/home/tanmay-ura/LAGCN/work_dir/hockey/joint_motion_fusion_w30_accel_CUDNN/epoch39_test_score_remapped.pkl"
ADJUSTED_PKL   = "/home/tanmay-ura/LAGCN/work_dir/hockey/joint_motion_fusion_w30_accel_CUDNN/epoch39_test_score_remapped_adjusted_w0.1.pkl"

ACTION_LABELS = [
    "GLID_FORW", "ACCEL_FORW", "GLID_BACK", "ACCEL_BACK",
    "TRANS_FORW_TO_BACK", "TRANS_BACK_TO_FORW", "POST_WHISTLE_GLIDING",
    "FACEOFF_BODY_POSITION", "MAINTAIN_POSITION", "PRONE", "ON_A_KNEE",
]

# Load ground truth from annotation pkl (same filtering as feeder)
with open(ANNOTATION_PKL, "rb") as f:
    content = pickle.load(f)
annotations = content["annotations"] if isinstance(content, dict) and "annotations" in content else content

gt = {}  # frame_dir -> true label
for sample in annotations:
    lbl = sample.get("label", -1)
    if lbl > 10 or lbl < 0:
        continue
    if "keypoint" not in sample or sample["keypoint"] is None:
        continue
    gt[sample["frame_dir"]] = int(lbl)

# Load predictions
with open(ORIGINAL_PKL, "rb") as f:
    orig_scores = pickle.load(f)
with open(ADJUSTED_PKL, "rb") as f:
    adj_scores = pickle.load(f)

# Build aligned lists
keys = [k for k in gt if k in orig_scores and k in adj_scores]
missing = len(gt) - len(keys)
if missing:
    print(f"Warning: {missing} samples missing from predictions, skipping them.")

y_true   = np.array([gt[k] for k in keys])
y_orig   = np.array([int(np.argmax(orig_scores[k])) for k in keys])
y_adj    = np.array([adj_scores[k]["pred"] for k in keys])

def per_class_acc(y_true, y_pred, n=11):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    per_cls = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    return per_cls, cm

orig_cls, orig_cm = per_class_acc(y_true, y_orig)
adj_cls,  adj_cm  = per_class_acc(y_true, y_adj)

orig_top1  = np.mean(y_true == y_orig) * 100
adj_top1   = np.mean(y_true == y_adj)  * 100
orig_mean  = np.mean(orig_cls) * 100
adj_mean   = np.mean(adj_cls)  * 100

print(f"\n{'':30s}  {'Original':>10}  {'Adjusted':>10}  {'Delta':>8}")
print("-" * 62)
print(f"{'Top-1 accuracy':30s}  {orig_top1:>9.2f}%  {adj_top1:>9.2f}%  {adj_top1-orig_top1:>+7.2f}%")
print(f"{'Mean class accuracy':30s}  {orig_mean:>9.2f}%  {adj_mean:>9.2f}%  {adj_mean-orig_mean:>+7.2f}%")
print()
print(f"{'Per-class accuracy':30s}  {'Original':>10}  {'Adjusted':>10}  {'Delta':>8}")
print("-" * 62)
for i, name in enumerate(ACTION_LABELS):
    o, a = orig_cls[i]*100, adj_cls[i]*100
    print(f"  {name:28s}  {o:>9.2f}%  {a:>9.2f}%  {a-o:>+7.2f}%")

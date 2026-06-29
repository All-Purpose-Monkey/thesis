"""
gen_test_preds.py — dump per-segment predictions on the IN-DISTRIBUTION test set
for the baseline + your best-5 configs, in the SAME CSV format as heldout_test.py.

No training. Loads each saved .pth + thresholds, runs inference, writes one CSV per
model. Feed those CSVs straight into bootstrap_ci.py.

It reproduces the exact 3-house test split by copying the preprocessing + split
block verbatim from compression_ablation.py / cnn_baseline.py (iterative_train_test_split
is deterministic, so the split is identical to the one the models were evaluated on).

Run on the server from the repo root:
    python gen_test_preds.py

>>> BEFORE RUNNING, CHECK TWO THINGS (marked with  # CHECK  below):
    1. The 5 config tags in CONFIGS match folder names under
       ~/thesis/results/cnn_mini/parameter_ablation/compression/
    2. The CNNmini(...) construction line matches your model signature
       (copied here exactly from your working heldout_test.py).
"""

import os
import numpy as np
import pandas as pd
import yaml
import torch
from torch.utils.data import DataLoader
from skmultilearn.model_selection import iterative_train_test_split

import preprocess
from data.dataset import STFTDataset
from data.loader import remove_nan_rows
from models.backbone import CNNBackbone, CNNmini
from models.heads import MultiHeadClassifier, MultiHeadClassifier_mini

# =========================================================
# WHAT TO RUN  — edit only this block
# =========================================================

OUT_DIR = os.path.expanduser("~/thesis/results/bootstrap_inputs/")

# Baseline (CNNBackbone + MultiHeadClassifier)
BASELINE = {
    "name":  "baseline",
    "model": os.path.expanduser("~/thesis/results/cnn_baseline/focal_loss/cnn_baseline_classifier.pth"),
    "thr":   os.path.expanduser("~/thesis/results/cnn_baseline/focal_loss/cnn_baseline_best_thresholds.csv"),
}

# Best-5 configs (CNNmini + MultiHeadClassifier_mini). tag = folder name.   # CHECK
COMP_DIR = os.path.expanduser("~/thesis/results/cnn_mini/parameter_ablation/compression/")
CONFIGS = [
    "pool_9_1_stride_3_kernel_5",
    "pool_9_1_stride_3_kernel_7",
    "pool_6_1_stride_3_kernel_7",
    "pool_6_1_stride_2_kernel_7",
    "pool_6_3_stride_3_kernel_7",
]
K_NUM, PADDING, SYM_PAD = 16, 0, False   # fixed values used in compression_ablation.py

# =========================================================

with open(os.path.expanduser("~/thesis/configs/cnn_baseline.yaml")) as f:
    cfg = yaml.safe_load(f)
APPS = cfg["appliances"]

device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")
os.makedirs(OUT_DIR, exist_ok=True)


def parse_tag(tag):
    """pool_9_1_stride_3_kernel_5 -> pool_size=(9,1), stride=3, kernel=5"""
    p = tag.split("_")
    return (int(p[1]), int(p[2])), int(p[4]), int(p[6])


# -------------------------
# REPRODUCE THE 3-HOUSE TEST SPLIT  (verbatim from compression_ablation.py)
# -------------------------
print("\n--- Rebuilding in-distribution test split ---")
X1, y1 = preprocess.mash_that("~/thesis/data/house_1/house1_binarized.csv", "~/thesis/data/house_1/stft_segments/2014/wk42/", path=True)
X2, y2 = preprocess.mash_that("~/thesis/data/house_2/house2_binarized.csv", "~/thesis/data/house_2/stft_segments/2013/wk38/", path=True)
X5, y5 = preprocess.mash_that("~/thesis/data/house_5/house5_binarized.csv", "~/thesis/data/house_5/stft_segments/2014/wk29/", path=True)

Big_X = np.concatenate([np.array(X1), np.array(X2), np.array(X5)], axis=0)
Big_y = np.concatenate([np.array(y1), np.array(y2), np.array(y5)], axis=0)

X_clean, y_clean = remove_nan_rows(Big_X, Big_y)
X_clean, y_clean = np.array(X_clean), np.array(y_clean)

X_train, y_train, X_test, y_test = iterative_train_test_split(
    X_clean, y_clean, test_size=cfg["data"]["test_size"]
)
print(f"Test size: {len(X_test)}   active/appliance: {y_test.sum(axis=0)}")

test_loader = DataLoader(
    STFTDataset(X_test, y_test),
    batch_size=cfg["downstream"]["batch_size"],
    shuffle=False, num_workers=4, pin_memory=True,
)


def load_thresholds(path):
    tdf = pd.read_csv(path, index_col=0)
    return {a: float(tdf.loc[a]["threshold"]) for a in APPS}


def err_type(t, p):
    return "TP" if (t and p) else "FP" if (p and not t) else "FN" if (t and not p) else "TN"


def run_and_dump(classifier, thresholds, out_name):
    classifier.eval()
    probs_all, targ_all = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            probs_all.append(torch.sigmoid(classifier(Xb.to(device))).cpu().numpy())
            targ_all.append(yb.numpy())
    probs = np.concatenate(probs_all, axis=0)
    targ = np.concatenate(targ_all, axis=0)
    thr = np.array([thresholds[a] for a in APPS])
    preds = (probs >= thr).astype(int)

    rows = []
    for s in range(len(probs)):
        row = {"timestamp": s}  # no real timestamp for in-dist split; row index keeps eda_heldout happy
        for i, a in enumerate(APPS):
            t, p, pr = int(targ[s, i]), int(preds[s, i]), float(probs[s, i])
            row[f"{a}_true"] = t
            row[f"{a}_pred"] = p
            row[f"{a}_prob"] = round(pr, 4)
            row[f"{a}_conf_diff"] = round(pr - thresholds[a], 4)
            row[f"{a}_error"] = err_type(t, p)
        rows.append(row)
    out = os.path.join(OUT_DIR, out_name)
    pd.DataFrame(rows).set_index("timestamp").to_csv(out)
    print(f"  saved {out}  ({len(rows)} rows)")


# -------------------------
# BASELINE
# -------------------------
print("\n=== baseline ===")
try:
    bb = CNNBackbone()
    clf = MultiHeadClassifier(bb, APPS).to(device)
    clf.load_state_dict(torch.load(BASELINE["model"], map_location=device))
    run_and_dump(clf, load_thresholds(BASELINE["thr"]), "baseline_indist_test.csv")
    del clf, bb
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
except Exception as e:
    print(f"  SKIPPED baseline: {e}")

# -------------------------
# BEST-5 CONFIGS
# -------------------------
for tag in CONFIGS:
    print(f"\n=== {tag} ===")
    try:
        pool, stride, kernel = parse_tag(tag)
        # CHECK: this line is copied from your working heldout_test.py
        bb = CNNmini(k_num=K_NUM, pool_size=pool, stride=stride, kernel=kernel,
                     padding=PADDING, sym_pad=SYM_PAD)
        clf = MultiHeadClassifier_mini(bb, APPS).to(device)
        model_path = os.path.join(COMP_DIR, tag, f"classifier_{tag}.pth")
        thr_path = os.path.join(COMP_DIR, tag, f"thresholds_{tag}.csv")
        clf.load_state_dict(torch.load(model_path, map_location=device))
        run_and_dump(clf, load_thresholds(thr_path), f"{tag}_indist_test.csv")
        del clf, bb
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    except Exception as e:
        print(f"  SKIPPED {tag}: {e}")

print(f"\nDone. CSVs in {OUT_DIR}")
print("Next: python bootstrap_ci.py <each csv> --n-boot 10000")
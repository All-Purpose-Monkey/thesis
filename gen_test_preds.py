"""
gen_test_preds_lazy.py — memory-light version of gen_test_preds.py.

Problem: preprocess.mash_that eagerly np.load()s every STFT into RAM (~50 GB for
3 houses), so reproducing the split won't fit on a head node.

This version never holds more than one spectrogram in memory:
  Phase 1  stream every segment once, in mash_that's exact traversal order,
           drop NaN rows (same rule as remove_nan_rows), keep only (path, label).
  Phase 2  reproduce the IDENTICAL iterative_train_test_split using the kept
           labels + a dummy index (the split decision uses y only), recover the
           TEST file paths.
  Phase 3  lazily load only the ~25k test spectrograms, batch by batch, run each
           model, and dump the per-segment CSV in heldout_test.py format.

Per-item transform copied from gradcam_analysis.load_spec (proven against the model).
Drop rule copied from heldout_test.remove_nan_rows_tracked.

Run it directly on the head node (no GPU needed, CPU fallback is automatic):
    python gen_test_preds_lazy.py

>>> CHECK (same two as gen_test_preds.py):
    1. CONFIGS tags match folders under .../compression/
    2. the CNNmini(...) line matches your signature (copied from heldout_test.py)
"""

import os
import csv
import numpy as np
import pandas as pd
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from skmultilearn.model_selection import iterative_train_test_split

from models.backbone import  CNNmini
from models.heads import  MultiHeadClassifier_mini

# =========================================================
# WHAT TO RUN  — edit only this block
# =========================================================
OUT_DIR = os.path.expanduser("~/thesis/results/bootstrap_inputs/")

HOUSES = [  # (labels csv, stft base)  — SAME ORDER as the training concat (H1, H2, H5)
    ("~/thesis/data/house_1/house1_binarized.csv", "~/thesis/data/house_1/stft_segments/2014/wk42/"),
    ("~/thesis/data/house_2/house2_binarized.csv", "~/thesis/data/house_2/stft_segments/2013/wk38/"),
    ("~/thesis/data/house_5/house5_binarized.csv", "~/thesis/data/house_5/stft_segments/2014/wk29/"),
]

BASELINE = {
    "name":  "baseline",
    "model": os.path.expanduser("~/thesis/results/cnn_mini/focal_loss/cnn_baseline_classifier.pth"),
    "thr":   os.path.expanduser("~/thesis/results/cnn_mini/focal_loss/cnn_baseline_best_thresholds.csv"),
}

COMP_DIR = os.path.expanduser("~/thesis/results/cnn_mini/parameter_ablation/compression/")
CONFIGS = [                                    # CHECK these folder names exist
    "pool_9_1_stride_3_kernel_5",
    "pool_9_1_stride_3_kernel_7",
    "pool_6_1_stride_3_kernel_7",
    "pool_6_1_stride_2_kernel_7",
    "pool_6_3_stride_3_kernel_7",
]
K_NUM, PADDING, SYM_PAD = 16, 0, False
BATCH = 256
# =========================================================

with open(os.path.expanduser("~/thesis/configs/cnn_baseline.yaml")) as f:
    cfg = yaml.safe_load(f)
APPS = cfg["appliances"]
TEST_SIZE = cfg["data"]["test_size"]

device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")
os.makedirs(OUT_DIR, exist_ok=True)


# -------------------------
# PHASE 1 — stream all segments, drop NaN, keep (path, label)
# (mirrors mash_that traversal + remove_nan_rows, constant memory)
# -------------------------
def stream_items():
    items = []  # (path, label_array) in exact mash_that order, NaN rows dropped
    for csv_path, stft_base in HOUSES:
        csv_path, stft_base = os.path.expanduser(csv_path), os.path.expanduser(stft_base)
        labels = {}
        with open(csv_path) as fh:
            r = csv.reader(fh); next(r)
            for row in r:
                labels[int(float(row[0]))] = [float(x) for x in row[1:]]  # positional, like mash_that
        n_house = 0
        for root_folder in sorted(os.listdir(stft_base)):
            root_path = os.path.join(stft_base, root_folder)
            if not os.path.isdir(root_path):
                continue
            for fname in sorted(os.listdir(root_path)):
                if not fname.endswith(".npy"):
                    continue
                try:
                    ts = int(float(fname.replace(".npy", "")))
                except ValueError:
                    continue
                if ts not in labels:
                    continue
                path = os.path.join(root_path, fname)
                arr = np.load(path)                      # loaded only to NaN-check, then discarded
                lab = np.array(labels[ts], dtype=np.float32)
                if np.isnan(arr).any() or np.isnan(lab).any():
                    continue                              # remove_nan_rows rule
                items.append((path, lab))
                n_house += 1
                if n_house % 20000 == 0:
                    print(f"  ...{n_house} kept from {os.path.basename(os.path.dirname(stft_base.rstrip('/')))}")
        print(f"  house done: {n_house} kept segments")
    print(f"Total kept (post-NaN): {len(items)}")
    return items


# -------------------------
# PHASE 3 — lazy dataset (transform copied from gradcam load_spec)
# -------------------------
class LazySTFT(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, lab = self.items[i]
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis, :]                      # (1, F, T)
        return torch.tensor(arr), torch.tensor(lab, dtype=torch.float32)


def load_thresholds(path):
    tdf = pd.read_csv(path, index_col=0)
    return {a: float(tdf.loc[a]["threshold"]) for a in APPS}


def err_type(t, p):
    return "TP" if (t and p) else "FP" if (p and not t) else "FN" if (t and not p) else "TN"


def run_and_dump(classifier, thresholds, loader, out_name):
    classifier.eval()
    probs_all, targ_all = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            probs_all.append(torch.sigmoid(classifier(Xb.to(device))).cpu().numpy())
            targ_all.append(yb.numpy())
    probs = np.concatenate(probs_all, axis=0)
    targ = np.concatenate(targ_all, axis=0)
    thr = np.array([thresholds[a] for a in APPS])
    preds = (probs >= thr).astype(int)

    rows = []
    for s in range(len(probs)):
        row = {"timestamp": s}                            # row index; eda_heldout only needs a unique index
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


def parse_tag(tag):
    p = tag.split("_")
    return (int(p[1]), int(p[2])), int(p[4]), int(p[6])


# =========================================================
print("\n--- Phase 1: streaming segments (constant memory) ---")
items = stream_items()

print("\n--- Phase 2: reproducing the test split (labels only) ---")
y = np.stack([lab for _, lab in items])                   # (N, 6)
idx = np.arange(len(items)).reshape(-1, 1)
_, _, test_idx, _ = iterative_train_test_split(idx, y, test_size=TEST_SIZE)
test_idx = test_idx.flatten()
test_items = [items[i] for i in test_idx]
print(f"Test segments: {len(test_items)}   active/appliance: {y[test_idx].sum(axis=0)}")

test_loader = DataLoader(LazySTFT(test_items), batch_size=BATCH,
                         shuffle=False, num_workers=0, pin_memory=False)

# -------------------------
# BASELINE
# -------------------------
print("\n=== baseline ===")
try:
    bb = CNNmini()
    clf = MultiHeadClassifier_mini(bb, APPS).to(device)
    clf.load_state_dict(torch.load(BASELINE["model"], map_location=device))
    run_and_dump(clf, load_thresholds(BASELINE["thr"]), test_loader, "baseline_indist_test.csv")
    del clf, bb
except Exception as e:
    print(f"  SKIPPED baseline: {e}")

# -------------------------
# BEST-5 CONFIGS
# -------------------------
for tag in CONFIGS:
    print(f"\n=== {tag} ===")
    try:
        pool, stride, kernel = parse_tag(tag)
        # CHECK: copied from your working heldout_test.py
        bb = CNNmini(k_num=K_NUM, pool_size=pool, stride=stride, kernel_size=kernel,
                     padding=PADDING, sym_pad=SYM_PAD)
        clf = MultiHeadClassifier_mini(bb, APPS).to(device)
        clf.load_state_dict(torch.load(os.path.join(COMP_DIR, tag, f"classifier_{tag}.pth"),
                                       map_location=device))
        run_and_dump(clf, load_thresholds(os.path.join(COMP_DIR, tag, f"thresholds_{tag}.csv")),
                     test_loader, f"{tag}_indist_test.csv")
        del clf, bb
    except Exception as e:
        print(f"  SKIPPED {tag}: {e}")

print(f"\nDone. CSVs in {OUT_DIR}")
print("Next (locally): python bootstrap_ci.py <each csv> --n-boot 10000")
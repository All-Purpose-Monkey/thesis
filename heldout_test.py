import yaml
import torch
import numpy as np
import pandas as pd
import preprocess
import os
import csv
from torch.utils.data import DataLoader
from data.dataset import STFTDataset
from data.loader import remove_nan_rows
from models.backbone import CNNmini
from models.heads import MultiHeadClassifier_mini
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# -------------------------
# CONFIG
# -------------------------

with open(os.path.expanduser("~/thesis/configs/cnn_baseline.yaml")) as f:
    cfg = yaml.safe_load(f)

# -------------------------
# PATHS
# -------------------------

MODEL_PATH = os.path.expanduser(
    "~/thesis/results/cnn_mini/parameter_ablation/compression/"
    "pool_9_1_stride_3_kernel_7/classifier_pool_9_1_stride_3_kernel_7.pth"
)

THRESHOLD_PATH = os.path.expanduser(
    "~/thesis/results/cnn_mini/parameter_ablation/compression/"
    "pool_9_1_stride_3_kernel_7/thresholds_pool_9_1_stride_3_kernel_7.csv"
)

LABELS_PATH  = "/home/u738865/thesis/data/test_set_2017/house1_jan2017_binarized.csv"
STFT_PATH    = "/home/u738865/thesis/data/test_set_2017/house_1/stft_segments/2017/wk04/current/"

MODEL_NAME        = "pool_9_1_stride_3_kernel_7"
OUTPUT_DIR        = os.path.expanduser("~/thesis/results/heldout/")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH       = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_heldout_test.csv")
FAILURE_PATH      = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_heldout_failure_analysis.csv")

# -------------------------
# DEVICE
# -------------------------

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# -------------------------
# LOAD HELD-OUT DATA
# -------------------------

print("\n--- Loading held-out test data ---")

X_held, y_held = preprocess.mash_that(LABELS_PATH, STFT_PATH, path=True)

# Collect timestamps in the same traversal order as mash_that.
# mash_that matches .npy filenames (which are Unix timestamps) against the CSV,
# so we replay the same sorted traversal to get the aligned timestamp vector.
def collect_timestamps(stft_base, labels_csv_path):
    """Mirror mash_that's traversal to return timestamps in the same order as X, y."""
    valid_timestamps = set()
    with open(labels_csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            valid_timestamps.add(int(float(row[0])))

    timestamps = []
    stft_base = os.path.expanduser(stft_base)
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
            if ts in valid_timestamps:
                timestamps.append(ts)
    return timestamps

raw_timestamps = collect_timestamps(STFT_PATH, LABELS_PATH)

X_held = np.array(X_held)
y_held = np.array(y_held)

print(f"Raw held-out: X={X_held.shape}, y={y_held.shape}, timestamps={len(raw_timestamps)}")
print(f"Active per appliance: {y_held.sum(axis=0)}")

# remove_nan_rows may drop rows — track which indices survive to keep timestamps aligned
def remove_nan_rows_tracked(X, y):
    """Returns cleaned arrays and the boolean mask of kept rows."""
    mask = []
    for i in range(len(X)):
        has_nan = np.isnan(X[i]).any() or np.isnan(y[i]).any()
        mask.append(not has_nan)
    mask = np.array(mask)
    return np.array(X)[mask], np.array(y)[mask], mask

X_held, y_held, kept_mask = remove_nan_rows_tracked(X_held, y_held)
timestamps = np.array(raw_timestamps)[kept_mask]

print(f"After NaN removal: {len(X_held)} segments ({kept_mask.sum()} kept, {(~kept_mask).sum()} dropped)")

# -------------------------
# DATASET & LOADER
# -------------------------

held_dataset = STFTDataset(X_held, y_held)

held_loader = DataLoader(
    held_dataset,
    batch_size=cfg["downstream"]["batch_size"],
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

# -------------------------
#model loading
# -------------------------

print("\n--- Loading model ---")

backbone   = CNNmini(k_num=16, pool_size=(9, 1), stride=3, kernel_size=7, padding=0, sym_pad=False)
classifier = MultiHeadClassifier_mini(backbone, cfg["appliances"]).to(device)
classifier.load_state_dict(torch.load(MODEL_PATH, map_location=device))
classifier.eval()

n_params = sum(p.numel() for p in classifier.parameters())
print(f"Loaded: {MODEL_PATH}")
print(f"Params: {n_params:,}  |  embed dim: {backbone.output_dim()}")

# -------------------------
# LOAD THRESHOLDS
# -------------------------

print("\n--- Loading thresholds ---")

threshold_df = pd.read_csv(THRESHOLD_PATH, index_col=0)
inference_thresholds = {
    app: float(threshold_df.loc[app]["threshold"])
    for app in cfg["appliances"]
}
print(inference_thresholds)

# -------------------------
# INFERENCE
# -------------------------

print("\n--- Running inference on held-out set ---")

all_probs, all_targets = [], []

with torch.no_grad():
    for X_batch, y_batch in held_loader:
        X_batch = X_batch.to(device)
        probs   = torch.sigmoid(classifier(X_batch))
        all_probs.append(probs.cpu().numpy())
        all_targets.append(y_batch.numpy())

all_probs   = np.concatenate(all_probs,   axis=0)
all_targets = np.concatenate(all_targets, axis=0)

thresh_arr   = np.array([inference_thresholds[a] for a in cfg["appliances"]])
binary_preds = (all_probs >= thresh_arr).astype(int)

# -------------------------
# METRICS
# -------------------------

print("\n--- Computing metrics ---")

rows = []
for i, app in enumerate(cfg["appliances"]):
    rows.append({
        "appliance": app,
        "f1":        f1_score(all_targets[:, i],     binary_preds[:, i], zero_division=0),
        "precision": precision_score(all_targets[:, i], binary_preds[:, i], zero_division=0),
        "recall":    recall_score(all_targets[:, i],    binary_preds[:, i], zero_division=0),
        "roc_auc":   roc_auc_score(all_targets[:, i],   all_probs[:, i]),
    })

rows.append({
    "appliance": "TOTAL (macro avg)",
    "f1":        np.mean([r["f1"]        for r in rows]),
    "precision": np.mean([r["precision"] for r in rows]),
    "recall":    np.mean([r["recall"]    for r in rows]),
    "roc_auc":   np.mean([r["roc_auc"]  for r in rows]),
})

results_df = pd.DataFrame(rows).set_index("appliance")

print("\n========== HELD-OUT TEST RESULTS ==========")
print(f"Model:  {MODEL_NAME}")
print(f"Data:   {STFT_PATH}")
print(f"Segs:   {len(X_held)}")
print()
print(results_df.to_string())

results_df.to_csv(OUTPUT_PATH)
print(f"\nSaved to: {OUTPUT_PATH}")

# -------------------------
# FAILURE ANALYSIS — per segment
# -------------------------

print("\n--- Building failure analysis ---")

def error_type(true, pred):
    if true == 1 and pred == 1: return "TP"
    if true == 0 and pred == 1: return "FP"
    if true == 1 and pred == 0: return "FN"
    return "TN"

failure_rows = []
for seg_idx in range(len(timestamps)):
    row = {"timestamp": timestamps[seg_idx]}
    for i, app in enumerate(cfg["appliances"]):
        t     = int(all_targets[seg_idx, i])
        p     = int(binary_preds[seg_idx, i])
        prob  = float(all_probs[seg_idx, i])
        thr   = inference_thresholds[app]
        row[f"{app}_true"]      = t
        row[f"{app}_pred"]      = p
        row[f"{app}_prob"]      = round(prob, 4)
        row[f"{app}_conf_diff"] = round(prob - thr, 4)   # >0 = called positive, <0 = called negative
        row[f"{app}_error"]     = error_type(t, p)
    failure_rows.append(row)

failure_df = pd.DataFrame(failure_rows).set_index("timestamp")
failure_df.sort_index(inplace=True)

failure_df.to_csv(FAILURE_PATH)
print(f"Saved failure analysis to: {FAILURE_PATH}")

# -------------------------
# FAILURE SUMMARY (printed)
# -------------------------

print("\n========== FAILURE SUMMARY ==========")
for app in cfg["appliances"]:
    col    = f"{app}_error"
    counts = failure_df[col].value_counts()
    fp     = counts.get("FP", 0)
    fn     = counts.get("FN", 0)
    tp     = counts.get("TP", 0)
    tn     = counts.get("TN", 0)
    total  = fp + fn

    # confident wrong: abs(conf_diff) > 0.2 and error in FP/FN
    mask_fp = (failure_df[col] == "FP")
    mask_fn = (failure_df[col] == "FN")
    conf_fp = failure_df.loc[mask_fp, f"{app}_conf_diff"].mean()
    conf_fn = failure_df.loc[mask_fn, f"{app}_conf_diff"].mean()

    print(f"  {app:20s}  TP={tp:4d}  TN={tn:5d}  FP={fp:4d}  FN={fn:4d}  "
          f"total_errors={total:4d}  "
          f"mean_conf_diff@FP={conf_fp:+.3f}  mean_conf_diff@FN={conf_fn:+.3f}")

print(f"\nColumns in failure CSV: timestamp + {len(cfg['appliances'])*5} per-appliance fields")
print(f"fields per appliance: _true  _pred  _prob  _conf_diff  _error")
import yaml
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import preprocess
import time
from torch.utils.data import DataLoader
import os
from data.dataset import STFTDataset
from data.loader import remove_nan_rows
from utils.metrics import threshold_sweep
from models.backbone import CNNmini
from models.heads import MultiHeadClassifier_mini
from training.downstream import train_downstream
from training.loss import classification_loss
from utils.logging import save_history
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from skmultilearn.model_selection import iterative_train_test_split

# =========================================================
# ABLATION SETTINGS  — edit only this block
# =========================================================

# --- fixed ---
PADDINGS  = 0
SYM_PADS  = False
K_NUM     = 16
POOL_SIZE = (1, 1)
KERNEL    = 3

# --- swept ---
STRIDE = (1, 2, 3)

BASE_OUTPUT_DIR = os.path.expanduser(
    "~/thesis/results/cnn_mini/parameter_ablation/stride/"
)
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
print(f"Base output directory: {BASE_OUTPUT_DIR}")

# -------------------------
# LOAD CONFIG
# -------------------------

with open(os.path.expanduser("~/thesis/configs/cnn_baseline.yaml")) as f:
    cfg = yaml.safe_load(f)

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
# PREPROCESSING  (once, outside the loop)
# -------------------------

X1, y1 = preprocess.mash_that("~/thesis/data/house_1/house1_binarized.csv",  "~/thesis/data/house_1/stft_segments/2014/wk42/", path=True)
X2, y2 = preprocess.mash_that("~/thesis/data/house_2/house2_binarized.csv",  "~/thesis/data/house_2/stft_segments/2013/wk38/", path=True)
X5, y5 = preprocess.mash_that("~/thesis/data/house_5/house5_binarized.csv",  "~/thesis/data/house_5/stft_segments/2014/wk29/", path=True)

Big_X = np.concatenate([np.array(X1), np.array(X2), np.array(X5)], axis=0)
Big_y = np.concatenate([np.array(y1), np.array(y2), np.array(y5)], axis=0)

print(f"X shape: {len(Big_X)}, y shape: {len(Big_y)}")
print(f"Example X[0] shape: {Big_X[0].shape}, y[0]: {Big_y[0]}")
print(f"Active samples per appliance: {Big_y.sum(axis=0)}")

# -------------------------
# CLEAN
# -------------------------

X_clean, y_clean = remove_nan_rows(Big_X, Big_y)
X_clean = np.array(X_clean)
y_clean = np.array(y_clean)

# -------------------------
# STRATIFIED SPLIT  (once, outside the loop)
# -------------------------

X_train, y_train, X_test, y_test = iterative_train_test_split(
    X_clean,
    y_clean,
    test_size=cfg["data"]["test_size"]
)

print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
print(f"Train active per appliance: {y_train.sum(axis=0)}")
print(f"Test  active per appliance: {y_test.sum(axis=0)}")

# -------------------------
# POS WEIGHTS  (once, from train split)
# -------------------------

y_arr      = np.array(y_train)
pos_counts = y_arr.sum(axis=0)
neg_counts = len(y_arr) - pos_counts
pos_weights = neg_counts / (pos_counts + 1e-6)
pos_weights = np.clip(pos_weights, 0, 10.0)
pos_weights = torch.tensor(pos_weights, dtype=torch.float32).to(device)

# -------------------------
# DATASETS & LOADERS  (once, outside the loop)
# -------------------------

train_dataset = STFTDataset(X_train, y_train)
test_dataset  = STFTDataset(X_test,  y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=cfg["downstream"]["batch_size"],
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=cfg["downstream"]["batch_size"],
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# =========================================================
# STRIDE ABLATION LOOP
# =========================================================

all_summary_rows = []
ablation_start = time.time()

for stride in STRIDE:

    config_start = time.time()
    tag          = f"pool_{POOL_SIZE[0]}_{POOL_SIZE[1]}_stride_{stride}_kernel_{KERNEL}"
    OUTPUT_DIR   = os.path.join(BASE_OUTPUT_DIR, tag)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  stride: {stride}  |  tag: {tag}")
    print(f"  fixed — k_num: {K_NUM}  pool: {POOL_SIZE}  kernel: {KERNEL}  padding: {PADDINGS}  sym_pad: {SYM_PADS}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"{'='*60}")

    # -------------------------
    # MODEL
    # -------------------------

    backbone = CNNmini(
        k_num=K_NUM,
        pool_size=POOL_SIZE,
        stride=stride,
        padding=PADDINGS,
        kernel_size=(KERNEL, KERNEL),
        sym_pad=SYM_PADS,
    )
    classifier = MultiHeadClassifier_mini(
        backbone,
        cfg["appliances"],
        k_num=K_NUM,
    ).to(device)

    n_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params:,}  |  embed dim: {backbone.output_dim()}")

    optimizer_class = optim.AdamW(
    classifier.parameters(),
    lr=cfg["downstream"]["lr"],
    weight_decay=cfg["downstream"]["weight_decay"],
    )

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_class,
        T_max=cfg["downstream"]["epochs"]
    )
    '''
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_class,
        patience=3,
        factor=0.5,
    )
    '''
    loss_fn = classification_loss(pos_weights=pos_weights, focal=True, gamma=2.0)

    # -------------------------
    # STAGE 1 — TRAIN WITHOUT THRESHOLDS
    # -------------------------

    print("\n--- Stage 1: Supervised training (no thresholds) ---")

    history = train_downstream(
        classifier,
        train_loader,
        test_loader,
        optimizer_class,
        loss_fn,
        device,
        cfg["appliances"],
        cfg["downstream"]["epochs"],
        thresholds=None,
        scheduler=cosine_scheduler,
        early_stopping_patience=5
    )

    train_time_min = (time.time() - config_start) / 60
    print(f"  Training time: {train_time_min:.1f} min")

    save_history(history, os.path.join(OUTPUT_DIR, f"history_{tag}.csv"))
    torch.save(
        classifier.state_dict(),
        os.path.join(OUTPUT_DIR, f"classifier_{tag}.pth")
    )

    # -------------------------
    # STAGE 2 — THRESHOLD SWEEP  (on train set)
    # -------------------------

    print(f"\n  --- Stage 2: Threshold sweep ({tag}) ---")

    all_preds, all_targets = [], []
    classifier.eval()

    with torch.no_grad():
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            probs   = torch.sigmoid(classifier(X_batch))
            all_preds.append(probs.cpu().numpy())
            all_targets.append(y_batch.numpy())

    all_preds   = np.concatenate(all_preds,   axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    best_thresholds, sweep_curves = threshold_sweep(
        all_preds, all_targets, cfg["appliances"]
    )

    pd.DataFrame(best_thresholds).T.to_csv(
        os.path.join(OUTPUT_DIR, f"thresholds_{tag}.csv")
    )

    inference_thresholds = {
        app: float(best_thresholds[app]["threshold"])
        for app in cfg["appliances"]
    }
    print(f"  Thresholds: {inference_thresholds}")

    # -------------------------
    # FINAL TEST EVALUATION
    # -------------------------

    print(f"\n  --- Final Evaluation ({tag}) ---")

    all_preds, all_targets = [], []
    classifier.eval()

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            probs   = torch.sigmoid(classifier(X_batch))
            all_preds.append(probs.cpu().numpy())
            all_targets.append(y_batch.numpy())

    all_preds    = np.concatenate(all_preds,   axis=0)
    all_targets  = np.concatenate(all_targets, axis=0)
    thresh_arr   = np.array([inference_thresholds[a] for a in cfg["appliances"]])
    binary_preds = (all_preds >= thresh_arr).astype(int)

    rows = []
    for i, app in enumerate(cfg["appliances"]):
        rows.append({
            "appliance": app,
            "f1":        f1_score(       all_targets[:, i], binary_preds[:, i], zero_division=0),
            "precision": precision_score(all_targets[:, i], binary_preds[:, i], zero_division=0),
            "recall":    recall_score(   all_targets[:, i], binary_preds[:, i], zero_division=0),
            "roc_auc":   roc_auc_score(  all_targets[:, i], all_preds[:, i]),
        })

    macro_f1  = np.mean([r["f1"]      for r in rows])
    macro_auc = np.mean([r["roc_auc"] for r in rows])

    rows.append({
        "appliance": "TOTAL (macro avg)",
        "f1":        macro_f1,
        "precision": np.mean([r["precision"] for r in rows]),
        "recall":    np.mean([r["recall"]    for r in rows]),
        "roc_auc":   macro_auc,
    })

    results_df = pd.DataFrame(rows).set_index("appliance")
    print(f"\n========== {tag} TEST RESULTS ==========")
    print(results_df.to_string())

    results_path = os.path.join(OUTPUT_DIR, f"test_results_{tag}.csv")
    results_df.to_csv(results_path)
    print(f"  Saved: {results_path}")

    config_elapsed_min = (time.time() - config_start) / 60
    print(f"  Total config time (incl. eval): {config_elapsed_min:.1f} min")

    all_summary_rows.append({
        "config":          tag,
        "stride":          stride,
        "pool_size":       str(POOL_SIZE),
        "kernel":          KERNEL,
        "n_params":        n_params,
        "embed_dim":       backbone.output_dim(),
        "macro_f1":        round(macro_f1,  4),
        "macro_auc":       round(macro_auc, 4),
        "train_time_min":  round(config_elapsed_min, 1),
    })

    del classifier, backbone, optimizer_class, cosine_scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# =========================================================
# SUMMARY
# =========================================================

total_elapsed_hr = (time.time() - ablation_start) / 3600

summary_df   = pd.DataFrame(all_summary_rows).set_index("config")
summary_path = os.path.join(BASE_OUTPUT_DIR, "stride_ablation_summary.csv")
summary_df.to_csv(summary_path)

print(f"\n{'='*60}")
print("  STRIDE ABLATION COMPLETE")
print(f"  Total time: {total_elapsed_hr:.2f} hrs")
print(f"{'='*60}")
print(summary_df.to_string())
print(f"\nSummary saved to: {summary_path}")
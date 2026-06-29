import yaml
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import preprocess
from torch.utils.data import DataLoader
import os
from data.dataset import STFTDataset
from data.loader import remove_nan_rows
from utils.metrics import threshold_sweep, compute_metrics
from models.backbone import CNNBackbone, CNNmini
from models.heads import MultiHeadClassifier_mini
from training.downstream import train_downstream
from training.loss import classification_loss
from utils.logging import save_history
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from skmultilearn.model_selection import iterative_train_test_split
 
# -------------------------
# POOL ABLATION CONFIGS
# (height, width) — i.e. (freq_bins, time_bins)
# baseline full model uses (12, 6)
# -------------------------
 
K_NUM = 16  # fix channel depth; swap to your RQ2a winner later
 
POOL_CONFIGS = [
    (3,  2),   # frequency heavy
    (6,  3),   
    (12, 6),   
    (3,  3),   # square 
    (6,  6),   
    (9, 9),  
    (3, 1),   # freq only
    (6,  1),
    (9,1),
    (12, 1),
    (1,3),  #time only
    (1,6),  
    (1,9),
    (1,12),
    (2,3), #time heavy
    (3,6),
    (6,12),
]
 
BASE_OUTPUT_DIR = os.path.expanduser("~/thesis/results/cnn_mini/pool_ablation/")
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
 
X1, y1 = preprocess.mash_that("~/thesis/data/house_1/house1_binarized.csv", "~/thesis/data/house_1/stft_segments/2014/wk42/", path=True)
X2, y2 = preprocess.mash_that("~/thesis/data/house_2/house2_binarized.csv", "~/thesis/data/house_2/stft_segments/2013/wk38/", path=True)
X5, y5 = preprocess.mash_that("~/thesis/data/house_5/house5_binarized.csv", "~/thesis/data/house_5/stft_segments/2014/wk29/", path=True)
 
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
 
y_arr = np.array(y_train)
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
# POOL ABLATION LOOP
# =========================================================
 
all_summary_rows = []  # collect one row per config for the aggregate CSV
 
for pool_h, pool_w in POOL_CONFIGS:
 
    pool_size  = (pool_h, pool_w)
    tag        = f"{pool_h}_{pool_w}"
    OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, tag)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
 
    print(f"\n{'='*60}")
    print(f"  Pool config: {pool_size}  |  k_num: {K_NUM}  |  tag: {tag}")
    print(f"  Output dir:  {OUTPUT_DIR}")
    print(f"{'='*60}")
 
    # -------------------------
    # MODEL
    # -------------------------
 
    backbone   = CNNmini(k_num=K_NUM, pool_size=pool_size)
    classifier = MultiHeadClassifier_mini(
        backbone,
        k_num=K_NUM,
        appliances= cfg["appliances"]
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

    save_history(history, os.path.join(OUTPUT_DIR, "cnn_baseline_nothresholds_history.csv"))
    torch.save(
        classifier.state_dict(),
        os.path.join(OUTPUT_DIR, "cnn_baseline_classifier.pth")
    )
 
    # -------------------------
    # STAGE 2 — THRESHOLD SWEEP
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
 
    threshold_path = os.path.join(OUTPUT_DIR, f"thresholds_{tag}.csv")
    pd.DataFrame(best_thresholds).T.to_csv(threshold_path)
 
    inference_thresholds = {
        app: float(pd.read_csv(threshold_path, index_col=0).loc[app]["threshold"])
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
            "f1":        f1_score(all_targets[:, i],     binary_preds[:, i], zero_division=0),
            "precision": precision_score(all_targets[:, i], binary_preds[:, i], zero_division=0),
            "recall":    recall_score(all_targets[:, i],    binary_preds[:, i], zero_division=0),
            "roc_auc":   roc_auc_score(all_targets[:, i],   all_preds[:, i])
        })
 
    macro_f1  = np.mean([r["f1"]      for r in rows])
    macro_auc = np.mean([r["roc_auc"] for r in rows])
 
    rows.append({
        "appliance": "TOTAL (macro avg)",
        "f1":        macro_f1,
        "precision": np.mean([r["precision"] for r in rows]),
        "recall":    np.mean([r["recall"]    for r in rows]),
        "roc_auc":   macro_auc
    })
 
    results_df = pd.DataFrame(rows).set_index("appliance")
    print(f"\n  === {tag} TEST RESULTS ===")
    print(results_df.to_string())
 
    results_path = os.path.join(OUTPUT_DIR, f"test_results_{tag}.csv")
    results_df.to_csv(results_path)
    print(f"  Saved: {results_path}")
 
    # collect for summary
    all_summary_rows.append({
        "pool_size":  str(pool_size),
        "embed_dim":  backbone.output_dim(),
        "n_params":   n_params,
        "macro_f1":   round(macro_f1,  4),
        "macro_auc":  round(macro_auc, 4),
    })
 
    # free GPU memory between runs
    del classifier, backbone, optimizer_class, cosine_scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
 
# =========================================================
# SUMMARY — one row per pool config
# =========================================================
 
summary_df = pd.DataFrame(all_summary_rows).set_index("pool_size")
summary_path = os.path.join(BASE_OUTPUT_DIR, "pool_ablation_summary.csv")
summary_df.to_csv(summary_path)
 
print(f"\n{'='*60}")
print("  POOL ABLATION COMPLETE")
print(f"{'='*60}")
print(summary_df.to_string())
print(f"\nSummary saved to: {summary_path}")
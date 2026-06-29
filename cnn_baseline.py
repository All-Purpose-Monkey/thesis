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
from training.active_learning import active_learning_round
from models.backbone import CNNBackbone
from models.heads import MultiHeadClassifier
from training.downstream import train_downstream
from training.loss import classification_loss
from utils.logging import save_history
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from skmultilearn.model_selection import iterative_train_test_split

# -------------------------
# LOAD CONFIG
# -------------------------

with open(os.path.expanduser("~/thesis/configs/cnn_baseline.yaml")) as f:
    cfg = yaml.safe_load(f)

# -------------------------
# OUTPUT DIRECTORY
# -------------------------

OUTPUT_DIR = os.path.expanduser("~/thesis/results/cnn_baseline/focal_loss/")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Saving all results to: {OUTPUT_DIR}")

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
# PREPROCESSING
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
# STRATIFIED SPLIT (iterative multilabel)
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
# POS WEIGHTS (class imbalance)
# -------------------------

y_arr = np.array(y_train)
pos_counts = y_arr.sum(axis=0)
neg_counts = len(y_arr) - pos_counts
pos_weights = neg_counts / (pos_counts + 1e-6)
pos_weights = np.clip(pos_weights, 0, 10.0)
pos_weights = torch.tensor(pos_weights, dtype=torch.float32).to(device)

# -------------------------
# DATASETS & LOADERS
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

# -------------------------
# MODEL — fully unfrozen CNN backbone + MLP head
# -------------------------

backbone = CNNBackbone()  # randomly initialised, no pretrained weights loaded

classifier = MultiHeadClassifier(
    backbone,
    cfg["appliances"]
).to(device)

# backbone stays trainable throughout — no requires_grad_(False) here

optimizer_class = optim.AdamW(
    classifier.parameters(),
    lr=cfg["downstream"]["lr"],
    weight_decay=cfg["downstream"]["weight_decay"],
)

cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_class,
    T_max=cfg["downstream"]["epochs"]
)

loss_fn = classification_loss(pos_weights=pos_weights, focal=True, gamma=2.0)

# -------------------------
# STAGE 1 — TRAIN WITHOUT THRESHOLDS
# -------------------------

print("\n--- Stage 1: Supervised training (no thresholds) ---")

history_stage1 = train_downstream(
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
    early_stopping_patience=7
)

save_history(history_stage1, os.path.join(OUTPUT_DIR, "cnn_baseline_nothresholds_history.csv"))
torch.save(
    classifier.state_dict(),
    os.path.join(OUTPUT_DIR, "cnn_baseline_classifier.pth")
)

# -------------------------
# STAGE 2 — THRESHOLD SWEEP
# -------------------------

print("\n--- Stage 2: Threshold sweep ---")

all_preds, all_targets = [], []
classifier.eval()

with torch.no_grad():
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        logits  = classifier(X_batch)
        probs   = torch.sigmoid(logits)
        all_preds.append(probs.cpu().numpy())
        all_targets.append(y_batch.numpy())

all_preds   = np.concatenate(all_preds,   axis=0)
all_targets = np.concatenate(all_targets, axis=0)

best_thresholds, sweep_curves = threshold_sweep(
    all_preds,
    all_targets,
    cfg["appliances"]
)

threshold_df   = pd.DataFrame(best_thresholds).T
threshold_path = os.path.join(OUTPUT_DIR, "cnn_baseline_best_thresholds.csv")
threshold_df.to_csv(threshold_path)
print(f"Saved thresholds to: {threshold_path}")
#simplified run for later in case you want to skip on adding extra results for validating te learning paths with optimal thresholding
threshold_df = pd.read_csv(threshold_path, index_col=0)
inference_thresholds = {
    app: float(threshold_df.loc[app]["threshold"])
    for app in cfg["appliances"]
}

print("Loaded thresholds:")
print(inference_thresholds)

# -------------------------
# STAGE 3 — REINITIALISE MLP - deprecrated i was stupid and didnt know how thresholding works and thought you need to retrain lol
# -------------------------
'''
print("\n--- Stage 3: Reinitialise classifier and reload thresholds ---")

backbone    = CNNBackbone()
classifier  = MultiHeadClassifier(backbone, cfg["appliances"]).to(device)
optimizer_class = optim.Adam(classifier.parameters(), lr=cfg["downstream"]["lr"], weight_decay=cfg["downstream"]["weight_decay"])

threshold_df = pd.read_csv(threshold_path, index_col=0)
inference_thresholds = {
    app: float(threshold_df.loc[app]["threshold"])
    for app in cfg["appliances"]
}

print("Loaded thresholds:")
print(inference_thresholds)

# -------------------------
# STAGE 4 — TRAIN WITH OPTIMAL THRESHOLDS
# -------------------------

print("\n--- Stage 4: Supervised training (with optimal thresholds) ---")

history_stage2 = train_downstream(
    classifier,
    train_loader,
    test_loader,
    optimizer_class,
    loss_fn,
    device,
    cfg["appliances"],
    cfg["downstream"]["epochs"],
    thresholds=inference_thresholds
)

save_history(history_stage2, os.path.join(OUTPUT_DIR, "cnn_baseline_thresholded_history.csv"))

torch.save(
    classifier.state_dict(),
    os.path.join(OUTPUT_DIR, "cnn_baseline_classifier_post_stage4.pth")
)

# -------------------------
# ACTIVE LEARNING - 2
# -------------------------

print("\n--- Active Learning ---")

for g in optimizer_class.param_groups:
    g["lr"] = 1e-4 #lr resent incase annealing has dropped it too low


al_pool_size  = int(0.5 * len(X_train))  # start with 50% of training data as unlabelled pool
X_al_labelled = X_train[al_pool_size:]
y_al_labelled = y_train[al_pool_size:]
X_pool        = X_train[:al_pool_size]
y_pool_oracle = y_train[:al_pool_size]

AL_ROUNDS = 10
N_QUERY   = 2000

for al_round in range(AL_ROUNDS):
    print(f"\n=== Active Learning Round {al_round + 1} ===")

    (X_al_labelled,
     y_al_labelled,
     X_pool,
     y_pool_oracle,
     queried) = active_learning_round(
        classifier,
        X_al_labelled, np.array(y_al_labelled),
        X_pool,        np.array(y_pool_oracle),
        device,
        n_query=N_QUERY,
        num_workers=4,
        pin_memory=True
    )

    al_train_dataset = STFTDataset(X_al_labelled, y_al_labelled)
    al_train_loader  = DataLoader(
        al_train_dataset,
        batch_size=cfg["downstream"]["batch_size"],
        shuffle=True
    )

    history_al = train_downstream(
        classifier,
        al_train_loader,
        test_loader,
        optimizer_class,
        loss_fn,
        device,
        cfg["appliances"],
        epochs=10,
        scheduler=None,
        early_stopping_patience=5
    )

    save_history(
        history_al,
        os.path.join(OUTPUT_DIR, f"cnn_baseline_al_round{al_round + 1}.csv")
    )

torch.save(
    classifier.state_dict(),
    os.path.join(OUTPUT_DIR, "cnn_baseline_classifier_final.pth")
)

# -------------------------
# STAGE 3 — THRESHOLD SWEEP
# -------------------------

print("\n--- Stage 2: Threshold sweep ---")

all_preds, all_targets = [], []
classifier.eval()

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        logits  = classifier(X_batch)
        probs   = torch.sigmoid(logits)
        all_preds.append(probs.cpu().numpy())
        all_targets.append(y_batch.numpy())

all_preds   = np.concatenate(all_preds,   axis=0)
all_targets = np.concatenate(all_targets, axis=0)

best_thresholds, sweep_curves = threshold_sweep(
    all_preds,
    all_targets,
    cfg["appliances"]
)

threshold_df   = pd.DataFrame(best_thresholds).T
threshold_path = os.path.join(OUTPUT_DIR, "cnn_baseline_best_thresholds.csv")
threshold_df.to_csv(threshold_path)
print(f"Saved thresholds to: {threshold_path}")
#simplified run for later in case you want to skip on adding extra results for validating te learning paths with optimal thresholding
threshold_df = pd.read_csv(threshold_path, index_col=0)
inference_thresholds = {
    app: float(threshold_df.loc[app]["threshold"])
    for app in cfg["appliances"]
}

print("Loaded thresholds:")
print(inference_thresholds)
'''
# -------------------------
# FINAL TEST EVALUATION
# -------------------------

print("\n--- Final Evaluation ---")

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
        "f1":        f1_score(all_targets[:, i],  binary_preds[:, i], zero_division=0),
        "precision": precision_score(all_targets[:, i], binary_preds[:, i], zero_division=0),
        "recall":    recall_score(all_targets[:, i],    binary_preds[:, i], zero_division=0),
        "roc_auc":   roc_auc_score(all_targets[:, i],   all_preds[:, i])
    })

rows.append({
    "appliance": "TOTAL (macro avg)",
    "f1":        np.mean([r["f1"]        for r in rows]),
    "precision": np.mean([r["precision"] for r in rows]),
    "recall":    np.mean([r["recall"]    for r in rows]),
    "roc_auc":   np.mean([r["roc_auc"]  for r in rows])
})

results_df = pd.DataFrame(rows).set_index("appliance")
print("\n========== FINAL TEST RESULTS ==========")
print(results_df.to_string())

results_path = os.path.join(OUTPUT_DIR, "cnn_baseline_test_results_final.csv")
results_df.to_csv(results_path)
print(f"\nSaved final test results to {results_path}")
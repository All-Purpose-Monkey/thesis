
import yaml
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import preprocess
import downloader
from torch.utils.data import DataLoader
import os
from data.dataset import STFTDataset
from data.loader import split_data
from data.loader import remove_nan_rows
from utils.metrics import threshold_sweep, compute_metrics
from training.active_learning import active_learning_round
import json
from models.backbone import CNNBackbone
from models.simsiam import SimSiam
from models.heads import MultiHeadClassifier
from training.training_ssl import train_ssl
from training.downstream import train_downstream
from training.loss import classification_loss
from utils.logging import save_history

#from carbontracker.tracker import CarbonTracker

# -------------------------
# LOAD CONFIG
# -------------------------

with open(os.path.expanduser("~/thesis/configs/experiment_simsiam.yaml")) as f:

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
# PREPROCESSING
# -------------------------
'''
appliances = ["kettle", "toaster", "microwave", "dishwasher", "fridge", "washing_machine"]


house1 = [10, 11, 13, 6, 12, 5]
#house2= [8, 16, 15, 13, 14, 12]
#house5 = [18, 15, 23, 22, 19, 24]

#------downloading y files-------

#downloader.download_dat_files(house=2, channels=house2, download_dir='~/thesis/data')
#downloader.download_dat_files(house=5, channels=house5, download_dir='~/thesis/data')

#------combining the IA files into csv and flooring by 6 seconds------
#preprocess.stitch_resample_6s(2, house2, "~/thesis/data/house_2/dat_files/", "~/thesis/data/house_2/")
#preprocess.stitch_resample_6s(5, house5, "~/thesis/data/house_5/dat_files/", "~/thesis/data/house_5/")
#preprocess.stitch_resample_6s(1, house1, "~/thesis/house_1/dat_files/", "~/thesis/house_1/")
#binarized_labels_2 = binarize_labels("~/thesis/data/house_2/house2_stitched.csv", appliances)
#binarized_labels_2.to_csv("~/thesis/data/house_2/house2_binarized.csv", index=False)
#binarized_labels_5 = binarize_labels("~/thesis/data/house_5/house5_stitched.csv", appliances)
#binarized_labels_5.to_csv("~/thesis/data/house_5/house5_binarized.csv", index=False)
#------downloading X files-------
#downloader.download_flac_files(house=1, week=42, year=2014, days=(2, 4, 6), active_hrs=True, active_range=(7, 23), download_dir="~/thesis/data")
#downloader.download_flac_files(house=2, week=24, year=2013, days=(2, 4, 6), active_hrs=True, active_range=(7, 23), download_dir="~/thesis/data")
#downloader.download_flac_files(house=5, week=33, year=2014, days=(2, 4, 6), active_hrs=True, active_range=(7, 23), download_dir="~/thesis/data")
#----binarizing the labels by differen thresholds-----

#h1_t=[10,10,10,10,10,10] 
#h2_t=[10,10,40,40,10,10]
#h5_t=[10,10,20,20,40,20]

#binarized_labels= preprocess.binarize_labels("~/thesis/data/house_1/house1_stitched.csv", appliances, threshold=h1_t)
#binarized_labels.to_csv("~/thesis/data/house_1/house1_binarized.csv", index=False)

#binarized_labels_2 = binarize_labels("~/thesis/data/house_2/house2_stitched.csv", appliances, threshold=h2_t)
#binarized_labels_2.to_csv("~/thesis/data/house_2/house2_binarized.csv", index=False)
#binarized_labels_5 = binarize_labels("~/thesis/data/house_5/house5_stitched.csv", appliances, threshold=h5_t)
#binarized_labels_5.to_csv("~/thesis/data/house_5/house5_binarized.csv", index=False)
##PIPELINE TESTING LABEL SETUP

preprocess.chop_flac(flac_folder="~/thesis/data/house_1/flac_files/2014/wk42/", cfg_file="~/Downloads/calibration_house_1.cfg", output_base="~/thesis/data/house_1/stft_segments/2014/wk42/", sample_rate=16000, window_sec=6, hop_samples=512, n_fft=1024, scale="db", scale_f=20, mode="none")
preprocess.chop_flac(flac_folder="~/thesis/data/house_2/flac_files/2014/wk42/", cfg_file="~/Downloads/calibration_house_2.cfg", output_base="~/thesis/data/house_1/stft_segments/2014/wk42/", sample_rate=16000, window_sec=6, hop_samples=512, n_fft=1024, scale="db", scale_f=20, mode="none")
preprocess.chop_flac(flac_folder="~/thesis/data/house_5/flac_files/2014/wk42/", cfg_file="~/Downloads/calibration_house_5.cfg", output_base="~/thesis/data/house_1/stft_segments/2014/wk42/", sample_rate=16000, window_sec=6, hop_samples=512, n_fft=1024, scale="db", scale_f=20, mode="none")
'''



Big_X, Big_y = preprocess.mash_that(
    "~/thesis/data/house_1/house1_binarized.csv",
    "~/thesis/data/house_1/stft_segments/2014/wk42/",
    path=True
)
Big_X = np.array(Big_X)
Big_y = np.array(Big_y)

print(f"X shape: {len(Big_X)}, y shape: {len(Big_y)}")
print(f"Example X[0] shape: {Big_X[0].shape}, y[0]: {Big_y[0]}")
#---loading appliacne thresholds from config for later use in evaluation---
#inference_thresholds = cfg.get("inference_thresholds", None)

# -------------------------
# SPLIT
# -------------------------
X_clean, y_clean = remove_nan_rows(Big_X, Big_y)
X_clean = np.array(X_clean)  # Convert list of arrays to a single numpy array
y_clean = np.array(y_clean)
X_train, X_test, y_train, y_test = split_data(
    X_clean,
    y_clean,
    test_size=cfg["data"]["test_size"],
    seed=cfg["seed"]
)

# -------------------------
# SSL DATASET
# -------------------------
os.makedirs(os.path.expanduser("~/thesis/logs/carbontracker/"), exist_ok=True)


ssl_dataset = STFTDataset(X_train)

ssl_loader = DataLoader(
    ssl_dataset,
    batch_size=cfg["ssl"]["batch_size"],
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# -------------------------
# MODELS
# -------------------------

backbone = CNNBackbone()

simsiam = SimSiam(backbone).to(device)

optimizer_ssl = optim.SGD(
    simsiam.parameters(),
    lr=cfg["ssl"]["lr"],
    momentum=0.9,
    weight_decay=1e-4
)
scheduler_ssl = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_ssl,
    T_max=cfg["ssl"]["epochs"]
)
# -------------------------
# SSL TRAINING
# -------------------------


train_ssl(
    simsiam,
    ssl_dataset,
    optimizer_ssl,
    device,
    cfg["ssl"]["epochs"],
    scheduler_ssl,
)


torch.save(
    backbone.state_dict(),
    os.path.expanduser("~/thesis/models/h1_server_experiment_simsiam_backbone.pth")
)
print("Saved SimSiam backbone to thesis/models/h1_server_experiment_simsiam_backbone.pth")


backbone_path = os.path.expanduser(
    "~/thesis/models/h1_server_experiment_simsiam_backbone.pth"
)

backbone.load_state_dict(
    torch.load(backbone_path, map_location=device)
)
print(f"Loaded pretrained backbone from {backbone_path}")

# -------------------------
# DOWNSTREAM DATA
# -------------------------

#pos weights for imbalanced data

y_arr = np.array(y_train)
pos_counts = y_arr.sum(axis=0)
neg_counts = len(y_arr) - pos_counts
pos_weights = neg_counts / (pos_counts + 1e-6)
pos_weights = np.clip(pos_weights, 0, 10.0)
pos_weights = torch.tensor(
    pos_weights,
    dtype=torch.float32
).to(device)

train_dataset = STFTDataset(
    X_train,
    y_train
)

test_dataset = STFTDataset(
    X_test,
    y_test
)

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
# STAGE 1 — TRAIN WITHOUT THRESHOLDS
# -------------------------

backbone.requires_grad_(False)

classifier = MultiHeadClassifier(
    backbone,
    cfg["appliances"]
).to(device)

optimizer_class = optim.Adam(
    classifier.parameters(),
    lr=cfg["downstream"]["lr"]
)

loss_fn = classification_loss(pos_weights=pos_weights)

history_stage1 = train_downstream(
    classifier,
    train_loader,
    test_loader,
    optimizer_class,
    loss_fn,
    device,
    cfg["appliances"],
    cfg["downstream"]["epochs"],
    thresholds=None
)
save_history(history_stage1, os.path.expanduser("~/thesis/h1_server_experiment_nothresholds_history.csv"))

# STAGE 2 — THRESHOLD SWEEP
# -------------------------

all_preds, all_targets = [], []

classifier.eval()

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        X_batch = X_batch.to(device)

        logits = classifier(X_batch)
        probs  = torch.sigmoid(logits)

        all_preds.append(probs.cpu().numpy())
        all_targets.append(y_batch.numpy())

all_preds   = np.concatenate(all_preds, axis=0)
all_targets = np.concatenate(all_targets, axis=0)

best_thresholds, sweep_curves = threshold_sweep(
    all_preds,
    all_targets,
    cfg["appliances"]
)

# save as csv instead of json
threshold_df = pd.DataFrame(best_thresholds).T
threshold_path = os.path.expanduser(
    "~/thesis/best_thresholds_simsiam.csv"
)

threshold_df.to_csv(threshold_path)

print("\nSaved thresholds to:")
print(threshold_path)

# -------------------------
# STAGE 3 — REINITIALIZE MLP
# -------------------------

classifier = MultiHeadClassifier(
    backbone,
    cfg["appliances"]
).to(device)

optimizer_class = optim.Adam(
    classifier.parameters(),
    lr=cfg["downstream"]["lr"]
)

# -------------------------
# LOAD THRESHOLDS
# -------------------------

threshold_df = pd.read_csv(
    threshold_path,
    index_col=0
)

inference_thresholds = {
    app: float(threshold_df.loc[app]["threshold"])
    for app in cfg["appliances"]
}

print("\nLoaded thresholds:")
print(inference_thresholds)

# -------------------------
# STAGE 4 — TRAIN AGAIN USING THRESHOLDS
# -------------------------

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
save_history(history_stage2, os.path.expanduser("~/thesis/h1_server_experiment_thresholded_history.csv"))

# --- Simulating a pool by holding out part of X_train ---

al_pool_size = int(0.5 * len(X_clean))
X_al_labelled = X_clean[al_pool_size:]
y_al_labelled = y_clean[al_pool_size:]
X_pool        = X_clean[:al_pool_size]
y_pool_oracle = y_clean[:al_pool_size]   # withheld; revealed only when queried

AL_ROUNDS  = 10
N_QUERY    = 200

for al_round in range(AL_ROUNDS):
    print(f"\n=== Active Learning Round {al_round + 1} ===")

    (X_al_labelled,
     y_al_labelled,
     X_pool,
     y_pool_oracle,
     queried) = active_learning_round(
        classifier,
        X_al_labelled, np.array(y_al_labelled),
        X_pool, np.array(y_pool_oracle),
        device,
        n_query=N_QUERY,
        number_workers=4,
        pin_memory=True
    )

    # Rebuild loader with expanded labelled set
    al_train_dataset = STFTDataset(X_al_labelled, y_al_labelled)
    al_train_loader  = DataLoader(
        al_train_dataset,
        batch_size=cfg["downstream"]["batch_size"],
        shuffle=True
    )

    # Fine-tune for a few epochs on expanded set
    history_al = train_downstream(
        classifier,
        al_train_loader,
        test_loader,
        optimizer_class,
        loss_fn,
        device,
        cfg["appliances"],
        epochs=5               # fewer epochs per AL round
    )

save_history(
        history_al,
        os.path.expanduser(f"~/thesis/h1_server_experiment_al_round{al_round + 1}.csv")
    )
torch.save(
    classifier.state_dict(),
    os.path.expanduser("~/thesis/models/h1_server_experiment_classifier_final.pth")
)

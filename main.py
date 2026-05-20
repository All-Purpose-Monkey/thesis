import yaml
import torch
import numpy as np
import torch.optim as optim
import preprocess
import downloader
from torch.utils.data import DataLoader
import os
from data.dataset import STFTDataset
from data.loader import split_data
from data.loader import remove_nan_rows

from models.backbone import CNNBackbone
from models.simsiam import SimSiam
from models.heads import MultiHeadClassifier


from training.training_ssl import train_ssl
from training.downstream import train_downstream

from training.loss import classification_loss

from utils.logging import save_history

# -------------------------
# LOAD CONFIG
# -------------------------

with open("/Users/yashsaraswat/thesis/configs/experiment_1.yaml") as f:

    cfg = yaml.safe_load(f)

# -------------------------
# DEVICE
# -------------------------

device = torch.device(cfg["device"])
# add this block after: device = torch.device(cfg["device"])

# -------------------------
# PREPROCESSING
# -------------------------

appliances = ["kettle", "toaster", "microwave", "dishwasher", "fridge", "washing_machine"]


house1 = [10, 11, 13, 6, 12, 5]
#house2= [8, 16, 15, 13, 14, 12]
#house5 = [18, 15, 23, 22, 19, 24]
#downloader.download_dat_files(house=2, channels=house2, download_dir='~/thesis/data')
#downloader.download_dat_files(house=5, channels=house5, download_dir='~/thesis/data')
#stitch_resample_6s(2, house2, "~/thesis/data/house_2/dat_files/", "~/thesis/data/house_2/")
#stitch_resample_6s(5, house5, "~/thesis/data/house_5/dat_files/", "~/thesis/data/house_5/")
#binarized_labels_2 = binarize_labels("~/thesis/data/house_2/house2_stitched.csv", appliances)
#binarized_labels_2.to_csv("~/thesis/data/house_2/house2_binarized.csv", index=False)
#binarized_labels_5 = binarize_labels("~/thesis/data/house_5/house5_stitched.csv", appliances)
#binarized_labels_5.to_csv("~/thesis/data/house_5/house5_binarized.csv", index=False)

downloader.download_flac_files(house=1, week=42, year=2014, days=(2, 4, 6), active_hrs=True, active_range=(7, 23), download_dir="~/thesis/data")
downloader.download_flac_files(house=2, week=24, year=2013, days=(2, 4, 6), active_hrs=True, active_range=(7, 23), download_dir="~/thesis/data")
downloader.download_flac_files(house=5, week=33, year=2014, days=(2, 4, 6), active_hrs=True, active_range=(7, 23), download_dir="~/thesis/data")

#preprocess.stitch_resample_6s(1, house1, "~/thesis/house_1/dat_files/", "~/thesis/house_1/")
binarized_labels= preprocess.binarize_labels("~/thesis/data/house_1/house1_stitched.csv", appliances)
binarized_labels.to_csv("~/thesis/data/house_1/house1_binarized.csv", index=False)
##PIPELINE TESTING LABEL SETUP

preprocess.chop_flac(flac_folder="~/thesis/data/house_1/flac_files/2014/wk42/", cfg_file="~/Downloads/calibration_house_1.cfg", output_base="~/thesis/data/house_1/stft_segments/2014/wk42/", sample_rate=16000, window_sec=6, hop_samples=512, n_fft=1024, scale="db", scale_f=20, mode="none")

X,y =preprocess.mash_that( binarized_labels, "~/thesis/data/house_1/stft_segments/2014/wk42/",path=False)

print(f"X shape: {len(X)}, y shape: {len(y)}")
print(f"Example X[0] shape: {X[0].shape}, y[0]: {y[0]}")
y = np.array(y)
print(f"active samples per appliance: {y.sum(axis=0)}")

# -------------------------
# SPLIT
# -------------------------
X_clean, y_clean = remove_nan_rows(X, y)
X_train, X_test, y_train, y_test = split_data(
    X_clean,
    y_clean,
    test_size=cfg["data"]["test_size"],
    seed=cfg["seed"]
)

# -------------------------
# SSL DATASET
# -------------------------

ssl_dataset = STFTDataset(X_train)

ssl_loader = DataLoader(
    ssl_dataset,
    batch_size=cfg["ssl"]["batch_size"],
    shuffle=True
)

# -------------------------
# MODELS
# -------------------------

backbone = CNNBackbone()

simsiam = SimSiam(backbone).to(device)

optimizer_ssl = optim.Adam(
    simsiam.parameters(),
    lr=cfg["ssl"]["lr"]
)
scheduler_ssl = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_ssl,
    T_max=cfg["ssl"]["epochs"]
)
# -------------------------
# SSL TRAINING
# -------------------------

'''
train_ssl(
    simsiam,
    ssl_loader,
    optimizer_ssl,
    device,
    cfg["ssl"]["epochs"],
    scheduler_ssl
)


torch.save(
    backbone.state_dict(),
    os.path.expanduser("~/thesis/models/backbone_ssl.pth")
)
print("Saved SSL backbone to thesis/models/backbone_ssl.pth")
'''

backbone_path = os.path.expanduser(
    "~/thesis/models/backbone_ssl.pth"
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
pos_weights = np.clip(pos_weights, 0, 5.0)
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
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=cfg["downstream"]["batch_size"],
    shuffle=False
)

# -------------------------
# Classifier with SimSiam backbone
# -------------------------
backbone.requires_grad_(False)  # Freezing backbone for downstream training
classifier = MultiHeadClassifier(
    backbone,
    cfg["appliances"]
).to(device)

optimizer_class = optim.Adam(
    classifier.parameters(),
    lr=cfg["downstream"]["lr"]
)


loss_fn = classification_loss(pos_weights=pos_weights)

# -------------------------
# TRAIN DOWNSTREAM
# -------------------------

history = train_downstream(
    classifier,
    train_loader,
    test_loader,
    optimizer_class,
    loss_fn,
    device,
    cfg["appliances"],
    cfg["downstream"]["epochs"]
)

# -------------------------
# SAVE METRICS
# -------------------------

save_history(
    history,
    "~/thesis/results_posw_4cnna_3mlpx2.csv"
)
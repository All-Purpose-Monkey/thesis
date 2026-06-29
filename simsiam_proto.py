import downloader
import preprocess
import numpy as np
import tests
import os
import librosa
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split

appliances = ["kettle", "toaster", "microwave", "dishwasher", "fridge", "washing_machine"]

house1 = [10, 11, 13, 6, 12, 5]

downloader.download_flac_files(house=1, week=38, year=2013, hours=24, download_dir="~/thesis")

preprocess.stitch_resample_6s(1, house1, "~/thesis/house_1/dat_files/", "~/thesis/house_1/")

##PIPELINE TESTING LABEL SETUP

preprocess.chop_flac(flac_folder="~/thesis/house_1/flac_files/2013/wk38/", cfg_file="~/Downloads/calibration_house_1.cfg", output_base="~/thesis/house_1/stft_segments/2013/wk38/", sample_rate=16000, window_sec=6, hop_samples=512, n_fft=1024, scale="db", mode="zscore")

X,y =preprocess.mash_that("~/thesis/house_1/house1_stitched.csv", "~/thesis/house_1/stft_segments/2013/wk38/")

print(f"X shape: {len(X)}, y shape: {len(y)}")
print(f"Example X[0] shape: {X[0].shape}, y[0]: {y[0]}")

#removing imputation logic for now - using non Nan label rows for final regression training as this SimSiam's inherent strength.
#y = np.nan_to_num(y, nan=0.0)

# -------------------------
# Train/test split
# -------------------------

# X = list of (1025,94) STFT arrays, y = list of 6-element labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
# -------------------------
# Dataset wrapper
# -------------------------
class STFTDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        stft = torch.tensor(self.X[idx], dtype=torch.float16).unsqueeze(0)  # (1, H, W)
        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float16)
            return stft, label
        return stft

# -------------------------
# Masking augmentations
# -------------------------
def time_mask(x, max_width=10):
    _, H, W = x.shape
    w = np.random.randint(1, min(max_width, W) + 1)  # clamp w <= W
    start = np.random.randint(0, W - w + 1)
    x[:, :, start:start+w] = 0
    return x

def freq_mask(x, max_width=10):
    _, H, W = x.shape
    w = np.random.randint(1, min(max_width, H) + 1)  # clamp w <= H
    start = np.random.randint(0, H - w + 1)
    x[:, start:start+w, :] = 0
    return x

# -------------------------
# SimSiam encoder
# -------------------------
class SimSiam(nn.Module):
    def __init__(self, flatten_size):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        self.prediction = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )

    def forward(self, x):
        feat = self.backbone(x)
        z = self.projection(feat)
        p = self.prediction(z)
        return z, p, feat

# -------------------------
# Cosine similarity loss
# -------------------------
cos = nn.CosineSimilarity(dim=1, eps=1e-6)
def negative_cosine_similarity(p, z):
    return -(cos(p, z.detach()).mean())

flatten_size = 128 * 129 * 12  # after 3 conv layers with stride 2 on (1,1025,94)
# -------------------------
# CPU device
# -------------------------
device = torch.device("cpu")

# -------------------------
# Pretraining: SimSiam SSL
# -------------------------
model = SimSiam(flatten_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

dataset_ssl = STFTDataset(X_train)  # only train STFTs
dataloader_ssl = DataLoader(dataset_ssl, batch_size=64, shuffle=True)

num_epochs_ssl = 10
for epoch in range(num_epochs_ssl):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader_ssl):
        batch = batch.to(device)
        view1 = batch.clone()
        view2 = batch.clone()
        for i in range(batch.shape[0]):
            view1[i] = time_mask(view1[i])
            view1[i] = freq_mask(view1[i])
            view2[i] = time_mask(view2[i])
            view2[i] = freq_mask(view2[i])
        z1, p1, _ = model(view1)
        z2, p2, _ = model(view2)
        loss = negative_cosine_similarity(p1, z2)/2 + negative_cosine_similarity(p2, z1)/2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"SSL Epoch {epoch+1}, Loss: {total_loss/len(dataloader_ssl):.4f}")

# -------------------------
# Regression head
# -------------------------
class RegressionHead(nn.Module):
    def __init__(self, backbone, flatten_size, output_dim=6):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(flatten_size, output_dim)

    def forward(self, x):
        feat = self.backbone(x)
        feat = feat.flatten(start_dim=1)
        out = torch.relu(self.fc(feat))
        return out


# -------------------------
# Regression: only clean rows (no NaNs in labels)
# -------------------------
reg_X_train, reg_y_train = [], []
for xi, yi in zip(X_train, y_train):
    if not np.isnan(yi).any():
        reg_X_train.append(xi)
        reg_y_train.append(yi)

reg_X_test, reg_y_test = [], []
for xi, yi in zip(X_test, y_test):
    if not np.isnan(yi).any():
        reg_X_test.append(xi)
        reg_y_test.append(yi)

print(f"Regression train samples: {len(reg_X_train)}")
print(f"Regression test samples: {len(reg_X_test)}")

# -------------------------
# Normalize per appliance
# -------------------------
y_train_arr = np.array(reg_y_train)
y_max = y_train_arr.max(axis=0)
y_max[y_max == 0] = 1.0  # prevent divide-by-zero
y_train_norm = y_train_arr / y_max
y_test_norm = np.array(reg_y_test) / y_max

# -------------------------
# Train regression
# -------------------------

reg_model = RegressionHead(model.backbone, flatten_size, output_dim=6).to(device)
optimizer_reg = optim.Adam(reg_model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()


dataset_reg = STFTDataset(reg_X_train, y_train_norm)
dataloader_reg = DataLoader(dataset_reg, batch_size=32, shuffle=True)
num_epochs_reg = 10

for epoch in range(num_epochs_reg):
    reg_model.train()
    total_loss = 0
    for batch_x, batch_y in tqdm(dataloader_reg):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        preds = reg_model(batch_x)
        loss = loss_fn(preds, batch_y)
        optimizer_reg.zero_grad()
        loss.backward()
        optimizer_reg.step()
        total_loss += loss.item()
    print(f"Regression Epoch {epoch+1}, Loss: {total_loss/len(dataloader_reg):.4f}")

# -------------------------
# Evaluate on test set
# -------------------------
reg_model.eval()
test_dataset = STFTDataset(reg_X_test, y_test_norm)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

test_loss = 0


all_preds = []
all_targets = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        preds = reg_model(batch_x)
        
        all_preds.append(preds.cpu().numpy())
        all_targets.append(batch_y.cpu().numpy())

all_preds = np.vstack(all_preds)
all_targets = np.vstack(all_targets)

# -------------------------
# Evaluation metrics
# -------------------------

# MSE
mse_per_column = np.mean((all_preds - all_targets)**2, axis=0)

# MAE
mae_per_column = np.mean(np.abs(all_preds - all_targets), axis=0)

# RMSE
rmse_per_column = np.sqrt(mse_per_column)

# R2
ss_res = np.sum((all_targets - all_preds) ** 2, axis=0)
ss_tot = np.sum((all_targets - np.mean(all_targets, axis=0)) ** 2, axis=0)
r2_per_column = 1 - (ss_res / ss_tot)

print("\nEvaluation metrics per appliance:\n")

for i, name in enumerate(appliances):
    print(
        f"{name:20s} | "
        f"MAE: {mae_per_column[i]:8.2f} | "
        f"MSE: {mse_per_column[i]:10.2f} | "
        f"RMSE: {rmse_per_column[i]:8.2f} | "
        f"R2: {r2_per_column[i]:6.3f}"
    )

#Output examples for visual checks

print("\nShowing 3 prediction examples...\n")

for i in range(3):

    stft = reg_X_test[i]
    true_label = y_test_norm[i]

    stft_tensor = torch.tensor(stft, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = reg_model(stft_tensor).cpu().numpy()[0]

    print(f"\nExample {i+1}")

    print("\nGround Truth:")
    for name, val in zip(appliances, true_label):
        print(f"{name:20s}: {val:.2f}")

    print("\nPrediction:")
    for name, val in zip(appliances, pred):
        print(f"{name:20s}: {val:.2f}")

    # Convert STFT to dB for visualization
    stft_db = 20 * np.log10(stft + 1e-6)

    # Plot directly
    plt.figure(figsize=(8,4))
    plt.imshow(stft_db, origin="lower", aspect="auto", cmap="magma")
    plt.colorbar(label="dB")
    plt.xlabel("Time frames")
    plt.ylabel("Frequency bins")
    plt.title(f"STFT Example {i+1}")
    plt.show()
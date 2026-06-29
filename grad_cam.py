"""
gradcam_analysis.py — GradCAM visualisation for CNNmini on STFT spectrograms

Outputs:
  ~/thesis/results/gradcam/gradcam_{appliance}.png   — 1×4 (TP/TN/FP/FN) per appliance
  ~/thesis/results/gradcam/gradcam_complex_1.png     — all 6 heads on one complex example
  ~/thesis/results/gradcam/gradcam_complex_2.png     — all 6 heads on a second complex example

Usage (on server):
  python gradcam_analysis.py

Fill in the three CONFIG lines before running.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── CONFIG — fill these in ────────────────────────────────────────────────────
MODEL_PATH     = os.path.expanduser("~/thesis/results/cnn_mini/parameter_ablation/compression/pool_6_3_stride_3_kernel_5/classifier_pool_6_3_stride_3_kernel_5.pth")
SPEC_DIR       = os.path.expanduser("~/thesis/data/test_set_2017/house_1/stft_segments/2017/wk04/current/")
FAILURE_CSV    = os.path.expanduser("~/thesis/results/heldout/pool_9_1_stride_3_kernel_5_heldout_failure_analysis.csv")
OUT_DIR        = os.path.expanduser("~/thesis/results/gradcam/")
SPEC_EXT       = ".npy"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────────────────────

APPLIANCES = ["kettle", "toaster", "microwave", "dishwasher", "fridge", "washing_machine"]
os.makedirs(OUT_DIR, exist_ok=True)


# ── Model definition (CNNmini) ────────────────────────────────────────────────
# Replace this with your actual import if the class lives in another file:
#   from backbone import CNNmini
# Otherwise this definition should match your trained architecture exactly.

class CNNmini(nn.Module):
    """
    Matches the actual checkpoint key structure:
      backbone.encoder.{0=Conv,1=BN,2=ReLU, 3=Conv,4=BN,5=ReLU, 6=Conv,7=BN,8=ReLU}
      shared.{0=Linear(embed,32), 1=Linear(32,32)}
      heads.{appliance}.{0=Linear(32,16), 1=Linear(16,16), 2=ReLU, 3=Linear(16,1)}
    """
    def __init__(self, pool_size=(6, 3), kernel=5, stride=3, embed_dim=1152):
        super().__init__()
        pad = kernel // 2
        self.backbone = nn.ModuleDict({
            "encoder": nn.Sequential(
                nn.Conv2d(1,  16, kernel, stride=stride, padding=pad),  # 0
                nn.BatchNorm2d(16),                                       # 1
                nn.ReLU(),                                                # 2
                nn.Conv2d(16, 32, kernel, stride=stride, padding=pad),  # 3
                nn.BatchNorm2d(32),                                       # 4
                nn.ReLU(),                                                # 5
                nn.Conv2d(32, 64, kernel, stride=stride, padding=pad),  # 6
                nn.BatchNorm2d(64),                                       # 7
                nn.ReLU(),                                                # 8
            )
        })
        self.pool   = nn.AdaptiveAvgPool2d(pool_size)
        self.shared = nn.Sequential(
            nn.Linear(embed_dim, 32),  # 0
            nn.LayerNorm(32),          # 1  ← LN (weight=[32], no running stats)
        )
        self.heads = nn.ModuleDict({
            a: nn.Sequential(
                nn.Linear(32, 16),  # 0
                nn.LayerNorm(16),   # 1  ← LN (weight=[16], no running stats)
                nn.ReLU(),          # 2
                nn.Linear(16, 1),   # 3
            )
            for a in APPLIANCES
        })

    def forward(self, x):
        x = self.backbone["encoder"](x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.shared(x)
        return torch.cat([self.heads[a](x) for a in APPLIANCES], dim=1)  # (B, 6)


# ── GradCAM ───────────────────────────────────────────────────────────────────

class GradCAM:
    def __init__(self, model):
        # target: last Conv2d in backbone.encoder (index 6)
        self._acts  = None
        self._grads = None
        target = model.backbone["encoder"][6]
        target.register_forward_hook(self._fwd)
        target.register_full_backward_hook(self._bwd)

    def _fwd(self, _, __, out):   self._acts  = out.detach()
    def _bwd(self, _, __, gout):  self._grads = gout[0].detach()

    def compute(self, model, tensor, appliance):
        """appliance: string name. Returns (H, W) CAM in [0,1]."""
        model.zero_grad()
        out = model(tensor)                               # (1, 6)
        idx = APPLIANCES.index(appliance)
        out[0, idx].backward()
        weights = self._grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self._acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_spec_index(spec_dir):
    """
    Walk spec_dir recursively and build {timestamp_int: full_path} dict.
    Handles any depth of subdirectories (hour dirs, date dirs, etc.).
    """
    index = {}
    for root, _, files in os.walk(spec_dir):
        for fname in files:
            if fname.endswith(SPEC_EXT):
                ts = fname.replace(SPEC_EXT, "")
                try:
                    index[int(ts)] = os.path.join(root, fname)
                except ValueError:
                    pass  # skip non-timestamp filenames
    print(f"Spec index built: {len(index)} files under {spec_dir}")
    return index


def load_spec(timestamp, spec_index):
    path = spec_index.get(int(timestamp))
    if path is None:
        raise FileNotFoundError(f"No spectrogram found for timestamp {timestamp}")
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :]
    return torch.tensor(arr).unsqueeze(0).to(DEVICE)


def overlay(spec_np, cam, ax, title, alpha=0.45):
    """Plot spectrogram with GradCAM heatmap overlaid (2D)."""
    ax.imshow(spec_np, origin="lower", aspect="auto", cmap="magma")
    ax.imshow(cam, origin="lower", aspect="auto", cmap="jet",
              alpha=alpha, vmin=0, vmax=1)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def overlay_3d(spec_np, cam, ax, title, step=4):
    """Plot spectrogram as 3D surface coloured by GradCAM.

    Z = normalised spectrogram magnitude.
    Face colour = CAM activation (jet).
    X = freq (left-right), Y = time (depth) — keeps low-freq ridge as a
    left-side wall so azim=-45 shows it without blocking the surface.
    Downsampled by `step` for render speed.
    """
    s = spec_np[::step, ::step]   # s[freq, time] after slicing
    c = cam[::step, ::step]
    n_freq, n_time = s.shape
    # X = freq axis (left→right), Y = time axis (front→back)
    X, Y = np.meshgrid(np.arange(n_freq), np.arange(n_time), indexing="ij")
    z = (s - s.min()) / (s.max() - s.min() + 1e-8)
    colors = cm.jet(c)
    ax.plot_surface(X, Y, z, facecolors=colors,
                    rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False)
    ax.view_init(elev=20, azim=-45)   # classic STFT angle: diagonal between freq+time
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("freq", fontsize=7)
    ax.set_ylabel("time", fontsize=7)
    ax.set_zlabel("mag",  fontsize=7)
    ax.tick_params(labelsize=6)


def pick_example(df, appliance, error_type):
    """Return one timestamp row matching appliance × error_type, or None."""
    col  = f"{appliance}_error"
    rows = df[df[col] == error_type]
    return rows.iloc[0] if len(rows) > 0 else None


def add_complexity(df):
    true_cols   = [f"{a}_true" for a in APPLIANCES]
    n_active    = df[true_cols].sum(axis=1)
    df = df.copy()
    df["complexity"] = n_active.map(
        lambda n: "negative" if n == 0 else ("simple" if n == 1 else "complex")
    )
    return df


# ── Main plots ────────────────────────────────────────────────────────────────

def plot_per_appliance(model, gcam, df, spec_index):
    """6 figures — one per appliance, 1×4 TP/TN/FP/FN panels (3D surface)."""
    for appl in APPLIANCES:
        fig = plt.figure(figsize=(20, 5))
        fig.suptitle(f"GradCAM — {appl.replace('_', ' ').title()}", fontsize=12)

        for i, etype in enumerate(["TP", "TN", "FP", "FN"]):
            ax  = fig.add_subplot(1, 4, i + 1, projection="3d")
            row = pick_example(df, appl, etype)
            if row is None:
                ax.set_title(f"{etype}\n(no example)", fontsize=9)
                continue
            try:
                tensor  = load_spec(row.name, spec_index)
                cam     = gcam.compute(model, tensor, appl)
                spec_np = tensor.squeeze().cpu().numpy()
                prob    = row[f"{appl}_prob"]
                overlay_3d(spec_np, cam, ax, f"{etype}  (p={prob:.2f})")
            except FileNotFoundError:
                ax.set_title(f"{etype}\n(spec not found)", fontsize=9)

        fig.tight_layout()
        out = os.path.join(OUT_DIR, f"gradcam_{appl}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")


def plot_complex_example(model, gcam, df, spec_index, example_num, timestamp):
    """One complex example — 2×3 grid of all 6 appliance CAMs (3D surface)."""
    try:
        tensor  = load_spec(timestamp, spec_index)
        spec_np = tensor.squeeze().cpu().numpy()
    except FileNotFoundError:
        print(f"[SKIP] complex example {example_num}: spec not found for ts={timestamp}")
        return

    row = df.loc[timestamp] if timestamp in df.index else None
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"GradCAM — Complex Example {example_num}  (ts={timestamp})", fontsize=12)

    for i, appl in enumerate(APPLIANCES):
        ax  = fig.add_subplot(2, 3, i + 1, projection="3d")
        cam = gcam.compute(model, tensor, appl)
        label = ""
        if row is not None:
            etype = row[f"{appl}_error"]
            prob  = row[f"{appl}_prob"]
            label = f"{etype}  p={prob:.2f}"
        overlay_3d(spec_np, cam, ax, f"{appl.replace('_', ' ')} | {label}")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"gradcam_complex_{example_num}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    # load model
    model = CNNmini().to(DEVICE)
    ckpt  = torch.load(MODEL_PATH, map_location=DEVICE)
    # handle both raw state_dict and checkpoint dicts
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()
    print(f"Model loaded from {MODEL_PATH}")

    gcam = GradCAM(model)

    # build spectrogram index once (walks all subdirs)
    spec_index = build_spec_index(SPEC_DIR)

    # load + index failure analysis by timestamp
    df = pd.read_csv(FAILURE_CSV, index_col="timestamp")
    df = add_complexity(df)

    # ── per-appliance TP/TN/FP/FN panels ────────────────────────────────────
    plot_per_appliance(model, gcam, df, spec_index)

    # ── two complex examples ─────────────────────────────────────────────────
    complex_ts = df[df["complexity"] == "complex"].index.tolist()
    if len(complex_ts) >= 2:
        plot_complex_example(model, gcam, df, spec_index, 1, complex_ts[0])
        plot_complex_example(model, gcam, df, spec_index, 2, complex_ts[len(complex_ts) // 2])
    else:
        print("[WARN] fewer than 2 complex examples found in CSV")

    print("\nDone. All outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
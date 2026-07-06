# Thesis Repository — Reviewer's Guide

This is a NILM (Non-Intrusive Load Monitoring) project that classifies appliance activity from high-frequency STFT spectrograms of UK-DALE mains data. This document is a navigation guide for reviewers: it explains what each file does and where to look first, so you don't have to read the whole repo to understand the work.

> **A note on the mess.** The research was done somewhat backwards — a compact baseline (`CNNmini`) unexpectedly outperformed the larger models, so several things had to be redone around it. As a result, a lot of files are experiments, ablations, or dead-end explorations. The sections below are ordered by importance, the later sections document artefacts of that exploration and can be skimmed or skipped.

---

## main model files

The primary model used throughout the thesis is **`CNNmini` + `MultiHeadClassifier_mini`**. Everything else is secondary.

| File | Description |
|------|-------------|
| `models/backbone.py` | Defines the CNN feature extractors. **`CNNmini`** is the main backbone used in the thesis; `CNNBackbone` is the larger original baseline. |
| `models/heads.py` | Defines the classifier heads. **`MultiHeadClassifier_mini`** is the main per-appliance head used with `CNNmini`; `MultiHeadClassifier_big` is the earlier larger variant. |
| `cnn_mini.py` | Main training/evaluation run for `CNNmini` + `MultiHeadClassifier_mini`. This is the headline experiment. |
| `cnn_baseline.py` | Training run for the original larger baseline model — the reference the mini model is compared against. |

---

## Core pipeline

Data loading, preprocessing, and shared utilities that the main model depends on.

### `data/`
| File | Description |
|------|-------------|
| `data/dataset.py` | `STFTDataset` — the PyTorch `Dataset` wrapping STFT spectrogram tensors and (optional) labels. |
| `data/loader.py` | Train/test splitting and NaN-row cleaning helpers (`split_data`, `remove_nan_rows`). |
| `data/augmentation.py` | STFT augmentation views (noise, time-shift) — used by the SSL experiments (see later section). |

### Preprocessing (root)
| File | Description |
|------|-------------|
| `preprocess.py` | FLAC signal normalization and the main STFT preprocessing pipeline that turns raw mains into spectrogram segments. |
| `preprocess_stft.py` | Standalone STFT segment generator with explicit settings (days, STFT params) mirroring the main pipeline. |
| `preprocess_test.py` | Downloads and prepares the 2017 held-out UK-DALE test set (`.h5`). |
| `downloader.py` | Downloads raw FLAC mains recordings for selected days/hours from the dataset source. |

### `utils/`
| File | Description |
|------|-------------|
| `utils/metrics.py` | Metric computation, including per-appliance threshold sweeping. |
| `utils/plotting.py` | Plotting helpers (e.g. STFT spectrogram visualization). |
| `utils/logging.py` | Training-history saving / logging helpers. |

### `training/`
| File | Description |
|------|-------------|
| `training/loss.py` | Loss functions — BCE, focal loss, and the SSL negative-cosine-similarity loss. |
| `training/downstream.py` | `train_downstream` — downstream classifier training loop (used after SSL pretraining). |

---

## Evaluation & analysis

Scripts for testing on held-out data and producing the analysis in the write-up.

| File | Description |
|------|-------------|
| `heldout_test.py` | Evaluates a trained model on the 2017 held-out test set. |
| `gen_test_preds.py` | Generates test-set predictions from a trained model for downstream analysis. |
| `eda.py` | Label-level exploratory data analysis (appliance activation stats). |
| `eda_2.py` | EDA of power buckets / NILM energy basins across appliances. |
| `eda_heldout.py` | Error analysis on the held-out set — bands how confidently/wrongly the model misclassified. |
| `experiment_analytics.py` | Produces the final thesis figures (e.g. channel-ablation plots) into `results/finale/`. |
| `grad_cam.py` | Grad-CAM saliency maps to visualize what the CNN attends to in the spectrograms. |
| `tests.py` | Sanity checks on the data (e.g. verifying 6-second timestamp spacing). |
| `cal_test.py` | One-off calibration check: FLAC → WAV → STFT using the house calibration config. |

---

## Ablation studies

Systematic sweeps around the `CNNmini` architecture. These support the design choices but are not the main result — skim as needed.

| File | Description |
|------|-------------|
| `cnn_mini_ksize.py` | Kernel-size ablation for the mini model. |
| `cnn_mini_stride.py` | Stride ablation for the mini model. |
| `cnn_mini_adpooling.py` | Adaptive-pooling shape ablation (frequency- vs time-heavy pooling). |
| `cnn_pad_loop.py` | Padding (amount + symmetry) ablation. |
| `cnn_mini_hp.py` | Hyperparameter (LR / weight-decay) sweep for the mini model. |
| `channel_abelation.py` | Channel-ablation study (current vs voltage input channels). |
| `island_study.py` | "Island" study on spectrogram segment structure/connectivity. |

---

## Run scripts (root)

Most experiments have a matching shell / SSH job script in the root directory that launches them on the compute server. They map 1:1 to the Python files above by name:

`run_cnn_mini.sh`, `run_cnn_baseline.sh`, `run_cnn_ksize.sh`, `run_cnn_stride.sh`, `run_cnn_adpooling.sh`, `run_cnn_pad_loop.sh`, `run_cnn_hp.sh`, `run_channel_test.sh`, `run_heldout.sh`, `run_gradscan.sh`, `run_server_test.sh`, `run_bootstrap_ci.sh`, `run_cnn_al.sh`, `run_simsiam.sh`.

`configs/` holds the corresponding YAML configs (`cnn_baseline.yaml`, `cnn_baseline_hp_abelatet.yaml`, `cnn_test.yaml`, `experiment_simsiam.yaml`).

---

## Artefacts of earlier / abandoned research

Everything below is from an earlier self-supervised-learning (SSL) direction and other exploratory work that was **not part of the final approach**. Kept for provenance; **reviewers can safely skip this section.**

### Self-supervised learning (SimSiam, pretraining, active learning, augments)
| File | Description |
|------|-------------|
| `models/simsiam.py` | SimSiam self-supervised model (projector/predictor over the backbone). |
| `main_simsiam.py` | Main SimSiam pretraining run. |
| `test_simsiam.py` | Evaluation run for SimSiam-pretrained representations. |
| `simsiam_proto.py` | Early SimSiam pipeline/label-setup prototype. |
| `training/training_ssl.py` | SSL training loop, including hardness-weighting logic. |
| `training/active_learning.py` | Active-learning uncertainty sampling over the unlabeled pool. |
| `cnn_al_loop.py` | Active-learning training loop experiment. |
| `data/augmentation.py` | (listed above) augmentation views, originally built for SSL. |

### Misc / dead-ends
| File | Description |
|------|-------------|
| `hot_hot_fix.py` | Ad-hoc patch script (requires manual edits before running). |
| `trashburner.py` | Scratch/throwaway script (duplicate of some preprocessing normalization code). |

---

*This README was written by Claude (Cowork mode) as a navigation guide based on the repository contents.*

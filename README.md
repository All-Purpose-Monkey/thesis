# What Does the Appliance Say?
### Audio-inspired feature extraction and vision-based deep learning for multi-appliance state classification in a sparse label regime

**MSc Thesis — Data Science & Society, Tilburg University (2026) | Author: Yash Saraswat**

## Summary

Non-Intrusive Load Monitoring (NILM) infers appliance activity from a single aggregate household signal. UK-DALE's 16 kHz current data is literally distributed as two-channel FLAC audio — this thesis takes that framing seriously and treats appliance-state detection as a **sound-classification problem**: STFT spectrograms of six-second current windows are passed as images to a compact 2D CNN that predicts six appliance states simultaneously (kettle, toaster, microwave, dishwasher, fridge, washing machine), every window, with no event-detection stage.

**Key results:**
- An ultra-light CNN (**<90k parameters**, kernel 5×5, stride 3, frequency-only adaptive pooling) reaches **macro F1 = 0.957** across three in-distribution UK-DALE 2015 households.
- Controlled ablations show the current channel far outperforms voltage (macro F1 0.73 vs. 0.34) and that **frequency-preserving pooling is the dominant architectural driver** — temporal structure within a window barely matters.
- On a strictly held-out 2017 household, F1 collapses to 0.39 while **macro AUROC holds at 0.84**: the encoder still ranks activations correctly, but decision thresholds don't transfer. A contiguity analysis shows the in-distribution figure is itself an optimistic ceiling, and comparison against HF-NILM baselines with true out-of-distribution protocols shows this collapse is a documented property of the regime.
- Grad-CAM confirms the network attends to coherent, appliance-specific harmonic bands, consistent with the harmonic-signature literature.
- Class imbalance is severe (fridge dominates; toaster is <0.2% of active segments) and is handled with positive-weighted focal loss plus per-appliance threshold sweeps.

**Compute:** all experiments ran on the university **GPU4EDU cluster** via SLURM-style shell job scripts (CUDA 11 / PyTorch 2.7); the `run_*.sh` launchers in the repo root map 1:1 to the experiment files.

**Stack:** PyTorch, TorchAudio, Librosa, SoundFile, NumPy, Pandas, SciPy, scikit-learn, Matplotlib.

## Repository guide

The research was done somewhat backwards — a compact baseline (CNNmini) unexpectedly outperformed larger models, so several things were redone around it. Files below are ordered by importance; later sections are exploration artefacts retained for provenance and future work.

**Main model:** `models/backbone.py` (CNNmini encoder), `models/heads.py` (per-appliance sigmoid heads), `cnn_mini.py` (headline experiment), `cnn_baseline.py` (larger reference model).

**Core pipeline:** `preprocess.py` / `preprocess_stft.py` (FLAC calibration → STFT segments), `preprocess_test.py` + `downloader.py` (data acquisition), `data/` (dataset, splits, augmentations), `training/loss.py` (BCE, focal, SSL losses), `training/downstream.py` (training loop), `utils/` (metrics with per-appliance threshold sweeping, plotting, logging).

**Evaluation & analysis:** `heldout_test.py` (2017 generalisation test), `gen_test_preds.py`, `eda*.py` (label statistics, energy basins, held-out error analysis), `experiment_analytics.py` (final thesis figures), `grad_cam.py` (saliency maps), `island_study.py` (label-contiguity / leakage analysis).

**Ablations:** `cnn_mini_ksize.py`, `cnn_mini_stride.py`, `cnn_mini_adpooling.py`, `cnn_pad_loop.py`, `cnn_mini_hp.py`, `channel_abelation.py` — the systematic sweeps behind the architecture choice.

**Earlier SSL direction (retained):** `models/simsiam.py`, `main_simsiam.py`, `training/training_ssl.py` (with hardness weighting), `training/active_learning.py` — a SimSiam self-supervised pretraining line that predates the final approach and is the basis for planned future work on cross-household transfer.

`configs/` holds the YAML experiment configs; each experiment has a matching `run_*.sh` cluster job script.

## AI usage

Code was developed manually by the author. AI assistance (Claude Sonnet 4.6) was used for code comments, sanity checks, logging utilities, and parts of the plotting/Grad-CAM implementations — flagged explicitly in the files where applicable. This README was compiled with AI assistance as a navigation guide based on repository contents.

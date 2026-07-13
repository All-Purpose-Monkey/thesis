# What Does the Appliance Say?
### Audio-inspired feature extraction and vision-based deep learning for multi-appliance state classification in a sparse label regime

**MSc Thesis — Data Science & Society, Tilburg University (2026) | Author: Yash Saraswat**

📄 **[Read the full thesis (PDF)](yash_DSS_Thesis__6_.pdf)**

## Summary

Non-Intrusive Load Monitoring (NILM) infers appliance activity from a single aggregate household signal. UK-DALE's 16 kHz current data is literally distributed as two-channel FLAC audio — this thesis takes that framing seriously and treats appliance-state detection as a **sound-classification problem**: STFT spectrograms of six-second current windows are passed as images to a compact 2D CNN that predicts six appliance states simultaneously (kettle, toaster, microwave, dishwasher, fridge, washing machine), every window, with no event-detection stage.

**Key results:**
- An ultra-light CNN (**<90k parameters**, kernel 5×5, stride 3, frequency-only adaptive pooling) reaches **macro F1 = 0.957** across three in-distribution UK-DALE 2015 households.
- Controlled ablations show the current channel far outperforms voltage (macro F1 0.73 vs. 0.34,  and that **frequency-preserving adaptive pooling is the dominant architectural driver**, temporal structure within a window barely matters.
- On a strictly held-out 2017 household , F1 collapses to 0.39 while **macro AUROC holds at 0.84**: the encoder still ranks activations correctly, but decision thresholds don't transfer. A label-contiguity analysis  shows the in-distribution figure is itself an optimistic ceiling, and comparison against HF-NILM baselines with true out-of-distribution protocols shows this collapse is a documented property of the regime, not a defect of the model.
- **Grad-CAM analysis is grounded in the physics of electrical loads** : the network attends to coherent, appliance-specific harmonic bands resistive loads like kettles light up broad high-energy bands, while washing machines show a consistent near-low-frequency signature. This matches the harmonic-signature literature (Dinar et al., 2022/2025: odd harmonics up to the 15th order carry the discriminative content), meaning the model independently rediscovers known appliance physics rather than fitting noise.
- Class imbalance is severe (fridge dominates; toaster is <0.2% of active segments) and is handled with positive-weighted focal loss plus per-appliance threshold sweeps.

**Compute:** all experiments ran on the university **GPU4EDU cluster** via shell job scripts (CUDA 11, PyTorch 2.7); the [`run_*.sh`](.) launchers in the repo root map 1:1 to the experiment files below.

**Stack:** PyTorch, TorchAudio, Librosa, SoundFile, NumPy, Pandas, SciPy, scikit-learn, Matplotlib.

## Repository guide

The research was done somewhat backwards — a compact baseline (CNNmini) unexpectedly outperformed the larger models, so several things were redone around it. Sections are ordered by importance; the later ones document artefacts of exploration retained for provenance and future work.

### Main model files

The primary model throughout the thesis is **CNNmini + MultiHeadClassifier_mini**. Everything else is secondary.

| File | Description |
|---|---|
| [`models/backbone.py`](models/backbone.py) | CNN feature extractors. `CNNmini` is the main backbone; `CNNBackbone` is the larger original baseline. |
| [`models/heads.py`](models/heads.py) | Classifier heads. `MultiHeadClassifier_mini` is the main per-appliance head used with CNNmini. |
| [`cnn_mini.py`](cnn_mini.py) | Main training/evaluation run for CNNmini — **the headline experiment**. |
| [`cnn_baseline.py`](cnn_baseline.py) | Training run for the larger baseline the mini model is compared against. |

### Core pipeline

| File | Description |
|---|---|
| [`data/dataset.py`](data/dataset.py) | `STFTDataset` — PyTorch Dataset wrapping STFT spectrogram tensors and labels. |
| [`data/loader.py`](data/loader.py) | Train/test splitting and NaN-row cleaning helpers. |
| [`data/augmentation.py`](data/augmentation.py) | STFT augmentation views (noise, time-shift) — built for the SSL experiments. |
| [`preprocess.py`](preprocess.py) | FLAC calibration and the main STFT preprocessing pipeline turning raw mains into spectrogram segments. |
| [`preprocess_stft.py`](preprocess_stft.py) | Standalone STFT segment generator with explicit settings; used for the channel-ablation set. |
| [`preprocess_test.py`](preprocess_test.py) | Downloads and prepares the 2017 held-out UK-DALE test set. |
| [`downloader.py`](downloader.py) | Downloads raw FLAC mains recordings and label files for selected days/hours. |
| [`training/loss.py`](training/loss.py) | Loss functions — BCE, focal loss, and the SSL negative-cosine-similarity loss. |
| [`training/downstream.py`](training/downstream.py) | `train_downstream` — classifier training loop (works for any supervised training, originally built for post-SSL). |
| [`utils/metrics.py`](utils/metrics.py) | Metric computation, including per-appliance threshold sweeping. |
| [`utils/plotting.py`](utils/plotting.py) | Plotting helpers (e.g. spectrogram visualisation). |
| [`utils/logging.py`](utils/logging.py) | Training-history saving / logging helpers. |

### Evaluation & analysis

| File | Description |
|---|---|
| [`heldout_test.py`](heldout_test.py) | Evaluates a trained model on the 2017 held-out household — the generalisation test. |
| [`gen_test_preds.py`](gen_test_preds.py) | Generates test-set predictions for downstream failure analysis. |
| [`eda.py`](eda.py) | Label-level EDA (appliance activation stats). |
| [`eda_2.py`](eda_2.py) | EDA of power buckets / NILM energy basins across appliances. |
| [`eda_heldout.py`](eda_heldout.py) | Held-out error analysis — bands how confidently/wrongly the model misclassified. *(developed with AI)* |
| [`experiment_analytics.py`](experiment_analytics.py) | Produces the final thesis figures into `results/finale/`. *(plotting developed with AI)* |
| [`grad_cam.py`](grad_cam.py) | Grad-CAM saliency maps — visualises which spectral bands the CNN attends to per appliance. *(adapted with AI from public implementations)* |
| [`island_study.py`](island_study.py) | Label-contiguity ("island") study quantifying split leakage in shuffled windows. |
| [`tests.py`](tests.py) | Sanity checks on the data (e.g. 6-second timestamp spacing). |
| [`cal_test.py`](cal_test.py) | One-off calibration check: FLAC → WAV → STFT with house calibration config. |

### Ablation studies

Systematic sweeps around the CNNmini architecture — these support the design choices but are not the main result.

| File | Description |
|---|---|
| [`cnn_mini_ksize.py`](cnn_mini_ksize.py) | Kernel-size ablation. |
| [`cnn_mini_stride.py`](cnn_mini_stride.py) | Stride ablation. |
| [`cnn_mini_adpooling.py`](cnn_mini_adpooling.py) | Adaptive-pooling shape ablation (frequency- vs time-heavy pooling) — **the decisive experiment**. |
| [`cnn_pad_loop.py`](cnn_pad_loop.py) | Padding (amount + symmetry) ablation. |
| [`cnn_mini_hp.py`](cnn_mini_hp.py) | Learning-rate / weight-decay sweep. |
| [`channel_abelation.py`](channel_abelation.py) | Channel ablation (current vs voltage input). |

### Run scripts & configs

Most experiments have a matching shell job script in the repo root that launches them on the GPU4EDU cluster (CUDA 11), mapping 1:1 by name: `run_cnn_mini.sh`, `run_cnn_baseline.sh`, `run_cnn_ksize.sh`, `run_cnn_stride.sh`, `run_cnn_adpooling.sh`, `run_cnn_pad_loop.sh`, `run_cnn_hp.sh`, `run_channel_test.sh`, `run_heldout.sh`, `run_gradscan.sh`, `run_server_test.sh`, `run_bootstrap_ci.sh`, `run_cnn_al.sh`, `run_simsiam.sh`.

[`configs/`](configs/) holds the corresponding YAML experiment configs.

### Earlier SSL direction (retained for future work)

A SimSiam self-supervised pretraining line that predates the final approach. Retained deliberately: the thesis's central open problem — cross-household threshold transfer — points to SSL pretraining as the clearest fix, so this code is the starting point for that follow-up.

| File | Description |
|---|---|
| [`models/simsiam.py`](models/simsiam.py) | SimSiam model (projector/predictor over the backbone). |
| [`main_simsiam.py`](main_simsiam.py) | Main SimSiam pretraining run. |
| [`test_simsiam.py`](test_simsiam.py) | Evaluation of SimSiam-pretrained representations. |
| [`simsiam_proto.py`](simsiam_proto.py) | Early SimSiam pipeline prototype. |
| [`training/training_ssl.py`](training/training_ssl.py) | SSL training loop, including hardness-weighting logic. |
| [`training/active_learning.py`](training/active_learning.py) | Active-learning uncertainty sampling over the unlabeled pool. |
| [`cnn_al_loop.py`](cnn_al_loop.py) | Active-learning training loop experiment. |

### Misc / dead-ends

| File | Description |
|---|---|
| [`hot_hot_fix.py`](hot_hot_fix.py) | Ad-hoc patch script (requires manual edits before running). |
| [`trashburner.py`](trashburner.py) | Scratch/throwaway script (duplicates some preprocessing normalisation code). |

## AI usage

Code was developed manually by the author. AI assistance (Claude Sonnet 4.6) was used for code comments, sanity checks, logging utilities, and the plotting/Grad-CAM implementations flagged in the tables above. This README was compiled with AI assistance as a navigation guide based on repository contents.

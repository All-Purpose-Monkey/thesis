"""
Download 2 days of House 1 FLAC files and generate voltage STFT segments.

Output hierarchy:
  ~/thesis/data/house_1/stft_segments/2014/wk42/voltage/<ts_folder>/<ts>.npy

Same STFT settings as current pipeline for a fair comparison.
Run locally, then upload voltage/ subfolder to the server.
"""

import os
import preprocess
import downloader

# =========================================================
# SETTINGS
# =========================================================

HOUSE       = 1
WEEK        = 42
YEAR        = 2014
DAYS        = (1, 2)          # first 2 days of wk42

CFG_FILE    = "~/Downloads/calibration_house_1.cfg"
BASE_DIR    = "~/thesis/data"

FLAC_DIR    = f"{BASE_DIR}/house_{HOUSE}/flac_files/{YEAR}/wk{WEEK}/"
STFT_OUT    = f"{BASE_DIR}/house_{HOUSE}/stft_segments/{YEAR}/wk{WEEK}/"

# STFT settings — identical to current pipeline
SAMPLE_RATE = 16000
WINDOW_SEC  = 6
HOP_SAMPLES = 512
N_FFT       = 1024
SCALE       = "db"
SCALE_F     = 20
MODE        = "none"

# =========================================================
# STEP 1 — DOWNLOAD
# =========================================================

print("=" * 60)
print(f"Downloading House {HOUSE}, week {WEEK}, {YEAR}, days {DAYS}")
print("=" * 60)

downloader.download_flac_files(
    house=HOUSE,
    week=WEEK,
    year=YEAR,
    days=DAYS,
    active_hrs=True,
    active_range=(7, 23),
    download_dir=BASE_DIR
)

# =========================================================
# STEP 2 — VOLTAGE STFT
# =========================================================

print("\n" + "=" * 60)
print("Generating voltage STFT segments (channel=0)")
print(f"Output → {os.path.expanduser(STFT_OUT)}/voltage/")
print("=" * 60)

preprocess.chop_flac(
    flac_folder=FLAC_DIR,
    cfg_file=CFG_FILE,
    output_base=STFT_OUT,
    sample_rate=SAMPLE_RATE,
    window_sec=WINDOW_SEC,
    hop_samples=HOP_SAMPLES,
    n_fft=N_FFT,
    scale=SCALE,
    scale_f=SCALE_F,
    mode=MODE,
    channel=0,             # voltage
)

print("\nDone. Voltage segments at:")
print(f"  {os.path.expanduser(STFT_OUT)}/voltage/")
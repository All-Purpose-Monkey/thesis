import os
import requests
import zipfile
import downloader
import preprocess
import numpy as np
import h5py
import hdf5plugin
import pandas as pd
import csv
from eda import run_label_eda
import matplotlib.pyplot as plt

def download_ukdale_h5(download_dir="~/thesis/data/test_set_2017"):

    download_dir = os.path.expanduser(download_dir)
    os.makedirs(download_dir, exist_ok=True)

    url       = "https://dap.ceda.ac.uk/edc/efficiency/residential/EnergyConsumption/Domestic/UK-DALE-2017/UK-DALE-FULL-disaggregated/ukdale.h5.zip"
    zip_path  = os.path.join(download_dir, "ukdale.h5.zip")
    h5_path   = os.path.join(download_dir, "ukdale.h5")

    if os.path.exists(h5_path):
        print(f"ukdale.h5 already exists, skipping download: {h5_path}")
    else:
        if os.path.exists(zip_path):
            print(f"Zip already exists, skipping download: {zip_path}")
        else:
            print(f"Downloading ukdale.h5.zip ...")
            try:
                r = requests.get(url, stream=True)
                r.raise_for_status()
            except requests.HTTPError as e:
                print(f"ERROR: Download failed: {e}")
                return

            total      = int(r.headers.get("content-length", 0))
            downloaded = 0

            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r  {pct:.1f}%  ({downloaded // (1024*1024)} MB / {total // (1024*1024)} MB)", end="", flush=True)

            print(f"\nSaved zip to: {zip_path}")

        print(f"Unzipping to: {download_dir}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            print(f"Contents: {zf.namelist()}")
            zf.extractall(download_dir)
        print(f"Unzipped successfully.")

        os.remove(zip_path)
        print(f"Removed zip file.")

    print(f"\nDone. H5 file at: {h5_path}")
    return h5_path


def explore_h5(path):
    with h5py.File(path, "r") as f:

        print("=== BUILDING 1 TOP-LEVEL KEYS (above elec) ===")
        for key in f["building1"].keys():
            print(f"  {key}")

        print("\n=== SAMPLE READ: meter5/table (first 5 rows) ===")
        table = f["building1/elec/meter5/table"]
        print(f"  dtype: {table.dtype}")
        sample = table[:5]
        for row in sample:
            ts_s  = row["index"] // 1_000_000_000
            watts = row["values_block_0"][0]
            print(f"  timestamp={ts_s}  watts={watts:.2f}")

def explore_h5_vairance(path):
    APPLIANCES  = ["kettle", "toaster", "microwave", "dishwasher", "fridge", "washing_machine"]
    H1_CHANNELS = [10, 11, 13, 6, 12, 5]
    channel_map = {f"meter{ch}": app for ch, app in zip(H1_CHANNELS, APPLIANCES)}

    with h5py.File(path, "r") as f:

        print(f"  {'meter':<10} {'rows':>10}  {'date range':<40}  {'mean W':>8}  {'max W':>8}  {'appliance'}")
        print(f"  {'-'*10} {'-'*10}  {'-'*40}  {'-'*8}  {'-'*8}  {'-'*20}")

        for meter_key, app in channel_map.items():
            table   = f[f"building1/elec/{meter_key}/table"][:]
            ts_s    = table["index"] // 1_000_000_000
            power   = table["values_block_0"].squeeze()
            t_start = pd.Timestamp(int(ts_s[0]),  unit="s")
            t_end   = pd.Timestamp(int(ts_s[-1]), unit="s")
            print(f"  {meter_key:<10} {len(table):>10}  {str(t_start)[:19] + ' → ' + str(t_end)[:19]:<40}  {power.mean():>8.2f}  {power.max():>8.2f}  {app}")

def extract_stitch_h5(h5_path, out_dir="~/thesis/data/test_set_2017"):
    """
    Reads 6 appliance meters from building1, applies same 6s bucketing
    as stitch_resample_6s in preprocess.py, outer joins, slices Jan 2017,
    saves CSV.
    """
    # channel order must match appliance order in binarize_labels call
    APPLIANCES  = ["kettle", "toaster", "microwave", "dishwasher", "fridge", "washing_machine"]
    H1_CHANNELS = [10, 11, 13, 6, 12, 5]

    JAN_2017_START = 1483228800   # 2017-01-01 00:00:00 UTC
    JAN_2017_END   = 1485907200   # 2017-02-01 00:00:00 UTC

    out_dir  = os.path.expanduser(out_dir)
    out_path = os.path.join(out_dir, "house1_2017_jan_stitched.csv")

    if os.path.exists(out_path):
        print(f"Already exists, skipping: {out_path}")
        return out_path

    channel_arrays  = []
    all_timestamps  = []

    with h5py.File(os.path.expanduser(h5_path), "r") as f:
        for ch in H1_CHANNELS:
            table  = f[f"building1/elec/meter{ch}/table"][:]
            ts_s   = (table["index"] // 1_000_000_000).astype(np.int64)
            vals   = table["values_block_0"].squeeze().astype(np.float64)

            # same 6s bucketing as stitch_resample_6s
            bucket_ts        = (ts_s // 6) * 6
            unique_ts, inv   = np.unique(bucket_ts, return_inverse=True)
            sums             = np.bincount(inv, weights=vals)
            counts           = np.bincount(inv)
            mean_vals        = sums / counts

            # slice to Jan 2017 before joining
            mask      = (unique_ts >= JAN_2017_START) & (unique_ts < JAN_2017_END)
            unique_ts = unique_ts[mask]
            mean_vals = mean_vals[mask]

            channel_arrays.append((unique_ts, mean_vals))
            all_timestamps.append(unique_ts)

    # outer join across all channels
    all_timestamps = np.unique(np.concatenate(all_timestamps))
    stitched_vals  = np.full((len(all_timestamps), len(H1_CHANNELS)), np.nan)

    for col_idx, (ts, vals) in enumerate(channel_arrays):
        idx = np.searchsorted(all_timestamps, ts)
        stitched_vals[idx, col_idx] = vals

    header = ["timestamp"] + [f"ch{ch}" for ch in H1_CHANNELS]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(len(all_timestamps)):
            row = [all_timestamps[i]] + stitched_vals[i].tolist()
            writer.writerow(row)

    print(f"Saved: {out_path}  ({len(all_timestamps)} rows)")
    return out_path

def daily_active_summary(binarized_csv):
    """
    Prints a DataFrame of active 6s windows per appliance per day of January 2017.
    Returns the DataFrame so you can save it with daily.to_csv(...) if needed.
    """

    APPLIANCES = ["kettle", "toaster", "microwave", "dishwasher", "fridge", "washing_machine"]

    df = pd.read_csv(os.path.expanduser(binarized_csv))
    df["date"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.date

    daily = df.groupby("date")[APPLIANCES].sum().reset_index()

    print(daily.to_string(index=False))

    return daily

if __name__ == "__main__":
    APPLIANCES = ["kettle", "toaster", "microwave", "dishwasher", "fridge", "washing_machine"]
    #download_ukdale_h5()
    #h5_path = os.path.expanduser("~/thesis/data/test_set_2017/ukdale.h5")
    #explore_h5(h5_path)
    #explore_h5_vairance(h5_path)
    #stitched_csv = extract_stitch_h5(h5_path)
    #run_label_eda(house="1_2017",labels_path=stitched_csv,hist_bucket_size=10,on_threshold=5,)
    #h1_t = [10, 10, 10, 10, 10, 10]  # [washing_machine, dishwasher, kettle, toaster, fridge, microwave]
    #binarized_csv = preprocess.binarize_labels(stitched_csv,appliance_names=APPLIANCES,threshold=h1_t)
    #binarized_csv.to_csv("~/thesis/data/test_set_2017/house1_jan2017_binarized.csv", index=False)
    #daily = daily_active_summary("~/thesis/data/test_set_2017/house1_jan2017_binarized.csv")
    #daily.to_csv("~/thesis/data/test_set_2017/daily_active_jan2017.csv", index=False)
    
    downloader.download_flac_files(base="https://dap.ceda.ac.uk/edc/efficiency/residential/EnergyConsumption/Domestic/UK-DALE-2017/UK-DALE-2017-16kHz/", house=1, week="04", year=2017, days=(4,), active_hrs=False, active_range=(7, 23), download_dir="~/thesis/data/test_set_2017")
    
    preprocess.chop_flac(flac_folder="~/thesis/data/test_set_2017/house_1/flac_files/2017/wk04/", cfg_file="~/Downloads/calibration_house_1.cfg", output_base="~/thesis/data/test_set_2017/house_1/stft_segments/2017/wk04/", sample_rate=16000, window_sec=6, hop_samples=512, n_fft=1024, scale="db", scale_f=20, mode="none", channel=1)
    
    X,y=preprocess.mash_that("~/thesis/data/test_set_2017/house1_jan2017_binarized.csv", "~/thesis/data/test_set_2017/house_1/stft_segments/2017/wk04/current/", path=True)
    X = np.array(X)
    y = np.array(y)
    print(f"X shape: {len(X)}, y shape: {len(y)}")
    print(f"Example X[0] shape: {X[0].shape}, y[0]: {y[0]}")

    print(f"active samples per appliance: {y.sum(axis=0)}")

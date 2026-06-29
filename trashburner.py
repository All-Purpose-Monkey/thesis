import numpy as np
import os
import datetime
import csv
import numpy as np
import configparser
import glob
import soundfile as sf
import librosa
import matplotlib.pyplot as plt

def normalize_flac_level(signal, mode="zscore"):

    """

    Normalization applied per FLAC file (before STFT)

    """

    if mode == "zscore":

        mean = np.mean(signal, axis=0, keepdims=True)

        std = np.std(signal, axis=0, keepdims=True) + 1e-8

        return (signal - mean) / std

    elif mode == "minmax":

        min_v = np.min(signal, axis=0, keepdims=True)

        max_v = np.max(signal, axis=0, keepdims=True)

        return (signal - min_v) / (max_v - min_v + 1e-8)

    elif mode == "none":

        return signal

    else:

        raise ValueError(f"Unknown normalization mode: {mode}")
    
def chop_flac(flac_folder, cfg_file, output_base, sample_rate=16000, window_sec=6, hop_samples=256, n_fft=2048, scale="db", mode="zscore", db_scale=20):

    ADC_HALF_RANGE = 2**31
    config = configparser.ConfigParser()
    cfg_path = os.path.expanduser(cfg_file)
    read_files = config.read(cfg_path)

    if not read_files:
        raise FileNotFoundError(f"Calibration file not found: {cfg_path}")

    volts_per_adc_step = float(config["Calibration"]["volts_per_adc_step"])
    amps_per_adc_step = float(config["Calibration"]["amps_per_adc_step"])

    print(f"Calibration loaded:\nVolts per ADC step: {volts_per_adc_step}\nAmps per ADC step: {amps_per_adc_step}")

    flac_path = os.path.expanduser(flac_folder)
    output_base = os.path.expanduser(output_base)
    os.makedirs(output_base, exist_ok=True)

    # ---------------------------------
    # HANDLE FILE vs FOLDER
    # ---------------------------------
    if os.path.isfile(flac_path) and flac_path.endswith(".flac"):
        flac_files = [flac_path]
    else:
        flac_files = sorted(glob.glob(os.path.join(flac_path, "*.flac")))

    print("Found FLAC files:", flac_files)

    for flac_file in flac_files:

        filename = os.path.basename(flac_file).split(".")[0]
        file_folder = os.path.join(output_base, filename)

        os.makedirs(file_folder, exist_ok=True)

        existing_segments = glob.glob(os.path.join(file_folder, "*.npy"))

        if len(existing_segments) >= 500:
            print(f"Skipping {filename}: already {len(existing_segments)} segments")
            continue

        print("Processing:", filename)

        audio, sr = sf.read(flac_file)

        if sr != sample_rate:
            raise ValueError(f"Expected sample rate {sample_rate}, got {sr}")

        if audio.ndim < 2 or audio.shape[1] < 2:
            raise ValueError("Expected at least 2 channels (Voltage, Current)")

        voltage = volts_per_adc_step * ADC_HALF_RANGE * audio[:, 0]
        current = amps_per_adc_step * ADC_HALF_RANGE * audio[:, 1]

        calibrated_audio = np.column_stack([voltage, current])
# FLAC-level normalization (choose mode)

        calibrated_audio = normalize_flac_level(calibrated_audio, mode=mode)
        win_samples = window_sec * sr
        num_segments = int(np.ceil(len(calibrated_audio) / win_samples))

        for seg_idx in range(num_segments):

            file_num = int(filename)
            timestamp = file_num + (seg_idx + 1) * 6

            output_path = os.path.join(file_folder, f"{timestamp}.npy")

            if os.path.exists(output_path):
                continue

            start = seg_idx * win_samples
            stop = start + win_samples

            segment = calibrated_audio[start:stop, :]

            if segment.shape[0] < win_samples:
                pad_len = win_samples - segment.shape[0]
                segment = np.pad(segment, ((0, pad_len), (0, 0)), mode='constant')

            stft_volt = np.abs(librosa.stft(
                segment[:, 0],
                n_fft=n_fft,
                hop_length=hop_samples,
                window='hann'
            ))
            epsilon = 1e-8

            if scale == "db":

                stft_volt = db_scale * np.log10(stft_volt + epsilon)
            
            elif scale == "log":
                stft_volt = np.log1p(stft_volt + epsilon)

            np.save(output_path, stft_volt.astype(np.float16))

        print(f"Processed {filename}: {num_segments} segments saved\n")

def draw_spectrogram(input="~/thesis/house_1/stft_segments/2013/wk38/"):

    input_path = os.path.expanduser(input)

    # ---------------------------------
    # HANDLE FILE vs FOLDER
    # ---------------------------------
    if os.path.isfile(input_path) and input_path.endswith(".npy"):
        npy_file = input_path
    else:
        npy_files = sorted(glob.glob(os.path.join(input_path, "*.npy")))

        if not npy_files:
            raise FileNotFoundError("No .npy files found in the given folder.")

        print("Available spectrograms:")
        for i, f in enumerate(npy_files[:20]):  # limit print
            print(f"{i}: {os.path.basename(f)}")

        idx = int(input(f"Select file index (0–{len(npy_files)-1}): "))
        npy_file = npy_files[idx]

    print(f"Loading: {npy_file}")

    # ---------------------------------
    # LOAD DATA
    # ---------------------------------
    spectrogram = np.load(npy_file)

    # ---------------------------------
    # PLOT
    # ---------------------------------
    plt.figure(figsize=(10, 6))
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='magma')
    plt.colorbar(label="Log Magnitude")
    plt.title(f"Spectrogram: {os.path.basename(npy_file)}")
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency Bins")
    plt.tight_layout()
    plt.show()

# Load CSV
chop_flac(flac_folder="~/thesis/house_1/flac_files/2013/wk38/1379289600.flac", cfg_file="~/Downloads/calibration_house_1.cfg", output_base="~/thesis/house_1/stft_segments/2013/wk38/", sample_rate=16000, window_sec=6, hop_samples=512, n_fft=1024, scale="db", mode="zscore", db_scale=50)
draw_spectrogram(input="~/thesis/house_1/stft_segments/2013/wk38/1379289600/1379290542.npy")
draw_spectrogram(input="~/thesis/house_1/stft_segments/2013/wk38/1379289600/1379290548.npy")
draw_spectrogram(input="~/thesis/house_1/stft_segments/2013/wk38/1379289600/1379290554.npy")
draw_spectrogram(input="~/thesis/house_1/stft_segments/2013/wk38/1379289600/1379290560.npy")
draw_spectrogram(input="~/thesis/house_1/stft_segments/2013/wk38/1379289600/1379290566.npy")
draw_spectrogram(input="~/thesis/house_1/stft_segments/2013/wk38/1379289600/1379290572.npy")
draw_spectrogram(input="~/thesis/house_1/stft_segments/2013/wk38/1379289600/1379290578.npy")
draw_spectrogram(input="~/thesis/house_1/stft_segments/2013/wk38/1379289600/1379290584.npy")
draw_spectrogram(input="~/thesis/house_1/stft_segments/2013/wk38/1379289600/1379290890.npy")
import os
import audioread
import numpy as np
import soundfile as sf
import configparser
from scipy.signal import stft
import matplotlib.pyplot as plt

# -----------------------------
# Paths
# -----------------------------
flac_file = os.path.expanduser("~/Downloads/flac1.flac")
wav_file = os.path.expanduser("~/Downloads/wav1.wav")
cfg_file = os.path.expanduser("~/Downloads/calibration_house_1.cfg")
out_dir = os.path.expanduser("~/Downloads/stft_images")

os.makedirs(out_dir, exist_ok=True)

# -----------------------------
# Read calibration.cfg
# -----------------------------
config = configparser.ConfigParser()
config.read(cfg_file)

volts_per_adc_step = float(config["Calibration"]["volts_per_adc_step"])
amps_per_adc_step = float(config["Calibration"]["amps_per_adc_step"])

ADC_HALF_RANGE = 2**31  # 32-bit signed ADC

# -----------------------------
# FLAC → WAV (using audioread)
# -----------------------------
samples = []
with audioread.audio_open(flac_file) as f:
    fs = f.samplerate
    channels = f.channels
    for buf in f:
        data = np.frombuffer(buf, dtype=np.int16)
        samples.append(data)

audio = np.concatenate(samples)

if channels > 1:
    audio = audio.reshape(-1, channels)

sf.write(wav_file, audio, fs)

# -----------------------------
# Load WAV and calibrate
# -----------------------------
wav_data, fs = sf.read(wav_file)

if wav_data.ndim < 2:
    raise ValueError("Expected at least 2 channels (Voltage, Current)")

# Assumption:
# Channel 0 = Voltage, Channel 1 = Current
voltage_adc = wav_data[:, 0]
current_adc = wav_data[:, 1]

voltage = volts_per_adc_step * ADC_HALF_RANGE * voltage_adc
current = amps_per_adc_step * ADC_HALF_RANGE * current_adc

# -----------------------------
# MATLAB STFT (first 4 segments)
# -----------------------------

win_len = 1024
hop = 512

for i in range(4):
    start = i * hop
    stop = start + win_len * 2
    v_seg = voltage[start:stop]

    f, t, Zxx = stft(
        v_seg,
        fs=fs,
        nperseg=win_len,
        noverlap=win_len - hop
    )

    plt.figure()
    plt.pcolormesh(t, f, 20 * np.log10(np.abs(Zxx) + 1e-12), shading='gouraud')
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.title(f"Voltage STFT {i+1}")
    plt.ylim(0, 500)
    plt.colorbar()

    plt.savefig(os.path.join(out_dir, f"voltage_stft_{i+1}.png"))
    plt.close()
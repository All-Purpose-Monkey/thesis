import matlab.engine
import os

# Start MATLAB
eng = matlab.engine.start_matlab()

# Input FLAC file
flac_file = os.path.expanduser("~/Downloads/flac1.flac")

# Read FLAC using MATLAB
x, fs = eng.audioread(flac_file, nargout=2)

# Output WAV file
wav_file = os.path.expanduser("~/Downloads/wav1.wav")

# Write WAV using MATLAB
eng.audiowrite(wav_file, x, fs, nargout=0)
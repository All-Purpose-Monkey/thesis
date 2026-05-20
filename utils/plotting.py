import matplotlib.pyplot as plt
import numpy as np

def plot_stft(stft):

    stft_db = 20 * np.log10(stft + 1e-6)

    plt.figure(figsize=(8,4))

    plt.imshow(
        stft_db,
        origin="lower",
        aspect="auto",
        cmap="magma"
    )

    plt.colorbar(label="dB")

    plt.xlabel("Time")

    plt.ylabel("Frequency")

    plt.show()
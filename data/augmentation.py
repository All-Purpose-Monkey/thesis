import numpy as np

def view_1(x, max_time_width=15, noise_std=0.05):
    """
    Structure-preserving view: mild noise + time shift + narrow time mask.
    Destroys temporal position but preserves frequency structure.
    x: np.ndarray (1, F, T)
    """
    x = x.copy()
    _, F, T = x.shape

    # additive gaussian noise with small stddev (relative to typical load signature amplitude)
    x += np.random.randn(*x.shape) * noise_std

    # random time shift (roll, not zero-pad, so energy is preserved)
    #shift = np.random.randint(-T // 8, T // 8)
    #x = np.roll(x, shift, axis=2)

    # narrow time mask — upper 25% of time axis only
    #w = np.random.randint(1, min(max_time_width, T // 4) + 1)
    #start = np.random.randint(0, T - w + 1)
    #x[:, :, start:start + w] = 0

    return x


def view_2(x, max_freq_width=50, high_freq_ratio=0.5):
    """
    Frequency-aware view: freq mask restricted to high-freq bins only.
    Preserves low-frequency load signature (fridge compressor, motor hum).
    x: np.ndarray (1, F, T)
    """
    x = x.copy()
    _, F, T = x.shape

    # mask only in the upper half of frequency bins
    #high_start = int(F * high_freq_ratio)
    #high_bins  = F - high_start

    w = np.random.randint(5, max_freq_width + 1)
    start = np.random.randint(0,  w + 1)
    x[:, start:start + w, :] = 0

    # time flip — symmetric on/off transients make this physically valid
    if np.random.rand() > 0.5:
        x = np.flip(x, axis=2).copy()

    return x
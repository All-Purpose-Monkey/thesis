import numpy as np

def time_mask(x, max_width=10):

    _, H, W = x.shape

    w = np.random.randint(
        1,
        min(max_width, W) + 1
    )

    start = np.random.randint(0, W - w + 1)

    x[:, :, start:start+w] = 0

    return x


def freq_mask(x, max_width=10):

    _, H, W = x.shape

    w = np.random.randint(
        1,
        min(max_width, H) + 1
    )

    start = np.random.randint(0, H - w + 1)

    x[:, start:start+w, :] = 0

    return x
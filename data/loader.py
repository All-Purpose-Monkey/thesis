import numpy as np
from sklearn.model_selection import train_test_split

def split_data(X, y, test_size=0.2, seed=42):

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        shuffle=True
    )


def remove_nan_rows(X, y):

    clean_X = []
    clean_y = []

    for xi, yi in zip(X, y):

        if not np.isnan(yi).any():

            clean_X.append(xi)
            clean_y.append(yi)

    return clean_X, clean_y


def normalize_labels(y_train, y_test):

    y_train_arr = np.array(y_train)

    y_max = y_train_arr.max(axis=0)

    y_max[y_max == 0] = 1.0

    y_train_norm = y_train_arr / y_max
    y_test_norm = np.array(y_test) / y_max

    return y_train_norm, y_test_norm, y_max
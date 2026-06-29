import numpy as np
import os

def test_timestamp_spacing_6_seconds(csv_path):
    """
    Test that consecutive timestamps in the stitched CSV are exactly 6 seconds apart.
    """
    csv_path = os.path.expanduser(csv_path)  # Expand ~ to home directory

    # Load only timestamp column for efficiency
    timestamps = np.loadtxt(
        csv_path,
        delimiter=",",
        skiprows=1,
        usecols=0
    )

    diffs = np.diff(timestamps)
    bad_rows = np.where(diffs != 6)[0]

    if len(bad_rows) == 0:
        print("✅ All timestamps are aligned: rows safe to use")
    else:
        raise AssertionError(f"Found invalid timestamp gaps at rows: {bad_rows[:10]} (showing first 10)")
"""Helpers for loading, resampling, trimming and chunking the recordings.

Sensor Logger writes the axes as z, y, x, so the loader puts them back in x, y, z order.
"""
import numpy as np
import pandas as pd
from scipy.signal import decimate

AXES = ["x", "y", "z"]
TARGET_FS = 50          # common sampling rate (Hz)
TRIM_S = 3.0            # seconds cut from each end of a clip
CHUNK_S = 10.0          # chunk length (s)
MIN_CHUNK_S = 5.0       # shorter leftovers are dropped


def load_sensor(folder, sensor):
    """Read one sensor file. Returns time in seconds and the x, y, z values."""
    df = pd.read_csv(f"{folder}/{sensor}.csv").sort_values("seconds_elapsed")
    return df["seconds_elapsed"].to_numpy(float), df[AXES].to_numpy(float)


def native_rate(t):
    """Sampling rate from the median time step."""
    return 1.0 / np.median(np.diff(t))


def to_uniform(t, X, fs):
    """Interpolate onto an evenly spaced time grid at fs Hz."""
    grid = np.arange(t[0], t[-1], 1.0 / fs)
    return grid, np.column_stack([np.interp(grid, t, X[:, k]) for k in range(X.shape[1])])


def harmonise(t, X, fs_native, fs_target=TARGET_FS):
    """Resample to fs_target Hz: interpolate to 100 Hz, low-pass filter, then decimate."""
    base = 100
    assert base % fs_target == 0
    grid, Xu = to_uniform(t, X, base)
    q = base // fs_target
    Xd = decimate(Xu, q, ftype="fir", axis=0, zero_phase=True)
    return grid[::q][: len(Xd)], Xd


def trim(t, X, seconds=TRIM_S):
    """Cut the first and last `seconds` of a clip."""
    keep = (t >= t[0] + seconds) & (t <= t[-1] - seconds)
    return t[keep], X[keep]


def make_chunks(n, fs, chunk_s=CHUNK_S, min_s=MIN_CHUNK_S):
    """Start and stop indices of non-overlapping chunks. A leftover of min_s or more is kept."""
    L = int(chunk_s * fs)
    out = [(s, s + L) for s in range(0, n - L + 1, L)]
    rest = n - (out[-1][1] if out else 0)
    if rest >= min_s * fs:
        out.append((n - rest, n))
    return out


def load_pair(raw_dir, rec):
    """Load the accelerometer and gyroscope CSVs of one recording from data/raw."""
    out = {}
    for s in ("accelerometer", "gyroscope"):
        df = pd.read_csv(f"{raw_dir}/{rec}_{s}.csv").sort_values("seconds_elapsed")
        out[s] = (df["seconds_elapsed"].to_numpy(float), df[AXES].to_numpy(float), df["time"].to_numpy(np.int64))
    return out

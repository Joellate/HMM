"""Data loading, quality control, resampling, trimming and chunking utilities.

All functions use only numpy, pandas and scipy. Sensor Logger exports columns
in the order ``time, seconds_elapsed, z, y, x``; the loader re-orders them to x, y, z.
"""
import numpy as np
import pandas as pd
from scipy.signal import decimate

AXES = ["x", "y", "z"]
TARGET_FS = 50          # harmonised sampling rate (Hz)
TRIM_S = 3.0            # seconds trimmed at both ends (phone handling)
CHUNK_S = 10.0          # chunk length (s)
MIN_CHUNK_S = 5.0       # remainders shorter than this are dropped


def load_sensor(folder, sensor):
    """Return (t, X) for one sensor file; t in seconds, X columns ordered x, y, z."""
    df = pd.read_csv(f"{folder}/{sensor}.csv").sort_values("seconds_elapsed")
    return df["seconds_elapsed"].to_numpy(float), df[AXES].to_numpy(float)


def native_rate(t):
    """Sampling rate estimated from the median timestamp spacing."""
    return 1.0 / np.median(np.diff(t))


def to_uniform(t, X, fs):
    """Linearly interpolate irregular samples onto a uniform grid at ``fs`` Hz."""
    grid = np.arange(t[0], t[-1], 1.0 / fs)
    return grid, np.column_stack([np.interp(grid, t, X[:, k]) for k in range(X.shape[1])])


def harmonise(t, X, fs_native, fs_target=TARGET_FS):
    """Resample to ``fs_target`` Hz: interpolate to an exact 100 Hz grid, then
    anti-alias (zero-phase FIR) and decimate by an integer factor."""
    base = 100
    assert base % fs_target == 0
    grid, Xu = to_uniform(t, X, base)
    q = base // fs_target
    Xd = decimate(Xu, q, ftype="fir", axis=0, zero_phase=True)
    return grid[::q][: len(Xd)], Xd


def trim(t, X, seconds=TRIM_S):
    """Drop the first and last ``seconds`` of a clip (phone-handling transients)."""
    keep = (t >= t[0] + seconds) & (t <= t[-1] - seconds)
    return t[keep], X[keep]


def make_chunks(n, fs, chunk_s=CHUNK_S, min_s=MIN_CHUNK_S):
    """Non-overlapping chunk index ranges (start, stop) over n samples."""
    L = int(chunk_s * fs)
    out = [(s, s + L) for s in range(0, n - L + 1, L)]
    rest = n - (out[-1][1] if out else 0)
    if rest >= min_s * fs:
        out.append((n - rest, n))
    return out


def load_pair(raw_dir, rec):
    """Load the labelled raw CSVs of one recording from the repo's data/raw folder.
    Returns {'accelerometer': (t, X, epoch_ns), 'gyroscope': (t, X, epoch_ns)}."""
    out = {}
    for s in ("accelerometer", "gyroscope"):
        df = pd.read_csv(f"{raw_dir}/{rec}_{s}.csv").sort_values("seconds_elapsed")
        out[s] = (df["seconds_elapsed"].to_numpy(float), df[AXES].to_numpy(float), df["time"].to_numpy(np.int64))
    return out

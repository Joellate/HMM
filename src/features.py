"""Windowing and feature extraction for the accelerometer and gyroscope."""
import numpy as np
import pandas as pd

FS = 50                  # Hz
WIN_S = 2.0              # window length in seconds
WIN = int(WIN_S * FS)    # 100 samples
STEP = WIN // 2          # 50% overlap
LOG_FEATS = ("var", "rms", "sma", "energy")   # skewed features, log10 before the Z-score
EPS = 1e-8

SENSORS = {"acc": ["ax", "ay", "az"], "gyr": ["gx", "gy", "gz"]}
BASE = ["mean", "var", "rms", "sma", "corr", "domfreq", "energy", "entropy"]
FEATURE_NAMES = [f"{s}_{b}" for s in SENSORS for b in BASE]


def window_indices(n, win=WIN, step=STEP):
    """Start and stop indices of the windows that fit inside n samples."""
    return [(s, s + win) for s in range(0, n - win + 1, step)]


def _spectrum(mag, fs=FS):
    """Power spectrum of the detrended, Hann-windowed signal without the DC term."""
    x = (mag - mag.mean()) * np.hanning(len(mag))
    P = np.abs(np.fft.rfft(x)) ** 2 / len(x)
    f = np.fft.rfftfreq(len(x), 1 / fs)
    return f[1:], P[1:]


def window_features(W):
    """Eight features from one sensor window W (n x 3).

    Time: mean magnitude, variance, RMS, SMA, axis correlation.
    Frequency (from the FFT of the magnitude): dominant frequency, spectral energy, spectral entropy.
    """
    mag = np.linalg.norm(W, axis=1)
    var = W.var(axis=0).sum()
    rms = np.sqrt((mag ** 2).mean())
    sma = np.abs(W).sum(axis=1).mean()
    C = np.corrcoef(W.T)
    C = np.nan_to_num(C, nan=0.0)
    corr = np.mean(np.abs([C[0, 1], C[0, 2], C[1, 2]]))
    f, P = _spectrum(mag)
    dom = f[np.argmax(P)]
    energy = P.sum()
    p = P / (P.sum() + EPS)
    ent = -(p * np.log2(p + EPS)).sum() / np.log2(len(p))   # scaled to 0-1
    return dict(mean=mag.mean(), var=var, rms=rms, sma=sma, corr=corr, domfreq=dom, energy=energy, entropy=ent)


def recording_windows(df, chunk_aware=True):
    """Cut a recording into windows and compute their features. With chunk_aware, windows stay inside a chunk."""
    rows = []
    groups = [(c, g) for c, g in df.groupby("chunk") if c >= 0] if chunk_aware else [(-1, df)]
    for c, g in groups:
        base = g.index[0]
        for a, b in window_indices(len(g)):
            seg = g.iloc[a:b]
            r = {"chunk": c, "t_start": seg.t.iloc[0], "t_end": seg.t.iloc[-1] + 1 / FS}
            if "t_raw" in seg:   # clock of the original recording, used to label the multi-activity clip
                r["t_raw_start"], r["t_raw_end"] = seg.t_raw.iloc[0], seg.t_raw.iloc[-1] + 1 / FS
            for s, cols in SENSORS.items():
                for k, v in window_features(seg[cols].to_numpy()).items():
                    r[f"{s}_{k}"] = v
            rows.append(r)
    return pd.DataFrame(rows)


def transform_features(F):
    """Take log10 of variance, RMS, SMA and energy, which are strongly skewed."""
    X = F[FEATURE_NAMES].copy()
    for c in X.columns:
        if c.split("_", 1)[1] in LOG_FEATS:
            X[c] = np.log10(X[c] + EPS)
    return X


class ZScaler:
    """Z-score scaler. Fit it on the training windows only."""
    def fit(self, X):
        X = np.asarray(X, float); self.mu = X.mean(0); self.sd = X.std(0) + 1e-12; return self
    def transform(self, X):
        return (np.asarray(X, float) - self.mu) / self.sd

# src/preprocess.py
import numpy as np, pandas as pd
from scipy.signal import butter, filtfilt
from pathlib import Path

FS   = 128          # Sampling rate in Hz
WIN  = 1 * FS       # 1-second window  (was 2*FS)
STEP = int(0.5 * FS)  # 0.5-second step (50 % overlap)

RAW_CSV = "data/interim/eeg_eye_state.csv"
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Band-pass filter
def bandpass(signal, fs=FS, lo=1, hi=45, order=4):
    ny = 0.5 * fs
    b, a = butter(order, [lo/ny, hi/ny], btype="band")
    return filtfilt(b, a, signal)

# ---------------- Load data
df = pd.read_csv(RAW_CSV)
labels = df["eyeDetection"].values
signals = df.drop(columns=["eyeDetection"]).values.T  # shape (14, 14980)

# ---------------- Filter each channel
signals = np.array([bandpass(ch) for ch in signals])  # shape (14, 14980)

# ---------------- Windowing
X, y = [], []
for start in range(0, signals.shape[1] - WIN + 1, STEP):
    seg = signals[:, start:start+WIN]          # (14, 256)
    lbl = int(labels[start:start+WIN].mean() > 0.5)
    X.append(seg); y.append(lbl)

X = np.array(X)  # (N, 14, 256)
y = np.array(y)  # (N,)

np.save(OUT_DIR / "X.npy", X)
np.save(OUT_DIR / "y.npy", y)

print(f"✅ Saved windows → {OUT_DIR} | X: {X.shape}, y: {y.shape}")

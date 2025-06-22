# src/features.py  (adds feature_names list)

import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from pathlib import Path
from tqdm import tqdm

FS = 128
BANDS = {"delta":(0.5,4),"theta":(4,8),"alpha":(8,13),"beta":(13,30)}

# Channel names from the dataset header
CHANNELS = [
    "AF3","F7","F3","FC5","T7","P7","O1",
    "O2","P8","T8","FC6","F4","F8","AF4"
]

def bandpower(sig):
    f, Pxx = welch(sig, fs=FS, nperseg=min(len(sig),128))
    out = {band: np.trapz(Pxx[(f>=lo)&(f<=hi)], f[(f>=lo)&(f<=hi)])
           for band,(lo,hi) in BANDS.items()}
    out["total"] = np.trapz(Pxx, f)
    return out

def spectral_entropy(sig):
    f,Pxx = welch(sig, fs=FS, nperseg=min(len(sig),128))
    Pn = Pxx/np.sum(Pxx)
    return -np.sum(Pn*np.log2(Pn+1e-12))

def hjorth(sig):
    d1 = np.diff(sig, prepend=sig[0]); d2 = np.diff(d1, prepend=d1[0])
    var0,var1,var2 = np.var(sig),np.var(d1),np.var(d2)
    act  = var0
    mob  = np.sqrt(var1/var0) if var0 else 0
    comp = np.sqrt(var2/var1)/mob if (var1 and mob) else 0
    return act, mob, comp

# ------------------------------------------------------------------
X = np.load("data/processed/X.npy")   # (N,14,L)
y = np.load("data/processed/y.npy")

feat_rows, feat_names = [], []

for ch_name in CHANNELS:
    # build once; names repeat for every sample
    feat_names.extend([
        f"{ch_name}_{band}_abs"  for band in BANDS.keys()          ] +
        [f"{ch_name}_{band}_rel"  for band in BANDS.keys()          ] +
        [f"{ch_name}_theta_alpha_ratio",
         f"{ch_name}_beta_alpha_ratio"] +
        [f"{ch_name}_{stat}"      for stat in ["mean","std","skew","kurt"]] +
        [f"{ch_name}_entropy"] +
        [f"{ch_name}_{p}"         for p in ["activity","mobility","complexity"]]
    )
# ---------------- feature extraction per sample -------------------
for window in tqdm(X, desc="Extracting"):
    row=[]
    for ch_sig,ch_name in zip(window,CHANNELS):
        bp = bandpower(ch_sig); total=bp.pop("total")
        rel = {f"{k}_rel":v/total for k,v in bp.items()}
        row.extend(bp.values())                    # abs 4
        row.extend(rel.values())                  # rel 4
        row.append(bp["theta"]/(bp["alpha"]+1e-6))
        row.append(bp["beta"] /(bp["alpha"]+1e-6))
        row.extend([np.mean(ch_sig),np.std(ch_sig),
                    skew(ch_sig),kurtosis(ch_sig)])
        row.append(spectral_entropy(ch_sig))
        row.extend(hjorth(ch_sig))                # 3
    feat_rows.append(row)

features = np.array(feat_rows)
Path("data/processed").mkdir(exist_ok=True)
np.save("data/processed/X_features.npy", features)
np.save("data/processed/feature_names.npy", np.array(feat_names))  # ✅ save names
np.save("data/processed/y.npy", y)
print("✅ Features:", features.shape, "| Names saved:", len(feat_names))

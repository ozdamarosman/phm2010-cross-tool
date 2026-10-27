# -*- coding: utf-8 -*-
"""Feature extraction for the NASA Ames milling data (see PREREG_nasa.md)."""
import sys
import numpy as np, pandas as pd, scipy.io as sio, pywt
from scipy.signal import welch
from scipy.stats import skew, kurtosis

SRC = sys.argv[1] if len(sys.argv) > 1 else "mill.mat"
OUT = sys.argv[2] if len(sys.argv) > 2 else "nasa_features.csv"
FS = 250.0
F_S = 826 / 60.0; F_T = 6 * F_S
CH = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
SETTING = {1: "S1", 9: "S1", 2: "S2", 12: "S2", 3: "S3", 11: "S3", 4: "S4", 10: "S4", 5: "S5", 16: "S5",
           6: "S6", 15: "S6", 7: "S7", 13: "S7", 8: "S8", 14: "S8"}


def segment(dc):
    idle = np.median(dc[:500]); thr = idle + 0.5 * (np.percentile(dc, 90) - idle)
    act = np.where(dc > thr)[0]
    a, b = act[0], act[-1]; L = b - a
    return a + int(0.1 * L), b - int(0.1 * L)


def band(f, p, f0, hw):
    return float(p[(f >= f0 - hw) & (f <= f0 + hw)].sum())


def feats(x, name):
    d = {}
    rms = np.sqrt(np.mean(x ** 2))
    d.update({f"{name}_mean": x.mean(), f"{name}_std": x.std(), f"{name}_rms": rms, f"{name}_skew": skew(x),
              f"{name}_kurt": kurtosis(x), f"{name}_p2p": np.ptp(x), f"{name}_crest": np.max(np.abs(x)) / rms})
    f, p = welch(x - x.mean(), fs=FS, nperseg=1024)
    es, e2, et = band(f, p, F_S, 1.0), band(f, p, 2 * F_S, 1.5), band(f, p, F_T, 3.0)
    lg = lambda v: np.log10(max(v, 1e-15))
    d.update({f"{name}_E_fs": lg(es), f"{name}_E_2fs": lg(e2), f"{name}_E_ft": lg(et),
              f"{name}_ratio_fs_ft": lg(es) - lg(et),
              f"{name}_centroid": float((f * p).sum() / p.sum()) if p.sum() > 0 else np.nan})
    wp = pywt.WaveletPacket(x - x.mean(), "db4", mode="symmetric", maxlevel=3)
    en = np.array([np.sum(n.data ** 2) for n in wp.get_level(3, order="freq")]); en = en / en.sum()
    d.update({f"{name}_wp{i}": v for i, v in enumerate(en)})
    return d


m = sio.loadmat(SRC, squeeze_me=True, struct_as_record=False)["mill"]
rows = []
for r in m:
    vb = float(r.VB) if np.size(r.VB) else np.nan
    dc = np.asarray(r.smcDC, float)
    a, b = segment(dc)
    row = {"case": int(r.case), "run": int(r.run), "setting": SETTING[int(r.case)], "time": float(r.time),
           "DOC": float(r.DOC), "feed": float(r.feed), "material": int(r.material), "VB": vb,
           "seg_start": a, "seg_end": b, "dc_clipped": float(np.mean(dc[a:b] >= 9.95))}
    for c in CH:
        row.update(feats(np.asarray(getattr(r, c), float)[a:b], c))
    rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(df.shape, "labelled", df.VB.notna().sum(), "seg len median", (df.seg_end - df.seg_start).median(),
      "runs with >5% DC clipping", int((df.dc_clipped > 0.05).sum()))

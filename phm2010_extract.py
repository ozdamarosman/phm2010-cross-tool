# -*- coding: utf-8 -*-
"""
PHM2010 feature extraction for the revised manuscript (V1).

Run on your own computer, next to the original 50 kHz data:

    pip install numpy pandas scipy PyWavelets
    python phm2010_extract.py "C:\\Users\\osman\\Desktop\\osman\\ML\\archive"

Expected layout (the original IEEE DataPort structure):
    archive\\c1\\c1_wear.csv
    archive\\c1\\c1\\c_1_001.csv ... c_1_315.csv
    (same for c4 and c6)

Outputs (written to archive\\phm2010_out\\):
    features.csv      one row per cut: cutter, cut, flute wear, VB_max, VB_mean, 152 features
    cnn_windows.npz   raw 7-channel windows for the 1D-CNN (decimated to 10 kHz, 4096 samples)
    feature_spec.txt  exact settings used (for Methods / Supplementary Table S1)

Upload the phm2010_out folder (zipped) to the chat.
"""
import sys, os, glob, re, time
import numpy as np
import pandas as pd
from scipy.signal import welch, decimate
from scipy.stats import skew, kurtosis
import pywt

# ---------------- settings (reported in the Methods) ----------------
FS = 50_000                      # Hz
RPM = 10_400
F_SPINDLE = RPM / 60.0           # 173.33 Hz
N_FLUTES = 3
F_TOOTH = N_FLUTES * F_SPINDLE   # 520 Hz
N_HARM = 5                       # tooth-passing harmonics 1..5 (520 ... 2600 Hz)
BAND_HW = 5.0                    # +/- Hz around each kinematic frequency
WELCH_NPERSEG = 16_384           # ~3.05 Hz resolution
CENTROID_MAX = 10_000.0          # Hz, spectral centroid band 0-10 kHz
AE_LOW = 1_000.0                 # Hz, AE-RMS low-frequency energy band
WP_WAVELET, WP_LEVEL = "db4", 3  # 8 terminal nodes
TRIM = 0.10                      # drop first/last 10 % of each cut (entry/exit transients)
CNN_DECIM = 5                    # 50 kHz -> 10 kHz
CNN_LEN = 4_096                  # samples per window (~0.41 s)
CUTTERS = ["c1", "c4", "c6"]
CHANNELS = ["force_x", "force_y", "force_z", "vib_x", "vib_y", "vib_z", "ae_rms"]


def band_energy(f, p, f0, hw=BAND_HW):
    m = (f >= f0 - hw) & (f <= f0 + hw)
    return float(p[m].sum())


def time_feats(x, name):
    rms = float(np.sqrt(np.mean(x ** 2)))
    return {
        f"{name}_mean": float(np.mean(x)),
        f"{name}_std": float(np.std(x)),
        f"{name}_rms": rms,
        f"{name}_skew": float(skew(x)),
        f"{name}_kurt": float(kurtosis(x)),
        f"{name}_p2p": float(np.ptp(x)),
        f"{name}_crest": float(np.max(np.abs(x)) / rms) if rms > 0 else np.nan,
    }


def spectral_feats(x, name):
    f, p = welch(x - x.mean(), fs=FS, nperseg=WELCH_NPERSEG)
    d = {}
    e_s = band_energy(f, p, F_SPINDLE)
    d[f"{name}_E_fs"] = e_s
    e_t = None
    for h in range(1, N_HARM + 1):
        e = band_energy(f, p, h * F_TOOTH)
        d[f"{name}_E_ft{h}"] = e
        if h == 1:
            e_t = e
    d[f"{name}_ratio_fs_ft"] = e_s / e_t if e_t > 0 else np.nan
    d[f"{name}_E_sideband"] = band_energy(f, p, F_TOOTH - F_SPINDLE) + band_energy(f, p, F_TOOTH + F_SPINDLE)
    m = f <= CENTROID_MAX
    d[f"{name}_centroid"] = float((f[m] * p[m]).sum() / p[m].sum()) if p[m].sum() > 0 else np.nan
    return d


def wavelet_feats(x, name):
    wp = pywt.WaveletPacket(data=x - x.mean(), wavelet=WP_WAVELET, mode="symmetric", maxlevel=WP_LEVEL)
    nodes = wp.get_level(WP_LEVEL, order="freq")
    en = np.array([np.sum(n.data ** 2) for n in nodes])
    en = en / en.sum() if en.sum() > 0 else en
    return {f"{name}_wp{i}": float(v) for i, v in enumerate(en)}


def ae_feats(x, name):
    d = time_feats(x, name)
    f, p = welch(x - x.mean(), fs=FS, nperseg=WELCH_NPERSEG)
    d[f"{name}_E_low"] = float(p[f <= AE_LOW].sum())
    return d


def process_cut(path):
    sig = pd.read_csv(path, header=None).values.astype(np.float64)
    assert sig.shape[1] == 7, f"{path}: expected 7 columns, found {sig.shape[1]}"
    n = len(sig)
    a, b = int(TRIM * n), int((1 - TRIM) * n)
    seg = sig[a:b]
    feats = {"n_samples": n}
    for j, ch in enumerate(CHANNELS):
        x = seg[:, j]
        if ch == "ae_rms":
            feats.update(ae_feats(x, ch))
        else:
            feats.update(time_feats(x, ch))
            feats.update(spectral_feats(x, ch))
            feats.update(wavelet_feats(x, ch))
    # CNN window: decimate the steady segment, take the central CNN_LEN samples
    dec = decimate(seg, CNN_DECIM, axis=0, zero_phase=True)
    c = len(dec) // 2
    win = dec[c - CNN_LEN // 2: c + CNN_LEN // 2].T.astype(np.float32)   # (7, CNN_LEN)
    return feats, win


def cut_number(path):
    m = re.search(r"_(\d+)\.csv$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def main(root):
    out = os.path.join(root, "phm2010_out")
    os.makedirs(out, exist_ok=True)
    rows, wins, keys = [], [], []
    t0 = time.time()
    for cutter in CUTTERS:
        cdir = os.path.join(root, cutter)
        wear_files = glob.glob(os.path.join(cdir, "**", "*wear*.csv"), recursive=True)
        assert wear_files, f"no wear file under {cdir}"
        wear = pd.read_csv(wear_files[0])
        wear.columns = [c.strip().lower() for c in wear.columns]
        cut_files = sorted([p for p in glob.glob(os.path.join(cdir, "**", "c_*_*.csv"), recursive=True)],
                           key=cut_number)
        print(f"{cutter}: {len(cut_files)} cut files, wear file {os.path.basename(wear_files[0])}")
        for k, path in enumerate(cut_files):
            cut = cut_number(path)
            feats, win = process_cut(path)
            w = wear.loc[wear["cut"] == cut].iloc[0]
            fl = [float(w["flute_1"]), float(w["flute_2"]), float(w["flute_3"])]
            row = {"cutter": cutter, "cut": cut, "flute_1": fl[0], "flute_2": fl[1], "flute_3": fl[2],
                   "VB_max_um": max(fl), "VB_mean_um": float(np.mean(fl))}
            row.update(feats)
            rows.append(row); wins.append(win); keys.append(f"{cutter}_{cut:03d}")
            if (k + 1) % 25 == 0:
                print(f"  {cutter} {k + 1}/{len(cut_files)}  ({time.time() - t0:.0f} s)")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "features.csv"), index=False)
    np.savez_compressed(os.path.join(out, "cnn_windows.npz"), X=np.stack(wins), keys=np.array(keys))
    with open(os.path.join(out, "feature_spec.txt"), "w") as fh:
        for k_, v_ in [("FS", FS), ("RPM", RPM), ("F_SPINDLE", F_SPINDLE), ("F_TOOTH", F_TOOTH),
                       ("N_HARM", N_HARM), ("BAND_HW", BAND_HW), ("WELCH_NPERSEG", WELCH_NPERSEG),
                       ("CENTROID_MAX", CENTROID_MAX), ("AE_LOW", AE_LOW), ("WP", f"{WP_WAVELET} level {WP_LEVEL}"),
                       ("TRIM", TRIM), ("CNN_DECIM", CNN_DECIM), ("CNN_LEN", CNN_LEN),
                       ("numpy", np.__version__), ("pandas", pd.__version__),
                       ("scipy", __import__("scipy").__version__), ("pywt", pywt.__version__),
                       ("n_features", len([c for c in df.columns if c.startswith(tuple(CHANNELS))]))]:
            fh.write(f"{k_}: {v_}\n")
    print(f"done: {len(df)} cuts, {df.shape[1]} columns -> {out}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")

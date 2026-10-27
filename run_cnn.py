# -*- coding: utf-8 -*-
"""1D-CNN on raw 7-channel windows (10 kHz, 4096 samples) under the same nested LOCO protocol.
Number of training epochs chosen on the grouped inner split (one training cutter -> the other, both
directions); final model refitted on both training cutters; predictions averaged over 5 seeds."""
import sys, time, os
sys.path.insert(0, ".")
import numpy as np, pandas as pd, torch, torch.nn as nn
from core import CUTTERS, recorded, load, rmse

torch.set_num_threads(2)
D = "phm2010_out/"
OUT = "results/"
target = sys.argv[1] if len(sys.argv) > 1 else "VB_max_um"
tag = "max" if target == "VB_max_um" else "mean"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 20

df, _ = load(D + "features.csv", target)
npz = np.load(D + "cnn_windows.npz")
X_all, keys = npz["X"], list(npz["keys"])
idx = {k: i for i, k in enumerate(keys)}
CROP, N_TEST_CROPS, MAX_EP, EVAL_EVERY, BATCH = 2048, 5, 150, 5, 8
SEEDS = [0, 1, 2, 3, 4]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        def blk(i, o, s):
            return nn.Sequential(nn.Conv1d(i, o, 7, stride=s, padding=3), nn.GroupNorm(8, o), nn.ReLU())
        self.f = nn.Sequential(blk(7, 32, 2), nn.MaxPool1d(4), blk(32, 64, 1), nn.MaxPool1d(4),
                               blk(64, 128, 1), nn.AdaptiveAvgPool1d(1))
        self.h = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 1))
    def forward(self, x): return self.h(self.f(x)).squeeze(-1)


def windows(sub):
    return X_all[[idx[f"{c}_{n:03d}"] for c, n in zip(sub.cutter, sub.cut)]]


def test_crops(X):
    starts = np.linspace(0, X.shape[-1] - CROP, N_TEST_CROPS).astype(int)
    return [X[..., s:s + CROP] for s in starts]


def train(Xtr, ytr, epochs, seed, Xval=None, yval=None):
    """Returns model and (if val given) val RMSE every EVAL_EVERY epochs."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    mu = Xtr.mean(axis=(0, 2), keepdims=True); sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-8
    ym, ys = ytr.mean(), ytr.std() + 1e-8
    net = Net(); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xn = (Xtr - mu) / sd; yn = (ytr - ym) / ys
    curve = []
    def predict(X):
        net.eval()
        with torch.no_grad():
            Xs = (X - mu) / sd
            p = np.mean([net(torch.tensor(c, dtype=torch.float32)).numpy() for c in test_crops(Xs)], axis=0)
        return p * ys + ym
    for ep in range(1, epochs + 1):
        net.train()
        perm = rng.permutation(len(Xn))
        for b in range(0, len(perm), BATCH):
            ii = perm[b:b + BATCH]
            s = rng.integers(0, Xn.shape[-1] - CROP + 1, size=len(ii))
            xb = np.stack([Xn[i, :, a:a + CROP] for i, a in zip(ii, s)])
            loss = nn.functional.mse_loss(net(torch.tensor(xb, dtype=torch.float32)),
                                          torch.tensor(yn[ii], dtype=torch.float32))
            opt.zero_grad(); loss.backward(); opt.step()
        if Xval is not None and ep % EVAL_EVERY == 0:
            curve.append(rmse(yval, predict(Xval)))
    return predict, curve


t0 = time.time()
rows, meta = [], []
for test in CUTTERS:
    tr = [c for c in CUTTERS if c != test]
    A = df[(df.cutter == tr[0]) & recorded(df.cut, K)]; B = df[(df.cutter == tr[1]) & recorded(df.cut, K)]
    te = df[df.cutter == test]
    XA, XB, XT = windows(A), windows(B), windows(te)
    _, cAB = train(XA, A.y.values, MAX_EP, 0, XB, B.y.values)
    _, cBA = train(XB, B.y.values, MAX_EP, 0, XA, A.y.values)
    curve = 0.5 * (np.array(cAB) + np.array(cBA))
    E = int((np.argmin(curve) + 1) * EVAL_EVERY)
    Xp = np.concatenate([XA, XB]); yp = np.concatenate([A.y.values, B.y.values])
    yhat = np.mean([train(Xp, yp, E, s)[0](XT) for s in SEEDS], axis=0)
    print(test, "epochs", E, "inner", round(curve.min(), 4), "test RMSE", round(rmse(te.y, yhat), 4),
          f"{time.time()-t0:.0f}s", flush=True)
    meta.append({"test": test, "epochs": E, "inner_curve": curve.tolist()})
    rows.append(pd.DataFrame({"model": "1D-CNN (raw signals)", "cutter": test, "cut": te.cut.values,
                              "y": te.y.values, "yhat": yhat}))
pd.concat(rows).to_csv(OUT + f"preds_cnn_{tag}_k{K}.csv", index=False)
pd.to_pickle(meta, OUT + f"cnn_meta_{tag}_k{K}.pkl")
print("pooled RMSE", round(rmse(pd.concat(rows).y, pd.concat(rows).yhat), 4))

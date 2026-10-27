# -*- coding: utf-8 -*-
"""NASA milling: nested grouped evaluation (see PREREG_nasa.md)."""
import sys, time, pickle, itertools
sys.path.insert(0, "..")
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from core import LEARNERS, wrap, rmse, grid, SEEDS

OUT = "results/"; import os; os.makedirs(OUT, exist_ok=True)
outer = sys.argv[1] if len(sys.argv) > 1 else "setting"     # setting | case
variant = sys.argv[2] if len(sys.argv) > 2 else "raw"        # raw | norm
d = pd.read_csv("nasa_features.csv")
d = d[~((d.case == 12) & (d.run == 1))]
CH = ["smcAC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
SF = [c for c in d.columns if c.startswith(tuple(c_ + "_" for c_ in CH))]
PROC = ["time", "DOC", "feed", "material"]

if variant == "norm":     # relative to the first valid run of each case
    pos = [f for f in SF if (d[f] > 0).all()]
    for c in d.case.unique():
        m = d.case == c; first = d[m].sort_values("run").iloc[0]
        for f in SF:
            d.loc[m, f] = np.log(d.loc[m, f] / first[f]) if f in pos else d.loc[m, f] - first[f]

L = d[d.VB.notna() & (d.case != 6)].reset_index(drop=True)
L[SF] = L[SF].replace([np.inf, -np.inf], np.nan).fillna(0.0)
G = L.setting if outer == "setting" else L.case


def cv_rmse(make, X, y, groups):
    gkf = GroupKFold(n_splits=min(5, groups.nunique()))
    p = np.zeros(len(y))
    for tr, va in gkf.split(X, y, groups):
        p[va] = make().fit(X[tr], y[tr]).predict(X[va])
    return rmse(y, p)


class Iso:
    def fit(self, X, y): self.m = IsotonicRegression(out_of_bounds="clip").fit(X[:, 0], y); return self
    def predict(self, X): return self.m.predict(X[:, 0])
class Cub:
    def fit(self, X, y): self.p = np.polyfit(X[:, 0], y, 3); return self
    def predict(self, X): return np.polyval(self.p, X[:, 0])
def inter(X):   # time x parameter interactions
    t = X[:, [0]]; return np.hstack([X, t * X[:, 1:]])
class RidgeInt:
    def __init__(self, a): self.a = a
    def fit(self, X, y): self.m = wrap(Ridge(alpha=self.a)).fit(inter(X), y); return self
    def predict(self, X): return self.m.predict(inter(X))


def run_fold(test):
    tr = L[G != test]; te = L[G == test]; gtr = tr.case
    y = tr.VB.values; out = {"test": test}
    preds = {}
    # (i) time-only baseline
    Xt, Xte = tr[["time"]].values, te[["time"]].values
    c = {"isotonic": Iso, "cubic": Cub}
    s = {k: cv_rmse(v, Xt, y, gtr) for k, v in c.items()}; b = min(s, key=s.get)
    preds["Time-only baseline"] = c[b]().fit(Xt, y).predict(Xte); out["time_variant"] = b
    # (ii) process baseline
    Xp, Xpte = tr[PROC].values, te[PROC].values
    cands = {f"ridge_int_{a}": (lambda a=a: RidgeInt(a)) for a in [0.01, 0.1, 1, 10, 100]}
    for p_ in grid({"max_depth": [2, 4, None], "min_samples_leaf": [1, 3]}):
        cands[f"rf_{p_}"] = (lambda p_=p_: wrap(RandomForestRegressor(n_estimators=300, random_state=0, **p_)))
    s = {k: cv_rmse(v, Xp, y, gtr) for k, v in cands.items()}; b = min(s, key=s.get); out["proc_variant"] = b
    if b.startswith("rf"):
        p_ = eval(b[3:])
        preds["Process baseline"] = np.mean([wrap(RandomForestRegressor(n_estimators=300, random_state=sd, **p_))
                                             .fit(Xp, y).predict(Xpte) for sd in SEEDS], axis=0)
    else:
        preds["Process baseline"] = cands[b]().fit(Xp, y).predict(Xpte)
    # sensor feature selection (Ridge proxy with process variables)
    order = tr[SF].corrwith(tr.VB).abs().fillna(0).sort_values(ascending=False).index.tolist()
    def sel(with_proc):
        best, bm = np.inf, 1
        for m in range(1, 21):
            cols = order[:m] + (PROC if with_proc else [])
            e = cv_rmse(lambda: wrap(Ridge(alpha=1.0)), tr[cols].values, y, gtr)
            if e < best - 1e-12: best, bm = e, m
        return order[:bm]
    for tag, wp in [("", True), (" (sensors only)", False)]:
        S = sel(wp); cols = S + (PROC if wp else []); out[f"selected{tag}"] = S
        for name, (mk, g, stoch) in LEARNERS.items():
            sc = [cv_rmse(lambda p=p: wrap(mk(p, 0)), tr[cols].values, y, gtr) for p in g]
            p = g[int(np.argmin(sc))]; ss = SEEDS if stoch else [0]
            preds[name + tag] = np.mean([wrap(mk(p, sd)).fit(tr[cols].values, y).predict(te[cols].values)
                                         for sd in ss], axis=0)
            out.setdefault("configs", {})[name + tag] = p
    rows = [pd.DataFrame({"model": k, "fold": test, "case": te.case.values, "run": te.run.values,
                          "y": te.VB.values, "yhat": v}) for k, v in preds.items()]
    out["preds"] = pd.concat(rows)
    return out


t0 = time.time(); folds = []
for test in sorted(G.unique(), key=str):
    folds.append(run_fold(test)); print(test, f"{time.time()-t0:.0f}s", folds[-1]["proc_variant"],
                                       len(folds[-1]["selected"]), flush=True)
P = pd.concat([f["preds"] for f in folds]); tag = f"{outer}_{variant}"
P.to_csv(OUT + f"preds_{tag}.csv", index=False)
pickle.dump([{k: v for k, v in f.items() if k != "preds"} for f in folds], open(OUT + f"folds_{tag}.pkl", "wb"))

# cluster bootstrap over held-out cases (paired)
rng = np.random.default_rng(0); cases = sorted(P.case.unique()); models = P.model.unique()
se = {m: {c: ((g.y - g.yhat) ** 2).values for c, g in P[P.model == m].groupby("case")} for m in models}
boot = {m: [] for m in models}
for _ in range(2000):
    cs = rng.choice(cases, size=len(cases), replace=True)
    for m in models:
        boot[m].append(np.sqrt(np.mean(np.concatenate([se[m][c] for c in cs]))))
boot = {m: np.array(v) for m, v in boot.items()}
q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
rows = []
for m in models:
    g = P[P.model == m]
    rows.append({"model": m, "RMSE": rmse(g.y, g.yhat),
                 "R2": 1 - np.sum((g.y - g.yhat) ** 2) / np.sum((g.y - g.y.mean()) ** 2), "CI": q(boot[m]),
                 "d_proc": rmse(g.y, g.yhat) - rmse(P[P.model == "Process baseline"].y, P[P.model == "Process baseline"].yhat),
                 "d_proc_CI": q(boot[m] - boot["Process baseline"]),
                 "d_time_CI": q(boot[m] - boot["Time-only baseline"]),
                 "folds_better": int(sum(rmse(h.y, h.yhat) < rmse(P[(P.model == "Process baseline") & (P.fold == f)].y,
                                                                    P[(P.model == "Process baseline") & (P.fold == f)].yhat)
                                         for f, h in g.groupby("fold"))), "n_folds": g.fold.nunique()})
S = pd.DataFrame(rows); S.to_pickle(OUT + f"summary_{tag}.pkl")
pd.set_option("display.width", 250)
print(S.assign(CI=S.CI.map(lambda t: tuple(round(x, 4) for x in t)), d_proc_CI=S.d_proc_CI.map(lambda t: tuple(round(x, 4) for x in t)),
               d_time_CI=S.d_time_CI.map(lambda t: tuple(round(x, 4) for x in t))).round(4).to_string())
print(f"done {time.time()-t0:.0f}s")

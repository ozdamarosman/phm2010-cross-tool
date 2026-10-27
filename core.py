# -*- coding: utf-8 -*-
"""Core pipeline: nested leave-one-cutter-out (LOCO) evaluation on PHM2010 features.

Every modelling decision (feature ranking, subset size, hyperparameters, baseline variant)
is made inside the training pool of an outer fold, with a grouped inner split
(fit on one training cutter, validate on the other, both directions).
"""
import itertools, warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
CUTTERS = ["c1", "c4", "c6"]
CHANNELS = ["force_x", "force_y", "force_z", "vib_x", "vib_y", "vib_z", "ae_rms"]
STAGES = {"initial": (1, 30), "steady": (31, 210), "accelerated": (211, 315)}
SEEDS = [0, 1, 2, 3, 4]


# ---------------------------------------------------------------- data
def load(path, target="VB_max_um"):
    df = pd.read_csv(path)
    feats = [c for c in df.columns if c.startswith(tuple(CHANNELS))]
    for c in feats:                                   # spectral energies are heavy-tailed -> log10
        if "_E_" in c or "ratio_fs_ft" in c:
            df[c] = np.log10(df[c].clip(lower=1e-15))
    df["y"] = df[target] / 1000.0                     # µm -> mm
    return df.sort_values(["cutter", "cut"]).reset_index(drop=True), feats


def recorded(cut, k):
    """Few-shot protocol: only cuts 1, 1+k, 1+2k, ... of a training cutter are recorded."""
    return (cut - 1) % k == 0


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# ---------------------------------------------------------------- models
def wrap(est):
    """Standardise X inside the pipeline and y via a target transformer."""
    return TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), est),
                                      transformer=StandardScaler())


def grid(params):
    keys = list(params)
    return [dict(zip(keys, v)) for v in itertools.product(*[params[k] for k in keys])]


LEARNERS = {
    "Ridge": (lambda p, s: Ridge(**p), grid({"alpha": [0.01, 0.1, 1, 10, 100]}), False),
    "Lasso": (lambda p, s: Lasso(max_iter=50000, **p), grid({"alpha": [0.001, 0.01, 0.05, 0.1, 0.3]}), False),
    "SVR": (lambda p, s: SVR(kernel="rbf", gamma="scale", **p),
            grid({"C": [0.1, 1, 10, 100], "epsilon": [0.05, 0.1, 0.3]}), False),
    "Random Forest": (lambda p, s: RandomForestRegressor(n_estimators=300, random_state=s, n_jobs=1, **p),
                      grid({"max_depth": [2, 4, None], "min_samples_leaf": [1, 3]}), True),
    "Gradient Boosting": (lambda p, s: GradientBoostingRegressor(random_state=s, subsample=0.8, **p),
                          grid({"n_estimators": [100, 300], "learning_rate": [0.03, 0.1], "max_depth": [1, 2]}), True),
    "XGBoost": (lambda p, s: XGBRegressor(random_state=s, subsample=0.8, n_jobs=1, verbosity=0, **p),
                grid({"n_estimators": [100, 300], "learning_rate": [0.03, 0.1], "max_depth": [1, 2]}), True),
    "MLP": (lambda p, s: MLPRegressor(max_iter=3000, random_state=s, **p),
            grid({"hidden_layer_sizes": [(16,), (32, 16)], "alpha": [0.001, 0.01, 0.1, 1.0],
                  "learning_rate_init": [0.001, 0.01]}), True),
}


def inner_rmse(make, XA, yA, XB, yB):
    """Grouped inner split: fit on one training cutter, validate on the other, both directions."""
    m1 = make().fit(XA, yA); m2 = make().fit(XB, yB)
    return 0.5 * (rmse(yB, m1.predict(XB)) + rmse(yA, m2.predict(XA)))


# ---------------------------------------------------------------- feature selection
def select_features(trA, trB, feats, kmax=20, alpha=1.0):
    pool = pd.concat([trA, trB])
    r = pool[feats].corrwith(pool["y"]).abs().fillna(0.0)
    order = r.sort_values(ascending=False).index.tolist()
    best, best_m, curve = np.inf, 1, []
    for m in range(1, kmax + 1):
        f = order[:m]
        e = inner_rmse(lambda: wrap(Ridge(alpha=alpha)), trA[f].values, trA.y.values, trB[f].values, trB.y.values)
        curve.append(e)
        if e < best - 1e-12:
            best, best_m = e, m
    return order[:best_m], curve


# ---------------------------------------------------------------- baseline
def fit_baseline(trA, trB):
    def iso():
        class M:
            def fit(self, c, y):
                self.m = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(c, y); return self
            def predict(self, c): return self.m.predict(c)
        return M()

    def poly():
        class M:
            def fit(self, c, y):
                self.p = np.polyfit(c, y, 3); return self
            def predict(self, c): return np.polyval(self.p, c)
        return M()

    cands = {"isotonic": iso, "cubic": poly}
    scores = {k: inner_rmse(f, trA.cut.values.astype(float), trA.y.values, trB.cut.values.astype(float), trB.y.values)
              for k, f in cands.items()}
    best = min(scores, key=scores.get)
    pool = pd.concat([trA, trB])
    return cands[best]().fit(pool.cut.values.astype(float), pool.y.values), best, scores


# ---------------------------------------------------------------- one outer fold
def run_fold(df, feats, test, k=20, learners=None, kmax=20, with_cut_ridge=True, seeds=SEEDS):
    learners = learners or list(LEARNERS)
    train = [c for c in CUTTERS if c != test]
    trA = df[(df.cutter == train[0]) & recorded(df.cut, k)]
    trB = df[(df.cutter == train[1]) & recorded(df.cut, k)]
    te = df[df.cutter == test]
    pool = pd.concat([trA, trB])
    out = {"test": test, "n_train": len(pool)}

    base, bname, bscores = fit_baseline(trA, trB)
    preds = {"Cut-index baseline": base.predict(te.cut.values.astype(float))}
    out["baseline_variant"] = bname

    sel, curve = select_features(trA, trB, feats, kmax=kmax)
    out["selected"] = sel; out["sel_curve"] = curve
    configs = {}

    def tune_and_predict(name, cols):
        make_est, g, stochastic = LEARNERS[name]
        scores = [inner_rmse(lambda p=p: wrap(make_est(p, 0)), trA[cols].values, trA.y.values,
                             trB[cols].values, trB.y.values) for p in g]
        p = g[int(np.argmin(scores))]
        ss = seeds if stochastic else [0]
        yhat = np.mean([wrap(make_est(p, s)).fit(pool[cols].values, pool.y.values).predict(te[cols].values)
                        for s in ss], axis=0)
        return p, yhat

    if with_cut_ridge:
        cols = sel + ["cut"]
        p, yhat = tune_and_predict("Ridge", cols)
        preds["Ridge + cut index"] = yhat; configs["Ridge + cut index"] = p
    for name in learners:
        p, yhat = tune_and_predict(name, sel)
        preds[name] = yhat; configs[name] = p
    out["configs"] = configs
    rows = []
    for m, yhat in preds.items():
        rows.append(pd.DataFrame({"model": m, "cutter": test, "cut": te.cut.values, "y": te.y.values, "yhat": yhat}))
    out["preds"] = pd.concat(rows)
    return out


# ---------------------------------------------------------------- evaluation
def block_indices(n, L, rng):
    starts = rng.integers(0, n - L + 1, size=int(np.ceil(n / L)))
    return np.concatenate([np.arange(s, s + L) for s in starts])[:n]


def bootstrap(preds, models, ref="Cut-index baseline", L=20, B=2000, seed=0):
    """Moving-block bootstrap of residual series within each held-out cutter.
    Same block draws for all models (paired). Returns pooled and per-cutter RMSE CIs and ΔRMSE CIs."""
    rng = np.random.default_rng(seed)
    P = {m: {c: preds[(preds.model == m) & (preds.cutter == c)].sort_values("cut") for c in CUTTERS}
         for m in set(models) | {ref}}
    res = {m: {"pooled": [], **{c: [] for c in CUTTERS}} for m in P}
    dres = {m: {"pooled": [], **{c: [] for c in CUTTERS}} for m in P}
    for _ in range(B):
        idx = {c: block_indices(len(P[ref][c]), L, rng) for c in CUTTERS}
        se = {}
        for m in P:
            se[m] = {c: (P[m][c].y.values[idx[c]] - P[m][c].yhat.values[idx[c]]) ** 2 for c in CUTTERS}
        for m in P:
            pooled = np.sqrt(np.mean(np.concatenate([se[m][c] for c in CUTTERS])))
            pooled_ref = np.sqrt(np.mean(np.concatenate([se[ref][c] for c in CUTTERS])))
            res[m]["pooled"].append(pooled); dres[m]["pooled"].append(pooled - pooled_ref)
            for c in CUTTERS:
                r1 = np.sqrt(np.mean(se[m][c])); r0 = np.sqrt(np.mean(se[ref][c]))
                res[m][c].append(r1); dres[m][c].append(r1 - r0)
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return ({m: {k: q(v) for k, v in d.items()} for m, d in res.items()},
            {m: {k: q(v) for k, v in d.items()} for m, d in dres.items()})


def summary(preds):
    rows = []
    for m, g in preds.groupby("model", sort=False):
        r = {"model": m, "RMSE_pooled": rmse(g.y, g.yhat),
             "R2_pooled": 1 - np.sum((g.y - g.yhat) ** 2) / np.sum((g.y - g.y.mean()) ** 2)}
        for c in CUTTERS:
            h = g[g.cutter == c]; r[f"RMSE_{c}"] = rmse(h.y, h.yhat)
        for s, (a, b) in STAGES.items():
            h = g[(g.cut >= a) & (g.cut <= b)]; r[f"RMSE_{s}"] = rmse(h.y, h.yhat)
        rows.append(r)
    return pd.DataFrame(rows)


def crossing(y_cut, yhat, cuts, lim):
    def first(v):
        i = np.where(v >= lim)[0]
        return int(cuts[i[0]]) if len(i) else None
    c_star, c_hat = first(y_cut), first(yhat)
    if c_star is None:
        return c_star, c_hat, None, None
    if c_hat is None:
        return c_star, c_hat, 0, int(cuts[-1] - c_star + 1)     # never predicted: late by at least this
    return c_star, c_hat, max(0, c_star - c_hat), max(0, c_hat - c_star)

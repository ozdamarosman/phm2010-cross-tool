# -*- coding: utf-8 -*-
"""Post-hoc analysis: per-tool baseline normalisation of features (see PREREG_normalisation.md)."""
import sys, time, pickle
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from core import *

FEAT = "phm2010_out/features.csv"
OUT = "results/"
what = sys.argv[1]
NCAL = int(sys.argv[2]) if len(sys.argv) > 2 else 5
df, feats = load(FEAT, "VB_max_um")


def normalise(df, feats, ncal):
    d = df.copy()
    pos = [f for f in feats if (df[f] > 0).all()]
    for c in CUTTERS:
        m = d.cutter == c
        ref = df.loc[m & (df.cut <= ncal), feats].mean()
        for f in feats:
            d.loc[m, f] = np.log(df.loc[m, f] / ref[f]) if f in pos else df.loc[m, f] - ref[f]
    return d, len(pos)


dn, npos = normalise(df, feats, NCAL)
print("strictly positive features (log-ratio):", npos, "of", len(feats))
t0 = time.time()
tag = f"norm{NCAL}"

if what == "main":
    folds = [run_fold(dn, feats, t, k=20) for t in CUTTERS]
    preds = pd.concat([o["preds"] for o in folds]); preds.to_csv(OUT + f"preds_{tag}.csv", index=False)
    S = summary(preds); ci, dci = bootstrap(preds, S.model.tolist())
    S["RMSE_CI"] = S.model.map(lambda m: ci[m]["pooled"]); S["dRMSE_CI"] = S.model.map(lambda m: dci[m]["pooled"])
    for c in CUTTERS: S[f"dRMSE_CI_{c}"] = S.model.map(lambda m: dci[m][c])
    S["dRMSE"] = S.RMSE_pooled - float(S.loc[S.model == "Cut-index baseline", "RMSE_pooled"].iloc[0])
    S.to_pickle(OUT + f"summary_{tag}.pkl")
    pickle.dump([{k: v for k, v in o.items() if k != "preds"} for o in folds], open(OUT + f"folds_{tag}.pkl", "wb"))
    # normalised vs raw, same learner (paired bootstrap)
    raw = pd.read_csv(OUT + "preds_main_max.csv")
    comp = {}
    for m in S.model:
        if m == "Cut-index baseline": continue
        P = pd.concat([raw[raw.model == m].assign(model="raw"), preds[preds.model == m].assign(model="norm")])
        _, d2 = bootstrap(P, ["norm"], ref="raw")
        comp[m] = d2["norm"]["pooled"]
    S["dRaw_CI"] = S.model.map(comp); S.to_pickle(OUT + f"summary_{tag}.pkl")
    pd.set_option("display.width", 250)
    for r in S.itertuples():
        print(f"{r.model:20s} RMSE {r.RMSE_pooled:.4f}  c1 {r.RMSE_c1:.4f} c4 {r.RMSE_c4:.4f} c6 {r.RMSE_c6:.4f} "
              f"acc {r.RMSE_accelerated:.4f}  dBase {r.dRMSE:+.4f} {np.round(r.dRMSE_CI,4)} "
              f"folds c1 {np.round(r.dRMSE_CI_c1,3)} c4 {np.round(r.dRMSE_CI_c4,3)} c6 {np.round(r.dRMSE_CI_c6,3)}  vsRaw {np.round(r.dRaw_CI,4) if isinstance(r.dRaw_CI, tuple) else ''}")
    for o in folds: print(o["test"], o["selected"])

elif what == "extras":
    preds = pd.read_csv(OUT + f"preds_{tag}.csv")
    rows = []
    for lim in [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16]:
        for m in ["Cut-index baseline", "Random Forest", "Gradient Boosting", "XGBoost", "Ridge + cut index", "Ridge"]:
            for c in CUTTERS:
                g = preds[(preds.model == m) & (preds.cutter == c)].sort_values("cut")
                cs, ch, de, dl = crossing(g.y.values, g.yhat.values, g.cut.values, lim)
                rows.append({"lim": lim, "model": m, "cutter": c, "c_star": cs, "c_hat": ch, "early": de, "late": dl})
    C = pd.DataFrame(rows); C.to_csv(OUT + f"crossing_{tag}.csv", index=False)
    print(C[C.lim == 0.15].to_string())
    C["abs"] = C.early.fillna(0) + C.late.fillna(0)
    print(C.groupby(["lim", "model"])["abs"].sum().unstack())
    out = []; lrows = []
    for k in [2, 5, 10, 15, 20, 30, 40]:
        ps = pd.concat([run_fold(dn, feats, t, k=k, learners=["Random Forest"], with_cut_ridge=False)["preds"]
                        for t in CUTTERS])
        lrows.append(ps.assign(k=k))
        ci, dci = bootstrap(ps, ["Random Forest"], B=1000)
        h = ps[ps.model == "Random Forest"]
        out.append({"k": k, "n_per_cutter": int(np.sum(recorded(np.arange(1, 316), k))), "RMSE": rmse(h.y, h.yhat),
                    "CI": ci["Random Forest"]["pooled"], "dCI": dci["Random Forest"]["pooled"]})
        print("k", k, round(out[-1]["RMSE"], 4), np.round(out[-1]["dCI"], 4), f"{time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(out).to_pickle(OUT + f"lcurve_{tag}.pkl"); pd.concat(lrows).to_csv(OUT + f"preds_lcurve_{tag}.csv", index=False)
print(f"done {time.time()-t0:.0f}s")

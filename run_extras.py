# -*- coding: utf-8 -*-
"""Explainability (out-of-fold), sensor-channel ablation, learning curve and threshold crossing,
all for the reference learner (Random Forest, the best sensor-based learner of the main analysis)."""
import sys, time, pickle
sys.path.insert(0, ".")
import numpy as np, pandas as pd, shap
from sklearn.inspection import permutation_importance
from core import *

FEAT = "phm2010_out/features.csv"
OUT = "results/"
REF = "Random Forest"
what = sys.argv[1]
target = sys.argv[2] if len(sys.argv) > 2 else "VB_max_um"
tag = "max" if target == "VB_max_um" else "mean"
df, feats = load(FEAT, target)
t0 = time.time()

if what == "explain":
    meta = pickle.load(open(OUT + f"folds_main_{tag}.pkl", "rb"))
    make_est = LEARNERS[REF][0]
    perm_rows, shap_rows = [], []
    for o in meta:
        test, sel, p = o["test"], o["selected"], o["configs"][REF]
        tr = [c for c in CUTTERS if c != test]
        pool = df[df.cutter.isin(tr) & recorded(df.cut, 20)]
        te = df[df.cutter == test]
        for s in SEEDS:
            m = wrap(make_est(p, s)).fit(pool[sel].values, pool.y.values)
            pi = permutation_importance(m, te[sel].values, te.y.values, n_repeats=20, random_state=s,
                                        scoring="neg_root_mean_squared_error")
            for f, v in zip(sel, pi.importances_mean):
                perm_rows.append({"fold": test, "seed": s, "feature": f, "dRMSE": v})
            # TreeSHAP on the forest (inputs standardised inside the pipeline; output in standardised y units)
            sc = m.regressor_.named_steps["standardscaler"]
            rf = m.regressor_.named_steps["randomforestregressor"]
            sv = shap.TreeExplainer(rf).shap_values(sc.transform(te[sel].values)) * m.transformer_.scale_[0]
            for j, f in enumerate(sel):
                for i in range(len(te)):
                    shap_rows.append({"fold": test, "seed": s, "feature": f, "cut": int(te.cut.values[i]),
                                      "value": float(te[f].values[i]), "shap": float(sv[i, j])})
    P = pd.DataFrame(perm_rows); Sh = pd.DataFrame(shap_rows)
    Sh = Sh.groupby(["fold", "feature", "cut"], as_index=False)[["value", "shap"]].mean()
    P.to_csv(OUT + f"perm_{tag}.csv", index=False); Sh.to_csv(OUT + f"shap_{tag}.csv", index=False)
    pt = P.groupby("feature").agg(dRMSE=("dRMSE", "mean"), folds=("fold", "nunique")).sort_values("dRMSE", ascending=False)
    st = Sh.assign(a=Sh.shap.abs()).groupby("feature").agg(mean_abs=("a", "mean"), folds=("fold", "nunique"))
    st["dir_corr"] = Sh.groupby("feature").apply(lambda g: np.corrcoef(g.value, g.shap)[0, 1] if g.value.std() > 0 else np.nan)
    print(pt.round(4)); print(st.sort_values("mean_abs", ascending=False).round(4))

elif what == "ablation":
    groups = {"All channels": CHANNELS, "Force + Vibration": CHANNELS[:6],
              "Force + AE-RMS": CHANNELS[:3] + ["ae_rms"], "Vibration + AE-RMS": CHANNELS[3:],
              "Force only": CHANNELS[:3], "Vibration only": CHANNELS[3:6], "AE-RMS only": ["ae_rms"]}
    rows, info = [], []
    for g, chans in groups.items():
        fg = [f for f in feats if f.startswith(tuple(chans))]
        for test in CUTTERS:
            o = run_fold(df, fg, test, k=20, learners=[REF], with_cut_ridge=False)
            pr = o["preds"]; pr = pr[pr.model == REF].assign(model=g)
            rows.append(pr); info.append({"group": g, "test": test, "n_cand": len(fg), "n_sel": len(o["selected"]),
                                          "selected": o["selected"]})
        print(g, f"{time.time()-t0:.0f}s", flush=True)
    preds = pd.concat(rows); preds.to_csv(OUT + f"preds_ablation_{tag}.csv", index=False)
    S = summary(preds); ci, dci = bootstrap(preds, S.model.tolist(), ref="All channels")
    S["RMSE_CI"] = S.model.map(lambda m: ci[m]["pooled"]); S["dRMSE_CI"] = S.model.map(lambda m: dci[m]["pooled"])
    S["dRMSE"] = S.RMSE_pooled - float(S.loc[S.model == "All channels", "RMSE_pooled"].iloc[0])
    S.to_pickle(OUT + f"summary_ablation_{tag}.pkl"); pd.to_pickle(info, OUT + f"ablation_info_{tag}.pkl")
    print(S[["model", "RMSE_pooled", "R2_pooled", "dRMSE", "dRMSE_CI"]].round(4).to_string())
    print(pd.DataFrame(info)[["group", "test", "n_cand", "n_sel"]].to_string())

elif what == "lcurve":
    rows = []
    for k in [2, 5, 10, 15, 20, 30, 40]:
        for test in CUTTERS:
            o = run_fold(df, feats, test, k=k, learners=[REF], with_cut_ridge=False)
            rows.append(o["preds"].assign(k=k, n_per_cutter=int(np.sum(recorded(np.arange(1, 316), k)))))
        print("k", k, f"{time.time()-t0:.0f}s", flush=True)
    preds = pd.concat(rows); preds.to_csv(OUT + f"preds_lcurve_{tag}.csv", index=False)
    out = []
    for k, g in preds.groupby("k"):
        ci, dci = bootstrap(g, [REF], B=1000)
        for m in ["Cut-index baseline", REF]:
            h = g[g.model == m]
            out.append({"k": k, "n_per_cutter": int(h.n_per_cutter.iloc[0]), "model": m, "RMSE": rmse(h.y, h.yhat),
                        "CI": ci[m]["pooled"], "dCI": dci[m]["pooled"] if m == REF else None})
    L = pd.DataFrame(out); L.to_pickle(OUT + f"lcurve_{tag}.pkl"); print(L.round(4).to_string())

elif what == "crossing":
    preds = pd.read_csv(OUT + f"preds_main_{tag}.csv")
    rows = []
    for lim in [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16]:
        for m in ["Cut-index baseline", REF, "Gradient Boosting", "XGBoost"]:
            for c in CUTTERS:
                g = preds[(preds.model == m) & (preds.cutter == c)].sort_values("cut")
                cs, ch, de, dl = crossing(g.y.values, g.yhat.values, g.cut.values, lim)
                rows.append({"lim": lim, "model": m, "cutter": c, "c_star": cs, "c_hat": ch, "early": de, "late": dl})
    C = pd.DataFrame(rows); C.to_csv(OUT + f"crossing_{tag}.csv", index=False)
    print(C[C.lim == 0.15].to_string())
    C["abs"] = C.early.fillna(0) + C.late.fillna(0)
    print(C.groupby(["lim", "model"])["abs"].sum().unstack().round(0))
print(f"done {time.time()-t0:.0f}s")

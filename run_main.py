# -*- coding: utf-8 -*-
import sys, json, time, pickle
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from core import *

FEAT = "phm2010_out/features.csv"
OUT = "results/"
import os; os.makedirs(OUT, exist_ok=True)
target = sys.argv[1] if len(sys.argv) > 1 else "VB_max_um"
tag = "max" if target == "VB_max_um" else "mean"

df, feats = load(FEAT, target)
t0 = time.time()
folds = []
for test in CUTTERS:
    o = run_fold(df, feats, test, k=20)
    print(test, "n_train", o["n_train"], "baseline", o["baseline_variant"], "n_sel", len(o["selected"]),
          f"{time.time()-t0:.0f}s", flush=True)
    folds.append(o)
preds = pd.concat([o["preds"] for o in folds])
preds.to_csv(OUT + f"preds_main_{tag}.csv", index=False)
S = summary(preds)
models = S.model.tolist()
ci, dci = bootstrap(preds, models)
S["RMSE_CI"] = S.model.map(lambda m: ci[m]["pooled"])
S["dRMSE"] = S.RMSE_pooled - float(S.loc[S.model == "Cut-index baseline", "RMSE_pooled"].iloc[0])
S["dRMSE_CI"] = S.model.map(lambda m: dci[m]["pooled"])
for c in CUTTERS:
    S[f"dRMSE_CI_{c}"] = S.model.map(lambda m: dci[m][c])
S.to_pickle(OUT + f"summary_main_{tag}.pkl")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(S.drop(columns=[f"dRMSE_CI_{c}" for c in CUTTERS]).round(4).to_string())
meta = [{k: v for k, v in o.items() if k != "preds"} for o in folds]
pickle.dump(meta, open(OUT + f"folds_main_{tag}.pkl", "wb"))
for o in meta:
    print(o["test"], o["selected"])
    print("   configs", o["configs"])
print(f"total {time.time()-t0:.0f}s")

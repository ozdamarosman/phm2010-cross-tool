import sys
sys.path.insert(0, ".")
import pandas as pd
from core import *
R = "results/"
tag = sys.argv[1] if len(sys.argv) > 1 else "max"
main = pd.read_csv(R + f"preds_main_{tag}.csv"); cnn = pd.read_csv(R + f"preds_cnn_{tag}_k20.csv")
P = pd.concat([main[main.model == "Cut-index baseline"], cnn])
S = summary(P); ci, dci = bootstrap(P, S.model.tolist())
S["RMSE_CI"] = S.model.map(lambda m: ci[m]["pooled"]); S["dRMSE_CI"] = S.model.map(lambda m: dci[m]["pooled"])
for c in CUTTERS: S[f"dRMSE_CI_{c}"] = S.model.map(lambda m: dci[m][c])
S["dRMSE"] = S.RMSE_pooled - float(S.loc[S.model == "Cut-index baseline", "RMSE_pooled"].iloc[0])
S = S[S.model != "Cut-index baseline"]; S.to_pickle(R + f"summary_cnn_{tag}.pkl")
print(S.round(4).T)

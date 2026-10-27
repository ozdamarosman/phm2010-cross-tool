# -*- coding: utf-8 -*-
import sys, pickle
sys.path.insert(0, ".")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core import CUTTERS, recorded, load, rmse

R = "results/"; F = "figs/"
import os; os.makedirs(F, exist_ok=True)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CUT_COL = {"c1": "#2a78d6", "c4": "#eb6834", "c6": "#1baf7a"}
BLUE, GRAY = "#2a78d6", "#8a8984"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "savefig.dpi": 300, "lines.linewidth": 1.6})
df, feats = load("phm2010_out/features.csv")
REF = "Random Forest"

# ---------- Fig 1: wear trajectories
fig, ax = plt.subplots(figsize=(5.6, 3.2))
for c in CUTTERS:
    g = df[df.cutter == c]
    ax.plot(g.cut, g.y, color=CUT_COL[c], label=c)
    r = g[recorded(g.cut, 20)]
    ax.plot(r.cut, r.y, "o", ms=4, color=CUT_COL[c], mec="white", mew=0.8)
ax.axhline(0.15, ls="--", lw=1, color=INK2); ax.text(4, 0.153, "VB_lim = 0.15 mm", color=INK2, fontsize=8)
ax.set_xlabel("Cut number"); ax.set_ylabel("VB_max (mm)"); ax.set_xlim(0, 318); ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig(F + "fig1_wear.png"); plt.close(fig)

# ---------- Fig 2: features selected per fold
meta = pickle.load(open(R + "folds_main_max.pkl", "rb"))
cnt = {}
for o in meta:
    for f in o["selected"]:
        cnt.setdefault(f, []).append(o["test"])
order = sorted(cnt, key=lambda f: (-len(cnt[f]), f))
fig, ax = plt.subplots(figsize=(5.6, 3.0))
for i, f in enumerate(order):
    for j, c in enumerate(CUTTERS):
        if c in cnt[f]:
            ax.plot(j, i, "s", ms=9, color=CUT_COL[c])
ax.set_yticks(range(len(order))); ax.set_yticklabels(order); ax.invert_yaxis()
ax.set_xticks(range(3)); ax.set_xticklabels([f"held-out {c}\n({len(m['selected'])} selected)" for c, m in zip(CUTTERS, meta)])
ax.set_xlim(-0.6, 2.6); ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(F + "fig2_selection.png"); plt.close(fig)

# ---------- Fig 3: model comparison
S = pd.read_pickle(R + "summary_main_max.pkl")
cnn = pd.read_pickle(R + "summary_cnn_max.pkl")
S = pd.concat([S, cnn], ignore_index=True)
S = S.sort_values("RMSE_pooled", ascending=False).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(5.6, 3.6))
base = float(S.loc[S.model == "Cut-index baseline", "RMSE_pooled"].iloc[0])
for i, r in S.iterrows():
    col = GRAY if r.model == "Cut-index baseline" else BLUE
    ax.plot(r.RMSE_CI, [i, i], color=col, lw=2, solid_capstyle="round")
    ax.plot(r.RMSE_pooled, i, "o", ms=6, color=col, mec="white", mew=1)
    for c in CUTTERS:
        ax.plot(r[f"RMSE_{c}"], i + 0.28, "|", ms=7, mew=1.6, color=CUT_COL[c])
ax.axvline(base, ls="--", lw=1, color=GRAY)
ax.set_yticks(range(len(S))); ax.set_yticklabels(S.model); ax.set_xlabel("Held-out RMSE (mm)")
from matplotlib.lines import Line2D
h = [Line2D([], [], marker="o", ls="-", color=BLUE, label="pooled, 95% CI")] + \
    [Line2D([], [], marker="|", ls="", ms=7, mew=1.6, color=CUT_COL[c], label=f"held-out {c}") for c in CUTTERS]
ax.legend(handles=h, loc="upper right", fontsize=7.5); ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(F + "fig3_models.png"); plt.close(fig)

# ---------- Fig 4: trajectories
P = pd.read_csv(R + "preds_main_max.csv")
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
for ax, c in zip(axs, CUTTERS):
    g = P[P.cutter == c]
    m = g[g.model == "Cut-index baseline"]; ax.plot(m.cut, m.y, color=INK, lw=1.8, label="measured")
    ax.plot(m.cut, m.yhat, color=GRAY, ls="--", lw=1.4, label="cut-index baseline")
    m = g[g.model == REF]; ax.plot(m.cut, m.yhat, color=BLUE, lw=1.2, label="Random Forest")
    ax.axhline(0.15, ls=":", lw=1, color=INK2); ax.set_title(f"held-out {c}", fontsize=9, color=INK)
    ax.set_xlabel("Cut number")
axs[0].set_ylabel("VB_max (mm)"); axs[0].legend(fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig(F + "fig4_trajectories.png"); plt.close(fig)

# ---------- Fig 5: SHAP
Sh = pd.read_csv(R + "shap_max.csv")
top = Sh.assign(a=Sh.shap.abs()).groupby("feature").a.mean().sort_values(ascending=False).index[:6].tolist()
fig, ax = plt.subplots(figsize=(5.6, 3.0))
rng = np.random.default_rng(0)
for i, f in enumerate(top):
    g = Sh[Sh.feature == f].copy()
    g["pct"] = g.groupby("fold").value.rank(pct=True)
    sc = ax.scatter(g.shap, i + rng.uniform(-0.25, 0.25, len(g)), c=g.pct, cmap="Blues", vmin=-0.2, vmax=1,
                    s=6, linewidths=0)
ax.axvline(0, color=INK2, lw=0.8)
ax.set_yticks(range(len(top))); ax.set_yticklabels(top); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
ax.set_xlabel("SHAP value (mm)")
cb = fig.colorbar(sc, ax=ax, pad=0.01); cb.set_label("feature value (percentile within held-out cutter)", fontsize=7)
cb.outline.set_visible(False)
fig.tight_layout(); fig.savefig(F + "fig5_shap.png"); plt.close(fig)

# ---------- Fig 6: ablation
A = pd.read_pickle(R + "summary_ablation_max.pkl").iloc[::-1].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(5.6, 2.8))
ref = float(A.loc[A.model == "All channels", "RMSE_pooled"].iloc[0])
for i, r in A.iterrows():
    ax.plot(r.RMSE_CI, [i, i], color=BLUE, lw=2, solid_capstyle="round")
    ax.plot(r.RMSE_pooled, i, "o", ms=6, color=BLUE, mec="white", mew=1)
ax.axvline(ref, ls="--", lw=1, color=GRAY)
ax.set_yticks(range(len(A))); ax.set_yticklabels(A.model); ax.set_xlabel("Held-out RMSE (mm), Random Forest")
ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(F + "fig6_ablation.png"); plt.close(fig)

# ---------- Fig 7: learning curve
L = pd.read_pickle(R + "lcurve_max.pkl")
fig, ax = plt.subplots(figsize=(5.6, 3.0))
for m, col, mk in [("Cut-index baseline", GRAY, "s"), (REF, BLUE, "o")]:
    g = L[L.model == m].sort_values("n_per_cutter")
    lo = [c[0] for c in g.CI]; hi = [c[1] for c in g.CI]
    ax.fill_between(g.n_per_cutter, lo, hi, color=col, alpha=0.15, lw=0)
    ax.plot(g.n_per_cutter, g.RMSE, marker=mk, color=col, ms=5, mec="white", label=m)
ax.set_xscale("log"); ax.set_xticks([8, 11, 16, 21, 32, 63, 158]); ax.set_xticklabels([8, 11, 16, 21, 32, 63, 158])
ax.minorticks_off()
ax.set_xlabel("Recorded and labelled cuts per training cutter"); ax.set_ylabel("Held-out RMSE (mm)")
ax.legend(loc="upper right")
fig.tight_layout(); fig.savefig(F + "fig7_lcurve.png"); plt.close(fig)
print("figs done")

# ---------- Fig 8: raw vs per-tool normalised features
Sr = pd.read_pickle(R + "summary_main_max.pkl").set_index("model")
Sn = pd.read_pickle(R + "summary_norm5.pkl").set_index("model")
ms = ["Ridge + cut index", "Ridge", "Lasso", "SVR", "Random Forest", "Gradient Boosting", "XGBoost", "MLP"]
fig, ax = plt.subplots(figsize=(5.6, 3.4))
base = float(Sr.loc["Cut-index baseline", "RMSE_pooled"])
for i, m in enumerate(ms):
    for off, S_, col, lab in [(-0.17, Sr, GRAY, "raw features"), (0.17, Sn, BLUE, "per-tool normalised")]:
        r = S_.loc[m]
        ax.plot(r.RMSE_CI, [i + off] * 2, color=col, lw=2, solid_capstyle="round")
        ax.plot(r.RMSE_pooled, i + off, "o", ms=6, color=col, mec="white", mew=1, label=lab if i == 0 else None)
ax.axvline(base, ls="--", lw=1, color=INK2, label="cut-index baseline")
ax.set_yticks(range(len(ms))); ax.set_yticklabels(ms); ax.invert_yaxis(); ax.set_xscale("log")
ax.set_xticks([0.015, 0.02, 0.03, 0.05, 0.08]); ax.set_xticklabels(["0.015", "0.02", "0.03", "0.05", "0.08"])
ax.minorticks_off(); ax.set_xlabel("Held-out RMSE (mm), pooled, 95% CI (log scale)"); ax.grid(axis="y", visible=False)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7.5)
fig.tight_layout(); fig.savefig(F + "fig8_normalisation.png"); plt.close(fig)
print("fig8 done")

# -*- coding: utf-8 -*-
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = "results/"; F = "../figs/"
INK2, GRID, BLUE, GRAY = "#52514e", "#e4e3df", "#2a78d6", "#8a8984"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 300})
raw = pd.read_pickle(R + "summary_setting_raw.pkl").set_index("model")
nor = pd.read_pickle(R + "summary_setting_norm.pkl").set_index("model")
ms = ["Ridge", "Lasso", "SVR", "Random Forest", "Gradient Boosting", "XGBoost", "MLP"]
fig, ax = plt.subplots(figsize=(5.6, 3.6))
rows = ["Time-only baseline", "Process baseline"] + ms
for i, m in enumerate(rows):
    if m in ms:
        for off, S, col, lab in [(-0.17, raw, GRAY, "sensors + process, raw"), (0.17, nor, BLUE, "sensors + process, normalised")]:
            r = S.loc[m]
            ax.plot(r.CI, [i + off] * 2, color=col, lw=2, solid_capstyle="round")
            ax.plot(r.RMSE, i + off, "o", ms=6, color=col, mec="white", mew=1, label=lab if m == ms[0] else None)
    else:
        r = raw.loc[m]
        ax.plot(r.CI, [i, i], color="#0b0b0b", lw=2, solid_capstyle="round")
        ax.plot(r.RMSE, i, "s", ms=6, color="#0b0b0b", mec="white", mew=1)
ax.axvline(raw.loc["Process baseline", "RMSE"], ls="--", lw=1, color=INK2, label="process baseline")
ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
ax.set_xlabel("Held-out RMSE (mm), 95% cluster-bootstrap CI")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7.5)
fig.tight_layout(); fig.savefig(F + "fig9_nasa.png"); plt.close(fig); print("fig9 done")

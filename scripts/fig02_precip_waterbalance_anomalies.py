#!/usr/bin/env python3
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fig_style

CF, CL = 1991, 2020
WINDOW = ["SON", "DJF", "MAM", "June"]
W = 5
BLACK = "#000000"
GREY, DARK = "#9a9a9a", "#333333"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.labelsize": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 9.5, "axes.linewidth": 0.8,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": False,
})

df = pd.read_csv("../data/all_seasons_combined.csv")
base = df[(df.season_year >= CF) & (df.season_year <= CL)]
mP = base.groupby("season").precip_chirps_mm.mean()
mE = base.groupby("season").pet_pm_mm.mean()
df["p_a"] = df.precip_chirps_mm - df.season.map(mP)
df["dem"] = -(df.pet_pm_mm - df.season.map(mE))
df["wb"]  = df.p_a + df.dem
df["wy"]  = np.where(df.season == "SON", df.season_year + 1, df.season_year)
df["in_ref"] = (df.season_year >= CF) & (df.season_year <= CL)

sj = df[df.season.isin(WINDOW)]
nper, nref = sj.groupby("wy").season.nunique(), sj.groupby("wy").in_ref.sum()
years = sorted(nper[nper == 4].index)
ref_years = [w for w in years if nref[w] == 4]
strad = [w for w in years if 0 < nref[w] < 4]
SEP = max(ref_years) + 0.5
ann = sj[sj.wy.isin(years)].groupby("wy")[["p_a", "dem", "wb", "precip_chirps_mm"]].sum()
clim_P = ann.precip_chirps_mm.loc[min(ref_years):max(ref_years) + 1].mean()

def wylab(w):
    return "%d/%02d" % (w - 1, w % 100)

roll = ann.wb.rolling(W).sum().dropna()
worst_y, last_y = roll.idxmin(), roll.index[-1]
rk = lambda v: int((roll < v).sum()) + 1
print("Sep-Jun climatological precipitation %.1f mm" % clim_P)
for lab, c in (("precipitation", ann.p_a), ("P - PET", ann.wb)):
    v = c.loc[2026]; h = c.loc[:2025]
    print("2025/26 %-14s %+7.1f mm  z %+5.2f  rank %2d of %d wettest"
          % (lab, v, (v - h.mean()) / h.std(), int((c > v).sum()) + 1, len(c)))
print("%d-year total: min %+.0f at %s (rank %d of %d); latest %+.0f (rank %d)"
      % (W, roll.min(), wylab(worst_y), rk(roll.min()), len(roll),
         roll.iloc[-1], rk(roll.iloc[-1])))

fig, axes = plt.subplots(3, 1, figsize=(9.6, 12.2),
                         gridspec_kw=dict(height_ratios=[1.0, 1.0, 1.0]))
ax1, ax2, ax3 = axes

def bars(ax, series, title, ylab):
    ax.axvspan(min(ref_years) - 0.5, SEP, color="#f2f2f2", zorder=0)
    ax.axhline(0, color=DARK, linewidth=1.0, zorder=3)
    for w in years:
        v = series.loc[w]
        ax.bar(w, v, width=0.74, color=BLACK, linewidth=0, zorder=2)
    for w in strad:
        ax.bar(w, series.loc[w], width=0.74, facecolor="none", edgecolor=DARK,
               linewidth=0.9, linestyle=(0, (2.2, 1.6)), zorder=4)
    ax.axvline(SEP, color=DARK, linewidth=1.0, linestyle=(0, (3, 2)), zorder=5)
    ax.set_ylabel(ylab)
    ax.set_title(title, loc="left", pad=7)

bars(ax1, ann.p_a, "(a)  September-June precipitation anomaly", "Precipitation anomaly (mm)")
bars(ax2, ann.wb, "(b)  September-June water-balance anomaly, P - PET",
     "P - PET anomaly (mm)")

lo = min(ann.p_a.min(), ann.wb.min()); hi = max(ann.p_a.max(), ann.wb.max())
pad = 0.12 * (hi - lo)
for ax in (ax1, ax2):
    ax.set_ylim(lo - 0.04 * (hi - lo), hi + pad)
    ax.set_yticks([-100, -50, 0, 50, 100, 150])
ax1.text(np.mean([min(ref_years), max(ref_years)]), ax1.get_ylim()[1] * 0.97,
         "1991-2020 reference", ha="center", va="top", fontsize=9.5, color="#666666")

rx, ry = roll.index.values.astype(float), roll.values
ax3.axhline(0, color=DARK, linewidth=1.0, zorder=3)
ax3.plot(rx, ry, color=BLACK, linewidth=2.1, zorder=5)
for yv in (worst_y, last_y):
    ax3.plot([yv], [roll.loc[yv]], "o", color=BLACK, markersize=6.0,
             markeredgecolor="white", markeredgewidth=1.1, zorder=6)
lo3, hi3 = ax3.get_ylim(); ax3.set_ylim(lo3 - 0.12 * (hi3 - lo3), hi3)
ax3.set_yticks([-400, -200, 0, 200])
ax3.set_ylabel("Five-year running total,\nP - PET anomaly (mm)")
ax3.set_title("(c)  Five-year running total of the water-balance anomaly",
              loc="left", pad=7)

for ax in axes:
    ax.set_xlim(years[0] - 0.9, years[-1] + 2.2)
    ax.set_xticks(years)
    ax.grid(axis="y", color="#eeeeee", linewidth=0.7, zorder=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(GREY); ax.spines["bottom"].set_color(GREY)
    ax.tick_params(colors="#4a4a4a", length=4)
for ax in axes:
    ax.set_xticklabels([wylab(w) for w in years], rotation=90, fontsize=9)
    ax.set_xlabel("Water year", labelpad=9)

L, WD, H = 0.088, 0.895, 0.20
for ax, bot in ((ax1, 0.73), (ax2, 0.415), (ax3, 0.10)):
    ax.set_position([L, bot, WD, H])

fig_style.apply(fig)
for ext, kw in (("png", dict(dpi=600)), ("pdf", {})):
    fig.savefig("../figures/fig02_precip_waterbalance_anomalies.%s" % ext, facecolor="white", **kw)
print("saved fig02")

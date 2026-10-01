import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fig_style
import matplotlib.ticker as mticker
import os

CLIM_FIRST, CLIM_LAST = 1991, 2020
WB_COL = "#000000"; ZERO_COL = "#333333"; FILL_COL = "#dddddd"; SPINE = "#9a9a9a"
P_COL = "#2b6cb0"; D_COL = "#b0562b"; SUM_COL = "#33475b"
PNG_DPI = 600

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.labelsize": 12, "axes.titlesize": 11,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
    "axes.linewidth": 0.8, "figure.facecolor": "white", "axes.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

def _find(name):
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    cands = [name, os.path.join(here, name), os.path.join(here, os.pardir, name),
             os.path.expanduser("~/Desktop/final non-AI dissertation scripts copy/" + name),
             os.path.expanduser("~/Desktop/GEE SEPT DATA NEW/" + name),
             os.path.expanduser("~/Desktop/Iran War Project/code/Water Balance/DATA/" + name)]
    for c in cands:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(name + " not found; looked in: " + ", ".join(cands))

df = pd.read_csv("../data/all_seasons_combined.csv")
df = df[df.season.isin(["SON", "DJF", "MAM", "JJA"])].copy()
df["start"] = pd.to_datetime(df.start_date)
base = df[(df.season_year >= CLIM_FIRST) & (df.season_year <= CLIM_LAST)]
mP = base.groupby("season").precip_chirps_mm.mean()
mE = base.groupby("season").pet_pm_mm.mean()
df["p_anom"] = df.precip_chirps_mm - df.season.map(mP)
df["pet_anom"] = df.pet_pm_mm - df.season.map(mE)
df["wb_anom"] = df.p_anom - df.pet_anom
df = df.sort_values("start").reset_index(drop=True)
df["t"] = df.start.dt.year + (df.start.dt.month - 1) / 12.0
df["cum_wb"] = df.wb_anom.cumsum()
df["cum_p"] = df.p_anom.cumsum()
df["cum_dem"] = (-df.pet_anom).cumsum()

resid = np.abs(df.cum_wb - (df.cum_p + df.cum_dem)).max()
assert resid < 1e-9, "decomposition does not close: %.3e" % resid
print("decomposition identity closes to %.2e mm" % resid)

pre = df[df.start < "2021-09-01"]
i_pk = pre.cum_wb.idxmax(); i_min = df.cum_wb.idxmin()
peak, trough, end = df.cum_wb[i_pk], df.cum_wb[i_min], df.cum_wb.iloc[-1]
drawdown = trough - peak; recovery = end - trough
seg_dn = df.loc[i_pk:i_min]; seg_up = df.loc[i_min + 1:]
dn_p, dn_d = seg_dn.p_anom.sum(), -seg_dn.pet_anom.sum()
up_p, up_d = seg_up.p_anom.sum(), -seg_up.pet_anom.sum()

def lab(i):
    r = df.loc[i]
    return ("DJF %d/%s" % (r.season_year - 1, str(int(r.season_year))[2:])
            if r.season == "DJF" else "%s %d" % (r.season, int(r.season_year)))

print("peak      %+8.1f mm at %s" % (peak, lab(i_pk)))
print("minimum   %+8.1f mm at %s" % (trough, lab(i_min)))
print("end       %+8.1f mm at %s" % (end, lab(df.index[-1])))
print("drawdown  %+8.1f mm   of which precipitation %+.1f, demand %+.1f "
      "(demand share %.0f%%)" % (drawdown, dn_p, dn_d, 100 * dn_d / (dn_p + dn_d)))
print("recovery  %+8.1f mm   of which precipitation %+.1f, demand %+.1f"
      % (recovery, up_p, up_d))
print("recovery is %.0f%% of the drawdown" % (100 * recovery / abs(drawdown)))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.6, 8.8), sharex=True)

ax1.axhline(0, color=ZERO_COL, linestyle="--", linewidth=1.0, zorder=2)
ax1.fill_between(df.t, df.cum_wb, 0, where=df.cum_wb < 0,
                 color=FILL_COL, alpha=1.0, linewidth=0, zorder=1)
ax1.plot(df.t, df.cum_wb, color=WB_COL, linewidth=1.7, zorder=4)
ax1.set_ylabel("Cumulative P - PET anomaly\nsince 1991 (mm)")
ax1.set_title("(a)", loc="left", pad=8)

ax2.axhline(0, color=ZERO_COL, linestyle="--", linewidth=1.0, zorder=2)
ax2.plot(df.t, df.cum_p, color=P_COL, linewidth=1.6, zorder=4, label="Contribution of precipitation")
ax2.plot(df.t, df.cum_dem, color=D_COL, linewidth=1.6, zorder=4, label="Contribution of evaporative demand")
ax2.plot(df.t, df.cum_wb, color=SUM_COL, linewidth=1.1, linestyle=(0, (4, 2)), zorder=3, label="Their sum (= panel a)")
ax2.set_ylabel("Accumulated contribution since 1991 (mm)", labelpad=8)
ax2.set_title("(b)", loc="left", pad=8)
ax2.legend(frameon=False, loc="upper right", fontsize=9, borderpad=0.3, handlelength=2.2, labelspacing=0.35)

for ax in (ax1, ax2):
    ax.set_xlim(df.t.min() - 0.6, df.t.max() + 1.0)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax.grid(axis="y", color="#eeeeee", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(SPINE)
    ax.spines["bottom"].set_color(SPINE)
    ax.tick_params(colors="#4a4a4a", length=4, labelbottom=True)
    for _lb in ax.get_xticklabels():
        _lb.set_rotation(90); _lb.set_va("top")
    ax.set_xlabel("Year", labelpad=22)

fig.subplots_adjust(left=0.115, right=0.965, top=0.955, bottom=0.135, hspace=0.5)

for ext, kw in (("png", dict(dpi=PNG_DPI)), ("pdf", {})):
    out = "../figures/fig03_accumulated_water_balance.%s" % ext
    fig_style.apply(fig)
    fig.savefig(out, facecolor="white", **kw)
    print("saved", out)
plt.close(fig)

df[["season", "season_year", "start_date", "p_anom", "pet_anom", "wb_anom",
    "cum_p", "cum_dem", "cum_wb"]].to_csv(
        "Table3_cumulative_water_balance.csv", index=False)
print("saved Table3_cumulative_water_balance.csv")

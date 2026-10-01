import glob, datetime as dt
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
from mono_common import style, BLACK, DARK, MED, GREY
style()
import matplotlib.pyplot as plt
import fig_style

TODAY = dt.date(2026, 9, 20)

def load(prefix):
    fs = sorted(glob.glob("../data/Iran_%s_seasonal_national_*.csv" % prefix))
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    d = d[d.season != "SepJun"].copy()
    d["end"] = pd.to_datetime(d.end_date).dt.date
    d = d[d.end <= TODAY]
    return d

smap = load("SMAP")[["season", "season_year", "sm_surface", "sm_rootzone"]]
gld = load("GLDAS")[["season", "season_year", "SoilMoi0_10cm_inst", "RootMoist_inst"]]
m = smap.merge(gld, on=["season", "season_year"], how="inner")

def zwithin(df, col):
    return df.groupby("season")[col].transform(lambda s: (s - s.mean()) / s.std(ddof=1))

PAIRS = [("(a)  Surface (SMAP 0-5 cm vs Noah 0-10 cm)", "sm_surface", "SoilMoi0_10cm_inst"),
         ("(b)  Root zone (SMAP 0-100 cm vs Noah root moisture)", "sm_rootzone", "RootMoist_inst")]
MARK = {"SON": "o", "DJF": "s", "MAM": "^", "JJA": "D", "June": "v"}

fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.9))
fig.subplots_adjust(left=0.075, right=0.985, top=0.93, bottom=0.135, wspace=0.24)

for ax, (title, scol, gcol) in zip(axes, PAIRS):
    sm_z = zwithin(m, scol); gl_z = zwithin(m, gcol)
    ok = np.isfinite(sm_z) & np.isfinite(gl_z)
    r, p = pearsonr(gl_z[ok], sm_z[ok]); rs, ps = spearmanr(gl_z[ok], sm_z[ok])
    for s, mk in MARK.items():
        sub = ok & (m.season == s)
        ax.scatter(gl_z[sub], sm_z[sub], marker=mk, s=34, facecolor="white",
                   edgecolor=BLACK, linewidth=0.9, zorder=3, label=s)
    lim = 2.9
    ax.plot([-lim, lim], [-lim, lim], color=GREY, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.axhline(0, color="#e6e6e6", lw=0.8, zorder=0); ax.axvline(0, color="#e6e6e6", lw=0.8, zorder=0)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.grid(color="#eeeeee", linewidth=0.7, zorder=0)
    ax.set_xlabel("GLDAS Noah, standardised")
    ax.set_ylabel("SMAP L4, standardised")
    ax.set_title(title, loc="left")
    ax.text(0.04, 0.96, "r = %+.2f\n$\\rho$ = %+.2f\nn = %d" % (r, rs, ok.sum()),
            transform=ax.transAxes, va="top", ha="left", fontsize=8.6, color=DARK)

axes[1].legend(title="Season", frameon=False, loc="lower right",
               handletextpad=0.3, borderpad=0.2, labelspacing=0.3)

fig_style.apply(fig, title=10.2, label=9.5, tick=8.6, legend=8.4)

fig.savefig("../figures/SMAP and GLDAS cross-check (Appendix).png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig("../figures/SMAP and GLDAS cross-check (Appendix).pdf", bbox_inches="tight", facecolor="white")
print("saved SMAP and GLDAS cross-check  n=%d" % len(m))

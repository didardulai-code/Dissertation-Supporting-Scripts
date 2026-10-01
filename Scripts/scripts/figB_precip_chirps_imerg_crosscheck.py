import numpy as np, pandas as pd
from mono_common import style, BLACK, DARK, MED, GREY, BAND
style()
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import fig_style

CLIM_FIRST, CLIM_LAST = 1991, 2020
ANTECEDENT = {"SON": 2024, "DJF": 2025, "MAM": 2025, "JJA": 2025, "June": 2025}
TEST = {"SON": 2025, "DJF": 2026, "MAM": 2026, "JJA": 2026, "June": 2026}

LINE = "#8a8a8a"; MARK = "#000000"; IMERG = "#bdbdbd"; MEANC = "#000000"

PANELS = [("SON", "(a)  SON"), ("DJF", "(b)  DJF"), ("MAM", "(c)  MAM"),
          ("JJA", "(d)  JJA"), ("June", "(e)  June")]

ch = pd.read_csv("../data/all_seasons_combined.csv")
ch = ch[ch.qc_complete == 1]

def imerg(season):
    f = {"SON": "son_imerg.csv", "DJF": "djf_imerg.csv", "MAM": "mam_imerg.csv",
         "JJA": "jja_imerg.csv", "June": "june_imerg.csv"}[season]
    d = pd.read_csv("../data/" + f)
    return d[["season_year", "precipitation_mm"]].astype(float)

fig, axes = plt.subplots(3, 2, figsize=(9.6, 8.4))
flat = axes.ravel()

for ax, (season, title) in zip(flat, PANELS):
    d = ch[ch.season == season].sort_values("season_year")
    x, y = d.season_year.values, d.precip_chirps_mm.values
    base = y[(x >= CLIM_FIRST) & (x <= CLIM_LAST)]
    m, sd = base.mean(), base.std(ddof=1)
    ax.axhspan(m - sd, m + sd, color="#e7e9ec", zorder=0)
    for b in (m - sd, m + sd):
        ax.axhline(b, color="#b3b9c2", linestyle=(0, (3, 2)), linewidth=0.7, zorder=1)
    ax.axhline(m, color=MEANC, linestyle="--", linewidth=1.0, zorder=2)
    ax.plot(x, y, "-", color=LINE, linewidth=1.0, zorder=3)
    ax.plot(x, y, "o", color=MARK, markersize=2.6, zorder=4)
    im = imerg(season)
    ax.plot(im.season_year, im.precipitation_mm, "-", color=IMERG, linewidth=1.1, zorder=1)
    for yr, mk, ms in ((ANTECEDENT[season], "D", 6.2), (TEST[season], "o", 7.6)):
        if yr in x:
            v = y[list(x).index(yr)]
            ax.plot([yr], [v], mk, markerfacecolor=("#000000" if v >= m else "white"),
                    markersize=ms, markeredgecolor="#000000", markeredgewidth=0.9, zorder=7)
    if season == "JJA":
        ax.text(0.985, 0.045, "JJA 2026 not in record", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7.2, style="italic", color="#6b6257", zorder=8)
    ax.set_title(title, loc="left", pad=5)
    ax.set_ylabel("Precipitation (mm)")
    ax.set_xlim(1990.2, 2026.8)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax.grid(axis="y", color="#f2f2f2", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

flat[-1].set_axis_off()

handles = [
    plt.Line2D([], [], color=LINE, marker="o", markerfacecolor=MARK, markeredgecolor=MARK,
               markersize=3.4, linewidth=1.0, label="CHIRPS v2.0, area-weighted national total"),
    plt.Line2D([], [], color=IMERG, linewidth=1.6, label="GPM IMERG V07, from 2001"),
    plt.Line2D([], [], color=MEANC, linestyle="--", linewidth=1.0, label="CHIRPS 1991-2020 seasonal mean"),
    plt.Line2D([], [], color="#e7e9ec", linewidth=9, label="±1 standard deviation of the CHIRPS 1991-2020 totals"),
    plt.Line2D([], [], color="#cfcfcf", marker="D", linestyle="none", markersize=6.2,
               markeredgecolor="#333333", label="Antecedent year, Sep 2024 - Aug 2025"),
    plt.Line2D([], [], color="#cfcfcf", marker="o", linestyle="none", markersize=7.6,
               markeredgecolor="#333333", markeredgewidth=0.9, label="Year under test, Sep 2025 - Jun 2026"),
    plt.Line2D([], [], marker="s", linestyle="none", markerfacecolor="#000000", markersize=6.5,
               markeredgecolor="#000000", label="filled: above the 1991-2020 seasonal mean"),
    plt.Line2D([], [], marker="s", linestyle="none", markerfacecolor="white", markersize=6.5,
               markeredgecolor="#000000", label="open: below the 1991-2020 seasonal mean"),
]
flat[-1].legend(handles=handles, loc="center", frameon=False, labelspacing=0.85,
                borderpad=0.0, handlelength=2.2)

fig.subplots_adjust(left=0.075, right=0.985, top=0.965, bottom=0.045, hspace=0.33, wspace=0.20)
fig_style.apply(fig, title=10.5, label=9.5, tick=8.6, legend=8.2)

fig.savefig("../figures/CHIRPS and IMERG cross-check (Appendix).png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig("../figures/CHIRPS and IMERG cross-check (Appendix).pdf", bbox_inches="tight", facecolor="white")
print("saved CHIRPS and IMERG cross-check")

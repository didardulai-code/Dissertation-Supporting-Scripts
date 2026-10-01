import numpy as np, pandas as pd
from mono_common import style, BLACK, MED, GREY
style()
import matplotlib.pyplot as plt
import fig_style
from scipy.stats import spearmanr
from matplotlib.lines import Line2D

s2 = pd.read_csv("../data/s2_crosscheck_perdate.csv")
s2["site"] = s2.site.replace({"Bakhtegan Tashk": "Bakhtegan-Tashk", "Esteghlal Minab": "Minab"})
s2 = s2[s2.valid_frac >= 0.70]
s2m = s2.groupby(["site", "year", "month"]).water_km2.median().reset_index().rename(columns={"water_km2": "s2"})

ls = pd.read_csv("../data/swm.csv")
ls = ls[ls.usable == True].rename(columns={"water_mndwi0_km2": "ls"})
lswy = ls[((ls.year == 2025) & (ls.month >= 9)) | ((ls.year == 2026) & (ls.month <= 6))][
    ["site", "site_class", "year", "month", "ls"]]

m = s2m.merge(lswy, on=["site", "year", "month"], how="inner")
m["cls"] = np.where(m.site_class == "reservoir", "reservoir", "lake")
rel = (m.s2 - m.ls) / m.ls.clip(lower=0.001)
rho, p = spearmanr(m.s2, m.ls); r = np.corrcoef(m.s2, m.ls)[0, 1]
mad = np.median(np.abs(rel))
nres = (m.cls == "reservoir").sum(); nlake = (m.cls == "lake").sum()
print("n=%d sites=%d rho=%.3f r=%.3f mad=%.3f" % (len(m), m.site.nunique(), rho, r, mad))

fig, ax = plt.subplots(figsize=(5.9, 6.1))
lo = min(m.s2.min(), m.ls.min()) * 0.65
hi = max(m.s2.max(), m.ls.max()) * 1.5
ax.plot([lo, hi], [lo, hi], ls="--", color=GREY, lw=0.9, zorder=1)
res = m[m.cls == "reservoir"]; lake = m[m.cls == "lake"]
ax.scatter(res.ls, res.s2, marker="o", s=34, color=BLACK, edgecolor="white", linewidths=0.5, zorder=3)
ax.scatter(lake.ls, lake.s2, marker="^", s=42, facecolor="none", edgecolor=BLACK, linewidths=0.9, zorder=3)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
ax.set_xlabel("Landsat water area (km$^2$)", labelpad=7)
ax.set_ylabel("Sentinel-2 water area (km$^2$)", labelpad=7)
box = ("$\\rho$ = %.2f\n$r$ = %.2f\nn = %d site-months\nmedian |diff| = %.0f%%" % (rho, r, len(m), mad * 100))
ax.text(0.05, 0.93, box, transform=ax.transAxes, va="top", ha="left", fontsize=9.5, color="#333333")
ax.legend(handles=[Line2D([], [], marker="o", ls="none", mfc=BLACK, mec="white", mew=0.5, label="Reservoir (n=%d)" % nres),
                   Line2D([], [], marker="^", ls="none", mfc="none", mec=BLACK, mew=0.9, label="Terminal lake/wetland (n=%d)" % nlake)],
          frameon=False, fontsize=9, loc="lower right", numpoints=1, handlelength=1.0, handletextpad=0.6)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)

fig_style.apply(fig)
fig.savefig("../figures/Sentinel and Landsat cross-check (Appendix).png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig("../figures/Sentinel and Landsat cross-check (Appendix).pdf", bbox_inches="tight", facecolor="white")
print("saved")

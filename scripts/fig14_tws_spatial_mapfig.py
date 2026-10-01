#!/usr/bin/env python3

import numpy as np, geopandas as gpd, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fig_style
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch

TEAL, BROWN, DARK, GREY = "#018571", "#a6611a", "#37474f", "#9aa5b1"
CM = LinearSegmentedColormap.from_list("s", ["#8c4415", "#d9a066", "#f4f1ec",
                                             "#7fbfae", "#018571"])
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                     "font.size": 8.4, "axes.unicode_minus": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

z = np.load("../data/grace_fields.npz")
LA, LO, inside = z["lat"], z["lon"], z["inside"]
slope, pval, resid = z["slope"], z["pval"], z["resid"]
iran = gpd.read_file("../data/gadm/gadm41_IRN_0.shp")
ext = [LO.min(), LO.max(), LA.min(), LA.max()]
msk = lambda a: np.where(inside, a, np.nan)[::-1]

fig, ax = plt.subplots(1, 2, figsize=(9.4, 4.4))

v1 = float(np.nanpercentile(np.abs(slope[inside]), 95))
im1 = ax[0].imshow(np.clip(msk(slope), -v1, v1), extent=ext, cmap=CM,
                   norm=TwoSlopeNorm(vmin=-v1, vcenter=0.0, vmax=v1), interpolation="none")
ax[0].set_title("(a)  TWS trend, 2002-2026",
                loc="left", fontsize=8.8, pad=6)

v2 = float(np.nanpercentile(np.abs(resid[inside]), 95))
im2 = ax[1].imshow(np.clip(msk(resid), -v2, v2), extent=ext, cmap=CM,
                   norm=TwoSlopeNorm(vmin=-v2, vcenter=0.0, vmax=v2), interpolation="none")
ax[1].set_title("(b)  2025/26 departure from pre-2025 trajectory",
                loc="left", fontsize=8.8, pad=6)

steep = slope <= np.nanpercentile(slope[inside], 50)
w = np.cos(np.deg2rad(LA))[:, None] * np.ones_like(inside, float)
for lab, m in (("faster decline, above own trend", steep & (resid > 0)),
               ("faster decline, below own trend", steep & (resid <= 0)),
               ("slower decline, above own trend", ~steep & (resid > 0)),
               ("slower decline, below own trend", ~steep & (resid <= 0))):
    v = inside & np.isfinite(slope)
    print("  %-34s %.0f%% of area" % (lab, 100*np.sum(w[v & m])/np.sum(w[v])))

for a in ax:
    for xg in np.arange(42, 67, 3):
        a.axvline(xg, color="#ffffff", lw=0.55, alpha=0.75, zorder=3)
    for yg in np.arange(24, 42, 3):
        a.axhline(yg, color="#ffffff", lw=0.55, alpha=0.75, zorder=3)
    iran.boundary.plot(ax=a, color=DARK, linewidth=0.9, zorder=4)
    a.set_xlim(43.5, 64.5); a.set_ylim(24.5, 40.5)
    a.set_aspect(1/np.cos(np.radians(32)))
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values(): sp.set_visible(False)

for a, im, lab in ((ax[0], im1, "cm equivalent water per year"),
                   (ax[1], im2, "mean monthly residual, observed minus expected,\ncm equivalent water")):
    cb = fig.colorbar(im, ax=a, orientation="horizontal", fraction=0.040,
                      pad=0.03, shrink=0.72, extend="both")
    cb.set_label(lab, fontsize=7.6); cb.ax.tick_params(labelsize=7)
    cb.outline.set_linewidth(0.5)

fig.tight_layout()
for e, kw in ((".png", dict(dpi=600)), (".pdf", {})):
    fig_style.apply(fig)
    fig.savefig("../figures/fig14_tws_spatial" + e, facecolor="white", bbox_inches="tight", **kw)
print("saved Fig_grace_spatial")

#!/usr/bin/env python3

import numpy as np
import geopandas as gpd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fig_style
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe

PNG_DPI = 600
SURF_LIM, ROOT_LIM = 5.0, 50.0
FROZEN_THRESH = 0.20

WB = LinearSegmentedColormap.from_list(
    "wb", ["#a6611a", "#dfc27d", "#f5f5f5", "#80cdc1", "#018571"])

plt.rcParams.update({
    "axes.unicode_minus": False,
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

ASPECT = 1.0 / np.cos(np.deg2rad(32.0))
SB_X0, SB_Y0, SB_H = 44.3, 26.1, 0.45
SB_STEP_KM, SB_NSEG = 150.0, 3
SB_SEG = SB_STEP_KM / (111.32 * np.cos(np.deg2rad(32.0)))

PANELS = [("Iran_SMAP_SON_2025_anomaly.tif",      "SON 2025",    "3-month mean"),
          ("Iran_SMAP_DJF_2025_26_anomaly.tif",   "DJF 2025/26", "3-month mean"),
          ("Iran_SMAP_MAM_2026_anomaly.tif",      "MAM 2026",    "3-month mean"),
          ("Iran_SMAP_June_2026_anomaly.tif",     "June 2026",   "1-month mean")]

prov = gpd.read_file("../data/gadm/gadm41_IRN_1.shp")
natl = gpd.read_file("../data/gadm/gadm41_IRN_0.shp")

def read(path):
    ds = rasterio.open("../data/rasters/"+path)
    d = {n: ds.read(i + 1).astype("float64") for i, n in enumerate(ds.descriptions)}
    tr = ds.transform
    ext = (tr.c, tr.c + tr.a * ds.width, tr.f + tr.e * ds.height, tr.f)
    lats = tr.f + (np.arange(ds.height) + 0.5) * tr.e
    W = np.broadcast_to(np.cos(np.deg2rad(lats))[:, None], (ds.height, ds.width))
    return d, ext, W

def wmean(a, W):
    m = np.isfinite(a)
    return float(np.sum(np.where(m, a * W, 0)) / np.sum(np.where(m, W, 0)))

def base(ax):
    prov.boundary.plot(ax=ax, linewidth=0.22, edgecolor="#a4a4a4", zorder=4)
    natl.boundary.plot(ax=ax, linewidth=0.65, edgecolor="#3a3a3a", zorder=5)
    ax.set_xlim(43.5, 63.6); ax.set_ylim(24.6, 40.2)
    ax.set_aspect(ASPECT)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

def north_arrow(ax):

    H = 2.1
    w = (H * ASPECT) / 1.7 / 2.0
    xc = SB_X0 + 1.65 * SB_SEG
    y_bot = SB_Y0 + SB_H + 1.10
    y_top, y_notch = y_bot + H, y_bot + 0.30 * H
    whole = [[xc, y_top], [xc - w, y_bot], [xc, y_notch], [xc + w, y_bot]]
    ax.add_patch(plt.Polygon(whole, closed=True, facecolor="white",
                             edgecolor="none", zorder=9))
    ax.add_patch(plt.Polygon(whole, closed=True, facecolor="none",
                             edgecolor="#222222", linewidth=0.7,
                             joinstyle="miter", zorder=10))

def scalebar(ax):
    for i in range(SB_NSEG):
        ax.add_patch(Rectangle((SB_X0 + i * SB_SEG, SB_Y0), SB_SEG, SB_H,
                               facecolor="#222222" if i % 2 == 0 else "white",
                               edgecolor="#222222", linewidth=0.4, zorder=8))
    for i in (0, SB_NSEG):
        lab = "%d km" % (i * SB_STEP_KM) if i == SB_NSEG else "%d" % (i * SB_STEP_KM)
        ax.text(SB_X0 + i * SB_SEG, SB_Y0 + SB_H + 0.12, lab,
                fontsize=9, ha="center", va="bottom", color="#222222", zorder=9)

fig, axes = plt.subplots(2, len(PANELS), figsize=(11.2, 5.4))

fig.subplots_adjust(left=0.030, right=0.918, top=0.925, bottom=0.052,
                    wspace=0.01, hspace=0.03)

stats, clip = {}, {}
for col, (path, lab, dur) in enumerate(PANELS):
    d, ext, W = read(path)
    if "frozen_frac" not in d:
        raise ValueError(path + " is missing frozen_frac; re-export this raster from the current GEE script.")
    for row, (key, depth, lim) in enumerate((("surf_anom", 50.0, SURF_LIM),
                                             ("root_anom", 1000.0, ROOT_LIM))):
        ax = axes[row, col]
        a = d[key] * depth
        ax.imshow(a, extent=ext, origin="upper", cmap=WB,
                  norm=Normalize(-lim, lim), interpolation="none", zorder=2)
        base(ax); scalebar(ax); north_arrow(ax)
        stats[(row, col)] = wmean(a, W)

        mm_ = np.isfinite(a)
        clip[(row, col)] = 100 * np.sum(np.where(mm_ & (np.abs(a) > lim), W, 0)) \
                           / np.sum(np.where(mm_, W, 0))

        fz = d["frozen_frac"]
        mask = np.where(np.isfinite(fz) & (fz > FROZEN_THRESH), 1.0, np.nan)
        ax.contourf(mask, levels=[0.5, 1.5], colors="none",
                    hatches=["////"], extent=ext, origin="upper", zorder=6)

        ax.text(0.985, 0.955, "%+d mm" % int(round(stats[(row, col)])),
                transform=ax.transAxes, fontsize=7.4, color="#1f1f1f",
                ha="right", va="top",
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])

        if row == 0:
            ax.set_title(lab + "\n" + dur, fontsize=9.0, pad=4,
                         linespacing=1.45)

for row, (name, lim) in enumerate((("SMAP L4 surface, 0-5 cm", SURF_LIM),
                                   ("SMAP L4 assimilated root zone, 0-100 cm", ROOT_LIM))):
    p0 = axes[row, 0].get_position()
    fig.text(0.008, (p0.y0 + p0.y1) / 2, name, rotation=90, va="center",
             ha="left", fontsize=8.8)
    cax = fig.add_axes([0.932, p0.y0 + p0.height * 0.20, 0.011,
                        p0.height * 0.58])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-lim, lim), cmap=WB),
                      cax=cax, ticks=[-lim, 0, lim])
    cb.ax.set_yticklabels(["-%d" % lim, "0", "+%d" % lim])
    cb.ax.tick_params(labelsize=6.6, length=2, pad=2)
    cb.outline.set_linewidth(0.4); cb.outline.set_edgecolor("#666666")
    cb.set_label("equivalent water depth (mm)", fontsize=6.6, labelpad=4)

legend_patch = Rectangle((0, 0), 1, 1, facecolor="white",
                         edgecolor="#333333", linewidth=0.5, hatch="////")
leg = fig.legend(
    handles=[legend_patch],

    labels=["Layer-1 soil temperature < 0 °C for >20% of the period"],
    loc="lower left", bbox_to_anchor=(0.030, 0.008), frameon=False,
    fontsize=7.4, handlelength=1.6, handleheight=1.1, borderpad=0.0)

print("national anomalies, mm equivalent")
for col, (_, lab, _dur) in enumerate(PANELS):
    print("  %-22s surface %+6.2f   root zone %+7.2f"
          % (lab, stats[(0, col)], stats[(1, col)]))

for ext_, kw in (("png", dict(dpi=PNG_DPI)), ("pdf", {})):
    out = "../figures/fig08_soil_moisture_maps.%s" % ext_
    fig_style.apply(fig)
    fig.savefig(out, facecolor="white", **kw); print("saved", out)
plt.close(fig)

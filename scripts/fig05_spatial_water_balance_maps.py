#!/usr/bin/env python3

import numpy as np
import geopandas as gpd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fig_style
from matplotlib.colors import (LinearSegmentedColormap, Normalize,
                               ListedColormap, BoundaryNorm)
from matplotlib.patches import Rectangle, Patch
import matplotlib.patheffects as pe

PNG_DPI = 600
WB = LinearSegmentedColormap.from_list(
    "wb", ["#a6611a", "#dfc27d", "#f5f5f5", "#80cdc1", "#018571"])

C_COLS = ["#8c4a26", "#efe8df", "#cfe6d8", "#9ad0b8", "#54a888", "#1b6b52"]
C_EDGES = [-1e6, 0, 10, 25, 50, 100, 1e6]
C_LABELS = ["further negative", "0-10", "10-25", "25-50", "50-100", ">= 100"]

plt.rcParams.update({

    "axes.unicode_minus": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

LABELS = [
    (51.39, 35.69, "Tehran",     "point", 0),
    (45.30, 37.70, "Lake Urmia", "point", 0),
    (59.60, 30.30, "Sistan and\nBaluchestan", "area", 0),
    (53.50, 33.20, "Central\nPlateau", "area", 0),
]

RANGES = [

    ([(49.6, 36.62), (51.3, 36.52), (53.0, 36.46), (54.8, 36.62),
      (56.6, 36.95)],
     "Alborz Mountains", 0),
    ([(45.9, 35.2), (47.6, 33.5), (49.4, 31.9), (51.4, 30.2), (53.4, 28.6)],
     "Zagros Mountains", 0),
]

def curved_label(ax, spine, txt, size=7.2, track=2.4, offset=0.0):

    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties

    lon, lat = np.asarray(spine, float).T
    t = np.linspace(0, 1, len(lon))
    tt = np.linspace(0, 1, 400)
    pts = ax.transData.transform(
        np.column_stack([np.interp(tt, t, lon), np.interp(tt, t, lat)]))
    seg = np.hypot(*np.diff(pts, axis=0).T)
    arc = np.concatenate([[0.0], np.cumsum(seg)])

    px = ax.figure.dpi / 72.0
    fp = FontProperties(size=size, style="italic")
    adv = [TextPath((0, 0), c if c != " " else "n",
                    prop=fp).get_extents().width + track for c in txt]
    total = sum(adv) * px

    pos = (arc[-1] - total) / 2.0 + offset * px
    halo = [pe.withStroke(linewidth=1.7, foreground="#4a4a4a")]
    for c, a in zip(txt, adv):
        d = pos + a * px / 2.0
        if d < 0 or d > arc[-1]:
            pos += a * px
            continue
        x = np.interp(d, arc, pts[:, 0])
        y = np.interp(d, arc, pts[:, 1])
        dx = np.interp(d + 6, arc, pts[:, 0]) - np.interp(d - 6, arc, pts[:, 0])
        dy = np.interp(d + 6, arc, pts[:, 1]) - np.interp(d - 6, arc, pts[:, 1])
        lx, ly = ax.transData.inverted().transform((x, y))
        if c != " ":
            ax.text(lx, ly, c, fontsize=size, style="italic", color="white",
                    ha="center", va="center", zorder=7,
                    rotation=np.degrees(np.arctan2(dy, dx)),
                    rotation_mode="anchor", path_effects=halo)
        pos += a * px

def read(path):
    with rasterio.open("../data/rasters/"+path) as ds:
        a = ds.read(1).astype("float64")
        if ds.nodata is not None:
            a[a == ds.nodata] = np.nan
        tr = ds.transform
    ext = (tr.c, tr.c + tr.a * a.shape[1], tr.f + tr.e * a.shape[0], tr.f)
    return a, ext

m1, ext = read("Iran_Map1_drought_2020_2025_wb_anom.tif")
m2, _ = read("Iran_Map2_recovery_2025_26_wb_anom.tif")
m3, _ = read("Iran_Map3_offset_percent.tif")

prov = gpd.read_file("../data/gadm/gadm41_IRN_1.shp")
natl = gpd.read_file("../data/gadm/gadm41_IRN_0.shp")

def base(ax):
    prov.boundary.plot(ax=ax, linewidth=0.28, edgecolor="#9a9a9a", zorder=4)
    natl.boundary.plot(ax=ax, linewidth=0.75, edgecolor="#3a3a3a", zorder=5)
    halo = [pe.withStroke(linewidth=1.6, foreground="white")]
    for lon, lat, txt, kind, _ in LABELS:
        if kind == "point":
            ax.plot(lon, lat, "o", ms=2.6, color="#222222", zorder=6)
            ax.annotate(txt, (lon, lat), xytext=(4, 3),
                        textcoords="offset points", fontsize=6.6,
                        color="#222222", zorder=7, path_effects=halo)
        else:
            ax.text(lon, lat, txt, fontsize=6.6, ha="center", va="center",
                    color="#3f3f3f", zorder=7, linespacing=1.15,
                    path_effects=halo)

    for spine, txt, off in RANGES:
        curved_label(ax, spine, txt, offset=off)
    ax.set_xlim(43.5, 63.6)
    ax.set_ylim(24.6, 40.2)
    ax.set_aspect(ASPECT)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

ASPECT = 1.0 / np.cos(np.deg2rad(32.0))

SB_X0, SB_Y0, SB_H = 44.3, 26.1, 0.45
SB_STEP_KM, SB_NSEG = 150.0, 3
SB_SEG = SB_STEP_KM / (111.32 * np.cos(np.deg2rad(32.0)))

def scalebar(ax):

    step_km, n_seg, seg = SB_STEP_KM, SB_NSEG, SB_SEG
    x0, y0, h = SB_X0, SB_Y0, SB_H

    for i in range(n_seg):
        ax.add_patch(Rectangle((x0 + i * seg, y0), seg, h,
                               facecolor="#222222" if i % 2 == 0 else "white",
                               edgecolor="#222222", linewidth=0.5, zorder=8))

    for i in (0, n_seg):
        lab = "%d km" % (i * step_km) if i == n_seg else "%d" % (i * step_km)
        ax.text(x0 + i * seg, y0 + h + 0.10, lab, fontsize=9,
                ha="center", va="bottom", color="#222222", zorder=9)

def north_arrow(ax):

    H = 2.4
    w = (H * ASPECT) / 1.7 / 2.0
    xc = SB_X0 + 1.49 * SB_SEG
    y_bot = SB_Y0 + SB_H + 0.80
    y_top, y_notch = y_bot + H, y_bot + 0.30 * H

    west = [[xc, y_top], [xc - w, y_bot], [xc, y_notch]]
    east = [[xc, y_top], [xc, y_notch], [xc + w, y_bot]]
    whole = [[xc, y_top], [xc - w, y_bot], [xc, y_notch], [xc + w, y_bot]]

    ax.add_patch(plt.Polygon(whole, closed=True, facecolor="white",
                             edgecolor="none", zorder=9))

    ax.add_patch(plt.Polygon(whole, closed=True, facecolor="none",
                             edgecolor="#222222", linewidth=0.9,
                             joinstyle="miter", zorder=10))

fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.9))
fig.subplots_adjust(left=0.005, right=0.995, top=0.935, bottom=0.135,
                    wspace=0.02)

PANELS = [
    (axes[0], m1, 800, "(a)  Accumulated anomaly, 2020-2025"),
    (axes[1], m2, 250, "(b)  Wet year, 2025/26"),
]

ims = []
for ax, arr, lim, ttl in PANELS:
    ims.append(ax.imshow(arr, extent=ext, origin="upper", cmap=WB,
                         norm=Normalize(-lim, lim), interpolation="none",
                         zorder=2))
    base(ax); scalebar(ax); north_arrow(ax)
    ax.set_title(ttl, fontsize=9.0, loc="left", pad=6)

axes[2].imshow(m3, extent=ext, origin="upper", cmap=ListedColormap(C_COLS),
               norm=BoundaryNorm(C_EDGES, len(C_COLS)),
               interpolation="none", zorder=2)
base(axes[2]); scalebar(axes[2]); north_arrow(axes[2])
axes[2].set_title("(c)  Wet year share of the deficit",
                  fontsize=9.0, loc="left", pad=6)

CB_W, CB_H, CB_Y = 0.215, 0.020, 0.088
for (ax, _, lim, _), im in zip(PANELS, ims):
    p = ax.get_position()
    cax = fig.add_axes([p.x0 + (p.width - CB_W) / 2, CB_Y, CB_W, CB_H])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal",
                      ticks=[-lim, -lim // 2, 0, lim // 2, lim])
    cb.ax.set_xticklabels(["-%d" % lim, "-%d" % (lim // 2), "0",
                           "+%d" % (lim // 2), "+%d" % lim])
    cb.ax.tick_params(labelsize=7.0, length=2.4, pad=2)
    cb.outline.set_linewidth(0.5); cb.outline.set_edgecolor("#666666")

    cb.ax.xaxis.set_label_position("top")
    cb.set_label("P - PET anomaly (mm)", fontsize=7.4, labelpad=4)

handles = [Patch(facecolor=c, edgecolor="#8a8a8a", linewidth=0.3, label=l)
           for c, l in zip(C_COLS, C_LABELS)]
handles = [handles[i] for i in (0, 3, 1, 4, 2, 5)]
p = axes[2].get_position()
leg = fig.legend(handles=handles, loc="upper center",
                 bbox_to_anchor=(p.x0 + p.width / 2, CB_Y + CB_H + 0.055),
                 ncol=3, frameon=False, fontsize=7.0, handlelength=1.5,
                 handleheight=0.9, columnspacing=1.4, labelspacing=0.45,
                 title="per cent of the accumulated anomaly offset")
leg.get_title().set_fontsize(7.4)

for ext_, kw in (("png", dict(dpi=PNG_DPI)), ("pdf", {})):
    out = "../figures/fig05_spatial_water_balance_maps.%s" % ext_
    fig_style.apply(fig)
    fig.savefig(out, facecolor="white", **kw)
    print("saved", out)
plt.close(fig)

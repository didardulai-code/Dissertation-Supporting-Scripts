#!/usr/bin/env python3

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fig_style
from matplotlib.colors import LinearSegmentedColormap, Normalize, ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle, Patch

PNG_DPI = 600
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

prov = gpd.read_file("../data/gadm/gadm41_IRN_1.shp")
natl = gpd.read_file("../data/gadm/gadm41_IRN_0.shp")

def furniture(ax):
    prov.boundary.plot(ax=ax, linewidth=0.26, edgecolor="#9a9a9a", zorder=4)
    natl.boundary.plot(ax=ax, linewidth=0.7, edgecolor="#3a3a3a", zorder=5)
    for i in range(SB_NSEG):
        ax.add_patch(Rectangle((SB_X0 + i*SB_SEG, SB_Y0), SB_SEG, SB_H,
                               facecolor="#222222" if i % 2 == 0 else "white",
                               edgecolor="#222222", linewidth=0.45, zorder=8))
    for i in (0, SB_NSEG):
        lab = "%d km" % (i*SB_STEP_KM) if i == SB_NSEG else "%d" % (i*SB_STEP_KM)
        ax.text(SB_X0 + i*SB_SEG, SB_Y0 + SB_H + 0.12, lab,
                fontsize=9, ha="center", va="bottom", color="#222222", zorder=9)
    H = 2.2; w = (H*ASPECT)/1.7/2.0
    xc = SB_X0 + 1.65*SB_SEG; yb = SB_Y0 + SB_H + 1.10
    kite = [[xc, yb+H], [xc-w, yb], [xc, yb+0.30*H], [xc+w, yb]]
    ax.add_patch(plt.Polygon(kite, closed=True, facecolor="white",
                             edgecolor="none", zorder=9))
    ax.add_patch(plt.Polygon(kite, closed=True, facecolor="none",
                             edgecolor="#222222", linewidth=0.75,
                             joinstyle="miter", zorder=10))
    ax.set_xlim(43.5, 63.6); ax.set_ylim(24.6, 40.2)
    ax.set_aspect(ASPECT); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

def extent(ds):
    t = ds.transform
    return (t.c, t.c + t.a*ds.width, t.f + t.e*ds.height, t.f)

wb_ds = rasterio.open("../data/rasters/Iran_MAM_2026_wb_anom.tif")
wb = wb_ds.read(1).astype("float64")
wb[wb == wb_ds.nodata] = np.nan

sm_ds = rasterio.open("../data/rasters/Iran_SMAP_MAM_2026_anomaly.tif")
d = {n: sm_ds.read(i+1).astype("float64") for i, n in enumerate(sm_ds.descriptions)}
root = d["root_anom"] * 1000.0

wb_on_sm = np.full(root.shape, np.nan)
reproject(source=wb, destination=wb_on_sm,
          src_transform=wb_ds.transform, src_crs=wb_ds.crs,
          dst_transform=sm_ds.transform, dst_crs=sm_ds.crs,
          resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan)

lats = sm_ds.transform.f + (np.arange(sm_ds.height) + 0.5) * sm_ds.transform.e
W = np.broadcast_to(np.cos(np.deg2rad(lats))[:, None], root.shape)
valid = np.isfinite(root) & np.isfinite(wb_on_sm)

def wcorr(a, b, w, m):
    a, b, w = a[m], b[m], w[m]
    ma, mb = np.sum(w*a)/np.sum(w), np.sum(w*b)/np.sum(w)
    ca, cb = a-ma, b-mb
    return float(np.sum(w*ca*cb)/np.sqrt(np.sum(w*ca**2)*np.sum(w*cb**2)))

r = wcorr(root, wb_on_sm, W, valid)

def wspearman(a, b, w, m):
    ra = np.argsort(np.argsort(a[m])).astype("float64")
    rb = np.argsort(np.argsort(b[m])).astype("float64")
    full_a = np.full(a.shape, np.nan); full_a[m] = ra
    full_b = np.full(b.shape, np.nan); full_b[m] = rb
    return wcorr(full_a, full_b, w, m)

rs = wspearman(root, wb_on_sm, W, valid)

def wsd(a, w, m):
    a_, w_ = a[m], w[m]
    mu = np.sum(w_*a_)/np.sum(w_)
    return float(np.sqrt(np.sum(w_*(a_-mu)**2)/np.sum(w_)))

sd_wb, sd_sm = wsd(wb_on_sm, W, valid), wsd(root, W, valid)
print("area-weighted SD: P - PET anomaly %.1f mm, root-zone anomaly %.1f mm"
      % (sd_wb, sd_sm))
for name, twb, tsm in (("no threshold (as mapped)", 0.0, 0.0),
                       ("fixed 10 mm / 2 mm", 10.0, 2.0),
                       ("0.10 SD", 0.10*sd_wb, 0.10*sd_sm),
                       ("0.25 SD", 0.25*sd_wb, 0.25*sd_sm)):
    keep = valid & (np.abs(wb_on_sm) > twb) & (np.abs(root) > tsm)
    agree = keep & (np.sign(wb_on_sm) == np.sign(root))
    kept = 100*np.sum(np.where(keep, W, 0))/np.sum(np.where(valid, W, 0))
    share = 100*np.sum(np.where(agree, W, 0))/np.sum(np.where(keep, W, 0))
    print("  %-24s cells retained %5.1f %%   signs agree %5.1f %% of those"
          % (name, kept, share))

cls = np.full(root.shape, np.nan)
cls[valid & (wb_on_sm > 0) & (root > 0)] = 0
cls[valid & (wb_on_sm > 0) & (root <= 0)] = 1
cls[valid & (wb_on_sm <= 0) & (root > 0)] = 2
cls[valid & (wb_on_sm <= 0) & (root <= 0)] = 3

COLS = ["#018571",
        "#dfc27d",
        "#80cdc1",
        "#a6611a"]
LABS = ["P - PET above normal, soil wetter",
        "P - PET above normal, soil drier",
        "P - PET below normal, soil wetter",
        "P - PET below normal, soil drier"]

LEG_ORDER = [0, 3, 1, 2]
frac = [100*np.sum(np.where(cls == k, W, 0))/np.sum(np.where(valid, W, 0))
        for k in range(4)]
print("area-weighted r = %.3f  (r2 = %.3f)   Spearman %.3f" % (r, r*r, rs))
for l, f in zip(LABS, frac):
    print("  %-22s %5.1f %%" % (l, f))

fig, ax = plt.subplots(figsize=(8.8, 4.9))
fig.subplots_adjust(left=0.015, right=0.585, top=0.925, bottom=0.02)

ax.imshow(cls, extent=extent(sm_ds), origin="upper",
          cmap=ListedColormap(COLS), norm=BoundaryNorm(np.arange(-0.5, 4.5), 4),
          interpolation="none", zorder=2)
furniture(ax)

fig.legend(handles=[Patch(facecolor=COLS[k], edgecolor="#8a8a8a", linewidth=0.35,
                          label="%s  %.0f%%" % (LABS[k], frac[k]))
                    for k in LEG_ORDER],
           loc="center left", bbox_to_anchor=(0.60, 0.62), ncol=1, frameon=False,
           fontsize=11, handlelength=1.7, labelspacing=0.5, borderpad=0.0)

for ext_, kw in (("png", dict(dpi=PNG_DPI)), ("pdf", {})):
    out = "../figures/fig10_waterbalance_vs_soil.%s" % ext_
    fig_style.apply(fig)
    fig.savefig(out, facecolor="white", **kw); print("saved", out)
plt.close(fig)

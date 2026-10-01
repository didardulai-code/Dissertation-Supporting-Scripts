#!/usr/bin/env python3

import numpy as np, pandas as pd, xarray as xr, geopandas as gpd, warnings
warnings.filterwarnings("ignore")
from scipy import stats
from shapely.geometry import box

NC = "GRCTellus.JPL.200204_202605.GLO.RL06.3M.MSCNv04CRI.nc"
d = xr.open_dataset(NC)
d = d.assign_coords(lon=(((d.lon + 180) % 360) - 180)).sortby("lon")
sub = d.sel(lat=slice(24, 41), lon=slice(43, 65))
lwe = sub.lwe_thickness
sf  = sub.scale_factor
lwe = lwe * sf
t   = pd.to_datetime(sub.time.values)
print("grid %d x %d, %d months, %s to %s"
      % (lwe.sizes["lat"], lwe.sizes["lon"], len(t), t[0].date(), t[-1].date()))

iran = gpd.read_file("../data/gadm/gadm41_IRN_0.shp").union_all()
LA, LO = sub.lat.values, sub.lon.values
res = float(LA[1] - LA[0])
inside = np.zeros((len(LA), len(LO)), bool)
for i, la in enumerate(LA):
    for j, lo in enumerate(LO):
        inside[i, j] = iran.intersects(box(lo-res/2, la-res/2, lo+res/2, la+res/2))
print("cells intersecting Iran: %d" % inside.sum())

A = lwe.values
yrs = (t - t[0]).days / 365.25

def mk(y):
    n = len(y); s = np.sign(np.subtract.outer(y, y))[np.triu_indices(n, 1)].sum()
    z = (s - np.sign(s)) / np.sqrt(n*(n-1)*(2*n+5)/18.0) if s else 0.0
    return z, 2*(1-stats.norm.cdf(abs(z)))

slope = np.full(inside.shape, np.nan); pval = np.full(inside.shape, np.nan)
slope_pre = np.full(inside.shape, np.nan)
pre = t < pd.Timestamp("2025-09-01")
for i in range(len(LA)):
    for j in range(len(LO)):
        if not inside[i, j]: continue
        y = A[:, i, j]
        m = np.isfinite(y)
        if m.sum() < 60: continue
        slope[i, j] = stats.theilslopes(y[m], yrs[m])[0]
        pval[i, j]  = mk(y[m])[1]
        mp = m & pre
        slope_pre[i, j] = stats.theilslopes(y[mp], yrs[mp])[0]

wy = (t >= "2025-09-01") & (t <= "2026-05-31")
print("\nsolutions in the 2025/26 window: %d, %s to %s"
      % (wy.sum(), t[wy][0].date(), t[wy][-1].date()))
anom  = np.full(inside.shape, np.nan)
resid = np.full(inside.shape, np.nan)
resid_full = np.full(inside.shape, np.nan)
mon = t.month

def _fit(y, fitmask):

    sl, ic = stats.theilslopes(y[fitmask], yrs[fitmask])[:2]
    r = y - (ic + sl * yrs)
    clim = pd.Series(r[fitmask]).groupby(mon[fitmask]).mean()
    return r - np.array([clim.get(k, np.nan) for k in mon])

for i in range(len(LA)):
    for j in range(len(LO)):
        if not inside[i, j]: continue
        y = A[:, i, j]
        m = np.isfinite(y)
        if m.sum() < 60: continue
        base = (t.year >= 2004) & (t.year <= 2020) & np.isin(mon, [9,10,11,12,1,2,3,4,5])
        anom[i, j] = np.nanmean(y[wy]) - np.nanmean(y[base & m])
        mp = m & pre
        if mp.sum() < 60: continue
        resid[i, j]      = np.nanmean(_fit(y, mp)[wy & m])
        resid_full[i, j] = np.nanmean(_fit(y, m)[wy & m])

w = np.cos(np.deg2rad(LA))[:, None] * np.ones_like(inside, float)
def wmean(x, m):
    v = m & np.isfinite(x); return float(np.sum(x[v]*w[v]) / np.sum(w[v]))
def wshare(mask, m):
    v = m & np.isfinite(slope); return 100*float(np.sum(w[v & mask]) / np.sum(w[v]))

print("\nTREND, full record 2002-2026")
print("  area-weighted mean %.2f cm/yr" % wmean(slope, inside))
print("  declining %.0f%% of area, of which significant at p<0.05 %.0f%%"
      % (wshare(slope < 0, inside), wshare((slope < 0) & (pval < 0.05), inside)))
print("  steepest %.2f cm/yr, shallowest %.2f" % (np.nanmin(slope[inside]), np.nanmax(slope[inside])))

print("\nDOES THE WET YEAR CHANGE THE TREND?")
print("  trend to Aug 2025      %.3f cm/yr" % wmean(slope_pre, inside))
print("  trend with 2025/26     %.3f cm/yr" % wmean(slope, inside))
print("  cells where the slope becomes less negative: %.0f%% of area"
      % wshare(slope > slope_pre, inside))
print("  cells where the sign flips to positive: %.0f%% of area"
      % wshare((slope_pre < 0) & (slope > 0), inside))

print("\n2025/26 ANOMALY, mean of the nine monthly solutions")
print("  area-weighted %.2f cm" % wmean(anom, inside))
print("  above baseline over %.0f%% of area" % wshare(anom > 0, inside))

def wmedian(x, m):
    v = m & np.isfinite(x); xs = x[v]; ws = w[v]
    o = np.argsort(xs); c = np.cumsum(ws[o]) / ws.sum()
    return float(np.interp(0.5, c, xs[o]))

print("\n2025/26 RESIDUAL, mean of the nine monthly residuals Sep 2025 to May 2026")
print("  fitted to Aug 2025, water year predicted out of sample")
print("    area-weighted mean %.2f cm, median %.2f cm  (left-skewed: the mean carries a"
      " north-western tail reaching %.1f cm)"
      % (wmean(resid, inside), wmedian(resid, inside), np.nanmin(resid[inside])))
print("    above its own trend over %.0f%% of area" % wshare(resid > 0, inside))
print("  sensitivity, fitted to the full record including 2025/26")
print("    area-weighted %.2f cm" % wmean(resid_full, inside))
print("    above its own trend over %.0f%% of area" % wshare(resid_full > 0, inside))
v = inside & np.isfinite(resid) & np.isfinite(resid_full)
print("    spatial correlation of the two versions r = %+.3f, sign agreement %.0f%%"
      % (np.corrcoef(resid[v], resid_full[v])[0, 1],
         100*np.mean(np.sign(resid[v]) == np.sign(resid_full[v]))))

steep = slope <= np.nanpercentile(slope[inside], 50)
print("\nIS THE RESPONSE WHERE THE DECLINE IS STEEPEST?")
for lab, m in (("steeper half, above own trend", steep & (resid > 0)),
               ("steeper half, below own trend", steep & (resid <= 0)),
               ("shallower half, above own trend", ~steep & (resid > 0)),
               ("shallower half, below own trend", ~steep & (resid <= 0))):
    print("  %-34s %.0f%% of area" % (lab, wshare(m, inside)))
v = inside & np.isfinite(slope) & np.isfinite(anom)
print("  trend vs RAW anomaly      r = %+.2f  (circular: the anomaly re-measures the trend)"
      % np.corrcoef(slope[v], anom[v])[0, 1])
v = inside & np.isfinite(slope) & np.isfinite(resid)
print("  trend vs RESIDUAL         r = %+.2f" % np.corrcoef(slope[v], resid[v])[0, 1])

np.savez("grace_fields.npz", lat=LA, lon=LO, inside=inside, slope=slope,
         pval=pval, slope_pre=slope_pre, anom=anom, resid=resid,
         resid_full=resid_full)
print("\nsaved grace_fields.npz")

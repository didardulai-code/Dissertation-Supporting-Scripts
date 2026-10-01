#!/usr/bin/env python3
import pandas as pd, numpy as np

d = pd.read_pickle("snow_all.pkl")
d = d[d.snow_frac.notna()].copy()
d["dt"] = pd.to_datetime(d["date"]); d["m"] = d.dt.dt.month

meta = d.groupby("hybas_id").agg(
    area_km2=("sub_area_km2","first"), elev_p50_m=("elev_p50","first"),
    elev_p90_m=("elev_p90","first"), forest_frac=("forest_frac","first"),
    obs_frac_mean=("obs_frac","mean"))

x = d[d.m.isin([9,10,11,12,1,2,3,4,5,6])].copy()
x["sy"] = np.where(x.m >= 9, x.year + 1, x.year)

def per(g):
    return pd.Series({
        "peak":   g.snow_frac.max(),
        "late":   g.loc[(g.window>=13)&(g.window<=22),"snow_frac"].mean(),
        "winter": g.loc[g.m.isin([12,1,2]),"snow_frac"].mean()})

BY   = x.groupby(["hybas_id","sy"]).apply(per, include_groups=False).reset_index()
base = BY[(BY.sy>=2004)&(BY.sy<=2025)]
cur  = BY[BY.sy==2026].set_index("hybas_id")
g    = base.groupby("hybas_id")
mp, lm, ls, wm = g.peak.mean(), g.late.mean(), g.late.std(), g.winter.mean()

T = pd.DataFrame({
    "area_km2": meta.area_km2.round(0), "elev_p50_m": meta.elev_p50_m.round(0),
    "elev_p90_m": meta.elev_p90_m.round(0), "forest_frac": meta.forest_frac.round(3),
    "obs_frac_mean": meta.obs_frac_mean.round(3),
    "mean_peak_snow_frac_2003_25": mp.round(4),
    "winter_2003_25": wm.round(4), "winter_2026": cur.winter.round(4),
    "winter_ratio": (cur.winter/wm).round(2),
    "late_2003_25": lm.round(4), "late_2026": cur.late.round(4),
    "late_ratio": (cur.late/lm).round(2),
    "late_z_2026": ((cur.late-lm)/ls).round(2)})
T["snow_carrying"] = (mp >= 0.20).astype(int)
T = T.sort_values(["snow_carrying","late_z_2026"], ascending=[False,False])
T.index.name = "hybas_id"
T.to_csv("snow_results_table.csv")
print("written snow_results_table.csv: %d basins, %d snow-carrying" % (len(T), T.snow_carrying.sum()))

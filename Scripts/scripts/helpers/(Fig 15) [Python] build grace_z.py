import pandas as pd
from scipy import stats
g = pd.read_csv("iran_grace_tws_monthly.csv"); g["date"] = pd.to_datetime(g.date)
g["year"] = g.date.dt.year; g["month"] = g.date.dt.month; g["tdec"] = g.year + (g.month-0.5)/12.0
sl, ic, *_ = stats.linregress(g.tdec, g.tws_cm); g["resid"] = g.tws_cm - (sl*g.tdec + ic)
clim = g[g.year<=2024].groupby("month").resid.mean(); g["des"] = g.resid - g.month.map(clim)
sdm = g[g.year<=2024].groupby("month").resid.std(ddof=1); g["z"] = g.des / g.month.map(sdm)
tgt = g[((g.year==2025)&(g.month>=9))|((g.year==2026)&(g.month<=6))].copy()
tgt["ym"] = tgt.year.astype(str) + "-" + tgt.month.map(lambda x: "%02d"%x)
tgt[["ym","z"]].round(3).to_csv("grace_z.csv", index=False)
print("wrote grace_z.csv")

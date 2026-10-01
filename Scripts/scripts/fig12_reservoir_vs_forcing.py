#!/usr/bin/env python3
import pandas as pd, numpy as np
from mono_common import style, BLACK, GREY, DARK
from _ci import spearman_ci
try:
    from adjustText import adjust_text; HAVE_AT=True
except ImportError:
    HAVE_AT=False
style()
import matplotlib.pyplot as plt
import fig_style

NAME={"Doroodzan":"Doroudzan","Karoon":"Karun-1","Sefidrood":"Sefid Rud",
      "Zarineh Rood":"Zarrineh Rud","Zayandeh Rood":"Zayandeh Rud"}
d=pd.read_csv("../data/swm.csv"); d["area"]=d.water_mndwi0_km2; d=d[d.usable==True].copy()
base=d[(d.year>=1991)&(d.year<=2020)]
ref=base.groupby(["site","month"]).area.agg(["mean","std","count"]); ref.columns=["m","sd","n"]
d=d.merge(ref,on=["site","month"],how="left"); d=d[(d.sd>0)&(d.n>=20)].copy()
d["z"]=(d.area-d.m)/d.sd; d["t"]=(d.year-2025)*12+d.month
wy=d[(d.t>=9)&(d.t<=18)]
def agg(g): return pd.Series(dict(
    change=g[g.t.isin([16,17,18])].z.mean()-g[g.t.isin([9,10,11])].z.mean(), cond=g.z.mean()))
site=wy.groupby("site").apply(agg).reset_index()
res=pd.read_csv("../data/reservoirs.csv"); res["HID"]=res.res_l5.astype('int64')
res["site"]=res.reservoir_name.replace(NAME)
m=site.merge(res[["site","HID"]],on="site",how="inner").merge(pd.read_csv("../data/basin_cwb.csv"),on="HID",how="left")

fig,(axA,axB)=plt.subplots(1,2,figsize=(9.8,4.0),gridspec_kw=dict(wspace=0.5))
for ax,yv,ylab,ttl in [(axA,"cond","Reservoir area anomaly\n(mean SD)","(a) Condition vs forcing"),
                       (axB,"change","Autumn to spring change (SD)","(b) Change vs forcing")]:
    ax.axhline(0,color=GREY,lw=0.7); ax.axvline(0,color=GREY,lw=0.7)
    ax.scatter(m.cwb,m[yv],s=50,color=BLACK,edgecolor="white",lw=0.6)
    _tx=[ax.text(x,y,t,fontsize=6.4,color="#8a8a8a",zorder=5) for x,y,t in zip(m.cwb,m[yv],m.site)]
    if HAVE_AT: adjust_text(_tx,ax=ax,arrowprops=dict(arrowstyle="-",color="#c2c2c2",lw=0.4),expand=(1.05,1.25),min_arrow_len=6)
    r,p,lo,hi,k=spearman_ci(m.cwb.values,m[yv].values)
    print("%s: rho=%.2f p=%.3f n=%d 95%% CI[%.2f,%.2f]"%(ttl,r,p,k,lo,hi))
    ax.set_xlabel("Basin WB anomaly (mm)", labelpad=8); ax.set_ylabel(ylab, labelpad=9)
    ax.set_title(ttl,loc="left",fontsize=8.9)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
fig_style.apply(fig)
for e in ("png","pdf"): fig.savefig("../figures/fig12_reservoir_vs_forcing."+e,dpi=300,bbox_inches="tight",facecolor="white")
print("Fig14 done")

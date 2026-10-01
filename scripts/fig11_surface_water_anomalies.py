#!/usr/bin/env python3
import numpy as np, pandas as pd
from mono_common import style, BLACK, MED, GREY
style()
import matplotlib.pyplot as plt
import fig_style
from matplotlib.lines import Line2D

d=pd.read_csv("../data/swm.csv"); d["area"]=d.water_mndwi0_km2; d=d[d.usable==True].copy()
base=d[(d.year>=1991)&(d.year<=2020)]
ref=base.groupby(["site","month"]).area.agg(["mean","std","count"]); ref.columns=["m","sd","n"]
d=d.merge(ref,on=["site","month"],how="left"); d=d[(d.sd>0)&(d.n>=20)].copy()
d["z"]=(d.area-d.m)/d.sd
d["grp"]=np.where(d.site_class=="reservoir","reservoir","lake")
d["t"]=(d.year-2025)*12+d.month
wy=d[(d.t>=9)&(d.t<=18)].copy()
MONTHS=["Sep","Oct","Nov","Dec","Jan","Feb","Mar","Apr","May","Jun"]
traj=wy.groupby(["grp","t"]).z.mean().reset_index()
def agg(g):
    aut=g[g.t.isin([9,10,11])].z.mean(); spr=g[g.t.isin([16,17,18])].z.mean()
    return pd.Series(dict(grp=g.grp.iloc[0],autumn=aut,spring=spr,change=spr-aut,cond=g.z.mean()))
by=wy.groupby("site").apply(agg).reset_index().dropna(subset=["change"])

fig=plt.figure(figsize=(11.6,4.6))
gs=fig.add_gridspec(1,3,width_ratios=[1.25,1.0,1.0],wspace=0.62)

axA=fig.add_subplot(gs[0,0]); axA.axhline(0,color=GREY,lw=0.8)
for g,lab,ls,mk,mf in [("reservoir","Reservoirs (n=12)","-","o",BLACK),
                       ("lake","Terminal lakes/wetland (n=5)","--","s","white")]:
    s=traj[traj.grp==g].sort_values("t")
    axA.plot(s.t-9,s.z,color=BLACK if g=="reservoir" else MED,lw=1.6,ls=ls,
             marker=mk,ms=4.5,mfc=mf,mec=BLACK if g=="reservoir" else MED,mew=1.0,label=lab)
axA.set_xticks(range(10)); axA.set_xticklabels(MONTHS,fontsize=7.6,rotation=90); axA.set_ylim(-1.08,0.62)
axA.set_xlabel("Month", labelpad=6)
axA.set_ylabel("Surface-water area anomaly\n(SD)")
axA.set_title("(a) Monthly trajectory",loc="left",fontsize=9.0,pad=5)
axA.legend(frameon=False,fontsize=7.4,loc="upper left")
for sp in ["top","right"]: axA.spines[sp].set_visible(False)

axB=fig.add_subplot(gs[0,1]); b=by.sort_values("change"); y=np.arange(len(b))
for yi,r in zip(y,b.itertuples()):
    axB.plot([r.autumn,r.spring],[yi,yi],color=GREY,lw=1.0,zorder=1)
    axB.scatter(r.autumn,yi,facecolor="white",edgecolor=BLACK,s=24,lw=0.9,zorder=2)
    axB.scatter(r.spring,yi,color=BLACK,s=28,zorder=3)
axB.axvline(0,color=BLACK,lw=0.7); axB.set_xlim(-2.9,1.25); axB.set_yticks(y); axB.set_yticklabels(b.site,fontsize=6.2)
axB.set_xlabel("SD anomaly"); axB.set_title("(b) Autumn to spring change",loc="left",fontsize=9.0,pad=5)
_hb=[Line2D([0],[0],marker="o",ls="none",mfc="white",mec=BLACK,mew=0.9,ms=6,label="Autumn (Sep-Nov)"),
     Line2D([0],[0],marker="o",ls="none",mfc=BLACK,mec=BLACK,ms=6,label="Spring (Apr-Jun)")]
axB.legend(handles=_hb,frameon=False,fontsize=6.2,loc="lower left",
           handletextpad=0.35,labelspacing=0.30,borderpad=0.2)
for sp in ["top","right","left"]: axB.spines[sp].set_visible(False)

axC=fig.add_subplot(gs[0,2]); c=by.sort_values("cond"); y=np.arange(len(c))
axC.barh(y,c.cond,color=MED,height=0.7)
axC.axvline(0,color=BLACK,lw=0.7); axC.set_yticks(y); axC.set_yticklabels(c.site,fontsize=6.2)
axC.set_xlabel("Water-year mean (SD)"); axC.set_title("(c) Overall condition",loc="left",fontsize=9.0,pad=5)
for sp in ["top","right","left"]: axC.spines[sp].set_visible(False)

fig_style.apply(fig, legend=7.5)
for e in ("png","pdf"): fig.savefig("../figures/fig11_surface_water_anomalies."+e,dpi=300,bbox_inches="tight",pad_inches=0.15,facecolor="white")
print("Fig12 n=%d res=%d lake=%d"%(len(by),(by.grp=='reservoir').sum(),(by.grp=='lake').sum()))

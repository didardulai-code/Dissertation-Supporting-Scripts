#!/usr/bin/env python3
import numpy as np, pandas as pd
from mono_common import style, BLACK, DARK, MED, GREY, BAND
style()
import matplotlib.pyplot as plt
import fig_style

d=pd.read_csv("../data/iran_grace_tws_monthly.csv",parse_dates=["date"]).sort_values("date")
d["yr"]=d.date.dt.year+(d.date.dt.dayofyear-1)/365.25
cut=pd.Timestamp("2025-09-01")
tr=d[d.date<cut]
X=np.column_stack([np.ones(len(tr)),tr.yr,np.sin(2*np.pi*tr.yr),np.cos(2*np.pi*tr.yr)])
beta,*_=np.linalg.lstsq(X,tr.tws_cm,rcond=None)
pred=lambda yr: beta[0]+beta[1]*yr+beta[2]*np.sin(2*np.pi*yr)+beta[3]*np.cos(2*np.pi*yr)
d["exp"]=pred(d.yr.values); d["resid"]=d.tws_cm-d["exp"]
wy=d[d.date>=cut]

fig,(axA,axB)=plt.subplots(2,1,figsize=(9.4,5.6),
    gridspec_kw=dict(height_ratios=[1.5,1],hspace=0.62))
axA.plot(d.yr,d.tws_cm,color=MED,lw=0.9,label="Observed TWS")
axA.plot(d.yr,d["exp"],color=BLACK,lw=1.2,ls="--",label="Pre-2025/26 trend + seasonal expectation")
axA.plot(wy.yr,wy.tws_cm,color=BLACK,lw=0,marker="o",ms=2.3,label="2025/26 observed")
axA.axvspan(2025.67,2026.5,color=BAND,alpha=0.55)
axA.set_ylabel("TWS anomaly\n(cm water equiv.)"); axA.set_xlabel("Year")
axA.set_xlim(2002,2026.6); axA.legend(frameon=False,fontsize=7.6,loc="lower left")
axA.set_title("(a) National terrestrial water storage and its pre-2025/26 expected trajectory",
    loc="left",fontsize=9.0,pad=5)
for sp in ["top","right"]: axA.spines[sp].set_visible(False)

axB.axhline(0,color=BLACK,lw=0.7)
axB.plot(d.yr,d.resid,color=MED,lw=0.7)
axB.bar(wy.yr,wy.resid,width=0.06,color=DARK)
axB.axvspan(2025.67,2026.5,color=BAND,alpha=0.55)
axB.set_ylabel("TWS departure from\nexpected trajectory (cm)"); axB.set_xlabel("Year"); axB.set_xlim(2002,2026.6)
axB.set_title("(b) Departure from the expected trajectory - the 2025/26 residual",
    loc="left",fontsize=9.0,pad=5)
for sp in ["top","right"]: axB.spines[sp].set_visible(False)

fig_style.apply(fig)
for e in ("png","pdf"): fig.savefig("../figures/fig13_tws_trajectory."+e,dpi=600,bbox_inches="tight",facecolor="white")
print("Fig10 residual mean %+.2f max %+.2f last %+.2f"%(wy.resid.mean(),wy.resid.max(),wy.resid.iloc[-1]))

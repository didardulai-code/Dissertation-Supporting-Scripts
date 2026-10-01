import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fig_style
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap, Normalize
MONTHS=["Sep","Oct","Nov","Dec","Jan","Feb","Mar","Apr","May","Jun"]
mnum=lambda p:(p.year-2025)*12+p.month
SEAS={"SON":(9,11),"DJF":(12,14),"MAM":(15,17)}
_as=pd.read_csv("../data/all_seasons_combined.csv"); _as["wb"]=_as.precip_chirps_mm-_as.pet_pm_mm
_tgt={"SON":2025,"DJF":2026,"MAM":2026}
def _seasz(col,seas):
    _sub=_as[_as.season==seas]; _b=_sub[(_sub.season_year>=1991)&(_sub.season_year<=2020)]
    return (float(_sub[_sub.season_year==_tgt[seas]][col].iloc[0])-_b[col].mean())/_b[col].std()
P={s:_seasz("precip_chirps_mm",s) for s in ["SON","DJF","MAM"]}
W={s:_seasz("wb",s) for s in ["SON","DJF","MAM"]}
def from_seasonal(d):
    r=np.full(10,np.nan)
    for s,(lo,hi) in SEAS.items():
        for t in range(lo,hi+1): r[t-9]=d[s]
    return r
def from_monthly(df,tc,zc):
    r=np.full(10,np.nan)
    for _,x in df.iterrows():
        i=int(x[tc])-9
        if 0<=i<10: r[i]=x[zc]
    return r
snow=pd.read_csv("../data/snow_z.csv"); snow["t"]=pd.PeriodIndex(snow.ym,freq="M").map(mnum)
sm=pd.read_csv("../data/sm_z.csv"); sm["t"]=(sm.year-2025)*12+sm.month
gr=pd.read_csv("../data/grace_z.csv"); gr["t"]=pd.PeriodIndex(gr.ym,freq="M").map(mnum)
sw=pd.read_csv("../data/sw_z.csv"); sw.columns=["t","z"]
ROWS=[("Precipitation","CHIRPS · national · 1991–2020",from_seasonal(P)),
      ("Water balance","P - PET · national · 1991–2020",from_seasonal(W)),
      ("Mountain snow cover","13 snow basins · 2003–2025",from_monthly(snow,"t","z")),
      ("Root-zone soil moisture","SMAP L4 0–100 cm · 2015–2025",from_monthly(sm,"t","z")),
      ("Surface-water extent","12 reservoirs · 1991–2020",from_monthly(sw,"t","z")),
      ("Terrestrial water storage","GRACE · detrended",from_monthly(gr,"t","z"))]
names=[r[0] for r in ROWS]; subs=[r[1] for r in ROWS]
M=np.vstack([r[2] for r in ROWS]); n=len(ROWS)
DARK="#4a4f54"; GREY="#d6dadf"
CMAP=LinearSegmentedColormap.from_list("s",["#90643f","#cdb391","#f1efe9","#a7c6ba","#4f8f7e"])
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["DejaVu Sans"],
    "font.size":9.0,"axes.linewidth":0.6,"axes.unicode_minus":False,"figure.facecolor":"white","pdf.fonttype":42})
vmax=float(np.nanmax(np.abs(M))); dnorm=TwoSlopeNorm(vmin=-vmax,vcenter=0.0,vmax=vmax)
TEAL=LinearSegmentedColormap.from_list("t",["#cfe0d8","#8fbcac","#4f8f7e"]); anorm=Normalize(0.0,3.0)
DOT=95
fig,(axA,axB)=plt.subplots(2,1,figsize=(10.6,8.0),gridspec_kw=dict(height_ratios=[1.0,1.12],hspace=0.62))
for ri in range(n):
    yi=n-1-ri; a=M[ri]
    axA.plot([0,9],[yi,yi],color=GREY,lw=0.5,zorder=1)
    pos=[i for i in range(10) if np.isfinite(a[i]) and a[i]>0]
    for i in pos:
        axA.scatter(i,yi,s=DOT,color=TEAL(anorm(a[i])),edgecolor="white",lw=0.6,zorder=3)
    if pos:
        axA.scatter(pos[0],yi,s=235,facecolor="none",edgecolor=DARK,lw=1.2,zorder=4)
        pk=int(np.nanargmax(a))
        axA.annotate("+%.1f"%np.nanmax(a),(pk,yi),xytext=(0,10),textcoords="offset points",ha="center",fontsize=8.8,color=DARK)
axA.set_yticks([n-1-ri for ri in range(n)]); axA.set_yticklabels(names,fontsize=10.2)
axA.set_xticks(range(10)); axA.set_xticklabels(MONTHS,fontsize=10.5); axA.set_xlim(-0.6,9.6); axA.set_ylim(-0.7,n-0.3); axA.tick_params(length=0)
for sp in axA.spines.values(): sp.set_visible(False)
axA.set_title("(a) Timing of positive anomalies",loc="left",fontsize=10.4,pad=8)
ref=[1,2,3]
hd=[axA.scatter([],[],s=DOT,color=TEAL(anorm(v)),edgecolor="white",lw=0.6,label="%d"%v) for v in ref]
hd.append(axA.scatter([],[],s=160,facecolor="none",edgecolor=DARK,lw=1.2,label="first positive month"))
leg=axA.legend(handles=hd,loc="upper center",bbox_to_anchor=(0.5,-0.16),ncol=4,frameon=False,
               fontsize=9.4,handletextpad=0.5,columnspacing=1.6,
               title="Standardised anomaly (SD)")
leg.get_title().set_fontsize(9.4)
im=axB.imshow(M,cmap=CMAP,norm=dnorm,aspect="auto",interpolation="none",extent=[-0.5,9.5,n-0.5,-0.5])
for i in range(n):
    for j in range(10):
        v=M[i,j]
        if np.isfinite(v): axB.text(j,i,"%.1f"%v,ha="center",va="center",fontsize=9.4,color="white" if abs(v)>0.66*vmax else "#2b3338")
        else: axB.text(j,i,"–",ha="center",va="center",fontsize=7.5,color=GREY)
axB.set_yticks(range(n)); axB.set_yticklabels([""]*n)
trans=axB.get_yaxis_transform()
for i,(nm,su) in enumerate(zip(names,subs)):
    axB.text(-0.015,i-0.15,nm,transform=trans,ha="right",va="center",fontsize=9.8,color="#2b3338")
    axB.text(-0.015,i+0.21,su,transform=trans,ha="right",va="center",fontsize=7.0,color="#9a9a9a")
axB.set_xticks(range(10)); axB.set_xticklabels(MONTHS,fontsize=10.5); axB.set_xlim(-0.6,9.6); axB.tick_params(length=0)
for k in range(1,10): axB.axvline(k-0.5,color="white",lw=0.6)
for k in range(1,n): axB.axhline(k-0.5,color="white",lw=0.6)
for sp in axB.spines.values(): sp.set_visible(False)
axB.set_title("(b) Monthly standardised anomalies",loc="left",fontsize=10.4,pad=8)
cb=fig.colorbar(im,ax=axB,fraction=0.030,pad=0.022); cb.set_label("Standardised anomaly (SD)",fontsize=9.0,labelpad=16); cb.ax.tick_params(labelsize=8.6); cb.outline.set_linewidth(0.4)
fig_style.apply(fig)
fig.savefig("../figures/fig15_synthesis.png",dpi=300,bbox_inches="tight",facecolor="white")
fig.savefig("../figures/fig15_synthesis.pdf",bbox_inches="tight",facecolor="white")
print("saved Fig15_combined (13 snow basins)")

import pandas as pd, numpy as np
from mono_common import style, BLACK, MED, GREY
style()
import matplotlib.pyplot as plt
import fig_style
try:
    from adjustText import adjust_text; HAVE_AT=True
except ImportError:
    HAVE_AT=False

NAMES={2050085980:'Semnan',2050643590:'Aras',2050085690:'Urmia',2050070510:'Sefid Rud',2050070520:'Lar',
 2050724400:'Kashaf Rud',2050085590:'Karaj-Jajrud',2050773470:'Sirvan',2050724600:'Hari Rud',
 2050816870:'Karkheh',2050085640:'Hamoun',2050828790:'Dez-Karun',2050085930:'Bakhtegan'}

sn=pd.read_csv("../data/snow_results_table.csv"); sn["HID"]=sn.hybas_id.astype('int64')
fr=pd.read_csv("../data/basin_iran_fraction.csv").set_index("hybas_id").frac_in_iran
DRAINS_IN={2050643590,2050724600,2050085640}
carry=sn[sn.mean_peak_snow_frac_2003_25>=0.20]
keep=carry[carry.HID.apply(lambda h: fr.get(h,0)>=0.50 or h in DRAINS_IN)]
s=keep.copy(); s["name"]=s.HID.map(NAMES); n=len(s)

fig,(axA,axB)=plt.subplots(1,2,figsize=(9.6,4.2),
    gridspec_kw=dict(width_ratios=[1,1.2],wspace=0.42))

axA.axhline(1,color=GREY,lw=0.8,ls="--"); axA.axvline(1,color=GREY,lw=0.8,ls="--")
axA.scatter(s.winter_ratio,s.late_ratio,s=42,color=BLACK,edgecolor="white",lw=0.5,zorder=3)
_tx=[axA.text(x,y,t,fontsize=6.6,color="#8a8a8a",zorder=5) for x,y,t in zip(s.winter_ratio,s.late_ratio,s.name)]
if HAVE_AT: adjust_text(_tx,ax=axA,arrowprops=dict(arrowstyle="-",color="#c2c2c2",lw=0.4),expand=(1.05,1.25),min_arrow_len=5)
axA.set_xlabel("Peak winter snow (ratio to normal)", labelpad=9)
axA.set_ylabel("Late-season snow (ratio to normal)", labelpad=9)
axA.set_title("(a) Peak vs persistence, by basin",loc="left",fontsize=9.0,pad=5)
for sp in ["top","right"]: axA.spines[sp].set_visible(False)

s2=s.sort_values("late_z_2026"); y=np.arange(n)
axB.barh(y,s2.late_z_2026,color=MED,height=0.72)
axB.axvline(0,color=BLACK,lw=0.8); axB.axvline(2,color=GREY,lw=0.7,ls=":")
axB.set_yticks(y); axB.set_yticklabels(s2.name,fontsize=7.4); axB.set_ylim(-0.6,n-0.4)
axB.set_xlabel("Late-season persistence anomaly\n(SD)", labelpad=9)
axB.set_title("(b) Persistence anomaly by basin",loc="left",fontsize=9.0,pad=5)
for sp in ["top","right","left"]: axB.spines[sp].set_visible(False)

for e in ("png","pdf"):
    fig_style.apply(fig)
    fig.savefig("../figures/fig06_snow_peak_vs_persistence."+e,dpi=300,bbox_inches="tight",pad_inches=0.15,facecolor="white")
print("Fig05 rebuilt, n=%d, order(top->bottom):"%n, list(s2.name)[::-1])

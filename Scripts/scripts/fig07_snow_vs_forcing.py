import geopandas as gpd, pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from mono_common import style, BLACK, GREY, DARK
from _ci import spearman_ci
try:
    from adjustText import adjust_text; HAVE_AT=True
except ImportError:
    HAVE_AT=False
style()
import matplotlib.pyplot as plt
import fig_style
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

INK="#37474f"
NAMES={2050085980:'Semnan',2050643590:'Aras',2050085690:'Urmia',2050070510:'Sefid Rud',2050070520:'Lar',
 2050724400:'Kashaf Rud',2050085590:'Karaj-Jajrud',2050773470:'Sirvan',2050724600:'Hari Rud',
 2050816870:'Karkheh',2050085640:'Hamoun',2050828790:'Dez-Karun',2050085930:'Bakhtegan'}
iran=gpd.read_file("../data/gadm/gadm41_IRN_0.shp")
hb=gpd.read_file("../data/hydrobasins/hybas_eu_lev05_v1c.shp"); hb["HID"]=hb.HYBAS_ID.astype('int64')
sn=pd.read_csv("../data/snow_results_table.csv"); sn["HID"]=sn.hybas_id.astype('int64')
fr=pd.read_csv("../data/basin_iran_fraction.csv").set_index("hybas_id").frac_in_iran
DRAINS_IN={2050643590,2050724600,2050085640}
carry=sn[sn.mean_peak_snow_frac_2003_25>=0.20]
keep=set(carry[carry.HID.apply(lambda h: fr.get(h,0)>=0.50 or h in DRAINS_IN)].HID)
cwb=pd.read_csv("../data/basin_cwb.csv"); s=sn[sn.HID.isin(keep)].merge(cwb,on="HID",how="left")
g=hb.merge(s[["HID","cwb","late_z_2026"]],on="HID",how="inner")
gp=g.to_crs(3857); g["cx"]=gp.geometry.centroid.to_crs(4326).x; g["cy"]=gp.geometry.centroid.to_crs(4326).y
g["name"]=g.HID.map(NAMES)
minx,miny,maxx,maxy=iran.total_bounds; b2=g.total_bounds
X0,Y0=min(minx,b2[0])-0.4,min(miny,b2[1])-0.4; X1,Y1=max(maxx,b2[2])+0.4,max(maxy,b2[3])+0.6

fig=plt.figure(figsize=(11.6,7.45)); gs=fig.add_gridspec(2,2,height_ratios=[1.8,1.0],hspace=0.12,wspace=0.20)
def draw(ax,col,cmap,norm,title,clabel):
    iran.plot(ax=ax,color="#eceef0",edgecolor="#c7ccd1",lw=0.5,zorder=1)
    g.plot(column=col,cmap=cmap,norm=norm,ax=ax,edgecolor="white",lw=0.4,zorder=2)
    for r in g.itertuples():
        ax.annotate(r.name,(r.cx,r.cy),fontsize=6.4,ha="center",va="center",color=INK,zorder=3,
                    bbox=dict(boxstyle="round,pad=0.12",fc="white",ec="none",alpha=0.6))
    ax.set_title(title,loc="left",fontsize=10.5,pad=8); ax.axis("off"); ax.set_xlim(X0,X1); ax.set_ylim(Y0,Y1); ax.set_anchor("N")
    cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=ax,fraction=0.033,pad=0.02,shrink=0.70)
    cb.set_label(clabel,fontsize=7.2); cb.ax.tick_params(labelsize=6.8); cb.outline.set_linewidth(0.4)
cnorm=TwoSlopeNorm(vmin=min(-1,g.cwb.min()),vcenter=0,vmax=g.cwb.max())
draw(fig.add_subplot(gs[0,0]),"cwb","BrBG",cnorm,"(a) Forcing: basin P-ET$_0$ anomaly","WB anomaly (mm)")
PT=LinearSegmentedColormap.from_list("pt",["#d6a0c0","#f4f1ec","#8fc9bb","#018571"])
snorm=TwoSlopeNorm(vmin=min(-1.05,g.late_z_2026.min()),vcenter=0,vmax=g.late_z_2026.max())
draw(fig.add_subplot(gs[0,1]),"late_z_2026",PT,snorm,"(b) Response: late-season snow persistence","persistence (SD)")

axC=fig.add_subplot(gs[1,:]); axC.axhline(0,color=GREY,lw=0.7); axC.axvline(0,color=GREY,lw=0.7)
axC.scatter(s.cwb,s.late_z_2026,s=70,color=BLACK,edgecolor="white",lw=0.7,zorder=3)
_tx=[axC.text(x,y,NAMES.get(int(h),""),fontsize=9.4,color="#8a8a8a",zorder=5) for x,y,h in zip(s.cwb,s.late_z_2026,s.HID)]
if HAVE_AT: adjust_text(_tx,ax=axC,arrowprops=dict(arrowstyle="-",color="#c2c2c2",lw=0.5),expand=(1.05,1.25),min_arrow_len=6)
r,p,lo,hi,k=spearman_ci(s.cwb.values,s.late_z_2026.values)
axC.set_xlabel("Basin WB anomaly (mm)", labelpad=9, fontsize=12); axC.set_ylabel("Late-season snow persistence\nanomaly (SD)", labelpad=9, fontsize=12)
axC.tick_params(axis="both", labelsize=12)
axC.set_title("(c) Snow response vs climatic forcing",loc="left",fontsize=9.8,pad=6)
for sp in ["top","right"]: axC.spines[sp].set_visible(False)
fig_style.apply(fig)
for e in ("png","pdf"): fig.savefig("../figures/fig07_snow_vs_forcing."+e,dpi=300,bbox_inches="tight",facecolor="white")
print("Fig06 rho=%.2f CI[%.2f,%.2f]"%(r,lo,hi))

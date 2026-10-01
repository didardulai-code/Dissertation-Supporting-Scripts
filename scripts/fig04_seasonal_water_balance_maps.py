import numpy as np, geopandas as gpd, rasterio, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fig_style
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe

PNG_DPI = 600
VMAX = 250.0
WB = LinearSegmentedColormap.from_list("wb", ["#a6611a","#dfc27d","#f5f5f5","#80cdc1","#018571"])
plt.rcParams.update({"axes.unicode_minus": False, "font.family": "sans-serif",
    "font.sans-serif": ["Arial","Helvetica","DejaVu Sans"], "pdf.fonttype": 42, "ps.fonttype": 42})

ASPECT = 1.0 / np.cos(np.deg2rad(32.0))
SB_X0, SB_Y0, SB_H = 44.3, 26.1, 0.60
SB_STEP_KM, SB_NSEG = 150.0, 3
SB_SEG = SB_STEP_KM / (111.32 * np.cos(np.deg2rad(32.0)))
LABELS = [(51.39,35.69,"Tehran"), (45.30,37.70,"Lake Urmia")]
RANGES = [([(49.6,36.62),(51.3,36.52),(53.0,36.46),(54.8,36.62),(56.6,36.95)],"Alborz Mountains"),
          ([(45.9,35.2),(47.6,33.5),(49.4,31.9),(51.4,30.2),(53.4,28.6)],"Zagros Mountains")]
PANELS = [("Iran_SON_2025_wb_anom.tif","(a)  SON 2025"),
          ("Iran_DJF_2026_wb_anom.tif","(b)  DJF 2025/26"),
          ("Iran_MAM_2026_wb_anom.tif","(c)  MAM 2026"),
          ("Iran_June_2026_wb_anom.tif","(d)  June 2026")]

GPKG="../data/iran_map_layers.gpkg"
prov=gpd.read_file(GPKG,layer="iran_provinces"); natl=gpd.read_file(GPKG,layer="iran_boundary")

def read(path):
    with rasterio.open("../data/rasters/"+path) as ds:
        a=ds.read(1).astype("float64")
        if ds.nodata is not None and np.isfinite(ds.nodata): a[a==ds.nodata]=np.nan
        tr=ds.transform
    return a,(tr.c, tr.c+tr.a*a.shape[1], tr.f+tr.e*a.shape[0], tr.f)

def curved_label(ax, spine, txt, size=3.9, track=1.35):
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties
    lon,lat=np.asarray(spine,float).T
    t,tt=np.linspace(0,1,len(lon)),np.linspace(0,1,400)
    pts=ax.transData.transform(np.column_stack([np.interp(tt,t,lon),np.interp(tt,t,lat)]))
    arc=np.concatenate([[0.0],np.cumsum(np.hypot(*np.diff(pts,axis=0).T))])
    px=ax.figure.dpi/72.0; fp=FontProperties(size=size,style="italic")
    adv=[TextPath((0,0),c if c!=" " else "n",prop=fp).get_extents().width+track for c in txt]
    pos=(arc[-1]-sum(adv)*px)/2.0; halo=[pe.withStroke(linewidth=1.6,foreground="#4a4a4a")]
    for c,a in zip(txt,adv):
        d=pos+a*px/2.0
        if 0<=d<=arc[-1] and c!=" ":
            x,y=np.interp(d,arc,pts[:,0]),np.interp(d,arc,pts[:,1])
            dx=np.interp(d+6,arc,pts[:,0])-np.interp(d-6,arc,pts[:,0])
            dy=np.interp(d+6,arc,pts[:,1])-np.interp(d-6,arc,pts[:,1])
            lx,ly=ax.transData.inverted().transform((x,y))
            ax.text(lx,ly,c,fontsize=size,style="italic",color="white",ha="center",va="center",
                    zorder=7,rotation=np.degrees(np.arctan2(dy,dx)),rotation_mode="anchor",path_effects=halo)
        pos+=a*px

def base(ax):
    prov.boundary.plot(ax=ax,linewidth=0.28,edgecolor="#9a9a9a",zorder=4)
    natl.boundary.plot(ax=ax,linewidth=0.75,edgecolor="#3a3a3a",zorder=5)
    halo=[pe.withStroke(linewidth=1.6,foreground="white")]
    for lon,lat,txt in LABELS:
        ax.plot(lon,lat,"o",ms=2.6,color="#222222",zorder=6)
        ax.annotate(txt,(lon,lat),xytext=(4,3),textcoords="offset points",fontsize=6.6,color="#222222",zorder=7,path_effects=halo)
    for spine,txt in RANGES: curved_label(ax,spine,txt)
    ax.set_xlim(43.5,63.6); ax.set_ylim(24.6,40.2); ax.set_aspect(ASPECT)
    ax.set_anchor("N"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)

def scalebar(ax):
    for i in range(SB_NSEG):
        ax.add_patch(Rectangle((SB_X0+i*SB_SEG,SB_Y0),SB_SEG,SB_H,
            facecolor="#222222" if i%2==0 else "white",edgecolor="#222222",linewidth=0.5,zorder=8))
    for i in (0, SB_NSEG):
        lab="%d km"%(i*SB_STEP_KM) if i==SB_NSEG else "%d"%(i*SB_STEP_KM)
        ax.text(SB_X0+i*SB_SEG,SB_Y0+SB_H+0.10,lab,fontsize=11,ha="center",va="bottom",color="#222222",zorder=9)

def north_arrow(ax):
    H=2.4; w=(H*ASPECT)/1.7/2.0; xc=SB_X0+1.49*SB_SEG
    y_bot=SB_Y0+SB_H+1.35; y_top,y_notch=y_bot+H,y_bot+0.30*H
    whole=[[xc,y_top],[xc-w,y_bot],[xc,y_notch],[xc+w,y_bot]]
    ax.add_patch(plt.Polygon(whole,closed=True,facecolor="white",edgecolor="none",zorder=9))
    ax.add_patch(plt.Polygon(whole,closed=True,facecolor="none",edgecolor="#222222",linewidth=0.9,joinstyle="miter",zorder=10))

fig,axes=plt.subplots(1,4,figsize=(11.2,4.3))
fig.subplots_adjust(left=0.006,right=0.994,top=0.93,bottom=0.175,wspace=0.02)
for ax,(path,ttl) in zip(axes,PANELS):
    arr,ext=read(path)
    ax.imshow(arr,extent=ext,origin="upper",cmap=WB,norm=Normalize(-VMAX,VMAX),interpolation="none",zorder=2)
    base(ax); scalebar(ax); north_arrow(ax)
    ax.set_title(ttl,fontsize=10,loc="left",pad=5)

sm=ScalarMappable(cmap=WB,norm=Normalize(-VMAX,VMAX)); sm.set_array([])
cax=fig.add_axes([0.36,0.115,0.28,0.030])
cb=fig.colorbar(sm,cax=cax,orientation="horizontal",ticks=[-VMAX,-VMAX/2,0,VMAX/2,VMAX])
cb.ax.set_xticklabels(["-250","-125","0","+125","+250"])
cb.ax.tick_params(labelsize=10.5,length=3,pad=3)
cb.outline.set_linewidth(0.5); cb.outline.set_edgecolor("#666666")
cb.ax.xaxis.set_label_position("top")
cb.set_label("P - PET anomaly (mm)",fontsize=11.5,labelpad=5)

for e,kw in (("png",dict(dpi=PNG_DPI)),("pdf",{})):
    fig_style.apply(fig)
    fig.savefig("../figures/fig04_seasonal_water_balance_maps.%s"%e,facecolor="white",**kw)
    print("saved",e)
plt.close(fig)

#!/usr/bin/env python3
import geopandas as gpd, pandas as pd
from rasterstats import zonal_stats
hb=gpd.read_file("hybas_eu_lev05_v1c.shp"); hb["HID"]=hb.HYBAS_ID.astype('int64')
sn=pd.read_csv("snow_results_table.csv"); res=pd.read_csv("reservoirs.csv")
snids=set(sn.hybas_id.astype('int64')); resids=set(res.res_l5.astype('int64'))
need=hb[hb.HID.isin(snids|resids)].copy()
zs=zonal_stats(need.geometry,"cwb.tif",stats=["mean"],all_touched=True,nodata=-9999)
need["cwb"]=[z["mean"] for z in zs]
need[["HID","cwb"]].to_csv("basin_cwb.csv",index=False)
print("wrote basin_cwb.csv",len(need),"basins")

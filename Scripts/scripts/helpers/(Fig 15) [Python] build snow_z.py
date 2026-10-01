import glob, pandas as pd, numpy as np

BY = pd.read_csv('snow_basin_year_metrics.csv')
mp = BY[BY.year<=2025].groupby('hybas_id').peak.mean()
FRAC = pd.read_csv('basin_iran_fraction.csv').set_index('hybas_id').frac_in_iran
iranian = set(FRAC.index[FRAC>=0.50]) | {2050643590,2050724600,2050085640}
snowy = sorted(b for b in mp.index[mp>=0.20] if b in iranian)
print('reporting basins: %d' % len(snowy))

m = pd.concat([pd.read_csv(f) for f in sorted(glob.glob('Iran_SNOWMONTH_MERGED_basin_*.csv'))],
              ignore_index=True)
m = m[m.hybas_id.isin(snowy) & (m.valid_km2days>0)]

pool = (m.groupby(['year','month'])[['snow_km2days','valid_km2days']].sum()
          .assign(frac=lambda d: d.snow_km2days/d.valid_km2days).reset_index())

TARGET = [(2025,9),(2025,10),(2025,11),(2025,12),(2026,1),(2026,2),(2026,3),(2026,4),(2026,5),(2026,6)]
rows=[]
for (Y,mo) in TARGET:
    tval = pool[(pool.year==Y)&(pool.month==mo)].frac
    if not len(tval): rows.append((f"{Y}-{mo:02d}", np.nan, np.nan, np.nan, np.nan, 0)); continue
    tval = tval.iloc[0]

    hi = 2024 if Y==2025 else 2025
    ref = pool[(pool.month==mo)&(pool.year>=2003)&(pool.year<=hi)].frac
    mu, sd = ref.mean(), ref.std(ddof=1)
    z = (tval-mu)/sd if sd>0 else np.nan
    rows.append((f"{Y}-{mo:02d}", round(z,3), round(tval,5), round(mu,5), round(sd,5), len(ref)))
out = pd.DataFrame(rows, columns=['ym','z','snow_frac','hist_mean','hist_sd','n_ref'])
out[['ym','z']].to_csv('snow_z.csv', index=False)
out.to_csv('snow_z_audit.csv', index=False)
print(out.to_string(index=False))

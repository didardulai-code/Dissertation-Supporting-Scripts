import glob, pandas as pd
m = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("Iran_SMAP_monthly_national_*.csv"))])
m = m[m.n_images > 0].copy(); m["root_mm"] = m.sm_rootzone * 1000
TARGET = [(2025,9),(2025,10),(2025,11),(2025,12),(2026,1),(2026,2),(2026,3),(2026,4),(2026,5),(2026,6)]
rows = []
for Y, mo in TARGET:
    tv = m[(m.year==Y)&(m.month==mo)].root_mm
    if not len(tv): continue
    tv = tv.iloc[0]
    hist = m[(m.month==mo)&(m.year>=2015)&(m.year!=Y)].root_mm
    rows.append((Y, mo, round((tv-hist.mean())/hist.std(ddof=1), 3)))
pd.DataFrame(rows, columns=["year","month","z"]).to_csv("sm_z.csv", index=False)
print("wrote sm_z.csv")

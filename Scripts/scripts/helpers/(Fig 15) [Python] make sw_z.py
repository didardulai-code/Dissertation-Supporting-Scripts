#!/usr/bin/env python3

import pandas as pd
d = pd.read_csv("swa/monthly_anomalies.csv")
w = d[d.usable].copy(); w["t"] = (w.year - 2025) * 12 + w.month
w = w[(w.t >= 9) & (w.t <= 18) & (w.waterbody_type == "reservoir")]
w.groupby("t").anom_z.mean().rename("anom_z").reset_index().to_csv("sw_z.csv", index=False)
print("wrote sw_z.csv, reservoirs only")

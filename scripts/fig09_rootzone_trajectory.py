#!/usr/bin/env python3

import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fig_style

PNG_DPI = 600
BLACK, MED, BAND, DARK, GREY = "#000000", "#8a8a8a", "#dcdcdc", "#333333", "#9a9a9a"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9, "axes.linewidth": 0.8,
    "axes.labelsize": 9.5, "axes.titlesize": 9.8,
    "xtick.labelsize": 8.8, "ytick.labelsize": 8.8, "legend.fontsize": 7.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#9a9a9a", "axes.axisbelow": True,
    "xtick.color": "#4a4a4a", "ytick.color": "#4a4a4a",
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "axes.unicode_minus": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

m = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("../data/Iran_SMAP_monthly_*.csv"))])
m = m[m.n_images > 0].copy()
m["root_mm"] = m.sm_rootzone * 1000

ORDER = [9, 10, 11, 12, 1, 2, 3, 4, 5, 6]
NAMES = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
lab = [NAMES[k - 1] for k in ORDER]
x = np.arange(len(ORDER))

def water_year(y):

    out = []
    for k in ORDER:
        yy = y if k >= 9 else y + 1
        r = m[(m.year == yy) & (m.month == k)]
        if r.empty:
            return None
        out.append(float(r.root_mm.iloc[0]))
    return np.array(out)

hist = {y: water_year(y) for y in range(2015, 2025)}
hist = {y: v for y, v in hist.items() if v is not None}
cur = water_year(2025)
H = np.vstack(list(hist.values()))

print("historical water years: %d" % len(hist))
print("Sep start   2025/26 %.1f   mean %.1f   anomaly %+.1f"
      % (cur[0], H[:, 0].mean(), cur[0] - H[:, 0].mean()))
print("Apr peak    2025/26 %.1f   mean of peaks %.1f"
      % (cur.max(), H.max(axis=1).mean()))
print("Jun end     2025/26 %.1f   mean %.1f   anomaly %+.1f"
      % (cur[-1], H[:, -1].mean(), cur[-1] - H[:, -1].mean()))

fig, (ax, axd) = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=False,
                              gridspec_kw=dict(height_ratios=[7.0, 3.0],
                                               hspace=0.42))

ax.fill_between(x, H.min(axis=0), H.max(axis=0), color=BAND, zorder=0)
for v in H:
    ax.plot(x, v, color=GREY, linewidth=0.5, zorder=1, alpha=0.55)
ax.plot(x, H.mean(axis=0), color=MED, linewidth=2.0, zorder=3)
ax.plot(x, cur, color=BLACK, linewidth=2.4, zorder=4)

ax.set_ylim(H.min() - 30, H.max() + 6)
ax.set_ylabel("SMAP L4 root-zone soil moisture,\n0-100 cm (mm equivalent)", labelpad=12)
ax.set_title("(a)  Monthly SMAP L4 root-zone soil moisture", loc="left")

dep = cur - H.mean(axis=0)
axd.axhline(0, color=BLACK, linewidth=1.0, zorder=4)
axd.fill_between(x, 0, dep, color=BAND, interpolate=True, zorder=1)
axd.plot(x, dep, color=BLACK, linewidth=2.0, zorder=3)
axd.set_ylabel("Root-zone soil-moisture anomaly\n(mm)", labelpad=12)
axd.set_title("(b)  Monthly SMAP L4 root-zone anomaly", loc="left")
axd.set_ylim(dep.min() - 11, dep.max() + 8)

for a in (ax, axd):
    a.set_xticks(x); a.set_xticklabels(lab)
    a.set_xlabel("Month of the water year", labelpad=12)
    a.set_xlim(-0.35, len(x) - 0.65)
    a.grid(axis="y", color="#eeeeee", linewidth=0.7, zorder=0)
    a.set_axisbelow(True)
    a.tick_params(length=4, pad=7)

handles = [
    plt.Line2D([], [], color=BLACK, linewidth=2.4, label="2025/26"),
    plt.Line2D([], [], color=MED, linewidth=2.0, label="2015/16 to 2024/25 mean"),
    plt.Line2D([], [], color=GREY, linewidth=0.9, label="Each historical year"),
    plt.Rectangle((0, 0), 1, 1, color=BAND, label="2015/16 to 2024/25 min-max"),
]
ax.legend(handles=handles, frameon=False, loc="upper left", borderpad=0.0,
          labelspacing=0.55)

fig.subplots_adjust(left=0.115, right=0.985, top=0.94, bottom=0.085)
for ext, kw in (("png", dict(dpi=PNG_DPI)), ("pdf", {})):
    out = "../figures/fig09_rootzone_trajectory.%s" % ext
    fig_style.apply(fig)
    fig.savefig(out, facecolor="white", **kw); print("saved", out)
plt.close(fig)

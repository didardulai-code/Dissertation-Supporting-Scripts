import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*invalid value encountered.*")
warnings.filterwarnings("ignore", message=".*Mean of empty slice.*")
warnings.filterwarnings("ignore", message=".*All-NaN slice.*")

def to_monthly_calendar(ts):

    monthly = ts.copy()
    monthly.index = monthly.index.to_period("M").to_timestamp()
    monthly = monthly.groupby(level=0).mean()

    full_months = pd.date_range(monthly.index.min(),
                                monthly.index.max(), freq="MS")
    return monthly.reindex(full_months)

CONFLICT_START = pd.Timestamp("2026-02-28")

SHOW_CONFLICT_INSET = True
INSET_MONTHS = 30

ROLLING_MIN_PERIODS = 6

DATASET_SHORT_NAME = "TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4"
FALLBACK_SHORT_NAME = "TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.1_V3"

ALLOW_VERSION_FALLBACK = True

FORCE_DOWNLOAD = False

STALE_AFTER_MONTHS = 6

REQUIRE_COUNTRY_BOUNDARY = True

BOUNDARY_LOCAL_PATH = None

APPLY_SCALE_FACTORS = False

TREND_UNIT = "mm"

_TREND_SCALE = {"mm": 10.0, "cm": 1.0}[TREND_UNIT]

OUTPUT_DIR = Path("./grace_iran_output")
NC_DIR = OUTPUT_DIR / "nc_files"

IRAN_BBOX = {
    "min_lon": 44.0, "max_lon": 63.7,
    "min_lat": 25.0, "max_lat": 39.8,
}

BOUNDARY_SOURCES = [
    "https://naturalearth.s3.amazonaws.com/"
    "50m_cultural/ne_50m_admin_0_countries.zip",
    "https://naciscdn.org/naturalearth/50m/cultural/"
    "ne_50m_admin_0_countries.zip",
    "https://naturalearth.s3.amazonaws.com/"
    "10m_cultural/ne_10m_admin_0_countries.zip",
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_50m_admin_0_countries.geojson",
]

SUBSAMPLE = 5

C_POSITIVE = "#4575b4"
C_NEGATIVE = "#d73027"
C_ROLLING = "#1a1a2e"
C_TREND = "#8B0000"
C_CONFLICT = "#7B1FA2"
C_GAP = "#808080"

def authenticate():

    try:
        import earthaccess
    except ImportError as exc:
        raise ImportError(
            "earthaccess is not installed.\n"
            "Run: pip3 install earthaccess"
        ) from exc

    print("Authenticating with NASA Earthdata...")

    try:
        earthaccess.login(strategy="netrc")
        print("  Credentials loaded from ~/.netrc")
        return earthaccess
    except Exception:
        pass

    print()
    print("  No saved credentials found — this is normal on a first run.")
    print()
    print("  " + "-" * 60)
    print("  IMPORTANT: Earthdata wants your USERNAME, not your email.")
    print("  If you registered with an email address, the username is")
    print("  the separate name you chose at sign-up.")
    print()
    print("  Check or reset at: https://urs.earthdata.nasa.gov/profile")
    print("  The password will NOT appear as you type. That is normal.")
    print("  " + "-" * 60)
    print()

    attempts = 3
    for attempt in range(1, attempts + 1):
        try:
            earthaccess.login(strategy="interactive", persist=True)
            print("\n  Login succeeded. Credentials saved to ~/.netrc "
                  "for future runs.")
            return earthaccess
        except Exception as exc:
            message = str(exc)
            if "invalid_credentials" in message or "Invalid user" in message:
                reason = "NASA rejected that username/password combination."
            else:
                reason = message.strip().splitlines()[0]

            print(f"\n  Attempt {attempt} of {attempts} failed: {reason}")

            if attempt < attempts:
                print("  Things to check:")
                print("    1. Are you typing your USERNAME, not your email?")
                print("    2. Have you changed your password recently?")
                print("    3. Is Caps Lock on?")
                print("    4. Is your account activated? Log in once at")
                print("       https://urs.earthdata.nasa.gov/ in a browser")
                print("       to confirm the credentials work there first.")
                print("\n  Try again.\n")

    raise SystemExit(
        "\nCould not authenticate with NASA Earthdata after "
        f"{attempts} attempts.\n"
        "Confirm your credentials by logging in at "
        "https://urs.earthdata.nasa.gov/ in a browser, then re-run "
        "this script.\n"
        "Note that the login needs your Earthdata USERNAME, which is "
        "not the same as your email address.\n"
    )

    return earthaccess

def download_grace(earthaccess):

    NC_DIR.mkdir(parents=True, exist_ok=True)

    existing = sorted(NC_DIR.glob("**/*.nc"))

    if existing and FORCE_DOWNLOAD:
        print(f"  FORCE_DOWNLOAD is on — removing {len(existing)} "
              "cached file(s) so the newest months are fetched.")
        for path in existing:
            path.unlink()
        existing = []

    if existing:
        print(f"  Found {len(existing)} existing NetCDF file(s). "
              "Skipping download.")
        print("  NOTE: PO.DAAC adds new months over time. If this file "
              "is old, the 'latest observation' will be stale.")
        print("  Set FORCE_DOWNLOAD = True at the top of this script "
              "for your final run.")
        return existing

    print(f"  Searching for: {DATASET_SHORT_NAME}")
    results = earthaccess.search_data(short_name=DATASET_SHORT_NAME)

    if not results and ALLOW_VERSION_FALLBACK:
        print(f"  Not found. Trying fallback: {FALLBACK_SHORT_NAME}")
        print("  WARNING: this is a DIFFERENT product version. The "
              "version actually loaded is read from the file metadata "
              "and used in the figure title, so the caption will stay "
              "accurate — but check it before submitting.")
        results = earthaccess.search_data(short_name=FALLBACK_SHORT_NAME)

    if not results:
        raise RuntimeError(
            f"The requested collection '{DATASET_SHORT_NAME}' was not "
            "found on PO.DAAC.\n"
            "Check the current short name at https://podaac.jpl.nasa.gov/ "
            "before continuing."
        )

    print(f"  Found {len(results)} downloadable item(s).")
    print("  Downloading — this may take several minutes.")
    downloaded = earthaccess.download(results, str(NC_DIR))

    files = [Path(p) for p in downloaded]
    print(f"  Downloaded {len(files)} file(s).")
    return files

def describe_product(ds):

    attrs = {k.lower(): str(v) for k, v in ds.attrs.items()}

    for key in ("title", "id", "short_name", "product_name", "source"):
        if key in attrs and attrs[key].strip():
            label = attrs[key].strip()
            break
    else:
        label = "JPL GRACE/GRACE-FO mascon (version not recorded in file)"

    version = attrs.get("product_version") or attrs.get("version")
    if version and version not in label:
        label = f"{label}, version {version}"

    if len(label) > 110:
        label = label[:107] + "..."

    print(f"  Product as recorded in the file: {label}")
    return label

def _fetch_to_file(url, dest):

    try:
        import requests
    except ImportError:
        import urllib.request
        with urllib.request.urlopen(url, timeout=90) as response:
            dest.write_bytes(response.read())
        return dest

    with requests.get(url, timeout=90, stream=True,
                      headers={"User-Agent": "Mozilla/5.0"}) as response:
        response.raise_for_status()
        with open(dest, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 16):
                handle.write(chunk)
    return dest

def _extract_iran(countries, name_fields):

    for field in name_fields:
        if field not in countries.columns:
            continue
        matches = countries[
            countries[field].astype(str)
            .str.contains("Iran", case=False, na=False)
        ]
        if not matches.empty:
            iran = matches.copy().to_crs("EPSG:4326").dissolve()
            geom = iran.geometry.iloc[0]
            print(f"  Iran boundary loaded. Bounds: "
                  f"{[round(v, 2) for v in geom.bounds]}")
            return geom
    return None

def load_iran_boundary():

    try:
        import geopandas as gpd
    except ImportError:
        if REQUIRE_COUNTRY_BOUNDARY:
            raise ImportError(
                "geopandas is not installed, so Iran's national "
                "boundary cannot be loaded.\n"
                "Run: pip3 install geopandas\n"
                "The analysis has stopped rather than fall back to a "
                "bounding box that includes neighbouring countries "
                "and adjacent seas."
            )
        print("  geopandas is not installed — falling back to a "
              "bounding box.")
        print("  Install with: pip3 install geopandas")
        return None

    name_fields = ["ADMIN", "SOVEREIGNT", "NAME", "NAME_LONG",
                   "FORMAL_EN", "name", "admin"]

    if BOUNDARY_LOCAL_PATH:
        local = Path(BOUNDARY_LOCAL_PATH).expanduser()
        if local.exists():
            print(f"  Using local boundary file: {local}")
            countries = gpd.read_file(local)
            geom = _extract_iran(countries, name_fields)
            if geom is not None:
                return geom
            print("    Iran not found in that file.")
        else:
            print(f"  BOUNDARY_LOCAL_PATH set but not found: {local}")

    cache_dir = OUTPUT_DIR / "boundary_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for url in BOUNDARY_SOURCES:
        name = url.split("/")[-1]
        cached = cache_dir / name

        try:
            if cached.exists() and cached.stat().st_size > 1000:
                print(f"  Using cached {name}")
            else:
                print(f"  Fetching {name} ...")
                _fetch_to_file(url, cached)
            countries = gpd.read_file(cached)
        except Exception as exc:

            detail = str(exc).strip().replace("\n", " ")[:180]
            print(f"    Failed: {type(exc).__name__}: {detail}")
            if cached.exists():
                cached.unlink(missing_ok=True)
            continue

        iran = None
        for field in name_fields:
            if field not in countries.columns:
                continue
            matches = countries[
                countries[field].astype(str)
                .str.contains("Iran", case=False, na=False)
            ]
            if not matches.empty:
                iran = matches.copy()
                break

        if iran is None or iran.empty:
            print("    Iran not found in this dataset.")
            continue

        iran = iran.to_crs("EPSG:4326").dissolve()
        geom = iran.geometry.iloc[0]
        print(f"  Iran boundary loaded. Bounds: "
              f"{[round(v, 2) for v in geom.bounds]}")
        return geom

    if REQUIRE_COUNTRY_BOUNDARY:
        raise RuntimeError(
            "Iran's national boundary could not be loaded from any "
            "source.\n"
            "The analysis has stopped rather than use a bounding box "
            "that includes neighbouring countries and adjacent seas.\n"
            "Check your internet connection, or set "
            "REQUIRE_COUNTRY_BOUNDARY = False to continue with the "
            "less accurate bounding-box average for troubleshooting."
        )

    print("  WARNING: all boundary sources failed. "
          "Falling back to a bounding box.")
    return None

def open_grace_dataset():

    import xarray as xr

    nc_files = sorted(NC_DIR.glob("*.nc"))
    if not nc_files:
        nc_files = sorted(NC_DIR.glob("**/*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No NetCDF files found in {NC_DIR}")

    print(f"  Opening {len(nc_files)} GRACE NetCDF file(s)...")

    if len(nc_files) == 1:
        ds = xr.open_dataset(str(nc_files[0]))
    else:
        try:
            ds = xr.open_mfdataset([str(p) for p in nc_files],
                                   combine="by_coords")
        except ImportError:
            print("  dask not installed — concatenating files directly.")
            parts = [xr.open_dataset(str(p)) for p in nc_files]
            ds = xr.concat(parts, dim="time").sortby("time")

    rename = {}
    if "longitude" in ds.coords and "lon" not in ds.coords:
        rename["longitude"] = "lon"
    if "latitude" in ds.coords and "lat" not in ds.coords:
        rename["latitude"] = "lat"
    if rename:
        ds = ds.rename(rename)

    missing = {"lon", "lat", "time"}.difference(ds.coords)
    if missing:
        raise KeyError(
            f"Missing required coordinates: {sorted(missing)}\n"
            f"Available coordinates: {list(ds.coords)}"
        )

    if float(ds.lon.max()) > 180:
        print("  Converting longitudes from 0–360 to -180–180.")
        ds = ds.assign_coords(
            lon=(((ds.lon + 180) % 360) - 180)
        ).sortby("lon")

    if float(ds.lat[0]) > float(ds.lat[-1]):
        print("  Re-sorting latitude to ascending order.")
        ds = ds.sortby("lat")

    print(f"  Variables: {list(ds.data_vars)}")
    print(f"  Time steps in file: {ds.sizes.get('time', 'unknown')}")
    return ds

def identify_tws_variable(ds):

    candidates = ["lwe_thickness", "lwe_thickness_cri", "twsa", "tws",
                  "liquid_water_equivalent_thickness"]

    for candidate in candidates:
        if candidate in ds.data_vars:
            print(f"  Using TWS variable: '{candidate}'")
            return candidate

    raise KeyError(
        "No recognised terrestrial water-storage variable was found.\n"
        f"Available variables: {list(ds.data_vars)}\n"
        "Add the correct name to the 'candidates' list."
    )

def build_area_fraction_mask(lons, lats, geometry):

    from shapely import contains_xy
    from shapely.prepared import prep

    dlon = float(np.median(np.diff(lons)))
    dlat = float(np.median(np.diff(lats)))

    steps = (np.arange(SUBSAMPLE) + 0.5) / SUBSAMPLE - 0.5
    off_lon = steps * dlon
    off_lat = steps * dlat

    lon_pts = (lons[:, None] + off_lon[None, :]).ravel()
    lat_pts = (lats[:, None] + off_lat[None, :]).ravel()
    grid_lon, grid_lat = np.meshgrid(lon_pts, lat_pts)

    try:
        inside = contains_xy(geometry, grid_lon, grid_lat)
    except Exception:

        from shapely.geometry import Point
        prepared = prep(geometry)
        flat_lon = grid_lon.ravel()
        flat_lat = grid_lat.ravel()
        inside = np.fromiter(
            (prepared.contains(Point(x, y))
             for x, y in zip(flat_lon, flat_lat)),
            dtype=bool, count=flat_lon.size,
        ).reshape(grid_lon.shape)

    n_lat, n_lon = len(lats), len(lons)
    fraction = (
        inside.reshape(n_lat, SUBSAMPLE, n_lon, SUBSAMPLE)
        .mean(axis=(1, 3))
    )
    return fraction

def extract_iran_timeseries(ds, geometry):

    var_name = identify_tws_variable(ds)
    lwe = ds[var_name]

    if geometry is not None:
        min_lon, min_lat, max_lon, max_lat = geometry.bounds
    else:
        min_lon, max_lon = IRAN_BBOX["min_lon"], IRAN_BBOX["max_lon"]
        min_lat, max_lat = IRAN_BBOX["min_lat"], IRAN_BBOX["max_lat"]

    buf = 1.0
    subset = lwe.sel(
        lon=slice(min_lon - buf, max_lon + buf),
        lat=slice(min_lat - buf, max_lat + buf),
    )

    if subset.size == 0:
        raise ValueError(
            "The Iran spatial subset is empty. Check the coordinate "
            "ranges in the file against the expected extent."
        )

    lons = subset.lon.values.astype(float)
    lats = subset.lat.values.astype(float)
    print(f"  Subset grid: {len(lats)} x {len(lons)} cells "
          f"(~{abs(np.median(np.diff(lons))):.2f} degrees)")

    if "land_mask" in ds.data_vars:
        try:
            land = ds["land_mask"].sel(
                lon=slice(min_lon - buf, max_lon + buf),
                lat=slice(min_lat - buf, max_lat + buf),
            )
            lwe_sub = subset.where(land > 0.5)
            print("  Product land mask applied.")
        except Exception:
            lwe_sub = subset
            print("  Note: land mask present but could not be applied.")
    else:
        lwe_sub = subset

    if "scale_factor" not in ds.data_vars:
        scale_note = ("no gain factors in file; CRI-filtered anomalies "
                      "used as distributed")
        print(f"  {scale_note}.")
    elif not APPLY_SCALE_FACTORS:
        scale_note = ("gain factors present but NOT applied; "
                      "CRI-filtered anomalies used as distributed")
        print("  Gain (scale) factors are present but have not been "
              "applied.")
        print("  The CRI-filtered anomalies are used directly. Set "
              "APPLY_SCALE_FACTORS = True to run a sensitivity check.")
    else:
        try:
            sf = ds["scale_factor"].sel(
                lon=slice(min_lon - buf, max_lon + buf),
                lat=slice(min_lat - buf, max_lat + buf),
            )
            lwe_sub = lwe_sub * sf.where(np.abs(sf) < 1e4)
            scale_note = "model-derived gain factors APPLIED"
            print("  Gain (scale) factors applied — this is the "
                  "sensitivity variant, not the default.")
        except Exception:
            scale_note = ("gain factors present but could not be "
                          "applied")
            print(f"  Note: {scale_note}.")

    import xarray as xr

    if geometry is not None:
        fraction = build_area_fraction_mask(lons, lats, geometry)
        covered = float((fraction > 0).sum())
        print(f"  Cells intersecting Iran: {int(covered)} "
              f"(fully inside: {int((fraction > 0.99).sum())})")
        if covered == 0:
            raise ValueError(
                "The country mask selected no cells. Check that the "
                "boundary and the grid use the same longitude "
                "convention."
            )
        country_weight = xr.DataArray(
            fraction, coords={"lat": lats, "lon": lons},
            dims=["lat", "lon"],
        )
        mask_note = "country boundary, area-fraction weighted"
    else:
        country_weight = xr.DataArray(
            np.ones((len(lats), len(lons))),
            coords={"lat": lats, "lon": lons}, dims=["lat", "lon"],
        )

        country_weight = country_weight.where(
            (country_weight.lon >= IRAN_BBOX["min_lon"])
            & (country_weight.lon <= IRAN_BBOX["max_lon"])
            & (country_weight.lat >= IRAN_BBOX["min_lat"])
            & (country_weight.lat <= IRAN_BBOX["max_lat"]), 0
        )
        mask_note = ("BOUNDING BOX ONLY — includes neighbouring "
                     "territory and adjacent seas")
        print("  WARNING: using a bounding box, not the country "
              "boundary. Note this limitation in your methods.")

    lat_weight = np.cos(np.deg2rad(country_weight.lat))
    weights = country_weight * lat_weight

    iran_mean = lwe_sub.weighted(weights.fillna(0)).mean(
        dim=["lat", "lon"], skipna=True
    )

    times = pd.to_datetime(iran_mean.time.values)
    values = np.asarray(iran_mean.values).squeeze()

    ts = pd.Series(values, index=times, name="tws_cm")
    ts.index.name = "date"

    ts = ts.replace([np.inf, -np.inf], np.nan)
    ts = ts.where(np.abs(ts) < 1e4)
    ts = ts.dropna().sort_index()

    if ts.empty:
        raise ValueError(
            "No valid GRACE observations remained after masking."
        )

    print("\n  Iran national TWS time series extracted")
    print(f"    Masking: {mask_note}")
    print(f"    Period: {ts.index[0]:%B %Y} to {ts.index[-1]:%B %Y}")
    print(f"    Valid monthly observations: {len(ts)}")
    print(f"    Minimum anomaly: {ts.min():.2f} cm")
    print(f"    Maximum anomaly: {ts.max():.2f} cm")
    print(f"    Mean anomaly:    {ts.mean():.2f} cm")

    conflict_obs = ts[ts.index >= CONFLICT_START]
    if len(conflict_obs):
        print(f"\n    Conflict-period observations available: "
              f"{len(conflict_obs)}")
        for date, value in conflict_obs.items():
            print(f"      {date:%B %Y}: {value:+.2f} cm")
        print("\n    CAUTION: these are MONTHLY solutions. A March 2026")
        print("    solution integrates conditions across the whole of")
        print("    March, so GRACE cannot resolve a change beginning on")
        print("    28 February. Terrestrial water storage also responds")
        print("    slowly. Treat these as broad monthly hydrological")
        print("    context, not a before-and-after conflict measure.")
    else:
        print("\n    No observations yet within the conflict period.")

    months_old = (pd.Timestamp.now() - ts.index[-1]).days / 30.44
    if months_old > STALE_AFTER_MONTHS:
        print(f"\n    WARNING: the newest observation is about "
              f"{months_old:.0f} months old.")
        print("    GRACE/GRACE-FO products are released with a "
              "processing delay that varies, but this file may still "
              "be out of date.")
        print("    Set FORCE_DOWNLOAD = True and re-run to fetch the "
              "newest months.")

    diag = {
        "fraction": fraction if geometry is not None else None,
        "lons": lons, "lats": lats, "mask_note": mask_note,
        "scale_note": scale_note,
    }
    return ts, diag

def fit_linear_trend(ts, label="Full record"):

    valid = ts.dropna()
    if len(valid) < 24:
        return None

    decimal_year = valid.index.year + (valid.index.dayofyear - 1) / 365.25
    x = np.asarray(decimal_year, dtype=float)
    y = valid.values.astype(float)

    slope, intercept, r_value, p_value, std_error = stats.linregress(x, y)
    trend = pd.Series(slope * x + intercept, index=valid.index,
                      name="linear_trend")
    total_change = slope * (x[-1] - x[0])

    u = TREND_UNIT
    k = _TREND_SCALE

    print(f"\n  {label} linear trend")
    print(f"    Slope: {slope * k:+.3f} {u}/year "
          f"(standard error {std_error * k:.3f})")
    print(f"    Estimated total change: {total_change * k:+.2f} {u}")
    print(f"    p-value: {p_value:.5f}    R-squared: {r_value ** 2:.3f}")
    if u != "cm":
        print(f"    [same trend in cm: {slope:+.3f} cm/year, "
              f"total change {total_change:+.2f} cm]")

    return {
        "slope_cm_per_year": slope,
        "slope_mm_per_year": slope * 10.0,
        "trend_series": trend,
        "p_value": p_value,
        "r_squared": r_value ** 2,
        "standard_error": std_error,
        "standard_error_mm": std_error * 10.0,
        "total_change_cm": total_change,
        "total_change_mm": total_change * 10.0,
        "n_observations": len(valid),
    }

def calculate_period_summaries(ts):

    full = ts.dropna()
    pre_conflict = full[full.index < CONFLICT_START]
    conflict = full[full.index >= CONFLICT_START]

    def safe(series, func, default=np.nan):
        return func(series) if not series.empty else default

    monthly = to_monthly_calendar(full)

    not_below = monthly[monthly >= 0]
    if not_below.empty:
        deficit_since = monthly.first_valid_index()
    elif not_below.index[-1] < monthly.index[-1]:
        after = monthly.loc[monthly.index > not_below.index[-1]]
        deficit_since = after.first_valid_index()
    else:
        deficit_since = pd.NaT

    if pd.notna(deficit_since):
        tail = monthly.loc[deficit_since:]
        deficit_gaps = int(tail.isna().sum())
    else:
        deficit_gaps = 0

    is_deficit = monthly.lt(0) & monthly.notna()
    run_lengths = is_deficit.groupby((~is_deficit).cumsum()).sum()
    longest_deficit_run = int(run_lengths.max()) if len(run_lengths) else 0

    summaries = {
        "full_record_start": full.index.min(),
        "full_record_end": full.index.max(),
        "full_record_n": len(full),
        "full_record_mean_cm": full.mean(),
        "full_record_min_cm": full.min(),
        "full_record_max_cm": full.max(),
        "pre_conflict_n": len(pre_conflict),
        "pre_conflict_mean_cm": safe(pre_conflict, lambda s: s.mean()),
        "latest_pre_conflict_date":
            safe(pre_conflict, lambda s: s.index[-1], pd.NaT),
        "latest_pre_conflict_anomaly_cm":
            safe(pre_conflict, lambda s: s.iloc[-1]),
        "conflict_period_n": len(conflict),
        "conflict_period_mean_cm": safe(conflict, lambda s: s.mean()),
        "conflict_period_min_cm": safe(conflict, lambda s: s.min()),
        "deficit_since": deficit_since,
        "deficit_since_gap_months": deficit_gaps,
        "longest_deficit_run_months": longest_deficit_run,
    }

    print("\n  Data-derived descriptors (no external drought "
          "definition assumed)")
    if pd.notna(deficit_since):
        gap_txt = ("" if deficit_gaps == 0 else
                   f" (note: {deficit_gaps} month(s) in this stretch "
                   "have no GRACE solution)")
        print(f"    All available observations have been below the "
              f"GRACE reference mean since "
              f"{deficit_since:%B %Y}{gap_txt}")
    print(f"    Longest run of consecutive calendar months below the "
          f"reference mean: {longest_deficit_run}")

    print("\n  Period summary")
    if not pre_conflict.empty:
        print(f"    Pre-conflict record, n = {len(pre_conflict)}")
        print(f"      Latest pre-conflict observation: "
              f"{pre_conflict.index[-1]:%B %Y}, "
              f"{pre_conflict.iloc[-1]:+.2f} cm")
    if not conflict.empty:
        print(f"    Conflict period, n = {len(conflict)}")
        print(f"      Mean anomaly: {conflict.mean():.2f} cm")

    return summaries

def plot_mask_check(diag, geometry, save_path):

    if diag["fraction"] is None:
        print("  Mask diagnostic skipped (bounding-box fallback).")
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    lons, lats = diag["lons"], diag["lats"]
    dlon = float(np.median(np.diff(lons)))
    dlat = float(np.median(np.diff(lats)))

    mesh = ax.pcolormesh(
        np.append(lons - dlon / 2, lons[-1] + dlon / 2),
        np.append(lats - dlat / 2, lats[-1] + dlat / 2),
        diag["fraction"], cmap="YlGnBu", vmin=0, vmax=1,
        shading="flat", edgecolors="white", linewidth=0.2,
    )
    fig.colorbar(mesh, ax=ax, label="Fraction of grid cell inside Iran")

    geoms = (geometry.geoms if hasattr(geometry, "geoms")
             else [geometry])
    for poly in geoms:
        x, y = poly.exterior.xy
        ax.plot(x, y, color="#B00020", linewidth=1.4)

    ax.set_title("Grid-cell weighting mask for Iran\n"
                 "GRACE mascon grid with national boundary overlaid",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Longitude (degrees east)")
    ax.set_ylabel("Latitude (degrees north)")
    ax.set_aspect("equal")

    fig.text(
        0.01, 0.005,
        "Diagnostic only. Colours show how much each cell contributes "
        "to the national average, NOT observed water storage. The "
        "0.5-degree grid is an interpolated representation of mascons "
        "roughly 3 degrees across, so this map does not imply "
        "sub-mascon spatial detail.",
        fontsize=7, style="italic", color="#555555", wrap=True,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(save_path, dpi=200, bbox_inches="tight",
                facecolor="white")
    print(f"  Mask diagnostic saved to: {save_path}")
    plt.close(fig)

def plot_tws(ts, trend_results, save_path, product_label=None):

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("white")

    ax.axvspan(pd.Timestamp("2017-07-01"), pd.Timestamp("2018-06-01"),
               color=C_GAP, alpha=0.20, zorder=1, linewidth=0)
    gap_patch = mpatches.Patch(color=C_GAP, alpha=0.20,
                               label="GRACE to GRACE-FO mission gap")

    conflict_patch = None
    has_conflict_data = CONFLICT_START <= ts.index[-1]
    if has_conflict_data:
        ax.axvspan(CONFLICT_START,
                   ts.index[-1] + pd.DateOffset(months=1),
                   color=C_CONFLICT, alpha=0.18, zorder=1, linewidth=0)
        conflict_patch = mpatches.Patch(
            color=C_CONFLICT, alpha=0.30,
            label="Conflict period, 28 February 2026 onwards")

    ax.axhline(0, color="#333333", linewidth=0.8, alpha=0.7, zorder=2)

    pre = ts[ts.index < CONFLICT_START]
    conflict = ts[ts.index >= CONFLICT_START]

    ax.bar(pre[pre >= 0].index, pre[pre >= 0].values, width=26,
           color=C_POSITIVE, alpha=0.60, zorder=3, linewidth=0)
    ax.bar(pre[pre < 0].index, pre[pre < 0].values, width=26,
           color=C_NEGATIVE, alpha=0.60, zorder=3, linewidth=0)

    pos_patch = mpatches.Patch(
        color=C_POSITIVE, alpha=0.60,
        label="Monthly anomaly above the GRACE reference mean")
    neg_patch = mpatches.Patch(
        color=C_NEGATIVE, alpha=0.60,
        label="Monthly anomaly below the GRACE reference mean")

    conflict_bar_patch = None
    if len(conflict):
        ax.bar(conflict.index, conflict.values, width=26,
               color=C_CONFLICT, alpha=0.95, zorder=4, linewidth=0)
        conflict_bar_patch = mpatches.Patch(
            color=C_CONFLICT, alpha=0.95,
            label="Monthly anomaly during the conflict period")

    monthly = to_monthly_calendar(ts)
    rolling = monthly.rolling(window=12, center=True,
                              min_periods=ROLLING_MIN_PERIODS).mean()

    roll_label = ("Centred 12-month rolling mean"
                  if ROLLING_MIN_PERIODS >= 12 else
                  f"Centred 12-month rolling mean "
                  f"(min. {ROLLING_MIN_PERIODS} observations)")
    line_roll, = ax.plot(rolling.index, rolling.values,
                         color=C_ROLLING, linewidth=2.2, zorder=5,
                         label=roll_label)

    trend_handle = None
    if trend_results is not None:

        slope = trend_results["slope_cm_per_year"] * _TREND_SCALE

        _fmt = "+.1f" if TREND_UNIT == "mm" else "+.2f"
        trend_handle, = ax.plot(
            trend_results["trend_series"].index,
            trend_results["trend_series"].values,
            color=C_TREND, linewidth=1.8, linestyle="--", zorder=6,
            label=f"Descriptive linear trend: "
                  f"{slope:{_fmt}} {TREND_UNIT}/year",
        )

    onset_handle = None
    if has_conflict_data:
        onset_handle = ax.axvline(
            CONFLICT_START, color=C_CONFLICT, linewidth=2,
            linestyle=":", zorder=7,
            label="Conflict onset, 28 February 2026")
    else:
        ax.annotate(
            "Conflict onset, 28 February 2026\n"
            "(after the latest GRACE observation)",
            xy=(ts.index[-1], ts.iloc[-1]),
            xytext=(ts.index[-1] - pd.DateOffset(months=40),
                    ts.min() + 0.20 * (ts.max() - ts.min())),
            fontsize=8.5, fontstyle="italic", color=C_CONFLICT,
            arrowprops={"arrowstyle": "->", "linewidth": 1,
                        "color": C_CONFLICT},
        )

    show_inset = SHOW_CONFLICT_INSET and has_conflict_data
    if not show_inset:
        ax.annotate(
            f"Latest observation:\n{ts.index[-1]:%B %Y}",
            xy=(ts.index[-1], ts.iloc[-1]),
            xytext=(ts.index[-1] - pd.DateOffset(months=34),
                    ts.max() - 0.12 * (ts.max() - ts.min())),
            fontsize=8, color="#444444",
            arrowprops={"arrowstyle": "->", "linewidth": 0.8,
                        "color": "#666666"},
        )

    subtitle = product_label or "JPL GRACE/GRACE-FO mascon, CRI-filtered"
    ax.set_title(
        "GRACE and GRACE-FO terrestrial water-storage anomalies across "
        f"Iran,\n{ts.index[0]:%B %Y} to {ts.index[-1]:%B %Y}\n"
        f"{subtitle}  |  area-weighted national mean",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Year", fontsize=11, labelpad=8)
    ax.set_ylabel("Terrestrial water-storage anomaly\n"
                  "(cm equivalent water height)", fontsize=11)

    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=45, labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.set_xlim(ts.index[0] - pd.DateOffset(months=3),
                ts.index[-1] + pd.DateOffset(months=6))

    handles = [pos_patch, neg_patch]
    if conflict_bar_patch is not None:
        handles.append(conflict_bar_patch)
    handles.append(line_roll)
    if trend_handle is not None:
        handles.append(trend_handle)
    if onset_handle is not None:
        handles.append(onset_handle)
    if conflict_patch is not None:
        handles.append(conflict_patch)
    handles.append(gap_patch)

    ax.legend(handles=handles, loc="lower left", fontsize=8,
              ncol=2, framealpha=0.95)

    fig.text(
        0.01, 0.012,
        "Anomalies are relative to the JPL 2004.0-2009.999 time-mean "
        "baseline. GRACE measures combined terrestrial water storage, "
        "including groundwater, soil moisture, surface water and snow; "
        "negative values do not independently demonstrate groundwater "
        "depletion. Native mascon resolution is approximately 3 "
        "degrees, so 0.5-degree cells are not independent "
        "observations and the grid does not resolve sub-mascon detail. "
        "Product measurement uncertainty is not propagated into this "
        "national mean or its trend.",
        fontsize=7.5, style="italic", color="#555555",
    )

    if SHOW_CONFLICT_INSET and has_conflict_data:
        cutoff = ts.index[-1] - pd.DateOffset(months=INSET_MONTHS)
        recent = ts[ts.index >= cutoff]

        if len(recent) >= 6:

            axins = ax.inset_axes([0.58, 0.60, 0.39, 0.32])
            axins.set_zorder(20)
            axins.patch.set_facecolor("white")
            axins.patch.set_alpha(1.0)

            axins.axvspan(CONFLICT_START,
                          recent.index[-1] + pd.DateOffset(months=1),
                          color=C_CONFLICT, alpha=0.18, linewidth=0)

            r_pre = recent[recent.index < CONFLICT_START]
            r_con = recent[recent.index >= CONFLICT_START]

            axins.bar(r_pre[r_pre >= 0].index, r_pre[r_pre >= 0].values,
                      width=22, color=C_POSITIVE, alpha=0.60, linewidth=0)
            axins.bar(r_pre[r_pre < 0].index, r_pre[r_pre < 0].values,
                      width=22, color=C_NEGATIVE, alpha=0.60, linewidth=0)
            axins.bar(r_con.index, r_con.values, width=22,
                      color=C_CONFLICT, alpha=0.95, linewidth=0)

            axins.axvline(CONFLICT_START, color=C_CONFLICT,
                          linewidth=1.4, linestyle=":")
            axins.axhline(0, color="#333333", linewidth=0.6, alpha=0.7)

            axins.set_title(
                f"Final {INSET_MONTHS} months  |  latest: "
                f"{ts.index[-1]:%b %Y}",
                fontsize=7.5, pad=4)
            axins.set_ylabel("cm", fontsize=6.5, labelpad=2)
            axins.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            axins.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
            axins.tick_params(axis="both", labelsize=6.5)
            axins.grid(axis="y", linestyle=":", alpha=0.3)
            for spine in axins.spines.values():
                spine.set_edgecolor("#999999")
                spine.set_linewidth(0.8)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(save_path, dpi=300, bbox_inches="tight",
                facecolor="white")
    print(f"  Figure saved to: {save_path}")
    plt.close(fig)

def save_outputs(ts, trend_results, summaries, diag):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    monthly_path = OUTPUT_DIR / "iran_grace_tws_monthly.csv"
    out = ts.to_frame()
    out["period"] = np.where(out.index >= CONFLICT_START,
                             "conflict", "pre_conflict")
    out.to_csv(monthly_path, header=True)

    rows = [
        ("Product as recorded in file",
         diag.get("product_label", "not recorded")),
        ("Gain (scale) factor treatment",
         diag.get("scale_note", "not recorded")),
        ("Masking approach", diag["mask_note"]),
        ("Full-record start", summaries["full_record_start"]),
        ("Full-record end", summaries["full_record_end"]),
        ("Full-record observations", summaries["full_record_n"]),
        ("Full-record mean anomaly (cm)", summaries["full_record_mean_cm"]),
        ("Full-record minimum anomaly (cm)", summaries["full_record_min_cm"]),
        ("Full-record maximum anomaly (cm)", summaries["full_record_max_cm"]),
    ]

    rows += [
        ("Anomaly baseline", "JPL 2004.0-2009.999 time mean"),
        ("Measurement uncertainty",
         "product uncertainty NOT propagated into the national "
         "mean or the trend; 0.5-degree cells represent ~3-degree "
         "mascons and are not independent observations"),
    ]

    if trend_results is not None:
        rows += [
            ("Descriptive linear trend (mm/year)",
             trend_results["slope_mm_per_year"]),
            ("Descriptive linear trend (cm/year)",
             trend_results["slope_cm_per_year"]),
            ("Trend standard error (mm/year)",
             trend_results["standard_error_mm"]),
            ("Trend standard error (cm/year)",
             trend_results["standard_error"]),
            ("Trend total change (mm)", trend_results["total_change_mm"]),
            ("Trend total change (cm)", trend_results["total_change_cm"]),
            ("Trend unit reported on figure", TREND_UNIT),
            ("Trend R-squared", trend_results["r_squared"]),
            ("Trend p-value (OLS, autocorrelation NOT corrected)",
             trend_results["p_value"]),
            ("Trend caveat",
             "GRACE residuals are seasonally autocorrelated; the OLS "
             "p-value overstates the evidence and is not reported on "
             "the figure"),
        ]
    rows += [
        ("All available observations below GRACE reference mean since",
         summaries["deficit_since"]),
        ("Months with no solution within that stretch",
         summaries["deficit_since_gap_months"]),
        ("Longest run of consecutive calendar months below reference "
         "mean", summaries["longest_deficit_run_months"]),
        ("Pre-conflict observations", summaries["pre_conflict_n"]),
        ("Pre-conflict mean anomaly (cm)",
         summaries["pre_conflict_mean_cm"]),
        ("Latest pre-conflict observation",
         summaries["latest_pre_conflict_date"]),
        ("Latest pre-conflict anomaly (cm)",
         summaries["latest_pre_conflict_anomaly_cm"]),
        ("Conflict-period observations", summaries["conflict_period_n"]),
        ("Conflict-period mean anomaly (cm)",
         summaries["conflict_period_mean_cm"]),
        ("Conflict-period minimum anomaly (cm)",
         summaries["conflict_period_min_cm"]),
    ]

    summary_path = OUTPUT_DIR / "iran_grace_tws_summary.csv"
    pd.DataFrame(rows, columns=["metric", "value"]).to_csv(
        summary_path, index=False)

    print(f"  Monthly CSV saved to: {monthly_path}")
    print(f"  Summary CSV saved to: {summary_path}")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("GRACE / GRACE-FO terrestrial water-storage analysis — Iran")
    print("=" * 65)

    print("\n[1/7] Authenticating with NASA Earthdata")
    earthaccess = authenticate()

    print("\n[2/7] Downloading GRACE data from PO.DAAC")
    download_grace(earthaccess)

    print("\n[3/7] Loading the Iran national boundary")
    geometry = load_iran_boundary()

    print("\n[4/7] Opening the GRACE dataset")
    ds = open_grace_dataset()
    product_label = describe_product(ds)

    print("\n[5/7] Extracting the national time series")
    ts, diag = extract_iran_timeseries(ds, geometry)
    diag["product_label"] = product_label

    print("\n[6/7] Calculating trends and summaries")
    trend_results = fit_linear_trend(ts)
    summaries = calculate_period_summaries(ts)

    print("\n[7/7] Saving outputs and producing figures")
    save_outputs(ts, trend_results, summaries, diag)
    plot_mask_check(diag, geometry,
                    OUTPUT_DIR / "iran_grace_mask_check.png")
    plot_tws(ts, trend_results,
             OUTPUT_DIR / "iran_grace_tws_anomaly.png",
             product_label=product_label)

    print("\n" + "=" * 65)
    print("Analysis complete")
    print(f"Outputs saved in: {OUTPUT_DIR.resolve()}")
    print("Check iran_grace_mask_check.png first to confirm the "
          "country mask is correct.")
    print("=" * 65)

if __name__ == "__main__":
    main()

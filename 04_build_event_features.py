"""
04_build_event_features.py
Turn street closures and downtown events into numeric daily features and merge
them into the daily panel (addresses the TA's "qualitative -> quantitative" comment).

Closures : DOMI Street Closures (WPRDC, dataset 'street-closures'), downloaded automatically.
           Each row is one road segment of a permit, with from_date/to_date, full_closure,
           and a WKT line geometry (lon lat) -> we compute miles and station proximity.
Events   : Downtown Pittsburgh Partnership calendar has no bulk download, so supply a small CSV
           at data/raw/events/downtown_events.csv (template is created on first run):
               title,start_date,end_date,category,major,lat,lon
           'major' (0/1), 'lat', 'lon' are optional. If 'major' is missing, a keyword
           heuristic is used (marathon, festival, parade, fireworks, ...).
Stations : optional data/raw/stations.csv (station_name,lat,lon) for proximity buffers.
           If absent, we try lat/lon columns inside data/processed/trips_clean.parquet.

Outputs  : data/processed/event_features.csv         (date + new feature columns only)
           data/processed/daily_panel_features.csv   (daily_panel.csv + new features)

Run after 03:  python 04_build_event_features.py [--radius 500] [--refresh]
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PANEL = Path("data/processed/daily_panel.csv")
CLOSURES_CSV = Path("data/raw/closures/street_closures.csv")
EVENTS_CSV = Path("data/raw/events/downtown_events.csv")
STATIONS_CSV = Path("data/raw/stations.csv")
TRIPS_PARQUET = Path("data/processed/trips_clean.parquet")
OUT_FEATURES = Path("data/processed/event_features.csv")
OUT_PANEL = Path("data/processed/daily_panel_features.csv")

CKAN = "https://data.wprdc.org/api/3/action/package_show"
# DOMI street closures CSV
FALLBACK_RESOURCE_ID = "a9a1d93a-9d3b-4c18-bd80-82ed6f86404a"

MAJOR_KEYWORDS = [
    "marathon", "festival", "parade", "fireworks", "regatta", "pedalpgh", "ride",
    "steelers", "penguins", "pirates", "riverhounds", "concert", "fashion week",
    "arts festival", "gala", "race", "run", "tournament", "draft", "convention",
]


# ================================================================ helpers
def hav_m(lat1, lon1, lat2, lon2):
    """Haversine distance in metres (vectorised)."""
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def to_bool(s):
    return s.astype(str).str.strip().str.lower().isin(["true", "t", "1", "yes", "y"])


def norm(s):
    """Normalise a station name for fuzzy matching."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def merge_intervals(d):
    """Merge overlapping/adjacent [start, end] day-number intervals per roadway_id."""
    d = d.sort_values(["roadway_id", "start"]).copy()
    d["cmax"] = d.groupby("roadway_id")["end"].cummax()
    d["prev"] = d.groupby("roadway_id")["cmax"].shift()
    new = d["prev"].isna() | (d["start"] > d["prev"] + 1)
    d["blk"] = new.cumsum()
    return d.groupby("blk").agg(roadway_id=("roadway_id", "first"),
                                start=("start", "min"), end=("end", "max")).reset_index(drop=True)


def daily_sum(starts, ends, weights, ndays):
    """Sum of `weights` of intervals active on each day 0..ndays-1 (difference array)."""
    starts, ends, weights = np.asarray(starts), np.asarray(
        ends), np.asarray(weights, float)
    keep = (ends >= 0) & (starts < ndays)
    starts, ends, weights = np.clip(starts[keep], 0, ndays), np.clip(
        ends[keep], 0, ndays - 1), weights[keep]
    arr = np.zeros(ndays + 1)
    np.add.at(arr, starts, weights)
    np.add.at(arr, ends + 1, -weights)
    return np.cumsum(arr)[:ndays]


# ================================================================ closures
def download_closures(refresh=False):
    if CLOSURES_CSV.exists() and not refresh:
        print(f"Using cached {CLOSURES_CSV}")
        return
    CLOSURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    rid = FALLBACK_RESOURCE_ID
    try:
        r = requests.get(CKAN, params={"id": "street-closures"}, timeout=60)
        r.raise_for_status()
        res = r.json()["result"]["resources"]
        csvs = [x for x in res if str(x.get("format", "")).upper(
        ) == "CSV" and x.get("datastore_active")]
        if csvs:
            rid = csvs[0]["id"]
    except Exception as e:  # fall back to the known resource id
        print(f"[warn] package_show failed ({e}); using fallback resource id")
    url = f"https://data.wprdc.org/datastore/dump/{rid}?format=csv"
    print(f"Downloading closures from {url} ...")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(CLOSURES_CSV, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    print(f"Saved {CLOSURES_CSV}")


def load_closures():
    c = pd.read_csv(CLOSURES_CSV, low_memory=False)
    print(f"Closures raw rows: {len(c):,}")
    c["from_date"] = pd.to_datetime(c["from_date"], errors="coerce")
    c["to_date"] = pd.to_datetime(c["to_date"], errors="coerce")
    c = c.dropna(subset=["from_date"])
    c["to_date"] = c["to_date"].fillna(c["from_date"])
    swap = c["to_date"] < c["from_date"]
    c.loc[swap, ["from_date", "to_date"]] = c.loc[swap,
                                                  ["to_date", "from_date"]].values
    c["full_closure"] = to_bool(
        c["full_closure"]) if "full_closure" in c else False
    print(
        f"Closures date coverage: {c['from_date'].min().date()} -> {c['to_date'].max().date()}")
    return c


def parse_geometry(c):
    """roadway_id -> (lons, lats) from WKT, plus length in miles."""
    geo = c.dropna(subset=["roadway_id", "wkt"]).drop_duplicates(
        "roadway_id")[["roadway_id", "wkt"]]
    lons, lats, miles = {}, {}, {}
    pat = re.compile(r"(-?\d+\.?\d*)\s+(-?\d+\.?\d*)")
    for rid, wkt in zip(geo["roadway_id"], geo["wkt"]):
        pts = np.array(pat.findall(str(wkt)), dtype=float)
        if len(pts) < 2:
            continue
        lon, lat = pts[:, 0], pts[:, 1]
        lons[rid], lats[rid] = lon, lat
        miles[rid] = hav_m(lat[:-1], lon[:-1], lat[1:],
                           lon[1:]).sum() / 1609.344
    print(f"Parsed geometry for {len(miles):,} road segments "
          f"(mean {np.mean(list(miles.values())):.3f} mi)")
    return lons, lats, miles


# ================================================================ stations
def load_stations(panel_cols):
    """Return {panel_station_column: (lat, lon)} matched by normalised name."""
    wanted = {col: norm(col.replace("trips_", "", 1))
              for col in panel_cols if col.startswith("trips_")}
    table = None
    if STATIONS_CSV.exists():
        s = pd.read_csv(STATIONS_CSV)
        name = next((c for c in s.columns if "name" in c.lower()
                    or c.lower() == "station"), None)
        lat = next((c for c in s.columns if "lat" in c.lower()), None)
        lon = next((c for c in s.columns if "lon" in c.lower()
                   or "lng" in c.lower()), None)
        if name and lat and lon:
            table = s[[name, lat, lon]].dropna().rename(
                columns={name: "name", lat: "lat", lon: "lon"})
    elif TRIPS_PARQUET.exists():
        t = pd.read_parquet(TRIPS_PARQUET)
        name = next((c for c in t.columns if "start" in c.lower()
                    and "name" in c.lower()), None)
        lat = next((c for c in t.columns if "start" in c.lower()
                   and "lat" in c.lower()), None)
        lon = next((c for c in t.columns if "start" in c.lower()
                   and ("lon" in c.lower() or "lng" in c.lower())), None)
        if name and lat and lon:
            table = (t.groupby(name)[[lat, lon]].median().reset_index()
                     .rename(columns={name: "name", lat: "lat", lon: "lon"}))
        else:
            print(
                f"[note] no start-station lat/lon in trips_clean.parquet (columns: {list(t.columns)})")
    if table is None:
        print("[note] no station coordinates found -> skipping station-proximity features.\n"
              "       Put POGOH station locations in data/raw/stations.csv (station_name,lat,lon).")
        return {}
    table["key"] = table["name"].map(norm)
    out = {}
    for col, key in wanted.items():
        m = table[table["key"] == key]
        if len(m):
            out[col] = (float(m["lat"].iloc[0]), float(m["lon"].iloc[0]))
    print(f"Matched coordinates for {len(out)}/{len(wanted)} stations")
    return out


# ================================================================ main
def main(radius, refresh, max_event_days, max_closure_days):
    panel = pd.read_csv(PANEL, parse_dates=["date"]).set_index(
        "date").sort_index()
    dates = pd.date_range(panel.index.min(), panel.index.max(), freq="D")
    n = len(dates)
    day0 = dates[0]
    feats = pd.DataFrame(index=dates)

    def dn(series):
        return (pd.to_datetime(series) - day0).dt.days.values

    # ---------------- closures
    download_closures(refresh)
    c = load_closures()
    c = c.dropna(subset=["roadway_id"]).copy()
    c["start"], c["end"] = dn(c["from_date"]), dn(c["to_date"])
    lons, lats, miles = parse_geometry(c)
    c = c[c["roadway_id"].isin(miles)]

    # Long-running permits (months/years, some to 2030) are "always on" background, not
    # day-specific disruptions, and they trend upward with ridership -> keep short closures only.
    dur = c["end"] - c["start"] + 1
    n_perm_all = c["permit_id"].nunique()
    print(f"\nClosure permit duration (days): median {dur.median():.0f}, "
          f"p90 {dur.quantile(0.9):.0f}, max {dur.max():.0f}")
    c = c[dur <= max_closure_days].copy()
    print(f"Keeping closures <= {max_closure_days} days: {c['permit_id'].nunique():,} of "
          f"{n_perm_all:,} permits ({len(c):,} segment rows)")

    # permits active (unique permit)
    permits = c.drop_duplicates(["permit_id", "start", "end"])
    feats["closure_permits_active"] = daily_sum(
        permits["start"], permits["end"], np.ones(len(permits)), n)
    started = np.zeros(n)
    ps = permits["start"].values
    ps = ps[(ps >= 0) & (ps < n)]
    np.add.at(started, ps, 1)
    feats["closure_permits_started"] = started

    # segments / miles active (merge intervals per roadway so overlapping permits are not double counted)
    m_all = merge_intervals(c[["roadway_id", "start", "end"]])
    m_full = merge_intervals(
        c[c["full_closure"]][["roadway_id", "start", "end"]])
    for tag, m in [("", m_all), ("_full", m_full)]:
        w_miles = m["roadway_id"].map(miles).values
        feats[f"closure_segments{tag}"] = daily_sum(
            m["start"], m["end"], np.ones(len(m)), n)
        feats[f"closure_miles{tag}"] = daily_sum(
            m["start"], m["end"], w_miles, n)
    feats["closure_any"] = (feats["closure_segments"] > 0).astype(int)

    # ---------------- station proximity (closures)
    stations = load_stations(panel.columns)
    for col, (slat, slon) in stations.items():
        key = col.replace("trips_", "", 1)
        near = {rid for rid in miles
                if hav_m(lats[rid], lons[rid], slat, slon).min() <= radius}
        for tag, m in [("near_closure", m_all), ("near_fullclosure", m_full)]:
            mm = m[m["roadway_id"].isin(near)]
            feats[f"{tag}_{key}"] = daily_sum(
                mm["start"], mm["end"], np.ones(len(mm)), n)

    # ---------------- events
    if not EVENTS_CSV.exists():
        EVENTS_CSV.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["title", "start_date", "end_date", "category", "major", "lat", "lon"]).to_csv(
            EVENTS_CSV, index=False)
        print(f"\n[note] created empty template {EVENTS_CSV}. Add downtown events "
              "(one row per event; major=1 for big ones) and re-run to populate event features.")
    ev = pd.read_csv(EVENTS_CSV)
    feats["event_count"] = 0.0
    feats["major_event_count"] = 0.0
    if len(ev):
        sc = next(c_ for c_ in ev.columns if c_.lower() in (
            "start_date", "start", "date", "from_date"))
        ec = next((c_ for c_ in ev.columns if c_.lower()
                  in ("end_date", "end", "to_date")), sc)
        ev["s"] = pd.to_datetime(ev[sc], errors="coerce")
        ev["e"] = pd.to_datetime(ev[ec], errors="coerce").fillna(ev["s"])
        ev = ev.dropna(subset=["s"])
        # multi-week exhibitions / rinks are not day-specific demand shocks -> drop
        long_ = (ev["e"] - ev["s"]).dt.days + 1 > max_event_days
        print(
            f"Events loaded: {len(ev)}  (dropping {int(long_.sum())} longer than {max_event_days} days)")
        ev = ev[~long_]
        if "major" in ev.columns and ev["major"].notna().any():
            ev["is_major"] = ev["major"].fillna(0).astype(int).astype(bool)
        else:
            text = (ev.get("title", "").astype(str) + " " +
                    ev.get("category", "").astype(str)).str.lower()
            ev["is_major"] = text.apply(
                lambda t: any(k in t for k in MAJOR_KEYWORDS))
            print(
                "[note] no 'major' column: using keyword heuristic - curate it by hand for the final report")
        s, e = dn(ev["s"]), dn(ev["e"])
        feats["event_count"] = daily_sum(s, e, np.ones(len(ev)), n)
        feats["major_event_count"] = daily_sum(s[ev["is_major"].values], e[ev["is_major"].values],
                                               np.ones(int(ev["is_major"].sum())), n)
        if {"lat", "lon"}.issubset(ev.columns) and stations:
            for col, (slat, slon) in stations.items():
                key = col.replace("trips_", "", 1)
                ok = ev["lat"].notna() & ev["lon"].notna()
                d = hav_m(ev["lat"].where(ok, 0).values,
                          ev["lon"].where(ok, 0).values, slat, slon)
                nearm = ok.values & (d <= radius)
                feats[f"near_event_{key}"] = daily_sum(
                    s[nearm], e[nearm], np.ones(int(nearm.sum())), n)
    feats["is_event_day"] = (feats["event_count"] > 0).astype(int)
    feats["is_major_event"] = (feats["major_event_count"] > 0).astype(int)

    # ---------------- save + report
    feats.index.name = "date"
    feats.to_csv(OUT_FEATURES)
    merged = panel.join(feats, how="left")
    merged.to_csv(OUT_PANEL)
    print(
        f"\nSaved {OUT_FEATURES} ({feats.shape[1]} features) and {OUT_PANEL} {merged.shape}")

    core = ["closure_permits_active", "closure_segments", "closure_segments_full", "closure_miles",
            "closure_miles_full", "event_count", "major_event_count"]
    print("\nFeature summary:\n", feats[core].describe(
    ).loc[["mean", "max"]].round(2).to_string())
    print("\nCorrelation with system_trips:\n",
          merged[core].corrwith(merged["system_trips"]).round(3).to_string())
    if c["to_date"].max() < panel.index.max():
        print(f"\n[warn] closures data ends {c['to_date'].max().date()}, before panel end "
              f"{panel.index.max().date()} - closure features are 0 after that; re-run with --refresh")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=500,
                    help="station buffer in metres")
    ap.add_argument("--refresh", action="store_true",
                    help="re-download closures")
    ap.add_argument("--max-event-days", type=int, default=7)
    ap.add_argument("--max-closure-days", type=int, default=30,
                    help="drop closure permits longer than this (long-running background closures)")
    a = ap.parse_args()
    main(a.radius, a.refresh, a.max_event_days, a.max_closure_days)

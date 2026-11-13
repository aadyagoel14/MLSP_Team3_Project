"""
Clean the raw POGOH trip XLSX files and build the analysis-ready daily panel:

    date | system_trips | trips_<top_station_1> | ... | temp_mean | precip | snow | dow | month | is_weekend | ...

Steps:
  1. Load and concatenate all monthly trip files (schema has drifted across
     months/years -- we normalize column names before concatenating).
  2. Clean: drop duplicate trip IDs, drop negative/zero/absurd durations,
     parse timestamps, flag and drop rows with missing start timestamps.
  3. Aggregate to DAILY counts: system-wide, and for the N busiest stations.
  4. Merge in daily weather (Open-Meteo CSV from step 2).
  5. Add calendar features: day-of-week, month, is_weekend, is_holiday.
  6. Save a tidy `daily_panel.csv` ready for modeling, plus the cleaned
     trip-level file for anything that needs trip granularity later.

Usage:
    python 03_clean_and_merge.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAW_TRIPS_DIR = Path("data/raw/trips")
WEATHER_DIR = Path("data/raw/weather")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Trip duration outlier bounds (minutes). Below 1 min = likely a false
# start/aborted unlock; above 24h = likely theft, lost bike, or a data error
# rather than a genuine ride.
MIN_DURATION_MIN = 1
MAX_DURATION_MIN = 24 * 60

N_TOP_STATIONS = 10

# Column-name variants seen across different months' files go here as they're
# discovered -- POGOH's exports are not perfectly consistent. Add new
# variants on the left as you encounter them.
COLUMN_ALIASES = {
    "Start Date": "start_time",
    "StartDate": "start_time",
    "start_date": "start_time",
    "End Date": "end_time",
    "EndDate": "end_time",
    "end_date": "end_time",
    "Start Station Id": "start_station_id",
    "StartStationId": "start_station_id",
    "Start Station Name": "start_station_name",
    "StartStationName": "start_station_name",
    "End Station Id": "end_station_id",
    "End Station Name": "end_station_name",
    "Trip Duration": "duration_sec",
    "TripDuration": "duration_sec",
    "Duration": "duration_sec",
    "Rider Type": "rider_type",
    "RiderType": "rider_type",
    "User Type": "rider_type",
    "Closed Status": "closed_status",
    "ClosedStatus": "closed_status",
}


def load_one_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        return pd.DataFrame()

    df = df.rename(columns=COLUMN_ALIASES)
    df["__source_file"] = path.name
    return df


def load_all_trips(raw_dir: Path) -> pd.DataFrame:
    files = sorted(raw_dir.glob("*.xlsx")) + sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No trip files found in {raw_dir}. Run 01_download_trip_data.py first."
        )

    frames = []
    for f in files:
        try:
            frames.append(load_one_file(f))
        except Exception as e:
            print(f"  [WARN] could not read {f.name}: {e}")

    df = pd.concat(frames, ignore_index=True, sort=False)
    print(f"Loaded {len(df):,} raw trip rows from {len(files)} files.")
    return df


def clean_trips(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)

    # Parse timestamps
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df["end_time"] = pd.to_datetime(df.get("end_time"), errors="coerce")

    # Drop rows with unparseable / missing start timestamps -- we can't
    # place these in the daily series at all.
    df = df[df["start_time"].notna()]
    n_after_ts = len(df)

    # Recompute duration in minutes where possible, preferring an explicit
    # duration column but falling back to end - start.
    if "duration_sec" in df.columns:
        dur_min = pd.to_numeric(df["duration_sec"], errors="coerce") / 60.0
    else:
        dur_min = pd.Series(np.nan, index=df.index)
    computed = (df["end_time"] - df["start_time"]).dt.total_seconds() / 60.0
    df["duration_min"] = dur_min.fillna(computed)

    # Drop duplicate trips (same start/end station + start time + duration
    # is a reasonable proxy for a duplicate record across overlapping
    # monthly exports).
    dedup_cols = [c for c in ["start_time", "start_station_id", "end_time",
                               "end_station_id", "duration_min"] if c in df.columns]
    df = df.drop_duplicates(subset=dedup_cols)
    n_after_dedup = len(df)

    # Drop implausible durations
    df = df[(df["duration_min"] >= MIN_DURATION_MIN) & (df["duration_min"] <= MAX_DURATION_MIN)]
    n_after_duration = len(df)

    # Exclude non-"normal" closed statuses if present (e.g. mechanical
    # lock failures) -- these aren't genuine completed rides for demand
    # purposes. Keep anything we don't recognize rather than silently
    # dropping it; print what we find so you can sanity-check.
    if "closed_status" in df.columns:
        statuses = df["closed_status"].value_counts(dropna=False)
        print("Closed-status value counts:\n", statuses)
        # Uncomment once you've confirmed which labels should be excluded:
        # df = df[df["closed_status"].isin(["normal"])]

    print(
        f"Cleaning: {n0:,} -> {n_after_ts:,} (valid timestamp) "
        f"-> {n_after_dedup:,} (deduped) -> {n_after_duration:,} (valid duration)"
    )
    return df


def build_daily_panel(trips: pd.DataFrame) -> pd.DataFrame:
    trips["date"] = trips["start_time"].dt.date
    daily = trips.groupby("date").size().rename("system_trips").reset_index()
    daily["date"] = pd.to_datetime(daily["date"])

    # Fill any missing calendar days with 0 trips (important: don't let
    # gaps silently disappear from a daily time series)
    full_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    daily = daily.set_index("date").reindex(full_range).rename_axis("date").reset_index()
    daily["system_trips"] = daily["system_trips"].fillna(0).astype(int)

    # Per-station counts for the busiest stations (useful for the
    # per-station anomaly check described in the proposal)
    if "start_station_name" in trips.columns:
        top_stations = (
            trips["start_station_name"].value_counts().head(N_TOP_STATIONS).index.tolist()
        )
        for station in top_stations:
            col = f"trips_{station.strip().replace(' ', '_').replace('/', '-')[:40]}"
            station_daily = (
                trips[trips["start_station_name"] == station]
                .groupby("date").size().rename(col)
            )
            station_daily.index = pd.to_datetime(station_daily.index)
            daily = daily.merge(station_daily, on="date", how="left")
            daily[col] = daily[col].fillna(0).astype(int)

    return daily


def add_calendar_features(daily: pd.DataFrame) -> pd.DataFrame:
    daily["dow"] = daily["date"].dt.dayofweek  # 0=Mon
    daily["month"] = daily["date"].dt.month
    daily["year"] = daily["date"].dt.year
    daily["is_weekend"] = daily["dow"].isin([5, 6]).astype(int)

    try:
        import holidays
        us_holidays = holidays.US(years=range(daily["year"].min(), daily["year"].max() + 1))
        daily["is_holiday"] = daily["date"].isin(us_holidays).astype(int)
    except ImportError:
        print("  [note] `pip install holidays` to add is_holiday automatically; skipping for now.")
        daily["is_holiday"] = 0

    return daily


def merge_weather(daily: pd.DataFrame, weather_dir: Path) -> pd.DataFrame:
    weather_files = sorted(weather_dir.glob("*.csv"))
    if not weather_files:
        print("  [WARN] no weather CSV found in data/raw/weather/ -- run 02_download_weather.py. "
              "Continuing without weather columns.")
        return daily

    weather = pd.read_csv(weather_files[-1])  # most recently created file
    weather["date"] = pd.to_datetime(weather["date"])
    merged = daily.merge(weather, on="date", how="left")

    missing_weather = merged["temperature_2m_mean"].isna().sum() if "temperature_2m_mean" in merged else len(merged)
    if missing_weather:
        print(f"  [WARN] {missing_weather} days have no matching weather row "
              f"-- check your weather date range covers the trip data range.")
    return merged


def main():
    trips_raw = load_all_trips(RAW_TRIPS_DIR)
    trips_clean = clean_trips(trips_raw)
    trips_clean.to_parquet(OUT_DIR / "trips_clean.parquet", index=False)
    print(f"Saved cleaned trip-level data -> {OUT_DIR / 'trips_clean.parquet'}")

    daily = build_daily_panel(trips_clean)
    daily = add_calendar_features(daily)
    daily = merge_weather(daily, WEATHER_DIR)

    daily.to_csv(OUT_DIR / "daily_panel.csv", index=False)
    print(f"Saved daily panel ({len(daily)} days, {daily.shape[1]} columns) -> "
          f"{OUT_DIR / 'daily_panel.csv'}")
    print("\nPreview:")
    print(daily.head())
    print("\nDate range:", daily["date"].min().date(), "to", daily["date"].max().date())


if __name__ == "__main__":
    main()

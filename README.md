# POGOH Daily Demand — Data Pipeline

## Why not run this here

WPRDC and Open-Meteo aren't reachable from this sandbox's network, so run
these scripts on your own laptop or a Colab/JupyterHub notebook with normal
internet access.

## Setup

```bash
pip install pandas numpy openpyxl holidays
```

## Run in order

```bash
# 1. Pull every monthly POGOH trip XLSX from WPRDC (auto-discovers files
#    via the CKAN API, so it doesn't need ~40 hardcoded URLs)
python 01_download_trip_data.py

# 2. Pull daily weather for Pittsburgh from Open-Meteo over your trip
#    data's full date range (POGOH launched May 2022)
#    WPRDC’s earliest available POGOH file is May 2022; we could not locate an April 2022 file despite the TA’s note.
python 02_download_weather.py --start 2022-05-01 --end 2026-10-31

# 3. Clean + aggregate to a daily panel, merge weather, add calendar features
python 03_clean_and_merge.py
```

Output: `data/processed/daily_panel.csv` — one row per calendar day, with
`system_trips`, per-station trip counts for your top stations, weather
columns, and day-of-week/month/weekend/holiday flags. This is what your
ARX/ARMAX/SARIMAX baselines and later gradient-boosted model should read in.

## Known rough edges to check once you have real files

- **Schema drift**: POGOH's monthly exports have changed column names over
  time (`COLUMN_ALIASES` in `03_clean_and_merge.py` handles the ones I
  know about from the data dictionary — add any new variants you see
  printed as unmapped columns).
- **`closed_status` filtering**: the script prints the value counts for
  this column but doesn't filter on it yet — decide with your team whether
  non-"normal" closes (e.g. mechanical issues) should count as real demand
  or get dropped, then uncomment the filter line.
- **Station renames/relocations**: WPRDC's changelog notes stations have
  been relocated/renumbered (e.g. Feb 2023, station 32). If you aggregate
  by station name/ID for the per-station anomaly check, confirm IDs are
  stable across the period you're using, or you'll split one physical
  station's history in two.
- **Duration bounds**: `MIN_DURATION_MIN=1` / `MAX_DURATION_MIN=1440` are a
  starting guess — look at the distribution for your data and adjust before
  the midterm so you can justify the cutoffs.
- **Pre-2022 data**: if you need more history for seasonality, the old
  "Healthy Ride" quarterly archive (pre-May-2022) is on WPRDC under a
  different dataset slug — the download script has a commented-out stub
  for it, but double check the exact slug on the WPRDC site since it isn't
  guaranteed to merge cleanly with POGOH's station IDs.

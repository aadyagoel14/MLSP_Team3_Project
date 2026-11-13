"""
Download daily historical weather for Pittsburgh from the Open-Meteo
Historical Weather API (free, no API key required).

Pulls: mean/max/min temperature, precipitation sum, snowfall sum.
Date range should cover the full span of your POGOH trip data.

Usage:
    python 02_download_weather.py --start 2022-05-01 --end 2026-10-31
"""

import argparse
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

# Downtown Pittsburgh coordinates (close enough for city-wide daily weather)
LATITUDE = 40.4406
LONGITUDE = -79.9959

OUT_DIR = Path("data/raw/weather")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DAILY_VARS = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "snowfall_sum",
    "windspeed_10m_max",
]


def fetch_weather(start_date: str, end_date: str) -> dict:
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(DAILY_VARS),
        "timezone": "America/New_York",
    }
    url = "https://archive-api.open-meteo.com/v1/archive?" + urlencode(params)
    req = Request(url, headers={"User-Agent": "research project"})
    with urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    args = parser.parse_args()

    print(f"Fetching Open-Meteo daily weather {args.start} to {args.end}...")
    data = fetch_weather(args.start, args.end)

    out_path = OUT_DIR / f"pittsburgh_weather_{args.start}_to_{args.end}.json"
    out_path.write_text(json.dumps(data, indent=2))
    print(f"Saved raw JSON to {out_path}")

    # Also write a flat CSV immediately for convenience
    import csv

    daily = data["daily"]
    out_csv = OUT_DIR / f"pittsburgh_weather_{args.start}_to_{args.end}.csv"
    fieldnames = ["date"] + DAILY_VARS
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for i, date in enumerate(daily["time"]):
            row = [date] + [daily[var][i] for var in DAILY_VARS]
            writer.writerow(row)
    print(f"Saved CSV to {out_csv}")


if __name__ == "__main__":
    main()

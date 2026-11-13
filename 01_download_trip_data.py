"""
Download all POGOH trip data files (monthly XLSX) from the WPRDC open data portal.

WPRDC runs CKAN, so we use the `package_show` API action to list every
resource (file) attached to the 'pogoh-trip-data' dataset, then download
each one. This avoids hardcoding ~40 individual monthly URLs by hand and
will pick up new months automatically as POGOH publishes them.

Run this on a machine with normal internet access (WPRDC is not reachable
from this sandbox's restricted network).

Usage:
    python 01_download_trip_data.py
"""

import json
import os
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

CKAN_BASE = "https://data.wprdc.org"
PACKAGE_ID = "pogoh-trip-data"
OUT_DIR = Path("data/raw/trips")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (research project; CMU coursework)"}


def fetch_json(url: str) -> dict:
    req = Request(url, headers=HEADERS)
    with urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_resources(package_id: str) -> list[dict]:
    """Query CKAN package_show to list every resource (file) in the dataset."""
    url = f"{CKAN_BASE}/api/3/action/package_show?id={package_id}"
    data = fetch_json(url)
    if not data.get("success"):
        raise RuntimeError(f"CKAN API call failed: {data}")
    return data["result"]["resources"]


def safe_filename(resource: dict) -> str:
    name = resource.get("name") or resource.get("id")
    name = name.strip().replace("/", "-").replace(" ", "_")
    fmt = (resource.get("format") or "xlsx").lower()
    if not name.lower().endswith(f".{fmt}"):
        name = f"{name}.{fmt}"
    return name


def download_resource(resource: dict, out_dir: Path, retries: int = 3) -> Path | None:
    url = resource.get("url")
    if not url:
        print(f"  [skip] no URL for resource {resource.get('id')}")
        return None

    fname = safe_filename(resource)
    out_path = out_dir / fname

    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"  [cached] {fname}")
        return out_path

    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers=HEADERS)
            with urlopen(req, timeout=120) as resp, open(out_path, "wb") as f:
                f.write(resp.read())
            print(f"  [ok] {fname}")
            return out_path
        except (URLError, HTTPError) as e:
            print(f"  [retry {attempt}/{retries}] {fname} -> {e}")
            time.sleep(2 * attempt)
    print(f"  [FAILED] {fname}")
    return None


def main():
    print(f"Listing resources for dataset '{PACKAGE_ID}'...")
    resources = get_resources(PACKAGE_ID)
    print(f"Found {len(resources)} resources.\n")

    downloaded, failed = [], []
    for r in resources:
        fmt = (r.get("format") or "").lower()
        if fmt not in ("xlsx", "csv", "xls"):
            # skip non-tabular resources (e.g. data dictionaries in PDF)
            continue
        path = download_resource(r, OUT_DIR)
        (downloaded if path else failed).append(r.get("name", r.get("id")))

    print(
        f"\nDone. {len(downloaded)} downloaded/cached, {len(failed)} failed.")
    if failed:
        print("Failed resources:")
        for f in failed:
            print(f"  - {f}")

    # Also grab the Healthy Ride (pre-POGOH, pre-May 2022) archive if you want
    # more history. Uncomment and check dataset slug on WPRDC if needed:
    # healthy_ride_resources = get_resources("healthy-ride-trip-data-archived")
    # for r in healthy_ride_resources:
    #     download_resource(r, Path("data/raw/trips_healthyride"))


if __name__ == "__main__":
    main()

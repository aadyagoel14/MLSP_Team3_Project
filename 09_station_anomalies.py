"""
09_station_anomalies.py  --  per-station anomaly detection (optional extension of 08).

Question: when the system has an anomalous day, is it system-wide (a snowstorm hits every station)
or local (one station jumps or collapses while the system is normal)?

Method (no model is trained per station):
  expected_s(t)  = system forecast(t)  x  share_s(t)
                   share_s(t) = station s's trips over days t-28..t-1 / system trips over the same days
                   (uses only the past, so no leakage; system forecast = results/xgb_predictions.csv)
  raw log ratio  = log((actual_s + 1) / (expected_s + 1))             -> station vs its own expectation
  local log ratio= raw log ratio  -  median of the raw log ratios of all stations that day
                   -> what is LEFT after removing the system-wide miss (a station-specific deviation);
                   the median (not the summed system miss) so that one jumping station cannot bend the others
  z              = robust z-score (median / 1.4826 x MAD) of each station's own series over the test days
  flag           = |z| > threshold AND |actual - expected| >= --min-abs trips (ignores tiny counts)
Only stations with a trailing mean >= --min-mean trips/day are scored.

Outputs
  results/station_anomalies_all.csv        every scored station-day (date, station, actual, expected, z_raw, z_local, flags)
  results/station_anomalies_flagged.csv    local flags, sorted by |z_local|
  results/station_anomalies_by_day.csv     per day: stations scored / raw-flagged / local-flagged, system z
  results/station_anomalies_by_station.csv per station: number of local flags, direction counts
  figures/anomalies/station_flag_counts.png, station_heatmap.png

Run from the repository root (after 06 and 08):   python 09_station_anomalies.py [--threshold 3]
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

PANEL = Path("data/processed/daily_panel_features.csv")
PRED = Path("results/xgb_predictions.csv")
OUT_DIR = Path("results")
FIG_DIR = Path("figures/anomalies")
SHARE_WINDOW = 28


def robust_z(x):
    med = x.median()
    mad = (x - med).abs().median() * 1.4826
    return (x - med) / mad if mad > 0 else x * 0.0


def main(args):
    OUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(PANEL, parse_dates=["date"]).set_index(
        "date").sort_index()
    pred = pd.read_csv(args.pred, parse_dates=[
                       "date"]).set_index("date").sort_index()
    pred = pred[["actual", "pred"]].dropna()

    st_cols = [c for c in panel.columns if c.startswith("trips_")]
    if not st_cols:
        raise SystemExit(
            "No trips_<station> columns in the panel; re-run 03_clean_and_merge.py.")
    st = panel[st_cols].fillna(0.0)
    sys_tot = panel["system_trips"].astype(float)
    print(f"{len(st_cols)} station columns; forecasts for {len(pred)} days "
          f"({pred.index[0].date()} to {pred.index[-1].date()})")

    # trailing share of the system (past 28 days only)
    sum_st = st.shift(1).rolling(SHARE_WINDOW).sum()
    sum_sys = sys_tot.shift(1).rolling(SHARE_WINDOW).sum()
    share = sum_st.div(sum_sys, axis=0)
    mean_st = sum_st / SHARE_WINDOW

    days = pred.index.intersection(share.dropna(how="all").index)
    P = pred.loc[days, "pred"]
    A = pred.loc[days, "actual"]

    # raw log ratio per station, then the system-wide component = median across stations that day
    raws, keep = {}, {}
    for c in st_cols:
        a = st.loc[days, c]
        e = share.loc[days, c] * P
        ok = (mean_st.loc[days, c] >= args.min_mean) & e.notna()
        if ok.sum() < 30:
            continue
        raws[c] = (np.log((a + 1) / (e + 1)).where(ok), a, e, ok)
    common = pd.DataFrame({c: v[0] for c, v in raws.items()})
    # robust to one station jumping (the summed forecast miss is not)
    sys_log = common.median(axis=1)
    sys_z = robust_z(sys_log)
    rows = []
    for c, (raw, a, e, ok) in raws.items():
        loc = raw - sys_log
        rows.append(pd.DataFrame({"date": days[ok], "station": c[len("trips_"):], "actual": a[ok].values,
                                  "expected": e[ok].values, "z_raw": robust_z(raw[ok]).values,
                                  "z_local": robust_z(loc[ok]).values}))
    allr = pd.concat(rows, ignore_index=True)
    big = (allr["actual"] - allr["expected"]).abs() >= args.min_abs
    allr["flag_raw"] = (allr["z_raw"].abs() > args.threshold) & big
    allr["flag_local"] = (allr["z_local"].abs() > args.threshold) & big
    allr["direction_local"] = np.where(allr["flag_local"], np.where(
        allr["z_local"] > 0, "surge", "drop"), "")
    allr["actual"] = allr["actual"].round(1)
    allr["expected"] = allr["expected"].round(1)
    allr[["z_raw", "z_local"]] = allr[["z_raw", "z_local"]].round(2)
    allr.to_csv(OUT_DIR / "station_anomalies_all.csv", index=False)
    print(f"Scored {allr['station'].nunique()} stations (trailing mean >= {args.min_mean}/day), "
          f"{len(allr)} station-days")

    # ---- by day
    by_day = allr.groupby("date").agg(stations=("station", "size"), raw_flagged=("flag_raw", "sum"),
                                      local_flagged=("flag_local", "sum"))
    by_day["share_raw_flagged_%"] = (
        by_day["raw_flagged"] / by_day["stations"] * 100).round(1)
    by_day["system_z"] = sys_z.reindex(by_day.index).round(2)
    by_day["system_flag"] = by_day["system_z"].abs() > args.threshold
    by_day.to_csv(OUT_DIR / "station_anomalies_by_day.csv")

    # ---- by station and flagged list
    fl = allr[allr["flag_local"]].copy()
    fl = fl.reindex(fl["z_local"].abs().sort_values(ascending=False).index)
    fl.to_csv(OUT_DIR / "station_anomalies_flagged.csv", index=False)
    by_st = allr.groupby("station").agg(days=("date", "size"), local_flags=("flag_local", "sum"),
                                        surges=("direction_local", lambda s: (
                                            s == "surge").sum()),
                                        drops=("direction_local", lambda s: (s == "drop").sum()))
    by_st["flag_rate_%"] = (by_st["local_flags"] /
                            by_st["days"] * 100).round(1)
    by_st = by_st.sort_values("local_flags", ascending=False)
    by_st.to_csv(OUT_DIR / "station_anomalies_by_station.csv")

    # ---- console summary
    sf = by_day[by_day["system_flag"]]
    nf = by_day[~by_day["system_flag"]]
    print(f"\nSystem-flagged days: {len(sf)}; on them {sf['share_raw_flagged_%'].mean():.1f}% of stations are "
          f"raw-flagged on average (other days: {nf['share_raw_flagged_%'].mean():.1f}%).")
    print("-> raw flags move together with system anomalies; local flags are what remains after removing the system miss.")
    print(
        f"Local station flags: {len(fl)} of {len(allr)} station-days ({len(fl) / len(allr) * 100:.2f}%)")
    print("\nDays with most locally flagged stations:")
    print(by_day.sort_values("local_flagged",
          ascending=False).head(10).to_string())
    print("\nStations with most local flags:")
    print(by_st.head(10).to_string())
    print("\nLargest local deviations:")
    print(fl.head(args.top).to_string(index=False))

    # ---- figures
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax[0].bar(by_day.index, by_day["share_raw_flagged_%"],
              width=1.0, color="tab:gray")
    ax[0].set_ylabel("% of stations\nraw-flagged")
    ax[0].scatter(sf.index, np.full(len(sf), by_day["share_raw_flagged_%"].max() * 1.03), marker="v",
                  color="tab:red", s=18, label="system-flagged day (08)")
    ax[0].legend(loc="upper right")
    ax[1].bar(by_day.index, by_day["local_flagged"],
              width=1.0, color="tab:blue")
    ax[1].set_ylabel("stations with a\nlocal flag")
    ax[0].set_title("Do station anomalies move with the system? (test year)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "station_flag_counts.png", dpi=150)
    plt.close(fig)

    top_st = by_st.head(15).index
    piv = allr[allr["station"].isin(top_st)].pivot(
        index="station", columns="date", values="z_local").reindex(top_st)
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(piv.clip(-6, 6).values, aspect="auto", cmap="RdBu_r", vmin=-6, vmax=6,
                   extent=[0, piv.shape[1], piv.shape[0], 0])
    ax.set_yticks(np.arange(len(top_st)) + 0.5)
    ax.set_yticklabels([s[:28] for s in top_st], fontsize=7)
    ticks = np.linspace(0, piv.shape[1] - 1, 7).astype(int)
    ax.set_xticks(ticks + 0.5)
    ax.set_xticklabels([piv.columns[i].strftime("%b %y") for i in ticks])
    fig.colorbar(im, label="local robust z (clipped at +-6)")
    ax.set_title("Local station deviations, 15 stations with most flags")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "station_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved CSVs to {OUT_DIR}/ and figures to {FIG_DIR}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=3.0, help="robust SDs")
    ap.add_argument("--pred", default=str(PRED),
                    help="CSV with columns date, actual, pred (system level)")
    ap.add_argument("--min-mean", type=float, default=5.0,
                    help="min trailing mean trips/day to score a station")
    ap.add_argument("--min-abs", type=float, default=5.0,
                    help="min |actual - expected| in trips to flag")
    ap.add_argument("--top", type=int, default=15)
    main(ap.parse_args())

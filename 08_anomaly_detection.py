"""
08_anomaly_detection.py
Flag days whose demand departs sharply from the forecast, then check whether the flagged days are
explained by holidays, weather extremes, street closures or downtown events.

Definition (from the proposal)
    A day is anomalous when its forecast residual lies more than THRESHOLD (default 3) robust
    standard deviations from normal, using the median as the centre and the median absolute
    deviation (MAD x 1.4826) as the spread.  Because weather is already a model input, flagged
    days should reflect something weather alone does not explain.

Residual
    --residual relative (default): (actual - forecast) / forecast.  Demand varies from ~100 to
                                   ~3,000 trips a day, so a +300 residual means very different
                                   things in January and July; relative residuals put them on one scale.
    --residual absolute          : actual - forecast, in trips.
    --scale-by season            : robust centre and spread estimated separately per season (default: global).

Where the forecasts come from
    default           : results/xgb_predictions.csv  (the test year, written by 06_xgboost.py)
    --pred FILE.csv   : any CSV with columns date, actual, pred (for example a DSP model's forecasts)
    --backtest        : refit the tuned XGBoost model in expanding windows and forecast each year
                        out of sample (default folds start 2023-11-01, 2024-11-01, 2025-11-01), which
                        gives about three years of forecasts to inspect instead of one. Saved to
                        results/xgb_backtest_predictions.csv.  (The first fold has only ~18 months of
                        training data, so its forecasts are the weakest.)

Context checks for each flagged day
    day of week, holiday, weather (temperature, rain, snow, wind), closure load (percentile within the
    same calendar year, because closures trend upward), event flags, and plain-language hints.
    For each context a Fisher exact test asks whether flagged days are over-represented.

Input : results/xgb_predictions.csv (or --pred / --backtest), data/processed/daily_panel_features.csv
Output: results/anomalies_all_days.csv, results/anomalies_flagged.csv, results/anomaly_episodes.csv,
        results/anomaly_enrichment.csv, results/anomaly_threshold_sensitivity.csv
        figures/anomalies/*.png

Run from the repository root:   python 08_anomaly_detection.py [--threshold 3] [--backtest]
"""
import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from metrics import season_of, print_table

PANEL = Path("data/processed/daily_panel_features.csv")
DEFAULT_PRED = Path("results/xgb_predictions.csv")
BACKTEST_PRED = Path("results/xgb_backtest_predictions.csv")
BEST = Path("results/xgb_best_params.json")
OUT_DIR = Path("results")
FIG_DIR = Path("figures/anomalies")
FOLD_STARTS = ["2023-11-01", "2024-11-01", "2025-11-01"]

# thresholds used for the plain-language hints (Open-Meteo units: C, mm, cm, km/h)
HEAVY_RAIN_MM = 10.0
COLD_C = -5.0
HOT_C = 30.0
WIND_KMH = 40.0
CLOSURE_TOP = 0.90   # within-year percentile


def load_module_06():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("xgb06", here / "06_xgboost.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ================================================================ forecasts
def backtest_predictions(fold_starts):
    """Expanding-window, one-step-ahead out-of-sample XGBoost forecasts, one fold per year."""
    x06 = load_module_06()
    with open(BEST) as f:
        params = json.load(f)
    n_trees, target_mode, objective = params.pop("n_trees"), params.pop("target"), params.pop("objective")
    params.pop("val_MAE", None)

    df = pd.read_csv(PANEL, parse_dates=["date"]).set_index("date").sort_index().asfreq("D")
    y = df["system_trips"].astype(float)
    X, base, _ = x06.build_features(df, target_mode)
    target = y / base if target_mode == "ratio" else y
    usable = X.notna().all(axis=1) & target.notna() & base.notna()
    w = pd.Series(x06.sample_weights(base, objective), index=df.index)

    parts = []
    for start in fold_starts:
        s = pd.Timestamp(start)
        e = min(s + pd.DateOffset(years=1), df.index[-1] + pd.Timedelta(days=1))
        tr = df.index[(df.index < s) & usable]
        te = df.index[(df.index >= s) & (df.index < e)]
        if len(te) == 0 or len(tr) < 200:
            print(f"  [skip] fold {start}: {len(tr)} training days, {len(te)} test days")
            continue
        _, p = x06.fit_predict(X.loc[tr], target[tr], w[tr], X.loc[te], base[te], params, objective, 0, n_trees)
        part = pd.DataFrame({"actual": y[te], "pred": p, "fold": start})
        print(f"  fold {start}: trained on {len(tr)} days, forecast {len(te)} days, "
              f"MAE {np.mean(np.abs(part['actual'] - part['pred'])):.1f}")
        parts.append(part)
    out = pd.concat(parts)
    out.index.name = "date"
    out.to_csv(BACKTEST_PRED)
    print(f"Saved {BACKTEST_PRED}")
    return out


# ================================================================ statistics
def robust_z(res, seasons=None):
    """(r - median) / (1.4826 * MAD), globally or within each season."""
    z = pd.Series(np.nan, index=res.index)
    pools = [np.ones(len(res), dtype=bool)] if seasons is None else [seasons == s for s in np.unique(seasons)]
    for m in pools:
        r = res[m]
        med = r.median()
        mad = (r - med).abs().median() * 1.4826
        z[m] = (r - med) / mad if mad > 0 else 0.0
    return z


def episodes(flag, z):
    """Group consecutive flagged days (gap of at most one day) into episodes."""
    days = flag.index[flag]
    out, cur = [], []
    for d in days:
        if cur and (d - cur[-1]).days > 2:
            out.append(cur)
            cur = []
        cur.append(d)
    if cur:
        out.append(cur)
    rows = []
    for c in out:
        zz = z[c]
        rows.append({"start": c[0].date(), "end": c[-1].date(), "n_flagged_days": len(c),
                     "peak_z": float(zz.loc[zz.abs().idxmax()]), "peak_day": zz.abs().idxmax().date(),
                     "direction": "surge" if zz.loc[zz.abs().idxmax()] > 0 else "drop"})
    return pd.DataFrame(rows)


# ================================================================ main
def main(args):
    OUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------- 1. forecasts
    if args.backtest:
        print("Building expanding-window out-of-sample forecasts ...")
        pred_df = backtest_predictions(args.fold_starts)
        source = "XGBoost backtest"
    else:
        path = Path(args.pred) if args.pred else DEFAULT_PRED
        pred_df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
        source = path.name
    pred_df = pred_df[["actual", "pred"] + [c for c in ["fold"] if c in pred_df]].dropna(subset=["actual", "pred"])
    print(f"Forecasts: {source}, {len(pred_df)} days, {pred_df.index[0].date()} to {pred_df.index[-1].date()}")

    panel = pd.read_csv(PANEL, parse_dates=["date"]).set_index("date").sort_index()
    d = pred_df.join(panel, how="left", rsuffix="_panel")

    # ---------------- 2. residuals and robust z-scores
    d["residual"] = d["actual"] - d["pred"]
    d["rel_residual"] = d["residual"] / d["pred"].clip(lower=1.0)
    res = d["rel_residual"] if args.residual == "relative" else d["residual"]
    d["season"] = season_of(d.index)
    d["z"] = robust_z(res, d["season"].values if args.scale_by == "season" else None)
    d["flag"] = d["z"].abs() > args.threshold
    d["direction"] = np.where(d["flag"], np.where(d["z"] > 0, "surge", "drop"), "")
    med = res.median()
    mad = (res - med).abs().median() * 1.4826
    unit = "relative residual" if args.residual == "relative" else "trips"
    print(f"\nResidual = {args.residual}; robust centre {med:.3f}, robust SD (1.4826 x MAD) {mad:.3f} ({unit})")
    print(f"Flagged |z| > {args.threshold}: {int(d['flag'].sum())} of {len(d)} days "
          f"({d['flag'].mean() * 100:.1f}%); surges {int((d['direction'] == 'surge').sum())}, "
          f"drops {int((d['direction'] == 'drop').sum())}")
    print("(for pure Gaussian noise about 0.3% of days would exceed 3 SD; "
          "heavier tails here are what the detector is looking for)")

    # ---------------- 3. context
    d["dow_name"] = d.index.day_name().str[:3]
    if "closure_miles" in d:
        d["closure_pctile_in_year"] = d.groupby(d.index.year)["closure_miles"].rank(pct=True)
    try:
        import holidays
        hol = holidays.country_holidays("US", subdiv="PA", years=sorted(set(d.index.year)))
        d["holiday_name"] = [hol.get(x.date(), "") for x in d.index]
    except Exception:
        d["holiday_name"] = ""

    def hints(row):
        h = []
        if row.get("is_holiday", 0) == 1 or row["holiday_name"]:
            h.append("holiday" + (f" ({row['holiday_name']})" if row["holiday_name"] else ""))
        if row.get("snowfall_sum", 0) > 0:
            h.append(f"snow {row['snowfall_sum']:.1f} cm")
        if row.get("precipitation_sum", 0) >= HEAVY_RAIN_MM:
            h.append(f"heavy rain {row['precipitation_sum']:.0f} mm")
        if row.get("temperature_2m_mean", 0) <= COLD_C:
            h.append(f"extreme cold {row['temperature_2m_mean']:.0f} C")
        if row.get("temperature_2m_mean", 0) >= HOT_C:
            h.append(f"extreme heat {row['temperature_2m_mean']:.0f} C")
        if row.get("windspeed_10m_max", 0) >= WIND_KMH:
            h.append(f"high wind {row['windspeed_10m_max']:.0f} km/h")
        if row.get("is_major_event", 0) == 1:
            h.append("major downtown event")
        elif row.get("is_event_day", 0) == 1:
            h.append("downtown event")
        if row.get("closure_pctile_in_year", 0) >= CLOSURE_TOP:
            h.append("heavy closure load")
        return "; ".join(h) if h else "none of the checked factors"

    d["hints"] = d.apply(hints, axis=1)

    cols = ["actual", "pred", "residual", "rel_residual", "z", "flag", "direction", "season", "dow_name",
            "holiday_name", "hints"] + [c for c in ["fold", "temperature_2m_mean", "precipitation_sum", "snowfall_sum",
                                                    "windspeed_10m_max", "is_holiday", "is_weekend", "closure_miles",
                                                    "closure_pctile_in_year", "is_event_day", "is_major_event"] if c in d]
    d[cols].round(4).to_csv(OUT_DIR / "anomalies_all_days.csv")
    flagged = d[d["flag"]].sort_values("z", key=np.abs, ascending=False)
    flagged[cols].round(4).to_csv(OUT_DIR / "anomalies_flagged.csv")

    if len(flagged):
        show = flagged.head(args.top).copy()
        show["date"] = show.index.date
        show["actual"] = show["actual"].round(0)
        show["pred"] = show["pred"].round(0)
        show["resid_%"] = (show["rel_residual"] * 100).round(0)
        show["z"] = show["z"].round(1)
        print(f"\n=== Top {min(args.top, len(flagged))} flagged days (by |z|) ===")
        print_table(show[["date", "dow_name", "actual", "pred", "resid_%", "z", "direction", "hints"]])
    else:
        print("\nNo day exceeds the threshold.")

    ep = episodes(d["flag"], d["z"])
    ep.to_csv(OUT_DIR / "anomaly_episodes.csv", index=False)
    if len(ep):
        print(f"\n{len(ep)} episodes (consecutive flagged days merged):")
        print_table(ep)

    # ---------------- 4. are flagged days over-represented under some context?
    ctx = {"holiday": (d.get("is_holiday", pd.Series(0, index=d.index)) == 1) | (d["holiday_name"] != ""),
           "weekend": d.get("is_weekend", pd.Series(0, index=d.index)) == 1,
           "snow day": d.get("snowfall_sum", pd.Series(0.0, index=d.index)) > 0,
           f"heavy rain (>= {HEAVY_RAIN_MM:.0f} mm)": d.get("precipitation_sum", pd.Series(0.0, index=d.index)) >= HEAVY_RAIN_MM,
           f"extreme cold (<= {COLD_C:.0f} C)": d.get("temperature_2m_mean", pd.Series(0.0, index=d.index)) <= COLD_C,
           f"high wind (>= {WIND_KMH:.0f} km/h)": d.get("windspeed_10m_max", pd.Series(0.0, index=d.index)) >= WIND_KMH,
           "closure load in top decile": d.get("closure_pctile_in_year", pd.Series(0.0, index=d.index)) >= CLOSURE_TOP,
           "downtown event day": d.get("is_event_day", pd.Series(0, index=d.index)) == 1,
           "major downtown event": d.get("is_major_event", pd.Series(0, index=d.index)) == 1}
    rows = []
    for name, m in ctx.items():
        m = m.fillna(False).astype(bool)
        if m.sum() == 0:
            continue
        a, b = int((m & d["flag"]).sum()), int((m & ~d["flag"]).sum())
        c_, e_ = int((~m & d["flag"]).sum()), int((~m & ~d["flag"]).sum())
        p = fisher_exact([[a, b], [c_, e_]])[1]
        rows.append({"context": name, "days": int(m.sum()), "flagged_days": a,
                     "flag_rate_%": 100 * a / m.sum(), "flag_rate_other_days_%": 100 * c_ / max(c_ + e_, 1),
                     "fisher_p": p})
    enr = pd.DataFrame(rows)
    enr.to_csv(OUT_DIR / "anomaly_enrichment.csv", index=False)
    if len(enr):
        print("\n=== Are flagged days over-represented in a context? (Fisher exact test) ===")
        print_table(enr, float_fmt="{:.3f}")
        print("With few flagged days these tests have little power; use them to guide the case studies,\n"
              "not as proof. Event rows appear only once the events CSV (04) has rows.")
    unexplained = int((flagged["hints"] == "none of the checked factors").sum()) if len(flagged) else 0
    print(f"\nFlagged days with none of the checked factors: {unexplained} of {len(flagged)} "
          "(candidates for manual investigation: news, university calendar, service outages)")

    # ---------------- 5. sensitivity to the threshold
    sens = pd.DataFrame({"threshold": [2.5, 3.0, 3.5, 4.0]})
    sens["flagged_days"] = [int((d["z"].abs() > t).sum()) for t in sens["threshold"]]
    sens["share_%"] = 100 * sens["flagged_days"] / len(d)
    sens.to_csv(OUT_DIR / "anomaly_threshold_sensitivity.csv", index=False)
    print("\n=== Threshold sensitivity ===")
    print_table(sens)

    # ---------------- 6. figures
    # (a) series with flags
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.plot(d.index, d["actual"], lw=0.9, label="actual")
    ax.plot(d.index, d["pred"], lw=0.9, alpha=0.8, label="forecast")
    for direction, color in [("surge", "C3"), ("drop", "C1")]:
        m = d["direction"] == direction
        ax.scatter(d.index[m], d.loc[m, "actual"], s=45, color=color, zorder=3, label=f"{direction} (|z| > {args.threshold:g})")
    ax.set(ylabel="Trips per day", title=f"Flagged days: {source}")
    ax.legend(ncol=4)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "flagged_timeseries.png", dpi=150)
    plt.close()

    # (b) distribution of robust z
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(d["z"].clip(-8, 8), bins=60)
    for t in (-args.threshold, args.threshold):
        ax.axvline(t, color="r", ls="--")
    ax.set(xlabel="Robust z-score of the residual (clipped at +-8)", ylabel="Days",
           title=f"Residual z-scores; red lines at +-{args.threshold:g}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "z_distribution.png", dpi=150)
    plt.close()

    # (c) case-study panels: window around the most extreme episodes
    top_eps = ep.reindex(ep["peak_z"].abs().sort_values(ascending=False).index).head(args.cases) if len(ep) else ep
    if len(top_eps):
        n = len(top_eps)
        ncol = 2 if n > 1 else 1
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(7 * ncol, 3.4 * nrow), squeeze=False)
        for ax, (_, e_) in zip(axes.ravel(), top_eps.iterrows()):
            c = pd.Timestamp(e_["peak_day"])
            w_ = d.loc[c - pd.Timedelta(days=14): c + pd.Timedelta(days=14)]
            ax.plot(w_.index, w_["actual"], marker="o", ms=3, lw=1, label="actual")
            ax.plot(w_.index, w_["pred"], lw=1, label="forecast")
            ax.axvspan(pd.Timestamp(e_["start"]) - pd.Timedelta(hours=12), pd.Timestamp(e_["end"]) + pd.Timedelta(hours=12),
                       color="r", alpha=0.15)
            ax.set_title(f"{c.date()} ({d.loc[c, 'dow_name']}), z = {e_['peak_z']:.1f}\n{d.loc[c, 'hints']}", fontsize=9)
            ax.tick_params(axis="x", labelrotation=30, labelsize=8)
        for ax in axes.ravel()[n:]:
            ax.axis("off")
        axes.ravel()[0].legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / "case_studies.png", dpi=150)
        plt.close()

    print(f"\nSaved tables to {OUT_DIR}/ and figures to {FIG_DIR}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=3.0, help="robust SDs (median / MAD)")
    ap.add_argument("--residual", choices=["relative", "absolute"], default="relative")
    ap.add_argument("--scale-by", choices=["global", "season"], default="global")
    ap.add_argument("--pred", default=None, help="CSV with columns date, actual, pred")
    ap.add_argument("--backtest", action="store_true", help="expanding-window out-of-sample XGBoost forecasts")
    ap.add_argument("--fold-starts", nargs="+", default=FOLD_STARTS)
    ap.add_argument("--top", type=int, default=15, help="rows of the flagged-day table to print")
    ap.add_argument("--cases", type=int, default=6, help="case-study panels to draw")
    main(ap.parse_args())

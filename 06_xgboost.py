"""
06_xgboost.py
Gradient-boosted (XGBoost) forecast of daily POGOH system trips, scored exactly like the
statistical baselines in 05_baselines.py.

Protocol (identical to 05_baselines.py)
    - Chronological split: test = 2025-11-01 to the end (last 12 months), via metrics.chrono_split.
    - One-step-ahead: the forecast for day t uses the ACTUAL trips up to day t-1 (lags, rolling
      means) and the OBSERVED weather / calendar / closure values of day t.
    - The model is fitted once; the test year is never used for tuning or early stopping.

Why the target is a ratio (default --target ratio)
    Trips grew from ~270/day (2022) to ~1,650/day (2026). Trees cannot predict above the values
    they saw in training, so a level target would cap the test-year forecasts. Instead the model
    predicts  y_t / base_t,  where base_t = mean of the previous 28 days (known at day t-1), and
    the forecast is  base_t * predicted_ratio.  Lag and rolling features are divided by the same
    base, so every feature is scale-free.  Use --target level for the raw-trips comparison.

Hyperparameter tuning
    Random search on a chronological validation split INSIDE the training years
    (train < 2024-11-01 <= validation < 2025-11-01), early stopping on validation error.
    The best configuration is then refitted on train + validation (number of trees = best
    iteration from the search) and evaluated once on the test year.

Feature groups (used for the ablation)
    lags      : trips 1, 2, 7, 14, 28 days ago; rolling means (7, 28) and rolling std (7)
    calendar  : day of week, month, weekend, holiday, sin/cos of day of year
    weather   : the four daily weather columns used by the baselines + previous-day rain and
                a 3-day snowfall sum
    closures  : closure_* columns from 04_build_event_features.py (constant columns dropped)
    events    : event_count / is_major_event ... (dropped automatically while they are all zero)
Station columns (trips_<station>) and `year` are NEVER used: the station counts of the same day
sum to the target (leakage) and `year` cannot extrapolate.

Input : data/processed/daily_panel_features.csv
Output: results/xgb_metrics.csv          (same columns as results/baseline_metrics.csv)
        results/xgb_predictions.csv      (date, actual, pred, residual: input for anomaly detection)
        results/xgb_ablation.csv         (leave-one-group-out WMAPE / MAE on the test year)
        results/xgb_importance.csv       (gain importance of every feature)
        results/xgb_best_params.json
        figures/xgb/*.png

Run from the repository root:   python 06_xgboost.py [--target ratio|level] [--n-iter 40]
                                                      [--objective squarederror|absoluteerror]
                                                      [--no-ablation] [--seed 0]
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from metrics import chrono_split, evaluate_by_season, same_weekday_baseline, print_table

# ---------------------------------------------------------------- settings
PANEL = Path("data/processed/daily_panel_features.csv")
BASELINES = Path("results/baseline_metrics.csv")   # written by 05_baselines.py (optional)
OUT_DIR = Path("results")
FIG_DIR = Path("figures/xgb")
TEST_START = "2025-11-01"    # same split as 05_baselines.py
VAL_START = "2024-11-01"     # validation = the 12 months before the test year
BASE_WINDOW = 28             # trailing window (days) of the ratio target's denominator
LAGS = [1, 2, 7, 14, 28]

WEATHER_COLS = ["temperature_2m_mean", "precipitation_sum", "snowfall_sum", "windspeed_10m_max"]
CALENDAR_BASE = ["is_weekend", "is_holiday"]

# random-search space (kept deliberately small: ~1,100 training days)
SPACE = {
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.1],
    "min_child_weight": [1, 3, 5, 10, 20],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.5, 0.7, 0.9, 1.0],
    "reg_lambda": [0.1, 1.0, 5.0, 10.0],
    "reg_alpha": [0.0, 0.1, 1.0],
    "gamma": [0.0, 0.5, 2.0],
}
MAX_TREES = 2000
EARLY_STOP = 50


# ---------------------------------------------------------------- features
def build_features(df, target):
    """Return (X, base, groups).  X has one row per day (NaN where lags are unavailable).
    base is the trailing mean used to rescale (all ones for target='level')."""
    y = df["system_trips"].astype(float)
    base = y.shift(1).rolling(BASE_WINDOW).mean() if target == "ratio" else pd.Series(1.0, index=df.index)

    feats, groups = {}, {}

    def add(name, series, group):
        feats[name] = series
        groups.setdefault(group, []).append(name)

    # --- lags (only information up to day t-1)
    for k in LAGS:
        lag = y.shift(k)
        add(f"lag_{k}", lag / base if target == "ratio" else lag, "lags")
    r7 = y.shift(1).rolling(7).mean()
    add("roll_mean_7", r7 / base if target == "ratio" else r7, "lags")
    if target == "level":   # in ratio mode the 28-day mean IS the scaling base (the feature would be 1)
        add("roll_mean_28", y.shift(1).rolling(28).mean(), "lags")
    add("roll_std_7", y.shift(1).rolling(7).std() / (base if target == "ratio" else 1.0), "lags")

    # --- calendar (known in advance)
    idx = df.index
    add("dow", pd.Series(idx.dayofweek, index=idx, dtype=float), "calendar")
    add("month", pd.Series(idx.month, index=idx, dtype=float), "calendar")
    doy = idx.dayofyear.values
    add("doy_sin", pd.Series(np.sin(2 * np.pi * doy / 365.25), index=idx), "calendar")
    add("doy_cos", pd.Series(np.cos(2 * np.pi * doy / 365.25), index=idx), "calendar")
    for c in CALENDAR_BASE:
        if c in df:
            add(c, df[c].astype(float), "calendar")

    # --- weather (observed values of day t stand in for a forecast, as in the baselines)
    for c in WEATHER_COLS:
        if c in df:
            add(c, df[c].astype(float), "weather")
    if "precipitation_sum" in df:
        add("precip_prev1", df["precipitation_sum"].shift(1), "weather")
    if "snowfall_sum" in df:
        add("snow_sum_3d", df["snowfall_sum"].rolling(3, min_periods=1).sum(), "weather")

    # --- closures and events from 04_build_event_features.py
    closure_cols = [c for c in df.columns if c.startswith(("closure_", "near_closure_", "near_fullclosure_"))]
    event_cols = [c for c in df.columns if c.startswith(("event_", "major_event_", "is_event", "is_major_event",
                                                         "near_event_"))]
    for c in closure_cols:
        add(c, df[c].astype(float), "closures")
    for c in event_cols:
        add(c, df[c].astype(float), "events")

    X = pd.DataFrame(feats, index=idx)

    # drop constant columns (e.g. event features while the events CSV is empty)
    const = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    if const:
        print(f"[note] dropping constant features: {const}")
        X = X.drop(columns=const)
        groups = {g: [c for c in cols if c not in const] for g, cols in groups.items()}
    groups = {g: cols for g, cols in groups.items() if cols}
    return X, base, groups


# ---------------------------------------------------------------- model helpers
def sample_weights(base, objective):
    """Make the training loss equal to the loss in TRIPS even though the target is a ratio."""
    if objective == "squarederror":
        return (base ** 2).values
    return base.values   # absoluteerror


def make_model(params, objective, seed, n_estimators=MAX_TREES, early_stopping=None):
    return xgb.XGBRegressor(
        objective=f"reg:{objective}",
        n_estimators=n_estimators,
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
        early_stopping_rounds=early_stopping,
        eval_metric="mae",
        **params,
    )


def fit_predict(X_tr, y_tr, w_tr, X_te, base_te, params, objective, seed, n_trees):
    m = make_model(params, objective, seed, n_estimators=n_trees)
    m.fit(X_tr, y_tr, sample_weight=w_tr)
    return m, pd.Series(m.predict(X_te) * base_te.values, index=X_te.index)


def wmape(y_true, y_pred):
    return float(np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100)


def search(X_tr, y_tr, w_tr, base_tr, X_va, y_va, w_va, base_va, trips_va, objective, n_iter, seed):
    """Random search; the score is the validation MAE in TRIPS (not in ratio units)."""
    rng = np.random.default_rng(seed)
    best = {"mae": np.inf}
    rows = []
    for i in range(n_iter):
        params = {k: v[rng.integers(len(v))] for k, v in SPACE.items()}
        params = {k: (v.item() if hasattr(v, "item") else v) for k, v in params.items()}
        m = make_model(params, objective, seed, early_stopping=EARLY_STOP)
        m.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_va, y_va)], sample_weight_eval_set=[w_va], verbose=False)
        pred = m.predict(X_va) * base_va.values
        mae = float(np.mean(np.abs(trips_va.values - pred)))
        n_trees = int(m.best_iteration) + 1
        rows.append({**params, "n_trees": n_trees, "val_MAE": mae, "val_WMAPE_%": wmape(trips_va.values, pred)})
        if mae < best["mae"]:
            best = {"mae": mae, "params": params, "n_trees": n_trees}
        print(f"  [{i + 1:>2}/{n_iter}] val MAE {mae:7.1f}  (best {best['mae']:7.1f})  trees {n_trees}", flush=True)
    return best, pd.DataFrame(rows).sort_values("val_MAE")


# ---------------------------------------------------------------- main
def main(args):
    df = pd.read_csv(PANEL, parse_dates=["date"]).set_index("date").sort_index().asfreq("D")
    print(f"Panel: {len(df)} days, {df.index[0].date()} to {df.index[-1].date()}")
    y = df["system_trips"].astype(float)

    X, base, groups = build_features(df, args.target)
    print(f"Target mode: {args.target} | objective: reg:{args.objective}")
    print("Feature groups:", {g: len(c) for g, c in groups.items()}, f"| {X.shape[1]} features")

    target = y / base if args.target == "ratio" else y
    usable = X.notna().all(axis=1) & target.notna() & base.notna()
    # the first BASE_WINDOW days have no complete lags; they are dropped from training only

    train_all, test = chrono_split(df, test_start=TEST_START)
    train, val, _ = chrono_split(df, test_start=TEST_START, val_start=VAL_START)
    i_tr, i_va, i_te = train.index[usable[train.index]], val.index[usable[val.index]], test.index
    assert usable[i_te].all(), "test rows contain NaN features"
    print(f"Train {len(i_tr)} days ({i_tr[0].date()}..{i_tr[-1].date()}), "
          f"validation {len(i_va)} ({i_va[0].date()}..{i_va[-1].date()}), "
          f"test {len(i_te)} ({i_te[0].date()}..{i_te[-1].date()})")

    w = pd.Series(sample_weights(base, args.objective), index=df.index)

    # ---------------- 1. tune on the validation year
    print(f"\n=== Random search ({args.n_iter} configs, validation = {i_va[0].date()}..{i_va[-1].date()}) ===")
    best, search_log = search(X.loc[i_tr], target[i_tr], w[i_tr], base[i_tr],
                              X.loc[i_va], target[i_va], w[i_va], base[i_va], y[i_va],
                              args.objective, args.n_iter, args.seed)
    print(f"\nBest validation MAE {best['mae']:.1f} with {best['n_trees']} trees:\n  {best['params']}")

    # ---------------- 2. refit on train + validation, evaluate once on the test year
    i_fit = i_tr.append(i_va)
    model, pred = fit_predict(X.loc[i_fit], target[i_fit], w[i_fit], X.loc[i_te], base[i_te],
                              best["params"], args.objective, args.seed, best["n_trees"])
    y_test = y[i_te]
    name = f"XGBoost ({args.target})"
    metrics_table = evaluate_by_season(y_test, pred, name)

    print("\n=== Test-year metrics, all days (one-step-ahead) ===")
    print_table(metrics_table[metrics_table["season"] == "All"])
    print("\n=== Test-year metrics by season ===")
    print_table(metrics_table[metrics_table["season"] != "All"])

    # compare with the statistical baselines if 05_baselines.py has been run
    if BASELINES.exists():
        base_tbl = pd.read_csv(BASELINES)
        both = pd.concat([base_tbl, metrics_table], ignore_index=True)
        both_all = both[both["season"] == "All"].sort_values("MAE")
        print("\n=== All models, test year, sorted by MAE ===")
        print_table(both_all)
        both.to_csv(OUT_DIR / "all_models_metrics.csv", index=False)
    else:
        sw = evaluate_by_season(y_test, same_weekday_baseline(y, 1).loc[i_te], "same_weekday")
        print("\n(results/baseline_metrics.csv not found; same-weekday reference:)")
        print_table(sw[sw["season"] == "All"])

    # ---------------- 3. save predictions, metrics, importance
    OUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    metrics_table.to_csv(OUT_DIR / "xgb_metrics.csv", index=False)
    out = pd.DataFrame({"actual": y_test, "pred": pred, "residual": y_test - pred})
    out.index.name = "date"
    out.to_csv(OUT_DIR / "xgb_predictions.csv")
    search_log.to_csv(OUT_DIR / "xgb_search_log.csv", index=False)
    with open(OUT_DIR / "xgb_best_params.json", "w") as f:
        json.dump({**best["params"], "n_trees": best["n_trees"], "target": args.target,
                   "objective": args.objective, "val_MAE": best["mae"]}, f, indent=2)

    imp = pd.Series(model.get_booster().get_score(importance_type="gain")).reindex(X.columns).fillna(0.0)
    imp = (imp / imp.sum() * 100).sort_values(ascending=False)
    group_of = {c: g for g, cols in groups.items() for c in cols}
    imp_df = pd.DataFrame({"feature": imp.index, "group": [group_of[c] for c in imp.index], "gain_%": imp.values})
    imp_df.to_csv(OUT_DIR / "xgb_importance.csv", index=False)
    print("\n=== Top 12 features by gain (%) ===")
    print(imp_df.head(12).round(2).to_string(index=False))
    print("\n=== Gain by feature group (%) ===")
    print(imp_df.groupby("group")["gain_%"].sum().sort_values(ascending=False).round(1).to_string())

    # ---------------- 4. leave-one-group-out ablation (same tuned hyperparameters)
    if not args.no_ablation:
        print("\n=== Ablation: drop one feature group, refit, score on the test year ===")
        rows = [{"dropped_group": "none (full model)", "MAE": float(np.mean(np.abs(y_test - pred))),
                 "WMAPE_%": wmape(y_test.values, pred.values)}]
        for g, cols in groups.items():
            keep = [c for c in X.columns if c not in cols]
            if not keep:
                continue
            _, p_g = fit_predict(X.loc[i_fit, keep], target[i_fit], w[i_fit], X.loc[i_te, keep], base[i_te],
                                 best["params"], args.objective, args.seed, best["n_trees"])
            rows.append({"dropped_group": g, "MAE": float(np.mean(np.abs(y_test - p_g))),
                         "WMAPE_%": wmape(y_test.values, p_g.values)})
        abl = pd.DataFrame(rows)
        abl["MAE_change"] = abl["MAE"] - abl["MAE"].iloc[0]
        abl.to_csv(OUT_DIR / "xgb_ablation.csv", index=False)
        print_table(abl)
        print("(a positive MAE_change means the group helped: removing it made the forecast worse)")
        if args.target == "ratio":
            print("note: in ratio mode the 28-day trailing mean still scales the target, so dropping 'lags'\n"
                  "      removes the day-to-day lag features only, not all recent-demand information.")

    # ---------------- 5. figures
    # (a) actual vs predicted over the test year
    fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    ax[0].plot(y_test.index, y_test, lw=1, label="actual")
    ax[0].plot(pred.index, pred, lw=1, label=name)
    ax[0].set(ylabel="Trips per day", title="XGBoost one-step-ahead forecast, test year")
    ax[0].legend()
    ax[1].bar(out.index, out["residual"], width=1.0, color="C3")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set(ylabel="Residual (actual - forecast)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "forecast_vs_actual.png", dpi=150)
    plt.close()

    # (b) feature importance
    top = imp_df.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top["feature"], top["gain_%"])
    ax.set(xlabel="Gain importance (%)", title="XGBoost feature importance (top 15)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_importance.png", dpi=150)
    plt.close()

    # (c) WMAPE by season, XGBoost vs baselines (if available)
    if BASELINES.exists():
        show = ["same_weekday", "ARX(1)", "ARMAX(9,2)", "ARIMAX(8,1,2)", name]
        seas = both[(both["season"] != "All") & both["model"].isin(show)].pivot(
            index="season", columns="model", values="WMAPE_%").reindex(["Winter", "Spring", "Summer", "Fall"])
        seas = seas[[m for m in show if m in seas.columns]]
        ax = seas.plot.bar(figsize=(9, 4.5), rot=0)
        ax.set(ylabel="WMAPE (%)", title="Test-year WMAPE by season")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "wmape_by_season.png", dpi=150)
        plt.close()

    print(f"\nSaved results to {OUT_DIR}/ and figures to {FIG_DIR}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["ratio", "level"], default="ratio")
    ap.add_argument("--objective", choices=["squarederror", "absoluteerror"], default="squarederror")
    ap.add_argument("--n-iter", type=int, default=40, help="random-search configurations")
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())

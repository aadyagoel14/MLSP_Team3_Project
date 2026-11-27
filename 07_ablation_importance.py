"""
07_ablation_importance.py
How much does each group of inputs contribute to the forecast, and under which conditions?
(Proposal: "feature ablation and importance analysis ... how different factors affect ridership
under different conditions".)  Run AFTER 06_xgboost.py: it reuses the feature builder and the tuned
hyperparameters saved in results/xgb_best_params.json.

Four analyses
  A. Ablation of the XGBoost model, repeated over several random seeds (subsampling makes a single
     fit noisy, so differences of a few trips are not meaningful on their own):
        - drop one feature group, refit, score on the test year;
        - keep only one feature group, refit, score on the test year.
  B. Permutation importance on the test year (feature level and group level): shuffle the
     columns of a fitted model and measure how much the MAE rises.
  C. TreeSHAP contributions (computed by XGBoost itself, no extra package): each day's forecast is
     split into additive contributions per feature, converted to TRIPS. Reported per group, per
     season, and for specific conditions (snow days, rainy days, weekends, holidays, temperature bins).
  D. The same group ablation for a regression with ARMA errors (the DSP side): drop the weather
     regressors, the calendar flags, or add the closure features.

Input : data/processed/daily_panel_features.csv, results/xgb_best_params.json
Output: results/xgb_ablation_multiseed.csv      (A)
        results/xgb_permutation_importance.csv  (B)
        results/xgb_shap_features.csv, results/xgb_shap_groups_by_season.csv,
        results/xgb_conditional_effects.csv     (C)
        results/dsp_ablation.csv                (D)
        figures/importance/*.png

Run from the repository root:
    python 07_ablation_importance.py [--seeds 5] [--repeats 20] [--arma-order 1,0,1] [--no-dsp]
"""
import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from metrics import chrono_split, season_of, print_table


def load_module_06():
    """06_xgboost.py starts with a digit, so it cannot be imported with a normal import statement."""
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "xgb06", here / "06_xgboost.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


x06 = load_module_06()

PANEL = x06.PANEL
BEST = Path("results/xgb_best_params.json")
OUT_DIR = Path("results")
FIG_DIR = Path("figures/importance")
TEST_START = x06.TEST_START
VAL_START = x06.VAL_START
SEASONS = ["Winter", "Spring", "Summer", "Fall"]


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def wmape(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.sum(np.abs(a - b)) / np.sum(np.abs(a)) * 100)


# ================================================================ D. ARMAX group ablation
def dsp_ablation(df, order):
    import warnings
    from statsmodels.tools.sm_exceptions import EstimationWarning
    from statsmodels.tsa.arima.model import ARIMA
    # "non-stationary starting parameters": harmless
    warnings.filterwarnings("ignore", category=EstimationWarning)

    y = df["system_trips"].astype(float)
    train_all, test = chrono_split(df, test_start=TEST_START)
    y_tr, y_te = y[train_all.index], y[test.index]

    W = [c for c in ["temperature_2m_mean", "precipitation_sum",
                     "snowfall_sum", "windspeed_10m_max"] if c in df]
    C = [c for c in ["is_weekend", "is_holiday"] if c in df]
    CL = [c for c in ["closure_permits_active",
                      "closure_segments", "closure_miles"] if c in df]
    configs = {
        "weather + calendar + closures": W + C + CL,
        "weather + calendar (baseline six)": W + C,
        "drop weather": C + CL,
        "drop calendar flags": W + CL,
        "no regressors (ARMA only)": [],
    }
    rows = []
    for name, cols in configs.items():
        try:
            if cols:
                mu, sd = df.loc[train_all.index, cols].mean(
                ), df.loc[train_all.index, cols].std().replace(0, 1)
                # standardised: helps the optimiser
                Xtr = ((df.loc[train_all.index, cols] - mu) / sd).astype(float)
                Xte = ((df.loc[test.index, cols] - mu) / sd).astype(float)
                fit = ARIMA(y_tr, exog=Xtr, order=order).fit(
                    method_kwargs={"method": "powell", "maxiter": 500, "xtol": 1e-8, "ftol": 1e-12})
                pred = fit.append(y_te, exog=Xte, refit=False).predict(
                    start=y_te.index[0], end=y_te.index[-1])
            else:
                fit = ARIMA(y_tr, order=order).fit(
                    method_kwargs={"method": "powell", "maxiter": 500, "xtol": 1e-8, "ftol": 1e-12})
                pred = fit.append(y_te, refit=False).predict(
                    start=y_te.index[0], end=y_te.index[-1])
            rows.append({"config": name, "n_regressors": len(cols), "MAE": mae(y_te, pred),
                         "WMAPE_%": wmape(y_te, pred)})
            print(
                f"  {name:<36s} MAE {rows[-1]['MAE']:7.1f}  WMAPE {rows[-1]['WMAPE_%']:5.2f}%")
        except Exception as e:   # keep going if one fit fails
            print(f"  [warn] {name}: {type(e).__name__}: {e}")
    out = pd.DataFrame(rows)
    if len(out):
        out["MAE_vs_baseline_six"] = out["MAE"] - out.loc[out["config"]
                                                          == "weather + calendar (baseline six)", "MAE"].iloc[0]
    return out


# ================================================================ main
def main(args):
    with open(BEST) as f:
        params = json.load(f)
    n_trees = params.pop("n_trees")
    target_mode = params.pop("target")
    objective = params.pop("objective")
    params.pop("val_MAE", None)
    print(
        f"Using tuned XGBoost settings from {BEST} (target={target_mode}, objective={objective}, trees={n_trees})")

    df = pd.read_csv(PANEL, parse_dates=["date"]).set_index(
        "date").sort_index().asfreq("D")
    y = df["system_trips"].astype(float)
    X, base, groups = x06.build_features(df, target_mode)
    target = y / base if target_mode == "ratio" else y
    usable = X.notna().all(axis=1) & target.notna() & base.notna()
    w = pd.Series(x06.sample_weights(base, objective), index=df.index)

    train, val, test = chrono_split(
        df, test_start=TEST_START, val_start=VAL_START)
    i_fit = train.index[usable[train.index]].append(
        val.index[usable[val.index]])
    i_te = test.index
    y_te, base_te, Xte_full = y[i_te], base[i_te], X.loc[i_te]
    seasons = pd.Series(season_of(i_te), index=i_te)
    OUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    def fit(cols, seed):
        return x06.fit_predict(X.loc[i_fit, cols], target[i_fit], w[i_fit], X.loc[i_te, cols], base_te,
                               params, objective, seed, n_trees)

    all_cols = list(X.columns)

    # ------------------------------------------------ A. multi-seed ablation
    print(
        f"\n=== A. XGBoost ablation, {args.seeds} seeds each (test-year MAE in trips) ===")
    configs = {"full model": all_cols}
    for g, cols in groups.items():
        rest = [c for c in all_cols if c not in cols]
        if rest:
            configs[f"drop {g}"] = rest
    for g, cols in groups.items():
        configs[f"only {g}"] = cols
    rows = []
    for name, cols in configs.items():
        maes, wm = [], []
        for s in range(args.seeds):
            _, p = fit(cols, s)
            maes.append(mae(y_te, p))
            wm.append(wmape(y_te, p))
        rows.append({"config": name, "n_features": len(cols), "MAE_mean": np.mean(maes), "MAE_sd": np.std(maes, ddof=1) if len(maes) > 1 else 0.0,
                     "WMAPE_mean_%": np.mean(wm)})
        print(
            f"  {name:<18s} MAE {np.mean(maes):7.1f} +- {rows[-1]['MAE_sd']:4.1f}", flush=True)
    abl = pd.DataFrame(rows)
    abl["MAE_change_vs_full"] = abl["MAE_mean"] - \
        abl.loc[abl["config"] == "full model", "MAE_mean"].iloc[0]
    abl.to_csv(OUT_DIR / "xgb_ablation_multiseed.csv", index=False)
    print_table(abl)
    print("A positive change for a 'drop' row means the group helped. Changes smaller than about two\n"
          "seed standard deviations (MAE_sd) should not be interpreted.")
    if target_mode == "ratio":
        print("Note: in ratio mode the trailing 28-day mean still scales the target, so 'drop lags' keeps that\n"
              "recent-demand information and 'only X' rows still use it for scaling.")

    # ------------------------------------------------ B. permutation importance (seed-0 full model)
    print("\n=== B. Permutation importance on the test year (MAE increase in trips) ===")
    model0, pred0 = fit(all_cols, 0)
    base_mae = mae(y_te, pred0)
    rng = np.random.default_rng(0)

    def perm_increase(cols):
        inc = []
        for _ in range(args.repeats):
            Xp = Xte_full.copy()
            order = rng.permutation(len(Xp))
            # one permutation for the whole group keeps rows consistent
            Xp[cols] = Xte_full[cols].values[order]
            inc.append(mae(y_te, model0.predict(
                Xp) * base_te.values) - base_mae)
        return np.mean(inc), np.std(inc, ddof=1)

    prow = []
    for g, cols in groups.items():
        m_, s_ = perm_increase(cols)
        prow.append({"level": "group", "name": g, "group": g,
                    "MAE_increase": m_, "sd": s_})
    for c in all_cols:
        m_, s_ = perm_increase([c])
        prow.append({"level": "feature", "name": c, "group": next(g for g, cs in groups.items() if c in cs),
                     "MAE_increase": m_, "sd": s_})
    perm = pd.DataFrame(prow).sort_values(
        ["level", "MAE_increase"], ascending=[True, False])
    perm.to_csv(OUT_DIR / "xgb_permutation_importance.csv", index=False)
    print(f"(full-model test MAE: {base_mae:.1f})")
    print_table(perm[perm["level"] == "group"][["name", "MAE_increase", "sd"]])
    print("\nTop 10 features:")
    print_table(perm[perm["level"] == "feature"].head(
        10)[["name", "group", "MAE_increase", "sd"]])
    print("Correlated features share credit, so read group-level numbers first.")

    # ------------------------------------------------ C. TreeSHAP in trips
    print("\n=== C. TreeSHAP contributions, in trips per day ===")
    contrib = model0.get_booster().predict(xgb.DMatrix(
        Xte_full), pred_contribs=True)   # last column = bias
    shap_trips = pd.DataFrame(
        contrib[:, :-1] * base_te.values[:, None], columns=all_cols, index=i_te)
    bias_trips = contrib[:, -1] * base_te.values
    add_err = np.abs(shap_trips.sum(axis=1).values +
                     bias_trips - pred0.values).max()
    print(
        f"additivity check (contributions + bias = forecast): max error {add_err:.4f} trips")

    feat = shap_trips.abs().mean().sort_values(ascending=False)
    feat_df = pd.DataFrame({"feature": feat.index,
                            "group": [next(g for g, cs in groups.items() if c in cs) for c in feat.index],
                            "mean_abs_shap_trips": feat.values})
    feat_df.to_csv(OUT_DIR / "xgb_shap_features.csv", index=False)

    group_shap = pd.DataFrame(
        {g: shap_trips[cols].sum(axis=1) for g, cols in groups.items()})
    by_season = pd.DataFrame(
        {s: group_shap[seasons == s].abs().mean() for s in SEASONS})
    by_season["All"] = group_shap.abs().mean()
    by_season.index.name = "group"
    by_season.to_csv(OUT_DIR / "xgb_shap_groups_by_season.csv")
    print("\nMean |contribution| of each group (trips per day):")
    print(by_season.round(1).to_string())

    # conditional effects: signed mean contribution of a feature under a specific condition
    def cond_rows():
        out = []

        def add(feature, label, mask):
            if feature in shap_trips and mask.sum() > 0:
                out.append({"feature": feature, "condition": label, "n_days": int(mask.sum()),
                            "mean_contribution_trips": float(shap_trips.loc[mask, feature].mean())})

        panel_te = df.loc[i_te]
        if "snowfall_sum" in df:
            add("snowfall_sum", "snow day (snowfall > 0)",
                panel_te["snowfall_sum"] > 0)
            add("snowfall_sum", "no snow", panel_te["snowfall_sum"] <= 0)
        if "precipitation_sum" in df:
            add("precipitation_sum", "rain >= 5 mm",
                panel_te["precipitation_sum"] >= 5)
            add("precipitation_sum", "dry (< 0.1 mm)",
                panel_te["precipitation_sum"] < 0.1)
        if "temperature_2m_mean" in df:
            t = panel_te["temperature_2m_mean"]
            add("temperature_2m_mean", "below 0 C", t < 0)
            add("temperature_2m_mean", "0 to 10 C", (t >= 0) & (t < 10))
            add("temperature_2m_mean", "10 to 20 C", (t >= 10) & (t < 20))
            add("temperature_2m_mean", "20 C and above", t >= 20)
        if "is_weekend" in df:
            add("is_weekend", "weekend", panel_te["is_weekend"] == 1)
        if "is_holiday" in df:
            add("is_holiday", "holiday", panel_te["is_holiday"] == 1)
        return pd.DataFrame(out)

    cond = cond_rows()
    cond.to_csv(OUT_DIR / "xgb_conditional_effects.csv", index=False)
    print("\nConditional effects (mean SHAP contribution of one feature on specific days, trips):")
    print_table(cond)

    # ------------------------------------------------ D. DSP ablation
    dsp = pd.DataFrame()
    if not args.no_dsp:
        order = tuple(int(v) for v in args.arma_order.split(","))
        print(
            f"\n=== D. Regression with ARMA errors, order {order}: regressor-group ablation (test year) ===")
        dsp = dsp_ablation(df, order)
        dsp.to_csv(OUT_DIR / "dsp_ablation.csv", index=False)
        print_table(dsp)

    # ------------------------------------------------ figures
    # 1. ablation bars
    fig, axes = plt.subplots(1, 2 if len(dsp) else 1,
                             figsize=(13 if len(dsp) else 7, 4.5))
    axes = np.atleast_1d(axes)
    drop = abl[abl["config"].str.startswith("drop")]
    axes[0].barh(drop["config"], drop["MAE_change_vs_full"],
                 xerr=2 * drop["MAE_sd"], color="C0", capsize=3)
    axes[0].axvline(0, color="k", lw=0.6)
    axes[0].set(xlabel="Change in test MAE vs full model (trips)  [error bar = 2 seed SD]",
                title="XGBoost: drop one feature group")
    if len(dsp):
        d = dsp[dsp["config"] != "weather + calendar (baseline six)"]
        axes[1].barh(d["config"], d["MAE_vs_baseline_six"], color="C1")
        axes[1].axvline(0, color="k", lw=0.6)
        axes[1].set(xlabel="Change in test MAE vs the baseline six regressors (trips)",
                    title=f"Regression with ARMA{order} errors: regressor groups")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ablation.png", dpi=150)
    plt.close()

    # 2. SHAP by group and season
    ax = by_season[SEASONS].T.plot.bar(figsize=(9, 4.5), rot=0)
    ax.set(ylabel="Mean |contribution| (trips per day)",
           title="XGBoost: what drives the forecast, by season (TreeSHAP)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_groups_by_season.png", dpi=150)
    plt.close()

    # 3. permutation importance, top features
    top = perm[perm["level"] == "feature"].head(15)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top["name"], top["MAE_increase"], xerr=top["sd"], capsize=2)
    ax.set(xlabel="MAE increase when shuffled (trips)",
           title="Permutation importance, test year (top 15)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "permutation_importance.png", dpi=150)
    plt.close()

    # 4. dependence plots
    dep = [(c, lab) for c, lab in [("temperature_2m_mean", "Mean temperature (C)"), ("snowfall_sum", "Snowfall (cm)"),
                                   ("precipitation_sum", "Precipitation (mm)")] if c in shap_trips]
    if dep:
        fig, axes = plt.subplots(1, len(dep), figsize=(4.6 * len(dep), 4))
        axes = np.atleast_1d(axes)
        for ax, (c, lab) in zip(axes, dep):
            ax.scatter(df.loc[i_te, c], shap_trips[c], s=8, alpha=0.5)
            ax.axhline(0, color="k", lw=0.6)
            ax.set(xlabel=lab, ylabel="Contribution to forecast (trips)")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "shap_dependence_weather.png", dpi=150)
        plt.close()

    print(f"\nSaved tables to {OUT_DIR}/ and figures to {FIG_DIR}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5,
                    help="random seeds per ablation configuration")
    ap.add_argument("--repeats", type=int, default=20,
                    help="shuffles per permutation-importance estimate")
    ap.add_argument("--arma-order", default="1,0,1",
                    help="order p,d,q for the DSP ablation (e.g. 9,0,2)")
    ap.add_argument("--no-dsp", action="store_true",
                    help="skip the ARMAX regressor-group ablation")
    main(ap.parse_args())

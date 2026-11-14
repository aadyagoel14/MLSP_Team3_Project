"""
05_baselines.py
Baseline forecasting models for the daily POGOH system trips.

Models (in this order):
     0. Same-weekday baseline : prediction = trips on the same weekday one week earlier
     1. AR(1)                 : ARIMA order (1, 0, 0)
     2. AR(9)                 : ARIMA order (9, 0, 0)
     3. ARMA(1,1)             : ARIMA order (1, 0, 1)
     4. ARMA(9,2)             : ARIMA order (9, 0, 2)
     5. ARX(1)                : ARIMA order (1, 0, 0) + exogenous regressors
     6. ARX(9)                : ARIMA order (9, 0, 0) + exogenous regressors
     7. ARMAX(1,1)            : ARIMA order (1, 0, 1) + exogenous regressors
     8. ARMAX(9,2)            : ARIMA order (9, 0, 2) + exogenous regressors
     9. ARIMAX(1,1,1)         : ARIMA order (1, 1, 1) + exogenous regressors
    10. ARIMAX(8,1,2)         : ARIMA order (8, 1, 2) + exogenous regressors

Every model family is fitted twice:
    - with the smallest orders (p = 1, q = 1), and
    - with the orders read from the ACF and PACF of the training data
      (see 05a_order_selection.py): p = 9 and q = 2 for the models without differencing,
      p = 8 and q = 2 for ARIMAX.
ARIMAX is differenced once (d = 1) in both versions.

All ten models use the same statsmodels class (ARIMA). Only `order` and `exog` change.
With exogenous regressors statsmodels fits a regression with AR / ARMA / ARIMA errors:
    y_t = const + X_t * beta + u_t      where u_t follows the AR / ARMA / ARIMA model
The models with d = 0 include a constant. ARIMAX (d = 1) has no constant, because a
constant cannot be estimated after differencing (this is the statsmodels default).

Forecast protocol: one-step-ahead on the test year.
    - The parameters are estimated once on the training set and are never updated.
    - The prediction for day t uses the actual trips up to day t-1.
    - The models with exogenous regressors also use the exogenous values of day t
      (observed weather and calendar flags; observed weather stands in for a forecast).

Input : data/processed/daily_panel_features.csv
Output: results/baseline_metrics.csv  (plus tables printed to the console)

Run from the repository root:  python 05_baselines.py
"""
from pathlib import Path

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from metrics import chrono_split, evaluate_by_season, same_weekday_baseline, print_table

# ---------------------------------------------------------------- settings
PANEL = Path("data/processed/daily_panel_features.csv")
OUT_DIR = Path("results")
TEST_START = "2025-11-01"   # test set = the last 12 months (README_midterm.md)

# Optimiser settings, used for every fit below: Powell's method with strict stopping rules.
# (The statsmodels default optimiser, L-BFGS, stops slightly before the likelihood maximum
# on the raw trip counts.)
MAX_ITER = 500   # iteration limit of the optimiser
X_TOL = 1e-10    # tolerance of each line search
F_TOL = 1e-14    # relative change of the objective at which the optimiser stops

# the six exogenous regressors listed in README_midterm.md, section 4
EXOG_COLS = [
    "temperature_2m_mean",
    "precipitation_sum",
    "snowfall_sum",
    "windspeed_10m_max",
    "is_weekend",
    "is_holiday",
]

# ---------------------------------------------------------------- 1. load the data
df = pd.read_csv(PANEL, parse_dates=["date"]).set_index("date")
df = df.asfreq("D")   # mark the index as daily (the panel has no missing days, so no rows are added)
print(f"Panel: {df.shape[0]} days, {df.index[0].date()} to {df.index[-1].date()}")

# ---------------------------------------------------------------- 2. chronological split
train, test = chrono_split(df, test_start=TEST_START)

y_train = train["system_trips"].astype(float)
y_test = test["system_trips"].astype(float)
X_train = train[EXOG_COLS].astype(float)
X_test = test[EXOG_COLS].astype(float)

first_test_day = y_test.index[0]
last_test_day = y_test.index[-1]

print(f"Train: {len(y_train)} days, {y_train.index[0].date()} to {y_train.index[-1].date()}")
print(f"Test : {len(y_test)} days, {first_test_day.date()} to {last_test_day.date()}")

# ---------------------------------------------------------------- 3. same-weekday baseline
# The baseline is computed on the full series, so the first test week can look back
# into the last training week. Then only the test days are kept.
y_all = df["system_trips"].astype(float)
pred_same_weekday = same_weekday_baseline(y_all, weeks_back=1)
pred_same_weekday = pred_same_weekday.loc[y_test.index]

# ---------------------------------------------------------------- 4. AR(1)
print("\n=== AR(1) ===")

# fit on the training set only
# ("method" inside method_kwargs selects the numerical optimiser)
model_ar1 = ARIMA(y_train, order=(1, 0, 0))
fit_ar1 = model_ar1.fit(method_kwargs={"method": "powell", "maxiter": MAX_ITER, "xtol": X_TOL, "ftol": F_TOL})
print("converged:", fit_ar1.mle_retvals["converged"])

# one-step-ahead predictions on the test year:
# append the test observations to the fitted model WITHOUT re-estimating the parameters,
# then read the one-step-ahead predictions of the test days
fit_ar1_appended = fit_ar1.append(y_test, refit=False)
pred_ar1 = fit_ar1_appended.predict(start=first_test_day, end=last_test_day)

# ---------------------------------------------------------------- 5. AR(9)
print("\n=== AR(9) ===")

model_ar9 = ARIMA(y_train, order=(9, 0, 0))
fit_ar9 = model_ar9.fit(method_kwargs={"method": "powell", "maxiter": MAX_ITER, "xtol": X_TOL, "ftol": F_TOL})
print("converged:", fit_ar9.mle_retvals["converged"])

fit_ar9_appended = fit_ar9.append(y_test, refit=False)
pred_ar9 = fit_ar9_appended.predict(start=first_test_day, end=last_test_day)

# ---------------------------------------------------------------- 6. ARMA(1,1)
print("\n=== ARMA(1,1) ===")

model_arma11 = ARIMA(y_train, order=(1, 0, 1))
fit_arma11 = model_arma11.fit(method_kwargs={"method": "powell", "maxiter": MAX_ITER, "xtol": X_TOL, "ftol": F_TOL})
print("converged:", fit_arma11.mle_retvals["converged"])

fit_arma11_appended = fit_arma11.append(y_test, refit=False)
pred_arma11 = fit_arma11_appended.predict(start=first_test_day, end=last_test_day)

# ---------------------------------------------------------------- 7. ARMA(9,2)
print("\n=== ARMA(9,2) ===")

model_arma92 = ARIMA(y_train, order=(9, 0, 2))
fit_arma92 = model_arma92.fit(method_kwargs={"method": "powell", "maxiter": MAX_ITER, "xtol": X_TOL, "ftol": F_TOL})
print("converged:", fit_arma92.mle_retvals["converged"])

fit_arma92_appended = fit_arma92.append(y_test, refit=False)
pred_arma92 = fit_arma92_appended.predict(start=first_test_day, end=last_test_day)

# ---------------------------------------------------------------- 8. ARX(1)
print("\n=== ARX(1) ===")

model_arx1 = ARIMA(y_train, exog=X_train, order=(1, 0, 0))
fit_arx1 = model_arx1.fit(method_kwargs={"method": "powell", "maxiter": MAX_ITER, "xtol": X_TOL, "ftol": F_TOL})
print("converged:", fit_arx1.mle_retvals["converged"])

# coefficient table of the exogenous regressors
coef_arx1 = pd.DataFrame({
    "coef": fit_arx1.params[EXOG_COLS],
    "std_err": fit_arx1.bse[EXOG_COLS],
    "p_value": fit_arx1.pvalues[EXOG_COLS],
})
print(coef_arx1.round(4))

# the exogenous regressors of the test days must be passed together with the test observations
fit_arx1_appended = fit_arx1.append(y_test, exog=X_test, refit=False)
pred_arx1 = fit_arx1_appended.predict(start=first_test_day, end=last_test_day)

# ---------------------------------------------------------------- 9. ARX(9)
print("\n=== ARX(9) ===")

model_arx9 = ARIMA(y_train, exog=X_train, order=(9, 0, 0))
fit_arx9 = model_arx9.fit(method_kwargs={"method": "powell", "maxiter": MAX_ITER, "xtol": X_TOL, "ftol": F_TOL})
print("converged:", fit_arx9.mle_retvals["converged"])

coef_arx9 = pd.DataFrame({
    "coef": fit_arx9.params[EXOG_COLS],
    "std_err": fit_arx9.bse[EXOG_COLS],
    "p_value": fit_arx9.pvalues[EXOG_COLS],
})
print(coef_arx9.round(4))

fit_arx9_appended = fit_arx9.append(y_test, exog=X_test, refit=False)
pred_arx9 = fit_arx9_appended.predict(start=first_test_day, end=last_test_day)

# ---------------------------------------------------------------- 10. ARMAX(1,1)
print("\n=== ARMAX(1,1) ===")

model_armax11 = ARIMA(y_train, exog=X_train, order=(1, 0, 1))
fit_armax11 = model_armax11.fit(method_kwargs={"method": "powell", "maxiter": MAX_ITER, "xtol": X_TOL, "ftol": F_TOL})
print("converged:", fit_armax11.mle_retvals["converged"])

coef_armax11 = pd.DataFrame({
    "coef": fit_armax11.params[EXOG_COLS],
    "std_err": fit_armax11.bse[EXOG_COLS],
    "p_value": fit_armax11.pvalues[EXOG_COLS],
})
print(coef_armax11.round(4))

fit_armax11_appended = fit_armax11.append(y_test, exog=X_test, refit=False)
pred_armax11 = fit_armax11_appended.predict(start=first_test_day, end=last_test_day)

# ---------------------------------------------------------------- 11. ARMAX(9,2)
print("\n=== ARMAX(9,2) ===")

model_armax92 = ARIMA(y_train, exog=X_train, order=(9, 0, 2))
fit_armax92 = model_armax92.fit(method_kwargs={"method": "powell", "maxiter": MAX_ITER, "xtol": X_TOL, "ftol": F_TOL})
print("converged:", fit_armax92.mle_retvals["converged"])

coef_armax92 = pd.DataFrame({
    "coef": fit_armax92.params[EXOG_COLS],
    "std_err": fit_armax92.bse[EXOG_COLS],
    "p_value": fit_armax92.pvalues[EXOG_COLS],
})
print(coef_armax92.round(4))

fit_armax92_appended = fit_armax92.append(y_test, exog=X_test, refit=False)
pred_armax92 = fit_armax92_appended.predict(start=first_test_day, end=last_test_day)

# ---------------------------------------------------------------- 12. ARIMAX(1,1,1)
print("\n=== ARIMAX(1,1,1) ===")

model_arimax111 = ARIMA(y_train, exog=X_train, order=(1, 1, 1))
fit_arimax111 = model_arimax111.fit(method_kwargs={"method": "powell", "maxiter": MAX_ITER, "xtol": X_TOL, "ftol": F_TOL})
print("converged:", fit_arimax111.mle_retvals["converged"])

coef_arimax111 = pd.DataFrame({
    "coef": fit_arimax111.params[EXOG_COLS],
    "std_err": fit_arimax111.bse[EXOG_COLS],
    "p_value": fit_arimax111.pvalues[EXOG_COLS],
})
print(coef_arimax111.round(4))

fit_arimax111_appended = fit_arimax111.append(y_test, exog=X_test, refit=False)
pred_arimax111 = fit_arimax111_appended.predict(start=first_test_day, end=last_test_day)

# ---------------------------------------------------------------- 13. ARIMAX(8,1,2)
print("\n=== ARIMAX(8,1,2) ===")

model_arimax812 = ARIMA(y_train, exog=X_train, order=(8, 1, 2))
fit_arimax812 = model_arimax812.fit(method_kwargs={"method": "powell", "maxiter": MAX_ITER, "xtol": X_TOL, "ftol": F_TOL})
print("converged:", fit_arimax812.mle_retvals["converged"])

coef_arimax812 = pd.DataFrame({
    "coef": fit_arimax812.params[EXOG_COLS],
    "std_err": fit_arimax812.bse[EXOG_COLS],
    "p_value": fit_arimax812.pvalues[EXOG_COLS],
})
print(coef_arimax812.round(4))

fit_arimax812_appended = fit_arimax812.append(y_test, exog=X_test, refit=False)
pred_arimax812 = fit_arimax812_appended.predict(start=first_test_day, end=last_test_day)

# ---------------------------------------------------------------- 14. evaluate on the test year
# evaluate_by_season returns one row per season plus one row for all test days ("All")
tables = [
    evaluate_by_season(y_test, pred_same_weekday, "same_weekday"),
    evaluate_by_season(y_test, pred_ar1, "AR(1)"),
    evaluate_by_season(y_test, pred_ar9, "AR(9)"),
    evaluate_by_season(y_test, pred_arma11, "ARMA(1,1)"),
    evaluate_by_season(y_test, pred_arma92, "ARMA(9,2)"),
    evaluate_by_season(y_test, pred_arx1, "ARX(1)"),
    evaluate_by_season(y_test, pred_arx9, "ARX(9)"),
    evaluate_by_season(y_test, pred_armax11, "ARMAX(1,1)"),
    evaluate_by_season(y_test, pred_armax92, "ARMAX(9,2)"),
    evaluate_by_season(y_test, pred_arimax111, "ARIMAX(1,1,1)"),
    evaluate_by_season(y_test, pred_arimax812, "ARIMAX(8,1,2)"),
]
metrics_table = pd.concat(tables, ignore_index=True)

print("\n=== Test-year metrics, all days (one-step-ahead) ===")
print_table(metrics_table[metrics_table["season"] == "All"])

print("\n=== Test-year metrics by season (one-step-ahead) ===")
print_table(metrics_table[metrics_table["season"] != "All"])

# ---------------------------------------------------------------- 15. save the metrics table
OUT_DIR.mkdir(exist_ok=True)
metrics_table.to_csv(OUT_DIR / "baseline_metrics.csv", index=False)
print(f"\nSaved {OUT_DIR / 'baseline_metrics.csv'}")

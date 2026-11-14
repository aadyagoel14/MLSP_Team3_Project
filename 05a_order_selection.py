"""
05a_order_selection.py
Choose the ARIMA orders p and q used in 05_baselines.py from the ACF and PACF of the
training data.

Reading rule: starting at lag 1, count how many consecutive lags are outside the 95% band
(+-1.96 / sqrt(n)).
    p = this count for the PACF
    q = this count for the ACF

Two series are examined. Only the training set is used; the test year is not looked at.
    1. the raw daily trips                -> p of the models without differencing (d = 0)
    2. the first difference of the trips  -> p and q of the model with differencing (d = 1)

The ACF of the raw trips decays very slowly and stays outside the band at every lag, so q
cannot be read from it. The models without differencing therefore borrow q from the
differenced series.

Input : data/processed/daily_panel_features.csv
Output: figures/baselines/acf_pacf_train.png  (plus the readings printed to the console)

Run from the repository root:  python 05a_order_selection.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf

from metrics import chrono_split

# ---------------------------------------------------------------- settings
PANEL = Path("data/processed/daily_panel_features.csv")
OUT_DIR = Path("figures/baselines")
TEST_START = "2025-11-01"   # same split as 05_baselines.py
N_LAGS = 35                 # number of lags examined (five weeks)

# ---------------------------------------------------------------- 1. load the training data
df = pd.read_csv(PANEL, parse_dates=["date"]).set_index("date")
df = df.asfreq("D")
train, test = chrono_split(df, test_start=TEST_START)

y_train = train["system_trips"].astype(float)
dy_train = y_train.diff().dropna()   # first difference: trips of a day minus trips of the day before

print(f"Train: {len(y_train)} days, {y_train.index[0].date()} to {y_train.index[-1].date()}")

# ---------------------------------------------------------------- 2. raw trips
band_raw = 1.96 / np.sqrt(len(y_train))
acf_raw = acf(y_train, nlags=N_LAGS)
pacf_raw = pacf(y_train, nlags=N_LAGS, method="ywm")

# count the consecutive lags outside the band, starting at lag 1
p_raw = 0
for lag in range(1, N_LAGS + 1):
    if abs(pacf_raw[lag]) > band_raw:
        p_raw = p_raw + 1
    else:
        break

q_raw = 0
for lag in range(1, N_LAGS + 1):
    if abs(acf_raw[lag]) > band_raw:
        q_raw = q_raw + 1
    else:
        break

# ---------------------------------------------------------------- 3. first difference of the trips
band_diff = 1.96 / np.sqrt(len(dy_train))
acf_diff = acf(dy_train, nlags=N_LAGS)
pacf_diff = pacf(dy_train, nlags=N_LAGS, method="ywm")

p_diff = 0
for lag in range(1, N_LAGS + 1):
    if abs(pacf_diff[lag]) > band_diff:
        p_diff = p_diff + 1
    else:
        break

q_diff = 0
for lag in range(1, N_LAGS + 1):
    if abs(acf_diff[lag]) > band_diff:
        q_diff = q_diff + 1
    else:
        break

# ---------------------------------------------------------------- 4. print the readings
values = pd.DataFrame({
    "acf_raw": acf_raw[1:15],
    "pacf_raw": pacf_raw[1:15],
    "acf_diff": acf_diff[1:15],
    "pacf_diff": pacf_diff[1:15],
}, index=range(1, 15))
values.index.name = "lag"

print("\n=== ACF and PACF, lags 1 to 14 ===")
print(values.round(3))

print("\n=== Readings (consecutive lags outside the 95% band, starting at lag 1) ===")
print(f"Raw trips        : band +-{band_raw:.3f}, PACF {p_raw} lags, ACF {q_raw} lags")
if q_raw == N_LAGS:
    print(f"                   (the ACF is outside the band at all {N_LAGS} lags, so q cannot be read)")
print(f"Differenced trips: band +-{band_diff:.3f}, PACF {p_diff} lags, ACF {q_diff} lags")

print("\n=== Orders for 05_baselines.py ===")
print(f"Models without differencing (AR, ARMA, ARX, ARMAX): p = {p_raw}, q = {q_diff} (q borrowed from the differenced series)")
print(f"Model with differencing (ARIMAX)                  : p = {p_diff}, d = 1, q = {q_diff}")

# ---------------------------------------------------------------- 5. save the figure
# bartlett_confint=False draws the same constant band (+-1.96 / sqrt(n)) that the rule uses
fig, axes = plt.subplots(2, 2, figsize=(13, 7))
plot_acf(y_train, lags=N_LAGS, zero=False, bartlett_confint=False, ax=axes[0, 0],
         title="ACF - raw trips (training set)")
plot_pacf(y_train, lags=N_LAGS, zero=False, method="ywm", ax=axes[0, 1],
          title="PACF - raw trips (training set)")
plot_acf(dy_train, lags=N_LAGS, zero=False, bartlett_confint=False, ax=axes[1, 0],
         title="ACF - first difference of the trips (training set)")
plot_pacf(dy_train, lags=N_LAGS, zero=False, method="ywm", ax=axes[1, 1],
          title="PACF - first difference of the trips (training set)")
axes[1, 0].set_xlabel("lag (days)")
axes[1, 1].set_xlabel("lag (days)")
plt.tight_layout()

OUT_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_DIR / "acf_pacf_train.png", dpi=150)
plt.close()
print(f"\nSaved {OUT_DIR / 'acf_pacf_train.png'}")

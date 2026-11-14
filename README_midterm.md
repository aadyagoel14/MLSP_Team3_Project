# Midterm Handoff: POGOH Ridership Forecasting (Team 3)

Status as of Nov 13, 2026. This covers what is done, what is left, and what goes in the midterm report.

Midterm requirements (from our proposal): initial results (cleaned and merged dataset, exploratory analysis, same-weekday baseline with MAE/RMSE), baselines, future methodology, and literature review.

---

## 1. What is DONE

| Item              | Script / file                | Notes                                                                               |
| ----------------- | ---------------------------- | ----------------------------------------------------------------------------------- |
| Download trips    | `01_download_trip_data.py`   | 54 monthly files, May 2022 to Oct 2026                                              |
| Download weather  | `02_download_weather.py`     | Open-Meteo daily, Apr 2022 to Oct 2026                                              |
| Clean + merge     | `03_clean_and_merge.py`      | 1,644 days, no gaps, no missing values; `is_holiday` now populated                  |
| DSP analysis      | `03b_spectral_analysis.py`   | Periodogram, ACF/PACF, ADF (TA comment 2)                                           |
| EDA               | `03c_eda.py`                 | 8 figures in `figures/eda/`                                                         |
| Closure features  | `04_build_event_features.py` | Permits of 30 days or less; miles, segments, full-closure versions                  |
| Metrics helper    | `metrics.py`                 | MAE, RMSE, MAPE, WMAPE, chronological split, same-weekday baseline, by-season table |
| Literature review | Claude Doc                   | Draft exists; needs trimming and reference check                                    |

Use `data/processed/daily_panel_features.csv` for modeling. It is the daily panel plus closure and event columns, regenerated after the holiday fix.

### Key facts and numbers (ready to cite)

- **Data:** 1,644 days, 2022-05-02 to 2026-10-31. Trips cleaned from 1,876,752 to 1,755,478 rows (about 6.4% removed for invalid duration, a few hundred for bad timestamps or duplicates).
- **TA's April 2022 date:** WPRDC's earliest POGOH file is **May 2022**. State this in the report.
- **Growth:** mean daily trips 267 (2022, partial), 544 (2023), 1,206 (2024), 1,499 (2025), 1,656 (2026, partial).
- **Daily trips:** mean 1,068, median 931, min 3, max 3,991.
- **Weekly pattern (weak):** Thursday highest (1,132), Sunday lowest (924), about a 20% swing.
- **Periodogram:** strongest peaks at about 183 and 329 days (semi-annual and annual, the 329-day peak is mixed with the growth trend). A small but real 7.0-day peak (about 3.5% of the top peak).
- **Stationarity (ADF):** raw p = 0.066 (borderline); first difference, seasonal difference (lag 7), and both together all p approximately 0. So `d = 1` suffices.
- **Weather:** snow days average 519 trips against 1,130 without snow (n = 168); wet days 967 against 1,134 dry. Correlations with trips: temperature +0.32, wind -0.21, snowfall -0.12, precipitation -0.10.
- **Calendar:** `is_weekend` -0.07, `is_holiday` -0.06, `dow` -0.04. The weak weekend effect fits campus-area stations dominating.
- **Stations:** the 10 top stations correlate 0.75 to 0.95 with the system total. S Bouquet & Sennott St is the busiest (161,913 trips); O'Hara St & University Place is the least tied to the total (0.75).
- **Closures:** after dropping permits longer than 30 days (27,285 of 54,415 permits kept), averages are about 158 active permits and 15 closed miles per day. Correlations with trips are 0.40 to 0.45, which is **very likely a shared upward trend and construction season, not a causal effect**.

---

## 2. What is LEFT

### Code (blocks the results section)

1. **`05_baselines.py`** (not written yet)
   - Same-weekday baseline, scored with `metrics.evaluate_by_season`.
   - ARX, ARMAX (partner), SARIMAX.
   - Coefficient table for the exogenous regressors.
   - A second run with closure and event columns.
2. **Decide on training start:** all data versus 2023-onward (2022 is launch ramp-up). Report both.
3. **Holidays check (30 seconds):** run `df["is_holiday"].sum()`; expect about 45 to 50 total. `03` printed a harmless `FutureWarning` about `isin`; the nonzero correlation shows holidays are populated.

### Data (optional for the midterm, needed for the final)

4. **Events:** fill `data/raw/events/downtown_events.csv` with about 20 to 40 verifiable major downtown events (`major=1`). Until then `event_count` and `major_event_count` are zero. Source: downtownpittsburgh.com events calendar.
5. **Station coordinates:** put POGOH station locations (WPRDC) in `data/raw/stations.csv` as `station_name,lat,lon` for station-proximity features (500 m buffers).

### Writing

6. Report sections (see section 3).
7. Shorten the literature review and verify every reference. Several came from preprints, theses, or short papers, so confirm each one exists before citing.

---

## 3. What to include in the report

### 3.1 Dataset and preprocessing

- Sources: POGOH trip data (WPRDC), Open-Meteo weather, DOMI Street Closures (WPRDC), Downtown Pittsburgh events calendar.
- Date range May 2022 to Oct 2026 (note the TA's April date does not exist on WPRDC).
- Cleaning steps: schema harmonization across monthly files, timestamp validation, de-duplication, duration filter (1 min to 24 h), daily aggregation system-wide and for the 10 busiest stations, missing calendar days filled with zero (none were missing), weather merge, calendar flags.
- Table: rows before and after each step (1,876,752 to 1,876,747 to 1,876,230 to 1,755,478).
- One-line justification for the duration filter and a note that about 6.4% of trips were removed.

### 3.2 Exploratory analysis (pick 4 to 6 figures)

| Figure                       | File                                    | Supports                    |
| ---------------------------- | --------------------------------------- | --------------------------- |
| Overview + growth            | `figures/eda/01_overview_growth.png`    | Growth and level shift      |
| Month by year                | `figures/eda/02_monthly_by_year.png`    | Annual seasonality          |
| Day-of-week / month boxplots | `figures/eda/03_dow_month_boxplots.png` | Weekly and seasonal pattern |
| Weather scatter              | `figures/eda/06_weather_scatter.png`    | Weather effects             |
| Correlation matrix           | `figures/eda/07_correlation_matrix.png` | Feature relationships       |
| Top stations                 | `figures/eda/08_top_stations.png`       | Station demand              |

Also include a small table of summary statistics and the yearly means.

### 3.3 DSP analysis (answers TA comment 2)

- Periodogram: `figures/spectral/periodogram.png`. Explain the annual and semi-annual peaks and the weak 7-day peak.
- ACF / PACF: `figures/spectral/acf_pacf_raw.png` and `acf_pacf_seasonal_diff_s7.png` (and `acf_pacf_seasonal_plus_first_diff.png`).
- ADF table (raw versus differenced).
- State how these choose SARIMAX orders: `d = 1`, seasonal period 7, with the annual cycle handled by regressors.

### 3.4 Event and closure features (answers TA comment 3)

- Describe how qualitative data became numbers: daily `closure_permits_active`, `closure_segments`, `closure_miles` (and full-closure versions), `event_count`, `is_major_event`, station-proximity buffers (planned, 500 m).
- Explain the 30-day permit filter and why (long permits stay "active" for years and just track the trend).
- State the trend caveat for closure correlations.

### 3.5 Initial results and baselines

- **Same-weekday baseline** with MAE, RMSE, and WMAPE (overall and by season).
- **ARX / ARMAX / SARIMAX** results table with the same metrics on the same split (train before 2025-11-01, test Nov 2025 to Oct 2026).
- Coefficient table: temperature, precipitation, snowfall, wind, weekend, holiday (and closure/event columns in the second run).
- Short discussion: which weather effects are significant, whether SARIMAX beats the same-weekday baseline, and where errors are worst (winter, low-demand days).
- Metrics: MAE, RMSE, and **WMAPE** (TA comment 4). Explain WMAPE is used over plain MAPE because near-zero winter days make MAPE unstable.

### 3.6 Future methodology (final report)

- Gradient-boosted regression with lag, calendar, weather, and event features; feature ablation and importance analysis.
- Anomaly detection: day-level forecast residual beyond 3 robust standard deviations (median and MAD); investigate flagged days against closures, events, and weather.
- Optional per-station detector to separate citywide disruptions (such as snowstorms) from local ones.
- Risks: heavy-tailed winter days (report errors by season), growth and ramp-up (training window choice), closure features confounded by trend.

### 3.7 Literature review

- Four themes: bikeshare demand forecasting, gradient boosting for demand with exogenous features, interpretable statistical baselines, anomaly detection and event attribution.
- Keep to the length your instructor specifies. Verify each reference before submission.

### 3.8 Updated proposal text (TA comments)

- Confirm dataset dates (May 2022 to Oct 2026).
- Add the DSP framing (periodogram/FFT, ACF/PACF).
- Add how events and closures become features.
- Add MAPE/WMAPE to the metrics.

---

## 4. Modeling notes for whoever writes `05`

```python
import pandas as pd, numpy as np
from metrics import chrono_split, evaluate, evaluate_by_season, same_weekday_baseline, print_table

df = pd.read_csv("data/processed/daily_panel_features.csv", parse_dates=["date"]).set_index("date")
train, test = chrono_split(df, test_start="2025-11-01")   # last 12 months, every season once
```

- Never shuffle. Always use `chrono_split`.
- Regressors: `temperature_2m_mean`, `precipitation_sum`, `snowfall_sum`, `windspeed_10m_max`, `is_weekend`, `is_holiday`. Add month dummies or Fourier terms for the annual cycle.
- Use `d = 1`. Read `(p, q)` from the ACF/PACF plots. A reasonable SARIMAX start is `(1,1,1)(1,0,1,7)`.
- Consider `log1p(system_trips)` because variance grows with level.
- Observed weather stands in for a weather forecast; say so in the report.
- Test closure features with trend and month terms, or on differenced data.

## 5. Known harmless warnings

- `03`: pandas datetime-format, `SettingWithCopy`, and `isin` FutureWarnings.
- `03c`: matplotlib `labels` deprecation.
- `04`: datetime-format warning, and a correlation warning for event columns until the events CSV has rows.
- `05`: numpy `RuntimeWarning` lines (`divide by zero`, `overflow`, `invalid value encountered in matmul`) printed before the tables. They appear when numpy runs on Apple's Accelerate library (pip-installed numpy on an Apple Silicon Mac), come from the start-value step inside statsmodels, and do not change any number.

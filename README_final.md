# Final Report Handoff: POGOH Ridership Forecasting (Team 3)

Continuation of `README_midterm.md`. Status as of Nov 26, 2026. Everything in sections 1 to 3 comes from the real run outputs of `06_xgboost.py`, `07_ablation_importance.py` and `08_anomaly_detection.py` (default settings). Anything not yet run is labelled **(proposed)** or **(not run)**.

Final deliverable (from the proposal): gradient-boosted model, comparison with baseline, anomaly detection, and event analysis.

---

## 1. What is DONE since the midterm

| Item                                                                               | Script                      | Output                                                                                                                                                                                                     |
| ---------------------------------------------------------------------------------- | --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| XGBoost forecast (tuned, one-step-ahead, same split and metrics as the DSP models) | `06_xgboost.py`             | `results/xgb_metrics.csv`, `xgb_predictions.csv`, `xgb_best_params.json`, `xgb_search_log.csv`, `xgb_importance.csv`, `xgb_ablation.csv`, `figures/xgb/`                                                   |
| Ablation, permutation importance, TreeSHAP, DSP regressor ablation                 | `07_ablation_importance.py` | `results/xgb_ablation_multiseed.csv`, `xgb_permutation_importance.csv`, `xgb_shap_features.csv`, `xgb_shap_groups_by_season.csv`, `xgb_conditional_effects.csv`, `dsp_ablation.csv`, `figures/importance/` |
| Anomaly detection (median/MAD, 3 robust SD) with context checks                    | `08_anomaly_detection.py`   | `results/anomalies_all_days.csv`, `anomalies_flagged.csv`, `anomaly_episodes.csv`, `anomaly_enrichment.csv`, `anomaly_threshold_sensitivity.csv`, `figures/anomalies/`                                     |

### How the XGBoost model is set up (for the Methods section)

- **Protocol identical to the DSP baselines.** Test year = 1 Nov 2025 to 31 Oct 2026 (365 days). One-step-ahead: the forecast for day t uses actual trips up to day t-1 and observed weather, calendar and closure values of day t. Parameters are fitted once and never updated.
- **Ratio target.** Trips grew from about 270 to about 1,650 a day, and trees cannot predict above what they saw in training. The model predicts `trips / (mean of previous 28 days)` and the forecast is scaled back. Lag and rolling features are divided by the same mean. Training loss is weighted so it equals the squared error in trips, not in ratio units.
- **Features (25).** Lags 7 (trips 1, 2, 7, 14, 28 days ago, 7-day rolling mean, 7-day rolling SD), calendar 6 (day of week, month, weekend, holiday, sin/cos of day of year), weather 6 (temperature, precipitation, snowfall, wind, previous-day rain, 3-day snow sum), closures 6.
  - Left out on purpose: the per-station `trips_*` columns (same-day station counts add up to the target, which would be leakage) and `year` (trees cannot extrapolate it).
  - Dropped automatically because they are constant: `closure_any` and all event columns (the events file is still empty).
- **Tuning.** Random search, 40 configurations, validation = Nov 2024 to Oct 2025 (inside the training years, test year never seen), early stopping. Best validation MAE 196.6. Then refit on train + validation and score once on the test year.
- **Chosen hyperparameters:** max_depth 4, learning_rate 0.03, min_child_weight 20, subsample 0.6, colsample_bytree 1.0, reg_lambda 0.1, reg_alpha 0.1, gamma 0.5, 179 trees (`results/xgb_best_params.json`).

---

## 2. Results (test year, one-step-ahead)

### 2.1 Overall accuracy: XGBoost ties the best DSP models, it does not beat them

| Model                      | MAE (trips) | RMSE (trips) | MAPE (%) | WMAPE (%) |
| -------------------------- | ----------- | ------------ | -------- | --------- |
| Same weekday               | 433.9       | 582.7        | 94.6     | 27.52     |
| ARX(1)                     | 239.2       | 318.3        | 40.4     | 15.17     |
| ARMAX(9,2)                 | 224.5       | **298.4**    | 44.9     | 14.24     |
| ARIMAX(8,1,2)              | 224.3       | 299.4        | 45.2     | 14.23     |
| **XGBoost (ratio target)** | **223.2**   | 304.7        | 46.1     | **14.16** |

Read it honestly:

- XGBoost cuts the same-weekday error by 49% (MAE 433.9 to 223.2, WMAPE 27.5% to 14.2%).
- Against the best DSP models the MAE gain is about 1 trip (0.5%), well inside the seed noise (see 2.3, MAE standard deviation of 3 to 4 trips across seeds). XGBoost's RMSE is actually worse (304.7 against 298.4).
- The proposal expected gradient boosting to beat the baselines by capturing nonlinear effects. On this data it does not, overall. Report it as such; a tie with a more flexible model is a legitimate finding.

The DSP rows above are from the midterm run. Your partner is re-tuning the DSP hyperparameters, so **refresh these rows from the final `results/baseline_metrics.csv` before submitting.**

### 2.2 By season: the two families are complementary

WMAPE (%), and MAE in trips in brackets:

| Model         | Winter         | Spring     | Summer         | Fall          |
| ------------- | -------------- | ---------- | -------------- | ------------- |
| Same weekday  | 62.9 (439)     | 31.8 (525) | 20.5 (334)     | 18.9 (438)    |
| ARX(1)        | **22.4** (156) | 17.2 (283) | 13.9 (227)     | 12.5 (289)    |
| ARMAX(9,2)    | 24.1 (168)     | 16.5 (272) | 12.8 (209)     | 10.7 (248)    |
| ARIMAX(8,1,2) | 24.3 (169)     | 16.6 (273) | 12.8 (208)     | 10.6 (246)    |
| **XGBoost**   | 30.4 (212)     | 18.0 (297) | **11.0** (180) | **8.8** (204) |

- XGBoost is best in **summer** (MAE 180 against 208, about 14% lower) and **fall** (204 against 246, about 17% lower).
- XGBoost is worse in **winter** (212 against 168, about 25% higher; ARX(1) is best at 156) and **spring** (297 against 272, about 9% higher).
- Winter is where the anomaly analysis finds the biggest misses (section 2.5), so these two findings support each other.
- **Why** XGBoost loses in winter is not tested. Plausible explanations to state as hypotheses: few winters in the training data and extreme cold/snow combinations that trees cannot extrapolate to; low winter volumes making ratios unstable. See section 4 for checks.

### 2.3 What drives the forecast (ablation, 5 seeds)

Test MAE with one feature group removed (full model 221.5 ± 3.8):

| Configuration | MAE (mean ± SD over seeds) | Change vs full                 |
| ------------- | -------------------------- | ------------------------------ |
| Full model    | 221.5 ± 3.8                | n/a                            |
| Drop lags     | 317.3 ± 5.4                | **+95.9** (helps most)         |
| Drop weather  | 269.3 ± 1.1                | **+47.9**                      |
| Drop calendar | 236.7 ± 1.3                | +15.3                          |
| Drop closures | 213.5 ± 3.4                | **-8.0** (forecast got better) |
| Only lags     | 277.4 ± 0.9                |                                |
| Only calendar | 337.2 ± 0.9                |                                |
| Only weather  | 425.7 ± 2.9                |                                |
| Only closures | 576.1 ± 2.8                |                                |

- **Lags matter most, then weather, then calendar.** Closures do not help; removing them improved test MAE by about 8 trips (roughly two seed SDs), so the honest statement is "no evidence that closures improve the forecast".
- Caveat for the write-up: in ratio mode the trailing 28-day mean still scales the target, so "drop lags" removes the day-to-day lag features only, not all recent-demand information. "Only X" rows also still use that scaling.

Permutation importance on the test year (MAE increase when shuffled): lags **348 ± 22**, weather **89 ± 8**, calendar **48 ± 6**, closures **7 ± 3**. Top single features: `lag_1` 225, `precipitation_sum` 68, `roll_mean_7` 65, `temperature_2m_mean` 28, `month` 21, `dow` 18. Correlated features share credit, so quote group-level numbers.

**The same conclusion holds for the DSP family** (regression with ARMA(1,1) errors, `dsp_ablation.csv`):

| Configuration                     | MAE   | WMAPE (%) | MAE vs baseline six |
| --------------------------------- | ----- | --------- | ------------------- |
| Weather + calendar + closures     | 228.3 | 14.48     | -1.0                |
| Weather + calendar (baseline six) | 229.3 | 14.54     | 0                   |
| Drop weather                      | 274.7 | 17.42     | **+45.4**           |
| Drop calendar flags               | 234.0 | 14.84     | +4.7                |
| No regressors (ARMA only)         | 285.8 | 18.13     | +56.5               |

Dropping the weather regressors costs about 45 trips in the DSP model and about 48 in XGBoost. Two different model families agree on the size of the weather effect, which makes a strong sentence for the report. Closures are again negligible (-1.0).

### 2.4 How the factors act, in trips per day (TreeSHAP)

Mean absolute contribution of each group to a day's forecast:

| Group    | Winter | Spring | Summer | Fall | All |
| -------- | ------ | ------ | ------ | ---- | --- |
| Lags     | 289    | 328    | 231    | 348  | 299 |
| Weather  | 73     | 163    | 111    | 171  | 130 |
| Calendar | 37     | 76     | 35     | 84   | 58  |
| Closures | 18     | 31     | 28     | 59   | 34  |

Conditional effects (mean contribution of one feature on specific days, `xgb_conditional_effects.csv`):

| Feature       | Condition      | Days | Mean contribution (trips) |
| ------------- | -------------- | ---- | ------------------------- |
| Precipitation | rain >= 5 mm   | 86   | **-268**                  |
| Precipitation | dry (< 0.1 mm) | 135  | +89                       |
| Temperature   | below 0 C      | 62   | -75                       |
| Temperature   | 0 to 10 C      | 90   | -73                       |
| Temperature   | 10 to 20 C     | 105  | +5                        |
| Temperature   | 20 C and above | 108  | +61                       |
| Snowfall      | snow day       | 53   | -29                       |
| Snowfall      | no snow        | 312  | +2                        |
| Weekend       | weekend        | 105  | 0.0                       |
| Holiday       | holiday        | 12   | -0.7                      |

- Rain is the largest single weather effect (about 270 fewer trips on rainy days, about 90 more on dry days). Warm days add about 60 trips; days below 10 C remove about 75.
- The snowfall feature's own contribution is small (-29) because temperature, rain and the seasonal terms already carry most of the winter signal. The model is not saying snow does not matter.
- The weekend contribution is exactly 0.0 because the model splits on `dow` (day of week) instead, which carries the same information. Do not write "weekends have no effect"; the weak weekly pattern is already documented in the midterm EDA (Thursday 1,133, Sunday 924).
- Weather matters most in spring and fall (about 165 to 170 trips), less in winter (73). Contributions are in trips and winter volume is low, so this does not mean winter is less weather-sensitive in relative terms.

### 2.5 Anomaly detection (default: XGBoost test year, relative residual, 3 robust SD)

- Robust centre -0.011, robust SD 0.161 (relative residual). **28 of 365 days flagged (7.7%)**, 15 surges and 13 drops, grouped into 12 episodes. A Gaussian would flag about 0.3%, so the residuals are heavy-tailed, concentrated in winter.
- Threshold sensitivity (`anomaly_threshold_sensitivity.csv`): 41 days at 2.5, **28 at 3.0**, 20 at 3.5, 14 at 4.0.

**Where the flags are** (Fisher exact tests, `anomaly_enrichment.csv`):

| Context                  | Days | Flagged | Flag rate (%) | Flag rate on other days (%) | p       |
| ------------------------ | ---- | ------- | ------------- | --------------------------- | ------- |
| Snow day                 | 53   | 12      | 22.6          | 5.1                         | < 0.001 |
| Extreme cold (<= -5 C)   | 30   | 11      | 36.7          | 5.1                         | < 0.001 |
| Weekend                  | 105  | 9       | 8.6           | 7.3                         | 0.67    |
| Heavy rain (>= 10 mm)    | 42   | 4       | 9.5           | 7.4                         | 0.55    |
| Closure load, top decile | 38   | 1       | 2.6           | 8.3                         | 0.34    |
| Holiday                  | 13   | 0       | 0             | 8.0                         | 0.61    |

Flagged days cluster in snow and extreme cold, not in holidays, weekends, rain or heavy closure load. Because weather is already a model input, this means the model **under-fits extreme winter conditions**, which matches its weak winter score. Event days cannot be tested yet (events file empty).

**Most extreme flagged days** (actual / forecast in trips):

| Date               | Day       | Actual    | Forecast  | z           | Hint from checks                                  |
| ------------------ | --------- | --------- | --------- | ----------- | ------------------------------------------------- |
| 2026-01-12         | Mon       | 1,459     | 459       | **+13.6**   | none of the checked factors                       |
| 2025-11-30         | Sun       | 729       | 268       | +10.7       | snow 0.2 cm                                       |
| 2026-01-13         | Tue       | 1,543     | 609       | +9.6        | none of the checked factors                       |
| 2026-01-26         | Mon       | 11        | 630       | -6.0        | snow 0.2 cm; extreme cold -10 C                   |
| 2026-02-20         | Fri       | 1,648     | 850       | +5.9        | none of the checked factors                       |
| 2026-01-27         | Tue       | 13        | 294       | -5.9        | extreme cold -13 C                                |
| 2026-01-25         | Sun       | 41        | 464       | -5.6        | snow 20.9 cm; heavy rain 30 mm; extreme cold -8 C |
| 2026-01-28         | Wed       | 48        | 249       | -4.9        | extreme cold -14 C                                |
| 2025-12-15 / 12-14 | Mon / Sun | 151 / 113 | 685 / 505 | -4.8 / -4.7 | extreme cold; snow                                |
| 2025-12-01         | Mon       | 1,701     | 984       | +4.6        | none of the checked factors                       |

Episodes (`anomaly_episodes.csv`): 25 Jan to 1 Feb 2026 is an **8-day drop** (trips 11 to 64 a day; a snowstorm of 20.9 cm followed by -8 to -17 C, which looks like a near-total shutdown), 11 to 16 Jan 2026 is a 5-day surge, 2 to 5 Jan is a 3-day surge, 30 Nov to 1 Dec 2025 is a 2-day surge, 14 to 15 Dec 2025 is a 2-day drop. **9 of 28 flagged days have none of the checked factors.**

---

## 3. The story for the final report (proposed)

1. **Forecasting.** A same-weekday baseline is easily beaten. Statistical regression with ARMA errors and a tuned gradient-boosted model reach the same overall accuracy (MAE about 223 to 225, WMAPE about 14.2%). The families are complementary: XGBoost is better in summer and fall, the linear models in winter and spring.
2. **Drivers.** Recent demand (lags) is the strongest signal, weather the second, calendar the third. Weather is worth about 45 to 48 trips of MAE in both model families, and rain is the biggest single effect. Closures add nothing measurable.
3. **Closures and events.** Closures, even after the 30-day filter, do not improve either family and are confounded with trend and season; this is a negative result worth reporting. Event analysis is still open (section 4).
4. **Anomalies.** Flagged days concentrate in extreme cold and snow, where the model under-predicts the shutdown of the system (trips of 11 to 64 against forecasts of 250 to 630). A handful of unexplained surges (mid-January 2026, 30 Nov to 1 Dec 2025, 20 Feb 2026) are the interesting case studies.
5. **Limitations and next steps** (section 6).

---

## 4. What is LEFT

### Code and runs

| #   | Task                                                                                                                                                                         | Why                                                                                                                                                                                                                    | Owner (proposed)        |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| 1   | `git pull` your partner's `results/baseline_metrics.csv`, then re-run `06_xgboost.py`                                                                                        | `06` printed "baseline_metrics.csv not found", so `all_models_metrics.csv` and `figures/xgb/wmape_by_season.png` (XGBoost against the DSP models) were not produced. The search is seeded, so results should reproduce | Aadya                   |
| 2   | Partner saves the DSP forecasts as `results/dsp_predictions.csv` (columns `date,actual,pred`)                                                                                | Lets `08` run on the DSP residuals: `python 08_anomaly_detection.py --pred results/dsp_predictions.csv`. Comparing which days both families flag is a robust anomaly definition                                        | Partner                 |
| 3   | Run `08` robustness variants and compare flag counts: `--backtest` (three out-of-sample years instead of one), `--scale-by season`, `--residual absolute`, `--threshold 3.5` | 7.7% flagged is high; winter probably dominates because low counts (11 trips) give huge relative residuals. **(not run)**                                                                                              | Aadya                   |
| 4   | Re-run `07` after the DSP re-tuning, with `--arma-order 9,0,2` to match the final DSP model                                                                                  | Keeps the two ablations on the same model                                                                                                                                                                              | Aadya                   |
| 5   | Optional: `09` per-station anomalies (does a snowstorm flag many stations at once?)                                                                                          | "If time allows" item in the proposal                                                                                                                                                                                  | Skip unless time allows |
| 6   | Add a SARIMAX baseline (weekly seasonal period 7)                                                                                                                            | Promised in the proposal and flagged as missing in the midterm                                                                                                                                                         | Partner                 |

### Data (this is the biggest gap)

- **Events.** `data/raw/events/downtown_events.csv` is still empty, so `event_count` and `is_major_event` are all zero and the "event analysis" in the proposal is not done. Minimum viable version: enter the major downtown events for **Nov 2025 to Oct 2026** (marathon, festivals, home games, big concerts; `major=1`), then re-run `04` -> `06` -> `07` -> `08`. Once filled, `08` automatically reports whether flagged days are enriched on event days.
- **Investigate the unexplained flags by hand.** Take the dates in section 2.5 and check news, the university calendar, POGOH service notices, and weather reports for each. 2026-01-12 and 2026-01-13 are the first to check: 1,459 and 1,543 trips in mid-winter against forecasts of about 460 and 610 is a tripling. Before calling it a real surge, **check the trip data for those days** (`trips_clean.parquet`: number of rider types, stations, trip durations) for a data artifact, a promotion, or a system test.
- **Station coordinates** (`data/raw/stations.csv`) are still needed for the 500 m buffers; only needed if `09` or event proximity is attempted.

### Analysis ideas (proposed, not run)

- **Ensemble.** Average the XGBoost and best DSP forecasts. Because each wins different seasons, the average will probably beat both. It is cheap to test with `xgb_predictions.csv` and the DSP predictions.
- **Winter fix.** Test an extreme-cold flag (for example mean temperature <= -5 C) or minimum temperature as features. It targets the model's biggest weakness. Treat any gain as a hypothesis check, and report it as an extension, not as the main model.
- **Why XGBoost loses in winter.** Compare XGBoost and ARMAX residuals on the Jan 2026 cold-snap days.
- **Multi-day horizon.** All results are one day ahead. A 7-day-ahead comparison would be a stronger test but is optional.

---

## 5. What to include in the final report

Suggested outline (reuse the midterm sections 3.1 to 3.4 of `README_midterm.md` for data, EDA, DSP analysis and features, and refresh any number the DSP re-tuning changes).

1. **Introduction and related work** (partner): bikeshare demand forecasting, gradient boosting for demand with exogenous features, robust residual-based anomaly detection. Fix or confirm every reference.
2. **Data and preprocessing**: as in the midterm; add the 30-day closure filter and the events limitation.
3. **Exploratory and spectral analysis**: as in the midterm (annual and semi-annual peaks, weak 7-day peak, `d = 1`).
4. **Models**:
   - same-weekday baseline, ARX/ARMAX/ARIMAX/SARIMAX (partner);
   - XGBoost: ratio target and why, features, leakage exclusions, tuning protocol, chosen hyperparameters.
5. **Results**:
   - overall table (2.1) and by-season table (2.2), plus `figures/xgb/wmape_by_season.png`;
   - forecast against actual: `figures/xgb/forecast_vs_actual.png`.
6. **What drives ridership**:
   - ablation table (2.3) and `figures/importance/ablation.png` (XGBoost and DSP side by side);
   - SHAP by season (`shap_groups_by_season.png`), weather dependence (`shap_dependence_weather.png`), permutation importance (`permutation_importance.png`);
   - the conditional-effects table (2.4).
7. **Anomaly detection and event analysis**:
   - definition (median/MAD, 3 robust SD, relative residual) and threshold sensitivity;
   - `figures/anomalies/flagged_timeseries.png`, `z_distribution.png`;
   - enrichment table (2.5);
   - **3 to 5 case studies** from `figures/anomalies/case_studies.png`: the 25 Jan to 1 Feb 2026 shutdown (explained by snow and extreme cold), the 11 to 16 Jan 2026 surge (unexplained, data check first), the 30 Nov to 1 Dec 2025 surge, one of 20 Feb or 19 Mar 2026, the 14 to 15 Dec 2025 drop;
   - the event enrichment result, once the events file is filled.
8. **Discussion**: the section 3 story; XGBoost ties DSP overall and the reasons, closures as a negative result, flags concentrated in extreme winter weather.
9. **Limitations** (section 6) and **future work** (ensemble, extreme-cold features, multi-day horizon, per-station detector).
10. **References** and a data and code availability note (repository link).

Metrics to report everywhere: MAE, RMSE, **WMAPE** (headline percentage error) and MAPE. Keep the explanation of why plain MAPE misleads (XGBoost's MAPE of 46% sits next to the best WMAPE of 14.2%, because a few very low winter days dominate MAPE).

---

## 6. Limitations to state in the report

- **One test year.** Differences of a few trips (XGBoost against ARMAX/ARIMAX overall) are not meaningful; seed noise alone is 3 to 4 trips of MAE.
- **One-step-ahead with actual lags.** `lag_1` alone is worth about 225 trips of MAE in the permutation test, so the model is largely "yesterday plus weather". Multi-day forecasts would be harder.
- **Observed weather replaces a forecast** in both families, making both look better than in operation.
- **Closures are confounded with trend and season** and add nothing measurable; events are not yet available.
- **Anomaly detection depends on the forecast model.** Flags are model residuals, so a model weakness (extreme winter) produces flags. The one-step-ahead structure also anchors forecasts to the previous day, so a rebound after a shutdown can look like a surge. Read neighbouring episodes together. Weather being a model input means flagged snow and cold days are under-fit days, not unexplained days.
- **Heavy tails and winter.** 7.7% of days exceed 3 robust SD; relative residuals on very low counts (11 to 64 trips) are extreme. State the sensitivity table and the robustness variants.
- **Single-city, single-system data** (POGOH, Pittsburgh), so effect sizes (for example -268 trips on rainy days) do not transfer to other systems.
- **Training data starts in the launch ramp-up** (2022); a shorter training window was not compared.

---

## 7. Run order and repo notes

```
pip install -r requirements.txt            # includes xgboost
python 03_clean_and_merge.py
python 03b_spectral_analysis.py
python 03c_eda.py
python 04_build_event_features.py          # re-run after filling the events CSV
python 05a_order_selection.py
python 05_baselines.py                     # partner; saves results/baseline_metrics.csv
python 06_xgboost.py                       # tuned XGBoost, ablation, importance
python 07_ablation_importance.py           # needs results/xgb_best_params.json from 06
python 08_anomaly_detection.py             # needs results/xgb_predictions.csv from 06
python 08_anomaly_detection.py --backtest  # three out-of-sample years
```

- `results/` and `figures/` are not in `.gitignore`, so commit the new CSVs and figures. They are small and your partner needs them to write the report.
- Run each script from the repository root; they use relative paths.
- Known harmless warnings: see section 5 of `README_midterm.md`. `06` prints a note about dropped constant event features until the events CSV has rows.

"""
03b_spectral_analysis.py
Spectral (FFT/periodogram) and autocorrelation (ACF/PACF) analysis of daily POGOH ridership.

Input : data/processed/daily_panel.csv  (needs columns: date, system_trips)
Output: figures/spectral/*.png and a printed summary used to justify SARIMAX orders.

Run after 03_clean_and_merge.py:  python 03b_spectral_analysis.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, periodogram
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

PANEL = Path("data/processed/daily_panel.csv")
OUT = Path("figures/spectral")
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- load
df = pd.read_csv(PANEL, parse_dates=["date"]).set_index("date").sort_index()
y = df["system_trips"].asfreq("D")
print(f"Series: {y.index.min().date()} -> {y.index.max().date()}, {len(y)} days, "
      f"{y.isna().sum()} missing")
y = y.interpolate(limit_direction="both")  # FFT/ACF need a gap-free series

# ---------------------------------------------------------------- 1. periodogram
# Linear detrend + Hann window to reduce spectral leakage. fs=1 sample/day, so
# frequency is in cycles/day and period = 1/frequency in days.
freqs, power = periodogram(y.values, fs=1.0, window="hann", detrend="linear",
                           scaling="spectrum")
mask = freqs > 0
freqs, power = freqs[mask], power[mask]
periods = 1.0 / freqs

peaks, _ = find_peaks(power, height=power.max() * 0.02)
top = peaks[np.argsort(power[peaks])[::-1][:8]]
print("\nTop spectral peaks (period in days, relative power):")
for i in top:
    print(f"  period = {periods[i]:7.1f} d   freq = {freqs[i]:.4f} cyc/day   "
          f"power = {power[i] / power.max():.3f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(freqs, power)
axes[0].set(xlabel="Frequency (cycles/day)", ylabel="Power",
            title="Periodogram of daily system trips")
for f, lab in [(1 / 7, "1/7 (weekly)"), (2 / 7, "2/7"), (1 / 365.25, "1/365 (annual)")]:
    axes[0].axvline(f, color="r", ls="--", alpha=0.5)
    axes[0].text(f, power.max() * 0.9, lab, rotation=90, va="top", fontsize=8)
axes[1].semilogx(periods, power)
axes[1].invert_xaxis()
axes[1].set(xlabel="Period (days, log scale)", ylabel="Power",
            title="Same spectrum vs. period")
for p in (7, 365.25):
    axes[1].axvline(p, color="r", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig(OUT / "periodogram.png", dpi=150)
plt.close()

# ---------------------------------------------------------------- 2. weekly structure
dow = y.groupby(y.index.dayofweek).mean()
dow.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
print("\nMean trips by day of week:\n", dow.round(1).to_string())

# ---------------------------------------------------------------- 3. stationarity


def adf_report(name, s):
    stat, p, *_ = adfuller(s.dropna(), autolag="AIC")
    print(f"  ADF {name:<34s} stat = {stat:7.3f}   p = {p:.4f}")


print("\nStationarity (ADF; p < 0.05 suggests stationary):")
d1 = y.diff().dropna()
d7 = y.diff(7).dropna()
adf_report("raw", y)
adf_report("first difference (d=1)", d1)
adf_report("seasonal difference (D=1, s=7)", d7)
adf_report("seasonal + first difference", y.diff(7).diff().dropna())

# ---------------------------------------------------------------- 4. ACF / PACF
LAGS = 42  # six weeks, shows weekly spikes at 7, 14, 21, ...
for name, s in [("raw", y), ("seasonal_diff_s7", d7), ("seasonal_plus_first_diff", y.diff(7).diff().dropna())]:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    plot_acf(s, lags=LAGS, ax=axes[0], title=f"ACF - {name}")
    plot_pacf(s, lags=LAGS, ax=axes[1], method="ywm", title=f"PACF - {name}")
    plt.tight_layout()
    plt.savefig(OUT / f"acf_pacf_{name}.png", dpi=150)
    plt.close()

# ---------------------------------------------------------------- 5. yearly view
fig, ax = plt.subplots(figsize=(12, 3.5))
ax.plot(y.index, y.values, lw=0.6, label="daily")
ax.plot(y.rolling(28, center=True).mean(), lw=2, label="28-day mean")
ax.set(title="Daily POGOH trips", ylabel="Trips")
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "series_overview.png", dpi=150)
plt.close()

print(f"\nFigures saved to {OUT}/")
print("Reading guide: a periodogram peak at ~7 d and ACF spikes at lags 7/14/21 "
      "justify s=7 in SARIMAX; the PACF/ACF of the differenced series suggest p, q, P, Q.")

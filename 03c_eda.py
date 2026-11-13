"""
03c_eda.py
Exploratory data analysis of the daily POGOH panel.

Input : data/processed/daily_panel.csv   (needs: date, system_trips; other columns auto-detected)
Output: figures/eda/*.png and data/processed/eda_summary_stats.csv

Run after 03_clean_and_merge.py:  python 03c_eda.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PANEL = Path("data/processed/daily_panel.csv")
OUT = Path("figures/eda")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})

df = pd.read_csv(PANEL, parse_dates=["date"]).set_index("date").sort_index()
print(f"Panel: {df.shape[0]} days x {df.shape[1]} columns")
print("Columns:", list(df.columns))

y = df["system_trips"]


def find_col(*keywords):
    """First column whose name contains all keywords (case-insensitive)."""
    for c in df.columns:
        if all(k in c.lower() for k in keywords):
            return c
    return None


temp_col = find_col("temperature", "mean") or find_col("temperature", "max") or find_col("temp")
precip_col = find_col("precipitation") or find_col("precip") or find_col("rain")
snow_col = find_col("snow")
wind_col = find_col("wind")
print(f"Detected -> temp: {temp_col}, precip: {precip_col}, snow: {snow_col}, wind: {wind_col}")

# ---------------------------------------------------------------- 1. data quality
print("\nMissing values per column (non-zero only):")
miss = df.isna().sum()
print(miss[miss > 0].to_string() if (miss > 0).any() else "  none")
print(f"Zero-trip days: {(y == 0).sum()}   |  days < 10 trips: {(y < 10).sum()}")

stats = y.describe()
stats.to_csv("data/processed/eda_summary_stats.csv", header=["system_trips"])
print("\nsystem_trips summary:\n", stats.round(1).to_string())

# ---------------------------------------------------------------- 2. overview + growth
fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
axes[0].plot(y.index, y, lw=0.5, alpha=0.6, label="daily")
axes[0].plot(y.rolling(7, center=True).mean(), lw=1.5, label="7-day mean")
axes[0].plot(y.rolling(30, center=True).mean(), lw=2.2, label="30-day mean")
axes[0].set(ylabel="Trips", title="Daily POGOH system trips")
axes[0].legend()
axes[1].plot(y.rolling(365, min_periods=300).mean(), color="C3")
axes[1].set(ylabel="Trips/day", title="Trailing 365-day mean (long-term growth)")
plt.tight_layout()
plt.savefig(OUT / "01_overview_growth.png")
plt.close()

# yearly totals / calendar-year comparison
yearly = y.groupby(y.index.year).agg(["sum", "mean", "count"])
print("\nTrips by calendar year (2022 and 2026 are partial):\n", yearly.round(1).to_string())
monthly = y.resample("MS").sum()
fig, ax = plt.subplots(figsize=(13, 4))
for yr, g in monthly.groupby(monthly.index.year):
    ax.plot(g.index.month, g.values, marker="o", label=str(yr))
ax.set(xlabel="Month", ylabel="Monthly trips", title="Monthly trips, one line per year",
       xticks=range(1, 13))
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "02_monthly_by_year.png")
plt.close()

# ---------------------------------------------------------------- 3. seasonality
dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
data_dow = [y[y.index.dayofweek == d] for d in range(7)]
axes[0].boxplot(data_dow, labels=dow_names, showfliers=False)
axes[0].set(title="Trips by day of week", ylabel="Trips")
data_m = [y[y.index.month == m] for m in range(1, 13)]
axes[1].boxplot(data_m, labels=["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"], showfliers=False)
axes[1].set(title="Trips by month", ylabel="Trips")
plt.tight_layout()
plt.savefig(OUT / "03_dow_month_boxplots.png")
plt.close()

# weekday x month heatmap (mean trips)
pivot = y.groupby([y.index.dayofweek, y.index.month]).mean().unstack()
fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
ax.set_yticks(range(7), dow_names)
ax.set_xticks(range(12), range(1, 13))
ax.set(xlabel="Month", title="Mean trips: day of week x month")
ax.grid(False)
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(OUT / "04_dow_month_heatmap.png")
plt.close()

# ---------------------------------------------------------------- 4. distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(y, bins=50)
axes[0].set(title="Distribution of daily trips", xlabel="Trips", ylabel="Days")
axes[1].hist(np.log1p(y), bins=50)
axes[1].set(title="log(1 + trips)", xlabel="log(1 + trips)")
plt.tight_layout()
plt.savefig(OUT / "05_distribution.png")
plt.close()

# ---------------------------------------------------------------- 5. weather relationships
wx = [(temp_col, "Temperature"), (precip_col, "Precipitation"), (snow_col, "Snowfall"), (wind_col, "Wind")]
wx = [(c, n) for c, n in wx if c]
if wx:
    fig, axes = plt.subplots(1, len(wx), figsize=(4.2 * len(wx), 4))
    axes = np.atleast_1d(axes)
    for ax, (c, n) in zip(axes, wx):
        ax.scatter(df[c], y, s=6, alpha=0.4)
        ax.set(xlabel=c, ylabel="Trips", title=f"Trips vs {n}")
    plt.tight_layout()
    plt.savefig(OUT / "06_weather_scatter.png")
    plt.close()

if precip_col:
    wet = df[precip_col] > 1.0
    print(f"\nMean trips: dry days {y[~wet].mean():.1f}  vs  wet days (>1 unit) {y[wet].mean():.1f}")
if snow_col:
    snowy = df[snow_col] > 0
    print(f"Mean trips: no-snow days {y[~snowy].mean():.1f}  vs  snow days {y[snowy].mean():.1f} (n={snowy.sum()})")

# ---------------------------------------------------------------- 6. correlations
num = df.select_dtypes("number")
num = num.loc[:, num.nunique() > 1]
corr = num.corr()
print("\nCorrelation with system_trips:\n", corr["system_trips"].drop("system_trips").sort_values(key=abs, ascending=False).round(2).to_string())
fig, ax = plt.subplots(figsize=(max(7, 0.45 * len(corr)), max(6, 0.4 * len(corr))))
im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
ax.set_yticks(range(len(corr)), corr.columns, fontsize=7)
ax.grid(False)
plt.colorbar(im, ax=ax)
ax.set_title("Correlation matrix")
plt.tight_layout()
plt.savefig(OUT / "07_correlation_matrix.png")
plt.close()

# ---------------------------------------------------------------- 7. top stations (if present)
station_cols = [c for c in df.columns if c.lower().startswith(("station", "trips_", "st_"))
                and c != "system_trips"]
station_cols = [c for c in station_cols if pd.api.types.is_numeric_dtype(df[c]) and df[c].sum() > 0]
if station_cols:
    totals = df[station_cols].sum().sort_values(ascending=False)
    print("\nTop station columns by total trips:\n", totals.head(10).to_string())
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    totals.head(10)[::-1].plot.barh(ax=axes[0])
    axes[0].set(title="Top stations by total trip starts")
    df[totals.head(5).index].rolling(30, center=True).mean().plot(ax=axes[1])
    axes[1].set(title="Top 5 stations, 30-day mean", ylabel="Trips")
    plt.tight_layout()
    plt.savefig(OUT / "08_top_stations.png")
    plt.close()
else:
    print("\n[note] no per-station columns detected; skipped station plots")

print(f"\nFigures saved to {OUT}/")
"""
metrics.py
Shared evaluation helpers so every model (baseline, SARIMAX, gradient boosting)
is scored the same way on the same chronological split.

Usage:
    from metrics import chrono_split, evaluate, evaluate_by_season, print_table
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------- point metrics
def _arr(a):
    return np.asarray(a, dtype=float)


def mae(y_true, y_pred):
    y_true, y_pred = _arr(y_true), _arr(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    y_true, y_pred = _arr(y_true), _arr(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred, eps=1.0):
    """Mean absolute percentage error (in %). Denominator is clipped at `eps`
    trips so near-zero winter days do not blow the metric up."""
    y_true, y_pred = _arr(y_true), _arr(y_pred)
    return float(np.mean(np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), eps)) * 100)


def wmape(y_true, y_pred):
    """Weighted MAPE (in %): sum|error| / sum|actual|. Stable when some days are
    near zero, and normalizes error against total demand."""
    y_true, y_pred = _arr(y_true), _arr(y_pred)
    denom = np.sum(np.abs(y_true))
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100) if denom else float("nan")


def evaluate(y_true, y_pred, name="model"):
    """Return a one-row dict of all metrics. NaN pairs are dropped."""
    y_true, y_pred = _arr(y_true), _arr(y_pred)
    ok = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[ok], y_pred[ok]
    return {
        "model": name,
        "n": int(ok.sum()),
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE_%": mape(y_true, y_pred),
        "WMAPE_%": wmape(y_true, y_pred),
    }


# ---------------------------------------------------------------- seasons
def season_of(index):
    """Meteorological seasons from a DatetimeIndex -> array of strings."""
    m = pd.DatetimeIndex(index).month
    return np.select(
        [np.isin(m, [12, 1, 2]), np.isin(m, [3, 4, 5]), np.isin(m, [6, 7, 8])],
        ["Winter", "Spring", "Summer"],
        default="Fall",
    )


def evaluate_by_season(y_true: pd.Series, y_pred: pd.Series, name="model"):
    """Metrics per season (y_true / y_pred must share a DatetimeIndex)."""
    y_pred = pd.Series(y_pred, index=y_true.index)
    seasons = season_of(y_true.index)
    rows = []
    for s in ["Winter", "Spring", "Summer", "Fall"]:
        m = seasons == s
        if m.sum():
            r = evaluate(y_true[m], y_pred[m], name)
            r["season"] = s
            rows.append(r)
    r = evaluate(y_true, y_pred, name)
    r["season"] = "All"
    rows.append(r)
    return pd.DataFrame(rows)[["model", "season", "n", "MAE", "RMSE", "MAPE_%", "WMAPE_%"]]


# ---------------------------------------------------------------- splitting
def chrono_split(df: pd.DataFrame, test_start="2025-11-01", val_start=None):
    """Chronological split (never shuffled).

    Default: train on everything before `test_start`, test on the rest. With the
    May 2022 - Oct 2026 panel, test_start='2025-11-01' leaves exactly the final
    12 months as test, so every season appears once. Pass `val_start` for a
    train / validation / test split (used for tuning gradient boosting).
    """
    df = df.sort_index()
    test_start = pd.Timestamp(test_start)
    test = df[df.index >= test_start]
    if val_start is not None:
        val_start = pd.Timestamp(val_start)
        train = df[df.index < val_start]
        val = df[(df.index >= val_start) & (df.index < test_start)]
        return train, val, test
    return df[df.index < test_start], test


def rolling_origin_splits(index, initial_train_days=730, test_days=30, step_days=30):
    """Yield (train_idx, test_idx) positional arrays for expanding-window
    cross-validation (a sturdier alternative to one fixed split)."""
    n = len(index)
    start = initial_train_days
    while start + test_days <= n:
        yield np.arange(0, start), np.arange(start, start + test_days)
        start += step_days


# ---------------------------------------------------------------- baseline
def same_weekday_baseline(y: pd.Series, weeks_back=1, average=False):
    """Predict each day using the same weekday `weeks_back` weeks earlier.
    If average=True, use the mean of the previous `weeks_back` same-weekdays."""
    y = y.asfreq("D")
    if average:
        preds = sum(y.shift(7 * k)
                    for k in range(1, weeks_back + 1)) / weeks_back
    else:
        preds = y.shift(7 * weeks_back)
    return preds.rename("same_weekday_baseline")


def print_table(df: pd.DataFrame, float_fmt="{:.2f}"):
    """Readable console table for results."""
    with pd.option_context("display.float_format", float_fmt.format, "display.width", 140):
        print(df.to_string(index=False))


if __name__ == "__main__":
    # quick self-test
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=400, freq="D")
    y = pd.Series(100 + 30 * np.sin(np.arange(400) * 2 *
                  np.pi / 7) + rng.normal(0, 5, 400), index=idx)
    base = same_weekday_baseline(y)
    train, test = chrono_split(pd.DataFrame({"y": y}), test_start="2025-01-01")
    print_table(evaluate_by_season(
        test["y"], base.loc[test.index], "same_weekday"))

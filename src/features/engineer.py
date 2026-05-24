"""
Feature engineering for StockIQ.

Computes 60+ technical features from raw OHLCV data with
ZERO look-ahead bias — every value at row t uses only data ≤ t.

Feature groups
──────────────
  1.  Returns (1d – 60d)
  2.  Price structure (body, shadow, gap)
  3.  Moving averages & crossovers
  4.  RSI (7 & 14)
  5.  MACD
  6.  Bollinger Bands
  7.  Stochastic oscillator
  8.  Average True Range / historical volatility
  9.  Volume signals (OBV, volume ratio)
  10. Williams %R
  11. Rate of Change
  12. Calendar (day-of-week, month, quarter)
  13. Lag features
  14. Streak / momentum features

Target
──────
  Binary: 1 if next-day Close > today's Close, else 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple

# Human-readable names used for SHAP display
FEATURE_DISPLAY_NAMES: dict[str, str] = {}


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - 100 / (1 + rs)


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build the feature matrix *X* and target series *y* from OHLCV *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns [Open, High, Low, Close, Volume] with a DatetimeIndex.

    Returns
    -------
    X : pd.DataFrame   — feature matrix (no NaN / Inf)
    y : pd.Series      — binary next-day-up target (aligned with X)
    """
    data  = df.copy()
    close = data["Close"]
    high  = data["High"]
    low   = data["Low"]
    open_ = data["Open"]
    vol   = data["Volume"]

    feat  = pd.DataFrame(index=data.index)

    # ── 1. Returns ────────────────────────────────────────────────
    for n in [1, 2, 3, 5, 10, 20, 60]:
        feat[f"return_{n}d"] = close.pct_change(n)

    feat["log_return_1d"]  = np.log(close / close.shift(1) + 1e-9)
    feat["log_return_5d"]  = np.log(close / close.shift(5) + 1e-9)
    feat["log_return_20d"] = np.log(close / close.shift(20) + 1e-9)

    # ── 2. Candle structure ────────────────────────────────────────
    hl_range = (high - low).replace(0, np.nan)
    feat["body_size"]    = (close - open_).abs() / (close.abs() + 1e-9)
    feat["upper_shadow"] = (high - close.where(close > open_, open_)) / hl_range
    feat["lower_shadow"] = (close.where(close < open_, open_) - low) / hl_range
    feat["gap"]          = (open_ - close.shift(1)) / (close.shift(1) + 1e-9)
    feat["hl_range"]     = hl_range / (close + 1e-9)

    # ── 3. Moving averages ─────────────────────────────────────────
    for n in [5, 10, 20, 50, 200]:
        sma = close.rolling(n, min_periods=n).mean()
        feat[f"price_sma{n}_ratio"] = close / (sma + 1e-9) - 1

    for n in [5, 10, 20, 50]:
        ema = close.ewm(span=n, adjust=False).mean()
        feat[f"price_ema{n}_ratio"] = close / (ema + 1e-9) - 1

    ema5  = close.ewm(span=5,  adjust=False).mean()
    ema10 = close.ewm(span=10, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    feat["ema5_20_cross"]  = (ema5  - ema20) / (close + 1e-9)
    feat["ema10_50_cross"] = (ema10 - ema50) / (close + 1e-9)
    feat["golden_cross"]   = ((ema5 > ema20) & (ema5.shift(1) <= ema20.shift(1))).astype(int)
    feat["death_cross"]    = ((ema5 < ema20) & (ema5.shift(1) >= ema20.shift(1))).astype(int)

    # ── 4. RSI ─────────────────────────────────────────────────────
    rsi7  = _rsi(close, 7)
    rsi14 = _rsi(close, 14)
    feat["rsi_7"]          = rsi7  / 100
    feat["rsi_14"]         = rsi14 / 100
    feat["rsi_oversold"]   = (rsi14 < 30).astype(int)
    feat["rsi_overbought"] = (rsi14 > 70).astype(int)
    feat["rsi_momentum"]   = (rsi14 - rsi14.shift(3)) / 100  # 3-day RSI change

    # ── 5. MACD ─────────────────────────────────────────────────────
    ema12     = close.ewm(span=12, adjust=False).mean()
    ema26     = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    sig_line  = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - sig_line

    feat["macd"]              = macd_line  / (close + 1e-9)
    feat["macd_signal"]       = sig_line   / (close + 1e-9)
    feat["macd_hist"]         = histogram  / (close + 1e-9)
    feat["macd_above_signal"] = (macd_line > sig_line).astype(int)
    feat["macd_bull_cross"]   = (
        (macd_line > sig_line) & (macd_line.shift(1) <= sig_line.shift(1))
    ).astype(int)
    feat["macd_bear_cross"]   = (
        (macd_line < sig_line) & (macd_line.shift(1) >= sig_line.shift(1))
    ).astype(int)

    # ── 6. Bollinger Bands ───────────────────────────────────────────
    sma20    = close.rolling(20, min_periods=20).mean()
    std20    = close.rolling(20, min_periods=20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_width = (bb_upper - bb_lower) / (sma20 + 1e-9)
    bb_pct   = (close - bb_lower) / (bb_upper - bb_lower + 1e-9)

    feat["bb_width"]       = bb_width
    feat["bb_position"]    = bb_pct.clip(0, 1)
    feat["bb_above_upper"] = (close > bb_upper).astype(int)
    feat["bb_below_lower"] = (close < bb_lower).astype(int)
    feat["bb_squeeze"]     = (bb_width < bb_width.rolling(50).mean()).astype(int)

    # ── 7. Stochastic ─────────────────────────────────────────────
    lo14  = low.rolling(14,  min_periods=14).min()
    hi14  = high.rolling(14, min_periods=14).max()
    stk   = (close - lo14) / (hi14 - lo14 + 1e-9) * 100
    std_d = stk.rolling(3).mean()

    feat["stoch_k"]          = stk   / 100
    feat["stoch_d"]          = std_d / 100
    feat["stoch_oversold"]   = (stk < 20).astype(int)
    feat["stoch_overbought"] = (stk > 80).astype(int)
    feat["stoch_bull_cross"] = ((stk > std_d) & (stk.shift(1) <= std_d.shift(1))).astype(int)

    # ── 8. ATR & Volatility ───────────────────────────────────────
    tr   = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=14).mean()

    feat["atr_ratio"]      = atr14 / (close + 1e-9)
    feat["hist_vol_20"]    = close.pct_change().rolling(20).std() * np.sqrt(252)
    feat["hist_vol_ratio"] = feat["hist_vol_20"] / (
        feat["hist_vol_20"].rolling(60).mean() + 1e-9
    )

    # ── 9. Volume ─────────────────────────────────────────────────
    vol_ma20 = vol.rolling(20, min_periods=20).mean()
    feat["volume_ratio"]   = vol / (vol_ma20 + 1e-9)
    feat["volume_trend"]   = (
        vol.rolling(5).mean() / (vol.rolling(20).mean() + 1e-9)
    )
    # On-Balance Volume trend
    obv       = (np.sign(close.diff()) * vol).cumsum()
    obv_ema20 = obv.ewm(span=20, adjust=False).mean()
    feat["obv_trend"]      = (obv - obv_ema20) / (obv_ema20.abs() + 1)
    feat["vol_return"]     = feat["return_1d"] * feat["volume_ratio"]

    # ── 10. Williams %R ───────────────────────────────────────────
    lo14_w = low.rolling(14,  min_periods=14).min()
    hi14_w = high.rolling(14, min_periods=14).max()
    feat["williams_r"] = (hi14_w - close) / (hi14_w - lo14_w + 1e-9)   # 0→1 (0=overbought)

    # ── 11. Rate of Change ────────────────────────────────────────
    for n in [5, 10, 20]:
        feat[f"roc_{n}"] = close.pct_change(n)

    # ── 12. Calendar ──────────────────────────────────────────────
    feat["day_of_week"] = data.index.dayofweek / 4.0
    feat["month"]       = data.index.month     / 12.0
    feat["quarter"]     = data.index.quarter   / 4.0

    # ── 13. Lag features ──────────────────────────────────────────
    for lag in [1, 2, 3]:
        feat[f"return_lag{lag}"]      = feat["return_1d"].shift(lag)
        feat[f"rsi14_lag{lag}"]       = (rsi14 / 100).shift(lag)
        feat[f"vol_ratio_lag{lag}"]   = feat["volume_ratio"].shift(lag)
        feat[f"macd_hist_lag{lag}"]   = feat["macd_hist"].shift(lag)

    # ── 14. Momentum streaks ──────────────────────────────────────
    up_days   = (close.diff() > 0).astype(int)
    down_days = (close.diff() < 0).astype(int)
    feat["up_days_5"]   = up_days.rolling(5).sum()   / 5
    feat["down_days_5"] = down_days.rolling(5).sum() / 5
    feat["up_days_10"]  = up_days.rolling(10).sum()  / 10

    # ── Target ────────────────────────────────────────────────────
    # Predict whether tomorrow's close will be higher than today's.
    # Shift(-1) so row t has the target for day t+1.
    target = (close.shift(-1) > close).astype(int)

    # ── Align & clean ─────────────────────────────────────────────
    feat   = feat.iloc[:-1]    # drop last row (no future target)
    target = target.iloc[:-1]

    # Replace infinities
    feat = feat.replace([np.inf, -np.inf], np.nan)

    # Drop rows where ANY feature or target is NaN
    valid = ~(feat.isna().any(axis=1) | target.isna())
    feat   = feat[valid].copy()
    target = target[valid].copy()

    return feat, target


def get_last_feature_row(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return the *single* most-recent feature row for live prediction.
    No target is computed — this is for inference only.
    """
    feat, _ = build_features(df)   # cheaply re-uses the same pipeline
    return feat.iloc[[-1]]         # keep DataFrame shape for scaler.transform

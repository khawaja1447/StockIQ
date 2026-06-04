"""
Prediction interface for StockIQ.

The StockPredictor class bundles:
  - model training  (walk-forward CV + final fit)
  - live inference  (ensemble probability + sentiment overlay)
  - SHAP explanation (why is this prediction UP or DOWN?)

Caching strategy
────────────────
  Use  @st.cache_resource(ttl=86400)  around the factory function
  so each ticker's model is trained once per day and reused across
  Streamlit reruns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
import warnings
warnings.filterwarnings("ignore")

from typing import Dict, Optional
from src.features.engineer   import build_features, get_last_feature_row
from src.models.trainer      import walk_forward_cv, train_final_model


class StockPredictor:
    """
    Encapsulates the full ML pipeline for a single ticker.

    Usage
    -----
        predictor = StockPredictor("BTC-USD")
        predictor.train(df_2yr)           # ~10-30 s first time
        result = predictor.predict(df_2yr, sentiment_score=0.3)
    """

    def __init__(self, ticker: str):
        self.ticker        = ticker
        self.xgb_model     = None
        self.lgb_model     = None
        self.scaler        = None
        self.feature_names: list[str] = []
        self.cv_metrics:    dict      = {}
        self._explainer     = None

    # ──────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame) -> None:
        """
        Build features, run walk-forward CV, then train final models.

        Parameters
        ----------
        df : pd.DataFrame  — 2-year OHLCV daily data for self.ticker
        """
        X, y = build_features(df)
        if len(X) < 100:
            raise ValueError(
                f"Insufficient data for {self.ticker}: only {len(X)} clean rows."
            )

        self.feature_names = list(X.columns)

        # 1. Walk-forward CV for honest accuracy estimate
        self.cv_metrics = walk_forward_cv(X, y, n_splits=5)

        # 2. Final model on all available data
        self.xgb_model, self.lgb_model, self.scaler = train_final_model(X, y)

        # 3. Pre-build SHAP explainer (TreeExplainer is fast for XGBoost)
        self._explainer = shap.TreeExplainer(self.xgb_model)

    # ──────────────────────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────────────────────

    def predict(
        self,
        df: pd.DataFrame,
        sentiment_score: float = 0.0,
        sentiment_weight: float = 0.12,
    ) -> Dict:
        """
        Generate a next-day prediction with SHAP explanation.

        Parameters
        ----------
        df               : current OHLCV data (same ticker, same period)
        sentiment_score  : composite VADER score in [-1, +1]
        sentiment_weight : how much sentiment nudges the technical probability

        Returns
        -------
        dict with keys:
            direction      "UP" | "DOWN"
            probability    float in [0, 1]  (P(UP))
            confidence     float in [50, 100]  (%)
            technical_prob float  — raw model probability (no sentiment)
            shap_values    np.ndarray of shape (n_features,)
            feature_names  list[str]
            last_features  pd.DataFrame of the feature row used
        """
        if self.xgb_model is None:
            raise RuntimeError("Model not trained — call .train(df) first.")

        # Get latest feature row
        last_row = get_last_feature_row(df)
        X_scaled = self.scaler.transform(last_row)

        # Ensemble probability
        p_xgb = float(self.xgb_model.predict_proba(X_scaled)[0, 1])
        p_lgb = float(self.lgb_model.predict_proba(X_scaled)[0, 1])
        p_technical = (p_xgb + p_lgb) / 2

        # Sentiment overlay
        # Convert sentiment_score [-1, +1] → probability contribution [0, 1]
        sentiment_prob = (sentiment_score + 1) / 2
        p_final = (
            p_technical * (1 - sentiment_weight)
            + sentiment_prob * sentiment_weight
        )
        p_final = float(np.clip(p_final, 0.0, 1.0))

        direction  = "UP" if p_final >= 0.50 else "DOWN"
        confidence = max(p_final, 1 - p_final) * 100

        # SHAP explanation (using XGBoost, most interpretable)
        shap_vals = self._explainer.shap_values(X_scaled)
        # For binary XGBoost: shap_values is 2-D array (1, n_features)
        if isinstance(shap_vals, list):
            # older SHAP returns list [neg_class, pos_class]
            shap_arr = shap_vals[1][0]
        else:
            shap_arr = shap_vals[0]

        return {
            "direction":      direction,
            "probability":    p_final,
            "confidence":     float(confidence),
            "technical_prob": p_technical,
            "p_xgb":          p_xgb,
            "p_lgb":          p_lgb,
            "shap_values":    shap_arr,
            "feature_names":  self.feature_names,
            "last_features":  last_row,
            "expected_value": float(self._explainer.expected_value)
                              if not isinstance(self._explainer.expected_value, (list, np.ndarray))
                              else float(self._explainer.expected_value[1]),
        }

    # ──────────────────────────────────────────────────────────────
    # Trade Planner — optimal exit prediction
    # ──────────────────────────────────────────────────────────────

    def predict_exit(self, df: pd.DataFrame, sentiment_score: float = 0.0) -> dict:
        """
        Given entry at today's close, find the optimal sell window to maximise
        risk-adjusted return over the next 30 days.

        Method
        ------
        1. Use the 1-day ML probability as the base signal strength.
        2. Decay the signal over time (momentum fades; decay = 0.82/day).
        3. Project expected price using the asset's historical mean daily return.
        4. Build 90 % confidence price bands via historical volatility × √h.
        5. Compute risk-adjusted score = P(up at h) × expected_return(h).
        6. Identify technical take-profit levels (ATR multiples, swing highs,
           Bollinger upper) and annotate each horizon.

        Returns a dict with scalars for the UI and a 'horizons' DataFrame for charts.
        """
        if self.xgb_model is None:
            raise RuntimeError("Model not trained.")

        close   = df["Close"]
        high    = df["High"]
        low     = df["Low"]
        entry   = float(close.iloc[-1])

        # ── ATR (14-day) ────────────────────────────────────────────
        tr   = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr  = float(tr.rolling(14).mean().iloc[-1])

        # ── Historical daily return statistics ──────────────────────
        rets       = close.pct_change().dropna()
        mean_ret   = float(rets.mean())          # e.g. 0.001 for BTC
        daily_vol  = float(rets.std())           # e.g. 0.025 for BTC

        # ── Base 1-day ML probability ───────────────────────────────
        base    = self.predict(df, sentiment_score=sentiment_score)
        p1      = base["probability"]            # P(up tomorrow)

        # ── Technical resistance levels ─────────────────────────────
        bb_upper = float(
            close.rolling(20).mean().iloc[-1] + 2 * close.rolling(20).std().iloc[-1]
        )
        swing_20 = float(high.tail(20).max())
        swing_50 = float(high.tail(50).max())
        tp1 = entry + 1.0 * atr
        tp2 = entry + 2.0 * atr
        tp3 = entry + 3.0 * atr
        sl  = entry - 1.5 * atr                 # stop-loss

        # ── Multi-horizon projection ─────────────────────────────────
        DECAY     = 0.82   # signal half-life ≈ 3.5 days
        HORIZONS  = [1, 2, 3, 5, 7, 10, 14, 21, 30]
        rows      = []
        for h in HORIZONS:
            # P(price at h > entry): starts at p1, decays toward 0.5
            p_h = 0.5 + (p1 - 0.5) * (DECAY ** (h - 1))

            # Expected price: compound mean daily return
            exp_price  = entry * ((1 + mean_ret) ** h)

            # 90 % confidence band (log-normal assumption)
            sigma_h    = daily_vol * np.sqrt(h)
            upper_90   = entry * np.exp( 1.65 * sigma_h)
            lower_90   = entry * np.exp(-1.65 * sigma_h)
            upper_50   = entry * np.exp( 0.67 * sigma_h)
            lower_50   = entry * np.exp(-0.67 * sigma_h)

            exp_ret    = (exp_price - entry) / entry * 100   # %
            # Risk-adjusted score: probability × return (Kelly-inspired)
            ra_score   = p_h * (exp_ret / 100)

            rows.append({
                "days":       h,
                "p_profit":   round(p_h, 4),
                "exp_price":  round(exp_price, 4),
                "exp_ret_pct": round(exp_ret, 2),
                "ra_score":   round(ra_score, 6),
                "upper_90":   round(upper_90, 4),
                "lower_90":   round(lower_90, 4),
                "upper_50":   round(upper_50, 4),
                "lower_50":   round(lower_50, 4),
            })

        horizons_df = pd.DataFrame(rows)

        # ── Optimal exit ────────────────────────────────────────────
        best_idx    = int(horizons_df["ra_score"].idxmax())
        best        = horizons_df.iloc[best_idx]
        opt_days    = int(best["days"])
        opt_price   = float(best["exp_price"])
        opt_ret     = float(best["exp_ret_pct"])
        opt_prob    = float(best["p_profit"])
        rr_ratio    = (opt_price - entry) / (entry - sl) if entry > sl else 0.0

        # ── Nearest resistance above entry ──────────────────────────
        resistances = {
            "TP1 (1×ATR)":      tp1,
            "TP2 (2×ATR)":      tp2,
            "TP3 (3×ATR)":      tp3,
            "Bollinger Upper":   bb_upper,
            "20-day Swing High": swing_20,
            "50-day Swing High": swing_50,
        }
        # Only keep levels above entry
        levels_above = {k: v for k, v in resistances.items() if v > entry}

        return {
            "entry":          entry,
            "stop_loss":      sl,
            "atr":            atr,
            "opt_days":       opt_days,
            "opt_price":      opt_price,
            "opt_ret_pct":    opt_ret,
            "opt_probability":opt_prob,
            "risk_reward":    rr_ratio,
            "resistances":    levels_above,
            "all_levels":     resistances,
            "horizons":       horizons_df,
            "p1":             p1,
            "mean_daily_ret": mean_ret,
            "daily_vol":      daily_vol,
        }

    # ──────────────────────────────────────────────────────────────
    # Feature Importance (for bar chart)
    # ──────────────────────────────────────────────────────────────

    def feature_importances(self) -> pd.DataFrame:
        """Return a DataFrame of XGBoost gain-based feature importances."""
        if self.xgb_model is None:
            return pd.DataFrame()
        imp = self.xgb_model.feature_importances_
        return (
            pd.DataFrame({"feature": self.feature_names, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

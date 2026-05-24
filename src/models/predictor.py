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

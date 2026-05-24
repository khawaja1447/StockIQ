# 🔮 StockIQ — AI-Powered Stock & Crypto Intelligence

> **Disclaimer:** Educational project only. Not financial advice.

StockIQ is a full-stack machine-learning dashboard that combines **technical analysis**, **NLP sentiment**, and an **XGBoost + LightGBM ensemble** to predict next-day price direction for both stocks and crypto — all inside a beautiful Streamlit dashboard.

---

## ✨ Features

| Feature | Detail |
|---|---|
| **Assets** | 10 Stocks (AAPL, TSLA, NVDA…) + 10 Crypto (BTC, ETH, SOL…) |
| **Live Prices** | yfinance — real-time OHLCV, 2-year history |
| **Fear & Greed** | Alternative.me Crypto Fear & Greed Index (free API) |
| **Sentiment NLP** | Reddit public feed + Yahoo Finance news → VADER scoring |
| **60+ Features** | RSI, MACD, Bollinger Bands, ATR, OBV, volume anomalies + more |
| **Ensemble Model** | XGBoost + LightGBM soft-voting ensemble |
| **Honest Accuracy** | Walk-forward cross-validation (no data leakage) |
| **SHAP** | Waterfall + bar chart — *why* the model said UP or DOWN |
| **Dashboard** | Streamlit: 4 tabs — Overview · Prediction · Sentiment · Explainability |

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/your-username/StockIQ.git
cd StockIQ
pip install -r requirements.txt
```

### 2. (Optional) Add API keys

```bash
cp .env.example .env
# Fill in REDDIT_CLIENT_ID etc. for richer sentiment
# The app works without any keys using free public APIs
```

### 3. Run

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🧠 How the AI Works

### Feature Engineering (60+ indicators)
- **Returns:** 1d, 2d, 3d, 5d, 10d, 20d, 60d price returns
- **Trend:** SMA(5/10/20/50/200), EMA(5/10/20/50), golden/death cross
- **Momentum:** RSI(7,14), MACD histogram, Rate of Change, Williams %R
- **Volatility:** Bollinger Bands (%B, width, squeeze), ATR(14), Historical Vol
- **Volume:** OBV trend, volume ratio, volume×return signal
- **Lags:** 1–3 day lags of key indicators to capture momentum
- **Calendar:** Day-of-week, month, quarter effects

### Model Architecture
```
Raw OHLCV  →  Feature Engineering (60+ features)
                        ↓
             RobustScaler (outlier-resistant)
                        ↓
          ┌─────────────┴────────────┐
          │                          │
    XGBoost (400 trees)    LightGBM (400 trees)
          │                          │
          └─────────────┬────────────┘
                   Soft Voting
                        ↓
              Technical Probability
                        ↓
           + Sentiment Overlay (12% weight)
                        ↓
               Final P(UP tomorrow)
```

### Walk-Forward Validation
- 5-fold `TimeSeriesSplit` — each fold trains only on past data
- Reported accuracy is **out-of-sample** (no data leakage)
- Typical range: **55–63%** directional accuracy

### Sentiment Overlay
- Reddit posts scored with VADER NLP (finance-tuned)
- Yahoo Finance news headlines scored
- Composite score nudges technical probability by ±12%

---

## 📊 Dashboard Tabs

| Tab | What you see |
|---|---|
| **Overview** | Candlestick chart + volume, RSI, MACD |
| **Prediction** | UP/DOWN card, confidence %, probability gauge, CV metrics |
| **Sentiment** | Fear & Greed index, Reddit posts, news headlines with scores |
| **Explainability** | SHAP waterfall + feature importance bar chart |

---

## 🗂 Project Structure

```
StockIQ/
├── app.py                     # Streamlit dashboard (entry point)
├── requirements.txt
├── .env.example
├── .streamlit/
│   └── config.toml            # Dark theme
└── src/
    ├── data/
    │   ├── fetcher.py         # yfinance + CoinGecko + Fear & Greed
    │   └── sentiment.py       # Reddit + Yahoo Finance + VADER
    ├── features/
    │   └── engineer.py        # 60+ technical features, no leakage
    └── models/
        ├── trainer.py         # Walk-forward CV + final model training
        └── predictor.py       # Inference interface + SHAP
```

---

## ⚡ Performance Tips

- First load per ticker: ~20–40 seconds (model training)
- Subsequent loads: instant (cached for 24 hours)
- Click **🔄 Refresh Data** to force re-fetch + retrain

---

## 🔧 Tech Stack

`streamlit` · `yfinance` · `xgboost` · `lightgbm` · `shap` ·
`scikit-learn` · `plotly` · `ta` · `vaderSentiment` · `requests`

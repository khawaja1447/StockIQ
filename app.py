"""
StockIQ — AI-Powered Stock & Crypto Intelligence Dashboard
==========================================================
Run with:   streamlit run app.py
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from src.data.fetcher   import (
    fetch_price_data, fetch_ticker_info, fetch_fear_greed_index,
    STOCKS, CRYPTO, TIMEFRAME_PERIODS, TRAINING_PERIOD,
)
from src.data.sentiment  import get_combined_sentiment
from src.features.engineer import build_features
from src.models.predictor  import StockPredictor

# ──────────────────────────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "StockIQ — AI Market Intelligence",
    page_icon  = "🔮",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ──────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Global ──────────────────────────────────────── */
.stApp { background-color: #0E1117; }

/* ── Metric cards ─────────────────────────────────── */
div[data-testid="stMetric"] {
    background    : #1A1F2E;
    border-radius : 12px;
    padding       : 16px 20px;
    border        : 1px solid #2D3748;
}
div[data-testid="stMetric"] > label {
    color         : #8892A4 !important;
    font-size     : 0.78rem !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
div[data-testid="stMetric"] > div {
    font-size     : 1.5rem !important;
    font-weight   : 700   !important;
}

/* ── Prediction card ──────────────────────────────── */
.pred-card {
    border-radius : 16px;
    padding       : 28px 32px;
    margin-bottom : 16px;
    text-align    : center;
}
.pred-up   { background: linear-gradient(135deg,#0D2137 0%,#0A3D2E 100%);
             border: 2px solid #00D4FF; }
.pred-down { background: linear-gradient(135deg,#2D0D0D 0%,#3D1515 100%);
             border: 2px solid #FF4B4B; }
.pred-arrow { font-size:4rem; line-height:1; }
.pred-label { font-size:2.2rem; font-weight:800; margin:4px 0; }
.pred-conf  { font-size:1.1rem; color:#C0C8D4; }

/* ── Sentiment gauge ──────────────────────────────── */
.sent-card {
    background    : #1A1F2E;
    border-radius : 12px;
    padding       : 20px;
    border        : 1px solid #2D3748;
    text-align    : center;
}
.sent-score {
    font-size  : 2.5rem;
    font-weight: 800;
}
.positive { color: #00D4FF; }
.negative { color: #FF4B4B; }
.neutral  { color: #F0B429; }

/* ── Sidebar ──────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background : #10141E;
}

/* ── Tabs ─────────────────────────────────────────── */
button[data-baseweb="tab"] {
    font-size  : 0.95rem !important;
    font-weight: 600     !important;
}

/* ── Scrollable table ─────────────────────────────── */
.scroll-table { max-height:300px; overflow-y:auto; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
# Cached Predictor  (trained once per ticker per day)
# ──────────────────────────────────────────────────────────────────

@st.cache_resource(ttl=86400, show_spinner=False)
def get_predictor(ticker: str) -> StockPredictor | None:
    """Train and cache a StockPredictor for *ticker*."""
    df = fetch_price_data(ticker, period=TRAINING_PERIOD)
    if df.empty or len(df) < 150:
        return None
    try:
        pred = StockPredictor(ticker)
        pred.train(df)
        return pred
    except Exception as e:
        st.error(f"Model training failed: {e}")
        return None


# ──────────────────────────────────────────────────────────────────
# Colour helpers
# ──────────────────────────────────────────────────────────────────

def _chg_color(val: float) -> str:
    return "#00D4FF" if val >= 0 else "#FF4B4B"

def _chg_arrow(val: float) -> str:
    return "▲" if val >= 0 else "▼"

def _sentiment_color(label: str) -> str:
    return {"Positive": "positive", "Negative": "negative"}.get(label, "neutral")

def _fg_color(value: int) -> str:
    if value <= 25:   return "#FF4B4B"
    if value <= 45:   return "#F0A500"
    if value <= 55:   return "#F0B429"
    if value <= 75:   return "#7ED321"
    return "#00C853"


# ──────────────────────────────────────────────────────────────────
# Chart helpers
# ──────────────────────────────────────────────────────────────────

CHART_LAYOUT = dict(
    paper_bgcolor = "#0E1117",
    plot_bgcolor  = "#0E1117",
    font          = dict(color="#E8ECF0", family="Inter, sans-serif"),
    xaxis         = dict(gridcolor="#1E2433", showgrid=True, zeroline=False),
    yaxis         = dict(gridcolor="#1E2433", showgrid=True, zeroline=False),
    margin        = dict(l=0, r=0, t=36, b=0),
    hovermode     = "x unified",
)


def build_candlestick_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Interactive OHLCV candlestick + volume chart."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.02,
    )

    # Candlestick
    up   = df["Close"] >= df["Open"]
    down = ~up
    colours = np.where(up, "#00D4FF", "#FF4B4B")

    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            increasing_line_color="#00D4FF",
            decreasing_line_color="#FF4B4B",
            name=ticker,
        ),
        row=1, col=1,
    )

    # 20-day SMA overlay
    sma20 = df["Close"].rolling(20).mean()
    fig.add_trace(
        go.Scatter(x=df.index, y=sma20, name="SMA 20",
                   line=dict(color="#F0B429", width=1.2, dash="dot")),
        row=1, col=1,
    )
    # 50-day SMA overlay
    sma50 = df["Close"].rolling(50).mean()
    fig.add_trace(
        go.Scatter(x=df.index, y=sma50, name="SMA 50",
                   line=dict(color="#A78BFA", width=1.2, dash="dot")),
        row=1, col=1,
    )

    # Volume bars
    fig.add_trace(
        go.Bar(
            x=df.index, y=df["Volume"],
            marker_color=colours,
            name="Volume", opacity=0.65,
        ),
        row=2, col=1,
    )

    fig.update_layout(
        **CHART_LAYOUT,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis_rangeslider_visible=False,
        height=520,
    )
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume",      row=2, col=1)
    return fig


def build_rsi_chart(df: pd.DataFrame) -> go.Figure:
    """RSI(14) line chart with overbought/oversold bands."""
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = 100 - 100 / (1 + gain / (loss + 1e-9))

    fig = go.Figure()
    fig.add_hrect(y0=70, y1=100, fillcolor="#FF4B4B", opacity=0.08, line_width=0)
    fig.add_hrect(y0=0,  y1=30,  fillcolor="#00D4FF", opacity=0.08, line_width=0)
    fig.add_hline(y=70, line_dash="dot", line_color="#FF4B4B", opacity=0.5)
    fig.add_hline(y=30, line_dash="dot", line_color="#00D4FF", opacity=0.5)
    fig.add_trace(go.Scatter(
        x=df.index, y=rsi, name="RSI(14)",
        line=dict(color="#7ED321", width=1.8),
        fill="tozeroy", fillcolor="rgba(126,211,33,0.06)",
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        yaxis_range=[0, 100], height=220,
        yaxis_title="RSI",
    )
    return fig


def build_macd_chart(df: pd.DataFrame) -> go.Figure:
    """MACD line, signal, and histogram."""
    close      = df["Close"]
    ema12      = close.ewm(span=12, adjust=False).mean()
    ema26      = close.ewm(span=26, adjust=False).mean()
    macd_line  = ema12 - ema26
    sig_line   = macd_line.ewm(span=9, adjust=False).mean()
    histogram  = macd_line - sig_line

    colours = np.where(histogram >= 0, "#00D4FF", "#FF4B4B")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df.index, y=histogram, name="Histogram",
        marker_color=colours, opacity=0.7,
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=macd_line, name="MACD",
        line=dict(color="#00D4FF", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=sig_line, name="Signal",
        line=dict(color="#FF4B4B", width=1.5, dash="dot"),
    ))
    fig.update_layout(**CHART_LAYOUT, yaxis_title="MACD", height=220)
    return fig


def build_fear_greed_chart(hist: pd.DataFrame) -> go.Figure:
    """Fear & Greed Index history line."""
    fig = go.Figure()
    fig.add_hrect(y0=0,  y1=25,  fillcolor="#FF4B4B", opacity=0.07, line_width=0)
    fig.add_hrect(y0=75, y1=100, fillcolor="#00C853", opacity=0.07, line_width=0)
    fig.add_trace(go.Scatter(
        x=hist["date"], y=hist["value"],
        fill="tozeroy",
        fillcolor="rgba(0,212,255,0.07)",
        line=dict(color="#00D4FF", width=2),
        name="Fear & Greed",
    ))
    fig.update_layout(**CHART_LAYOUT, yaxis_range=[0, 100], height=200,
                      yaxis_title="Index")
    return fig


def build_shap_bar(shap_vals: np.ndarray, feature_names: list[str]) -> go.Figure:
    """Horizontal bar chart of top SHAP values."""
    df_shap = (
        pd.DataFrame({"feature": feature_names, "shap": shap_vals})
        .reindex(pd.RangeIndex(len(feature_names)))
        .assign(abs_shap=lambda d: d["shap"].abs())
        .sort_values("abs_shap", ascending=True)
        .tail(20)
    )
    colours = df_shap["shap"].apply(lambda v: "#00D4FF" if v >= 0 else "#FF4B4B")

    fig = go.Figure(go.Bar(
        x=df_shap["shap"],
        y=df_shap["feature"],
        orientation="h",
        marker_color=colours,
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        height=500,
        xaxis_title="SHAP value (impact on prediction)",
        yaxis_title=None,
        title="Feature Impact on Today's Prediction",
    )
    return fig


# ──────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        "<h1 style='color:#00D4FF;margin-bottom:0'>🔮 StockIQ</h1>"
        "<p style='color:#8892A4;margin-top:4px;font-size:0.85rem'>"
        "AI-Powered Market Intelligence</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    asset_type = st.radio(
        "Asset Type",
        ["📈 Stocks", "🪙 Crypto"],
        horizontal=True,
    )

    if asset_type == "📈 Stocks":
        asset_options = {f"{t} — {n}": t for t, n in STOCKS.items()}
    else:
        asset_options = {f"{t.replace('-USD','')} — {n}": t for t, n in CRYPTO.items()}

    selected_label = st.selectbox("Select Asset", list(asset_options.keys()))
    ticker         = asset_options[selected_label]

    timeframe = st.select_slider(
        "Chart Timeframe",
        options=list(TIMEFRAME_PERIODS.keys()),
        value="3 Months",
    )

    st.divider()

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    st.markdown("""
    <p style='color:#8892A4;font-size:0.75rem;margin-top:16px'>
    <b>Model:</b> XGBoost + LightGBM ensemble<br>
    <b>Validation:</b> Walk-forward CV (5 folds)<br>
    <b>Features:</b> 60+ technical indicators<br>
    <b>Sentiment:</b> Reddit + Yahoo Finance
    </p>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
# Load Data
# ──────────────────────────────────────────────────────────────────

period_for_chart = TIMEFRAME_PERIODS[timeframe]

with st.spinner(f"Fetching data for {ticker}…"):
    df_full  = fetch_price_data(ticker, period=TRAINING_PERIOD)
    df_chart = fetch_price_data(ticker, period=period_for_chart)
    info     = fetch_ticker_info(ticker)

if df_full.empty:
    st.error(f"❌ Could not fetch data for **{ticker}**. Check your connection.")
    st.stop()

# ──────────────────────────────────────────────────────────────────
# Header Row
# ──────────────────────────────────────────────────────────────────

current_price = float(df_full["Close"].iloc[-1])
prev_price    = float(df_full["Close"].iloc[-2])
chg_abs       = current_price - prev_price
chg_pct       = chg_abs / prev_price * 100
chg_color     = _chg_color(chg_pct)
chg_arrow     = _chg_arrow(chg_pct)

asset_name = info.get("name", ticker)

col_title, col_price, col_chg, col_mktcap = st.columns([2.5, 1.5, 1.5, 2])

with col_title:
    is_crypto = "-USD" in ticker
    icon = "🪙" if is_crypto else "📈"
    st.markdown(
        f"<h2 style='margin:0;color:#E8ECF0'>{icon} {asset_name}</h2>"
        f"<p style='color:#8892A4;margin:0;font-size:0.85rem'>{ticker} · {info.get('sector','')}</p>",
        unsafe_allow_html=True,
    )

with col_price:
    st.metric("Current Price", f"${current_price:,.4f}" if current_price < 1 else f"${current_price:,.2f}")

with col_chg:
    st.metric(
        "24h Change",
        f"{chg_arrow} {abs(chg_pct):.2f}%",
        delta=f"${chg_abs:+.2f}",
        delta_color="normal",
    )

with col_mktcap:
    mktcap = info.get("market_cap", 0)
    if mktcap > 1e12:
        mktcap_str = f"${mktcap/1e12:.2f}T"
    elif mktcap > 1e9:
        mktcap_str = f"${mktcap/1e9:.2f}B"
    elif mktcap > 1e6:
        mktcap_str = f"${mktcap/1e6:.2f}M"
    else:
        mktcap_str = "N/A"
    st.metric("Market Cap", mktcap_str)

st.divider()


# ──────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────

tab_overview, tab_predict, tab_sentiment, tab_explain = st.tabs([
    "📊 Overview", "🎯 Prediction", "💬 Sentiment", "🔍 Explainability"
])


# ═══════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════

with tab_overview:
    # 52-week stats
    c1, c2, c3, c4 = st.columns(4)
    h52 = info.get("52w_high") or float(df_full["Close"].max())
    l52 = info.get("52w_low")  or float(df_full["Close"].min())
    vol_avg = float(df_full["Volume"].tail(30).mean())

    c1.metric("52-Week High", f"${h52:,.2f}")
    c2.metric("52-Week Low",  f"${l52:,.2f}")
    c3.metric("Avg Volume (30d)", f"{vol_avg/1e6:.1f}M" if vol_avg > 1e6 else f"{vol_avg:,.0f}")
    c4.metric("Days of Data", f"{len(df_full):,}")

    st.markdown("#### Price Chart")
    if df_chart.empty:
        df_chart = df_full.tail(90)
    st.plotly_chart(build_candlestick_chart(df_chart, ticker),
                    use_container_width=True)

    col_rsi, col_macd = st.columns(2)
    with col_rsi:
        st.markdown("#### RSI (14)")
        st.plotly_chart(build_rsi_chart(df_chart), use_container_width=True)
    with col_macd:
        st.markdown("#### MACD")
        st.plotly_chart(build_macd_chart(df_chart), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 2 — PREDICTION
# ═══════════════════════════════════════════════════════════════════

with tab_predict:

    # Load sentiment for overlay
    with st.spinner("Analyzing sentiment…"):
        sent_data = get_combined_sentiment(ticker)
    sent_score = sent_data["score"] if sent_data["available"] else 0.0

    # Train / load model
    with st.spinner("🤖 Loading AI model (first run may take ~30s)…"):
        predictor = get_predictor(ticker)

    if predictor is None:
        st.error("Could not train model — insufficient historical data.")
        st.stop()

    # Run prediction
    result = predictor.predict(df_full, sentiment_score=sent_score)

    direction  = result["direction"]
    prob       = result["probability"]
    confidence = result["confidence"]
    tech_prob  = result["technical_prob"]
    cv         = predictor.cv_metrics

    is_up = direction == "UP"

    # ── Main prediction card ───────────────────────────────────────
    col_card, col_stats = st.columns([1.2, 1])

    with col_card:
        card_class = "pred-up" if is_up else "pred-down"
        arrow      = "▲" if is_up else "▼"
        label_color= "#00D4FF" if is_up else "#FF4B4B"

        st.markdown(f"""
        <div class="pred-card {card_class}">
            <div class="pred-arrow">{arrow}</div>
            <div class="pred-label" style="color:{label_color}">{direction}</div>
            <div class="pred-conf">Confidence: <b>{confidence:.1f}%</b></div>
            <br>
            <div style="color:#C0C8D4;font-size:0.9rem">
                Probability of price going UP tomorrow:<br>
                <span style="font-size:1.8rem;font-weight:800;color:{label_color}">
                    {prob*100:.1f}%
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_stats:
        st.markdown("##### Model Performance (Backtested)")
        mean_acc = cv.get("mean_accuracy", 0)
        mean_auc = cv.get("mean_auc", 0)
        mean_f1  = cv.get("mean_f1", 0)
        std_acc  = cv.get("std_accuracy", 0)

        m1, m2 = st.columns(2)
        m1.metric("CV Accuracy",  f"{mean_acc*100:.1f}%", f"±{std_acc*100:.1f}%")
        m2.metric("ROC-AUC",      f"{mean_auc:.3f}")
        m1.metric("F1 Score",     f"{mean_f1:.3f}")
        m2.metric("XGBoost P",    f"{result['p_xgb']*100:.1f}%")

        # Fold breakdown
        folds = cv.get("fold_accuracies", [])
        if folds:
            st.markdown("**Fold Accuracies:**")
            fold_df = pd.DataFrame({
                "Fold": [f"Fold {i+1}" for i in range(len(folds))],
                "Accuracy": [f"{a*100:.1f}%" for a in folds],
            })
            st.dataframe(fold_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Probability gauge ──────────────────────────────────────────
    st.markdown("##### Prediction Probability Breakdown")
    gauge_fig = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = prob * 100,
        delta = {"reference": 50, "suffix": "%"},
        number= {"suffix": "%", "font": {"size": 52, "color": "#E8ECF0"}},
        gauge = {
            "axis":     {"range": [0, 100], "tickcolor": "#8892A4"},
            "bar":      {"color": "#00D4FF" if is_up else "#FF4B4B"},
            "bgcolor":  "#1A1F2E",
            "steps": [
                {"range": [0,  40], "color": "rgba(255,75,75,0.15)"},
                {"range": [40, 60], "color": "rgba(240,180,41,0.10)"},
                {"range": [60,100], "color": "rgba(0,212,255,0.15)"},
            ],
            "threshold": {
                "line":  {"color": "#FFFFFF", "width": 3},
                "value": 50,
            },
        },
        title = {"text": "P(UP Tomorrow)", "font": {"color": "#8892A4"}},
    ))
    gauge_fig.update_layout(
        paper_bgcolor="#0E1117", font_color="#E8ECF0",
        height=300, margin=dict(l=30, r=30, t=30, b=10),
    )
    st.plotly_chart(gauge_fig, use_container_width=True)

    # ── Sentiment influence ────────────────────────────────────────
    if sent_data["available"]:
        st.info(
            f"🧠 Sentiment overlay: **{sent_data['label']}** "
            f"(score {sent_score:+.2f}) — shifted technical probability "
            f"({tech_prob*100:.1f}%) → **final {prob*100:.1f}%**"
        )


# ═══════════════════════════════════════════════════════════════════
# TAB 3 — SENTIMENT
# ═══════════════════════════════════════════════════════════════════

with tab_sentiment:

    with st.spinner("Fetching sentiment data…"):
        sent = get_combined_sentiment(ticker)

    fg  = fetch_fear_greed_index()

    # ── Top row ────────────────────────────────────────────────────
    c_sent, c_fg, c_redd, c_news = st.columns(4)

    with c_sent:
        label_class = _sentiment_color(sent["label"])
        score_pct   = int((sent["score"] + 1) / 2 * 100)
        st.markdown(f"""
        <div class="sent-card">
            <div style="color:#8892A4;font-size:0.78rem;text-transform:uppercase;
                        letter-spacing:.05em">Market Sentiment</div>
            <div class="sent-score {label_class}">{sent['score']:+.2f}</div>
            <div style="font-size:1.1rem;font-weight:700;
                        color:{'#00D4FF' if sent['label']=='Positive' else '#FF4B4B' if sent['label']=='Negative' else '#F0B429'}">
                {sent["label"]}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_fg:
        fg_val   = fg["value"]
        fg_class = fg["classification"]
        fg_col   = _fg_color(fg_val)
        st.markdown(f"""
        <div class="sent-card">
            <div style="color:#8892A4;font-size:0.78rem;text-transform:uppercase;
                        letter-spacing:.05em">Fear & Greed</div>
            <div class="sent-score" style="color:{fg_col}">{fg_val}</div>
            <div style="font-size:1.1rem;font-weight:700;color:{fg_col}">{fg_class}</div>
        </div>
        """, unsafe_allow_html=True)

    with c_redd:
        r_score = sent["reddit_score"]
        r_label = sent["label"] if sent["available"] else "N/A"
        r_col   = _chg_color(r_score)
        st.markdown(f"""
        <div class="sent-card">
            <div style="color:#8892A4;font-size:0.78rem;text-transform:uppercase;
                        letter-spacing:.05em">Reddit Score</div>
            <div class="sent-score" style="color:{r_col}">{r_score:+.2f}</div>
            <div style="font-size:0.9rem;color:#8892A4">
                {len(sent.get('reddit_posts', []))} posts analysed
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_news:
        n_score = sent["news_score"]
        n_col   = _chg_color(n_score)
        st.markdown(f"""
        <div class="sent-card">
            <div style="color:#8892A4;font-size:0.78rem;text-transform:uppercase;
                        letter-spacing:.05em">News Score</div>
            <div class="sent-score" style="color:{n_col}">{n_score:+.2f}</div>
            <div style="font-size:0.9rem;color:#8892A4">
                {len(sent.get('news_articles', []))} articles analysed
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # ── Fear & Greed history ───────────────────────────────────────
    if not fg["history"].empty:
        st.markdown("#### Crypto Fear & Greed — 30-Day History")
        st.plotly_chart(build_fear_greed_chart(fg["history"]),
                        use_container_width=True)

    # ── Post feeds ────────────────────────────────────────────────
    col_r, col_n = st.columns(2)

    with col_r:
        st.markdown("#### 📢 Reddit Posts")
        posts = sent.get("reddit_posts", [])
        if posts:
            for p in posts[:10]:
                score_col = _chg_color(p["score"])
                label_bg  = "#0A3D2E" if p["score"] > 0.05 else (
                            "#2D0D0D" if p["score"] < -0.05 else "#1A1F2E")
                st.markdown(f"""
                <div style="background:{label_bg};border-radius:8px;
                            padding:10px 14px;margin-bottom:8px;
                            border-left:3px solid {score_col}">
                    <div style="font-size:0.82rem;color:{score_col};font-weight:700">
                        {p['label']}  {p['score']:+.3f}  ·  {p['source']}
                    </div>
                    <div style="font-size:0.88rem;color:#C0C8D4;margin-top:3px">
                        {p['title']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Reddit data unavailable (rate limited — try again in a moment).")

    with col_n:
        st.markdown("#### 📰 News Headlines")
        articles = sent.get("news_articles", [])
        if articles:
            for a in articles[:10]:
                score_col = _chg_color(a["score"])
                label_bg  = "#0A3D2E" if a["score"] > 0.05 else (
                            "#2D0D0D" if a["score"] < -0.05 else "#1A1F2E")
                st.markdown(f"""
                <div style="background:{label_bg};border-radius:8px;
                            padding:10px 14px;margin-bottom:8px;
                            border-left:3px solid {score_col}">
                    <div style="font-size:0.82rem;color:{score_col};font-weight:700">
                        {a['label']}  {a['score']:+.3f}  ·  {a['source']}
                    </div>
                    <div style="font-size:0.88rem;color:#C0C8D4;margin-top:3px">
                        {a['title']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("News data unavailable.")


# ═══════════════════════════════════════════════════════════════════
# TAB 4 — EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════

with tab_explain:

    # Predictor must already be loaded from Tab 2
    if predictor is None:
        st.error("Model not available.")
        st.stop()

    result_e = predictor.predict(df_full, sentiment_score=sent_data.get("score", 0.0))
    shap_vals = result_e["shap_values"]
    feat_names = result_e["feature_names"]

    st.markdown(
        "#### Why did the model predict **"
        + result_e["direction"]
        + "**? — SHAP Analysis"
    )
    st.markdown(
        "_SHAP (SHapley Additive exPlanations) shows how much each "
        "feature **pushed** the prediction toward UP (blue, positive) "
        "or DOWN (red, negative)._"
    )

    # Plotly SHAP bar chart
    st.plotly_chart(
        build_shap_bar(shap_vals, feat_names),
        use_container_width=True,
    )

    st.divider()

    # ── SHAP waterfall (matplotlib) ────────────────────────────────
    st.markdown("#### SHAP Waterfall — Top 15 Drivers")

    try:
        # Build a proper shap.Explanation object
        expected_val = result_e["expected_value"]
        explanation  = shap.Explanation(
            values       = shap_vals,
            base_values  = expected_val,
            data         = result_e["last_features"].values[0],
            feature_names= feat_names,
        )

        plt.figure(figsize=(10, 7))
        plt.rcParams.update({
            "figure.facecolor": "#0E1117",
            "axes.facecolor":   "#0E1117",
            "text.color":       "#E8ECF0",
            "axes.labelcolor":  "#E8ECF0",
            "xtick.color":      "#E8ECF0",
            "ytick.color":      "#E8ECF0",
        })
        shap.plots.waterfall(explanation, max_display=15, show=False)
        plt.tight_layout()
        st.pyplot(plt.gcf(), clear_figure=True)
        plt.close("all")

    except Exception as e:
        st.warning(f"Waterfall chart unavailable: {e}")

    # ── Feature importance table ───────────────────────────────────
    st.markdown("#### Top Feature Importances (XGBoost Gain)")
    fi_df = predictor.feature_importances().head(20)
    if not fi_df.empty:
        fi_fig = px.bar(
            fi_df, x="importance", y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale=["#2D3748", "#00D4FF"],
        )
        fi_fig.update_layout(
            **CHART_LAYOUT,
            height=500,
            showlegend=False,
            coloraxis_showscale=False,
            xaxis_title="Importance (Gain)",
            yaxis_title=None,
        )
        fi_fig.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fi_fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    "<p style='text-align:center;color:#4A5568;font-size:0.78rem'>"
    "StockIQ — Educational AI dashboard. Not financial advice. "
    "Predictions are probabilistic and based on historical patterns."
    "</p>",
    unsafe_allow_html=True,
)

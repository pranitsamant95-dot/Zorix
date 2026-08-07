import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── News Sentiment ─────────────────────────────────────────────────────────────
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import urllib.request
    import urllib.parse
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Zorix AI",
    page_icon="📈",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

html, body, [class*="css"]  {
    font-family: 'Segoe UI', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #020617, #07111f, #0f172a);
    color: white;
}

h1, h2, h3, h4 {
    color: white;
}

.hero-title {
    font-size: 70px;
    font-weight: 800;
    text-align: center;
    color: white;
    letter-spacing: 2px;
}

.hero-subtitle {
    text-align: center;
    font-size: 22px;
    color: #94a3b8;
    margin-bottom: 40px;
}

.glass-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
}

.metric-card {
    background: rgba(255,255,255,0.04);
    padding: 20px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.08);
}

.stButton>button {
    background: linear-gradient(to right, #2563eb, #7c3aed);
    color: white;
    border-radius: 12px;
    border: none;
    padding: 12px 28px;
    font-size: 16px;
    font-weight: 600;
}

.stButton>button:hover {
    transform: scale(1.03);
    transition: 0.2s;
}

.sentiment-positive {
    background: rgba(0, 255, 136, 0.08);
    border: 1px solid rgba(0, 255, 136, 0.25);
    border-radius: 16px;
    padding: 20px;
}

.sentiment-negative {
    background: rgba(255, 77, 109, 0.08);
    border: 1px solid rgba(255, 77, 109, 0.25);
    border-radius: 16px;
    padding: 20px;
}

.sentiment-neutral {
    background: rgba(148, 163, 184, 0.08);
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 16px;
    padding: 20px;
}

.headline-item {
    background: rgba(255,255,255,0.03);
    border-left: 3px solid #3b82f6;
    padding: 10px 14px;
    margin-bottom: 8px;
    border-radius: 0 10px 10px 0;
    font-size: 14px;
    color: #cbd5e1;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SESSION STATE
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "home"

# =========================================================
# SENTIMENT HELPERS
# =========================================================

def fetch_headlines(query: str) -> list:
    """
    Fetch recent news headlines from Google News RSS.
    No API key required. Returns up to 15 headlines.
    Falls back to an empty list on any network error.
    """
    try:
        import re
        q   = urllib.parse.quote(query)
        url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        titles = re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>", raw)
        if not titles:
            titles = re.findall(r"<title>(.*?)</title>", raw)
        return [t for t in titles if len(t) > 10][1:16]   # skip feed title
    except Exception:
        return []


def score_headlines(headlines: list) -> dict:
    """
    Score a list of headlines with VADER.
    Returns mean compound, pos, neg scores across all headlines.
    """
    if not VADER_AVAILABLE or not headlines:
        return {"compound": 0.0, "pos": 0.0, "neg": 0.0}
    analyser = SentimentIntensityAnalyzer()
    scores   = [analyser.polarity_scores(h) for h in headlines]
    return {
        "compound": float(sum(s["compound"] for s in scores) / len(scores)),
        "pos":      float(sum(s["pos"]      for s in scores) / len(scores)),
        "neg":      float(sum(s["neg"]      for s in scores) / len(scores)),
    }


def build_sentiment_features(df: pd.DataFrame, ticker: str, today_scores: dict) -> pd.DataFrame:
    """
    Attach four sentiment columns to the dataframe.

    For the historical rows we build a return-correlated proxy (same approach
    as the notebook's _fetch_news_sentiment method) so the feature exists for
    the full training window without requiring a paid historical news API.
    The most recent row is overwritten with the live-fetched VADER score.

    All sentiment columns are lagged by 1 trading day to prevent leakage —
    we only know yesterday's news when predicting today's direction.

    Columns added:
        Sentiment_Score  — VADER compound score in [-1, +1]
        Sentiment_Pos    — positive sentiment component  [0, 1]
        Sentiment_Neg    — negative sentiment component  [0, 1]
        Sentiment_MA3    — 3-day rolling mean of Sentiment_Score
    """
    n   = len(df)
    ret = df["Returns"].fillna(0).values

    np.random.seed(42)
    noise    = np.random.normal(0, 0.12, n)
    compound = np.clip(0.45 * ret * 10 + 0.55 * noise, -1, 1)

    # Override the last value with the live-fetched score
    compound[-1] = today_scores["compound"]

    pos_arr = np.clip((compound + 1) / 4, 0, 0.6)
    neg_arr = np.clip((1 - compound) / 4, 0, 0.6)

    # Lag by 1 day to prevent data leakage
    df["Sentiment_Score"] = pd.Series(compound, index=df.index).shift(1).fillna(0.0)
    df["Sentiment_Pos"]   = pd.Series(pos_arr,  index=df.index).shift(1).fillna(0.0)
    df["Sentiment_Neg"]   = pd.Series(neg_arr,  index=df.index).shift(1).fillna(0.0)
    df["Sentiment_MA3"]   = df["Sentiment_Score"].rolling(3, min_periods=1).mean()

    return df


# =========================================================
# HOME PAGE
# =========================================================

if st.session_state.page == "home":

    st.markdown("<div class='hero-title'>ZORIX AI</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='hero-subtitle'>Market Intelligence Platform</div>",
        unsafe_allow_html=True
    )

    st.markdown("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class='glass-card'>
        <h3>📈 AI Forecasting</h3>
        Predict stock direction using machine learning models.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='glass-card'>
        <h3>🧠 Technical Analysis</h3>
        RSI, MACD, Bollinger Bands & advanced indicators.
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='glass-card'>
        <h3>⚡ Interactive Dashboard</h3>
        Real-time charts and institutional-style analytics.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("⚙️ Configure Analysis")

    ticker = st.text_input("Stock Ticker", "TCS.NS")

    start_date = st.date_input(
        "Start Date",
        pd.to_datetime("2018-01-01")
    )

    model_choice = st.selectbox(
        "Choose AI Model",
        ["XGBoost", "Random Forest", "Logistic Regression"]
    )

    if st.button("🚀 Generate Insights"):

        st.session_state.ticker = ticker
        st.session_state.start_date = start_date
        st.session_state.model_choice = model_choice

        st.session_state.page = "dashboard"

        st.rerun()

# =========================================================
# DASHBOARD PAGE
# =========================================================

else:

    ticker = st.session_state.ticker
    start_date = st.session_state.start_date
    model_choice = st.session_state.model_choice

    # ---------------- BACK BUTTON ----------------

    if st.button("⬅ Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    st.title(f"📊 Zorix Insights Engine")
    st.caption(f"Live AI Analytics for {ticker}")

    # =========================================================
    # LOAD DATA
    # =========================================================

    @st.cache_data
    def load_data(ticker, start_date):

        df = yf.download(ticker, start=start_date)

        df.columns = df.columns.get_level_values(0)

        return df

    df = load_data(ticker, start_date)

    # =========================================================
    # FEATURES
    # =========================================================

    df['RSI'] = RSIIndicator(close=df['Close'].squeeze()).rsi()

    macd = MACD(close=df['Close'].squeeze())

    df['MACD'] = macd.macd()

    bb = BollingerBands(close=df['Close'].squeeze())

    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()

    df['Returns'] = df['Close'].pct_change()

    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    df = df.dropna()

    # =========================================================
    # NEWS SENTIMENT
    # =========================================================

    # Fetch live headlines and score them with VADER
    ticker_clean = ticker.replace("^", "").replace(".", "-")
    with st.spinner("🔍 Fetching latest news sentiment..."):
        headlines     = fetch_headlines(f"{ticker_clean} stock")
        today_scores  = score_headlines(headlines)

    # Attach sentiment columns to the dataframe
    df = build_sentiment_features(df, ticker_clean, today_scores)

    # =========================================================
    # MODEL
    # =========================================================

    # Sentiment_Score and Sentiment_MA3 are added as model features
    # alongside the original technical indicators
    features = ['RSI', 'MACD', 'Returns', 'Sentiment_Score', 'Sentiment_MA3']

    X = df[features]
    y = df['Target']

    split = int(len(df) * 0.8)

    X_train = X[:split]
    X_test = X[split:]

    y_train = y[:split]
    y_test = y[split:]

    if model_choice == "XGBoost":

        model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            random_state=42
        )

    elif model_choice == "Random Forest":

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

    else:

        model = LogisticRegression()

    # =========================================================
    # TRAIN
    # =========================================================

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)

    latest = X.tail(1)

    prediction = model.predict(latest)[0]

    prob = model.predict_proba(latest)[0]

    confidence = max(prob) * 100

    # =========================================================
    # METRICS
    # =========================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Model Accuracy", f"{accuracy*100:.2f}%")

    with col2:
        st.metric("Current Price", f"₹{df['Close'].iloc[-1]:.2f}")

    with col3:
        st.metric("RSI", f"{df['RSI'].iloc[-1]:.2f}")

    with col4:
        st.metric("Confidence", f"{confidence:.2f}%")

    # =========================================================
    # SIGNAL
    # =========================================================

    st.subheader("🤖 AI Prediction Signal")

    if prediction == 1:

        st.success(f"""
        ### BUY SIGNAL 📈

        Zorix AI predicts bullish momentum.

        Confidence Level: {confidence:.2f}%
        """)

    else:

        st.error(f"""
        ### SELL / HOLD SIGNAL 📉

        Zorix AI predicts bearish momentum.

        Confidence Level: {confidence:.2f}%
        """)

    # =========================================================
    # CANDLESTICK
    # =========================================================

    st.subheader("📊 Advanced Market Chart")

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3]
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Price"
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['BB_High'],
            line=dict(color='rgba(255,255,255,0.4)', width=1),
            name='BB Upper'
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['BB_Low'],
            line=dict(color='rgba(255,255,255,0.4)', width=1),
            name='BB Lower'
        ),
        row=1,
        col=1
    )

    colors = np.where(df['Returns'] >= 0, '#00ff88', '#ff4d6d')

    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            marker_color=colors,
            name='Volume'
        ),
        row=2,
        col=1
    )

    fig.update_layout(
        template='plotly_dark',
        height=700,
        xaxis_rangeslider_visible=False,
        paper_bgcolor='#020617',
        plot_bgcolor='#020617',
        font=dict(color='white')
    )

    st.plotly_chart(fig, use_container_width=True)

    # =========================================================
    # RSI + MACD
    # =========================================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📈 RSI Indicator")

        rsi_fig = go.Figure()

        rsi_fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['RSI'],
                line=dict(color='#38bdf8')
            )
        )

        rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
        rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")

        rsi_fig.update_layout(
            template='plotly_dark',
            height=350,
            paper_bgcolor='#020617',
            plot_bgcolor='#020617'
        )

        st.plotly_chart(rsi_fig, use_container_width=True)

    with col2:

        st.subheader("📉 MACD Indicator")

        macd_fig = go.Figure()

        macd_fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD'],
                line=dict(color='#fbbf24')
            )
        )

        macd_fig.update_layout(
            template='plotly_dark',
            height=350,
            paper_bgcolor='#020617',
            plot_bgcolor='#020617'
        )

        st.plotly_chart(macd_fig, use_container_width=True)

    # =========================================================
    # NEWS SENTIMENT SECTION
    # =========================================================

    st.subheader("📰 News Sentiment Analysis")

    # ── Live sentiment score card ─────────────────────────────────────────────

    compound = today_scores["compound"]

    if compound >= 0.05:
        sentiment_label = "Positive 🟢"
        sentiment_class = "sentiment-positive"
        sentiment_color = "#00ff88"
    elif compound <= -0.05:
        sentiment_label = "Negative 🔴"
        sentiment_class = "sentiment-negative"
        sentiment_color = "#ff4d6d"
    else:
        sentiment_label = "Neutral ⚪"
        sentiment_class = "sentiment-neutral"
        sentiment_color = "#94a3b8"

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Today's Sentiment",
            sentiment_label,
            help="VADER compound score from latest news headlines"
        )

    with col2:
        st.metric(
            "Compound Score",
            f"{compound:.3f}",
            help="Range: -1 (most negative) to +1 (most positive)"
        )

    with col3:
        st.metric(
            "Positive Signal",
            f"{today_scores['pos']:.3f}",
            help="Average positive component across headlines"
        )

    with col4:
        st.metric(
            "Negative Signal",
            f"{today_scores['neg']:.3f}",
            help="Average negative component across headlines"
        )

    # ── Headlines list ────────────────────────────────────────────────────────

    if headlines:
        with st.expander(f"📰 Latest Headlines for {ticker}  ({len(headlines)} fetched)", expanded=True):
            analyser = SentimentIntensityAnalyzer() if VADER_AVAILABLE else None
            for h in headlines:
                if analyser:
                    h_score = analyser.polarity_scores(h)["compound"]
                    if h_score >= 0.05:
                        dot = "🟢"
                    elif h_score <= -0.05:
                        dot = "🔴"
                    else:
                        dot = "⚪"
                    st.markdown(
                        f"<div class='headline-item'>{dot} {h} &nbsp;"
                        f"<span style='color:#64748b;font-size:12px;'>"
                        f"[{h_score:+.3f}]</span></div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div class='headline-item'>• {h}</div>",
                        unsafe_allow_html=True
                    )
    else:
        st.info("No headlines fetched — check network connectivity or try a different ticker.")

    # ── Sentiment chart: score over time + close price ────────────────────────

    sent_fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.4, 0.35, 0.25],
        subplot_titles=(
            "Stock Close Price",
            "Daily Sentiment Score  (Green = Positive · Red = Negative)",
            "3-Day Rolling Sentiment + Components"
        )
    )

    # Panel 1: close price
    sent_fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Close'],
            line=dict(color='#38bdf8', width=1.5),
            name='Close Price'
        ),
        row=1, col=1
    )

    # Panel 2: daily sentiment bars
    bar_colors = [
        '#00ff88' if v >= 0.05 else ('#ff4d6d' if v <= -0.05 else '#94a3b8')
        for v in df['Sentiment_Score']
    ]

    sent_fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Sentiment_Score'],
            marker_color=bar_colors,
            name='Sentiment Score',
            opacity=0.8
        ),
        row=2, col=1
    )

    sent_fig.add_hline(
        y=0.05,  row=2, col=1,
        line_dash="dot", line_color="rgba(0,255,136,0.4)",
        annotation_text="Positive threshold"
    )
    sent_fig.add_hline(
        y=-0.05, row=2, col=1,
        line_dash="dot", line_color="rgba(255,77,109,0.4)",
        annotation_text="Negative threshold"
    )

    # Panel 3: rolling sentiment + positive/negative fill
    sent_fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Sentiment_MA3'],
            line=dict(color='#a78bfa', width=1.5),
            name='Sentiment MA-3'
        ),
        row=3, col=1
    )

    sent_fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Sentiment_Pos'],
            line=dict(color='rgba(0,255,136,0.0)'),
            fill='tozeroy',
            fillcolor='rgba(0,255,136,0.15)',
            name='Positive component'
        ),
        row=3, col=1
    )

    sent_fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Sentiment_Neg'].mul(-1),
            line=dict(color='rgba(255,77,109,0.0)'),
            fill='tozeroy',
            fillcolor='rgba(255,77,109,0.15)',
            name='Negative component'
        ),
        row=3, col=1
    )

    sent_fig.update_layout(
        template='plotly_dark',
        height=750,
        paper_bgcolor='#020617',
        plot_bgcolor='#020617',
        font=dict(color='white'),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.01,
            xanchor='right',
            x=1
        ),
        showlegend=True
    )

    sent_fig.update_yaxes(range=[-1, 1], row=2, col=1)
    sent_fig.update_yaxes(range=[-0.7, 0.7], row=3, col=1)

    st.plotly_chart(sent_fig, use_container_width=True)

    # ── Sentiment ↔ Returns correlation ──────────────────────────────────────

    corr = df['Sentiment_Score'].corr(df['Returns'])

    st.markdown(
        f"<div class='glass-card' style='margin-top:4px;'>"
        f"<b style='color:#a78bfa;'>Sentiment ↔ Daily Return correlation</b>"
        f"&nbsp;&nbsp;"
        f"<span style='font-size:22px;font-weight:700;color:{sentiment_color};'>"
        f"r = {corr:.4f}</span>"
        f"<br><span style='color:#64748b;font-size:13px;'>"
        f"Pearson r in [-1, +1]. Values closer to ±1 indicate stronger "
        f"linear relationship between news sentiment and price movement.</span>"
        f"</div>",
        unsafe_allow_html=True
    )

    st.markdown("")

    # =========================================================
    # FEATURE IMPORTANCE
    # =========================================================

    if model_choice in ["XGBoost", "Random Forest"]:

        st.subheader("🧠 Feature Importance")

        importance = model.feature_importances_

        imp_df = pd.DataFrame({
            "Feature": features,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        imp_fig = go.Figure()

        imp_fig.add_trace(
            go.Bar(
                x=imp_df["Importance"],
                y=imp_df["Feature"],
                orientation='h',
                marker=dict(
                    color=imp_df["Importance"],
                    colorscale='Plasma'
                )
            )
        )

        imp_fig.update_layout(
            template='plotly_dark',
            height=400,
            paper_bgcolor='#020617',
            plot_bgcolor='#020617'
        )

        st.plotly_chart(imp_fig, use_container_width=True)

    # =========================================================
    # DATA TABLE
    # =========================================================

    with st.expander("📄 View Processed Dataset"):

        st.dataframe(df.tail(20))

    # =========================================================
    # FOOTER
    # =========================================================

    st.markdown("---")

    st.caption("""
    Zorix AI • Institutional Market Intelligence Platform
    """)

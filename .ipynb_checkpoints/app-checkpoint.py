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

</style>
""", unsafe_allow_html=True)

# =========================================================
# SESSION STATE
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "home"

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

    features = ['RSI', 'MACD', 'Returns']

    X = df[features]
    y = df['Target']

    split = int(len(df) * 0.8)

    X_train = X[:split]
    X_test = X[split:]

    y_train = y[:split]
    y_test = y[split:]

    # =========================================================
    # MODEL
    # =========================================================

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
                orientation='h'
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
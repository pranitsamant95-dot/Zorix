# Hybrid Stock Market Prediction System
### XGBoost + LSTM Ensemble for Next-Day Direction Forecasting

> **Portfolio project** — B.Tech Data Science | Quant Finance & ML

---

## What This Project Does

Predicts whether a stock will close **UP or DOWN** the next trading day using a hybrid ensemble of XGBoost and a Bidirectional LSTM. It compares five models, backtests a trading strategy, and produces 14 publication-quality charts.

**This is a classification problem, not price regression.**  
The output is a probability: `P(next day close > today's close)`.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (default: AAPL, 5 years)
python stock_predictor.py

# 3. To change the stock, edit Config at the top of the file:
#    TICKER = "TSLA"   # or MSFT, RELIANCE.NS, TCS.NS, etc.
```

---

## Project Structure

```
stock_predictor.py         ← Full pipeline (single-file, modular)
requirements.txt           ← All dependencies
README.md                  ← This file
outputs/                   ← Auto-created: all charts and CSVs
saved_models/              ← Auto-created: trained model files
```

---

## Models Implemented

| Model | Type | Why Included |
|---|---|---|
| Logistic Regression | Linear baseline | Fast, interpretable, establishes baseline |
| Random Forest | Ensemble trees | Non-linear, handles feature interactions |
| XGBoost | Gradient boosting | Best single-model for tabular finance data |
| LSTM (Bidirectional) | Deep learning | Captures temporal sequential patterns |
| **Hybrid Ensemble** | **XGBoost + LSTM** | **Combines strengths of both** |

---

## Features Engineered (~60 features)

- SMA / EMA (5, 10, 20, 50, 200 day)
- RSI (7, 14 day)
- MACD + Signal + Histogram
- Bollinger Bands (width, % position)
- ATR (Average True Range)
- ADX (trend strength)
- OBV (On-Balance Volume), VWAP, MFI
- Stochastic Oscillator, Williams %R, CCI
- Daily returns, log returns, momentum
- Rolling volatility, skew, mean
- Lag features (returns, RSI, volume × 5 lags)
- S&P 500 correlation (Beta-20)
- Sentiment score placeholder (extensible)

---

## Outputs Generated

| File | Description |
|---|---|
| `01_price_history.png` | Close price + moving averages + volume |
| `02_candlestick.html` | Interactive Plotly candlestick + Bollinger |
| `03_correlation_heatmap.png` | Feature correlation matrix |
| `04_technical_indicators.png` | RSI, MACD, volume on last 1 year |
| `05_roc_curves.png` | ROC curves for all 5 models |
| `06_feature_importance.png` | Top 25 XGBoost features |
| `07_lstm_training_history.png` | Loss + accuracy curves |
| `08_model_comparison.png` | Grouped bar chart of all metrics |
| `09_cumulative_returns.png` | Strategy backtest vs buy-and-hold |
| `10_confusion_matrices.png` | Confusion matrices for all models |
| `11_prediction_confidence.png` | Predicted probability distributions |
| `12_shap_bar.png` | SHAP global feature importance |
| `13_shap_beeswarm.png` | SHAP direction + magnitude |
| `14_monte_carlo.png` | 500-path GBM price simulation |
| `predictions_Hybrid.csv` | Full prediction log with signals |

---

## Key Design Decisions

**No random shuffling** — train/test split is strictly chronological.  
Using a random shuffle leaks future data into training, inflating accuracy by ~10-15 pp.

**RobustScaler not StandardScaler** — financial returns have heavy tails and outliers.  
RobustScaler uses median and IQR, making it less sensitive to extreme values.

**Weighted ensemble (55% XGBoost, 45% LSTM)** — adjustable in `Config`.  
XGBoost gets slightly higher weight because it consistently outperforms on tabular features.

**EarlyStopping on LSTM** — prevents overfitting to recent regime patterns.

---

## Extending This Project

**Add real sentiment:**
```python
# Replace the placeholder in _sentiment_placeholder():
from transformers import pipeline
finbert = pipeline("text-classification", model="ProsusAI/finbert")
# Score news headlines and join to df by date
```

**Change the stock:**
```python
# In Config:
TICKER    = "TCS.NS"     # NSE India
BENCHMARK = "^NSEI"      # NIFTY 50
```

**Add live prediction:**
```python
# After training, fetch today's data and run inference:
today = collector.fetch(cfg.TICKER, start="2024-01-01", end=datetime.today().strftime("%Y-%m-%d"))
```

---

## Limitations & Disclaimers

- This is an **educational portfolio project**, not a live trading system.
- No transaction costs, slippage, or market impact are modelled.
- Past directional accuracy does not guarantee future performance.
- Financial ML models degrade as market regimes change.
- Do not use this for actual investment decisions.

---

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · TensorFlow/Keras  
Matplotlib · Seaborn · Plotly · yfinance · ta · SHAP

---

*Built for Data Science internship portfolios and quant finance interview preparation.*

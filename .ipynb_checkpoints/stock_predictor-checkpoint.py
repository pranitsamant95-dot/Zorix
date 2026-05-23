# =============================================================================
# HYBRID STOCK MARKET PREDICTION SYSTEM
# XGBoost + LSTM Ensemble for Next-Day Direction Forecasting
# =============================================================================
# Author  : Data Science / Quant Finance Portfolio Project
# Purpose : Predict whether a stock will close UP or DOWN the next day
# Models  : Logistic Regression, Random Forest, XGBoost, LSTM, Hybrid Ensemble
# =============================================================================

# ── Standard Library ──────────────────────────────────────────────────────────
import os
import warnings
import logging
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Data & Numerics ───────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

# ── Finance Data ──────────────────────────────────────────────────────────────
import yfinance as yf

# ── Technical Analysis ────────────────────────────────────────────────────────
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    logger.warning("'ta' library not found. Install via: pip install ta")

# ── Visualisation ─────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")          # non-interactive backend; works in all environments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not found. Interactive charts will be skipped.")

# ── Machine Learning ──────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

import xgboost as xgb

# ── Deep Learning ─────────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        LSTM, Dense, Dropout, BatchNormalization,
        Input, Bidirectional, GRU
    )
    from tensorflow.keras.callbacks import (
        EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    )
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
    # Suppress TensorFlow INFO / WARNING logs
    tf.get_logger().setLevel("ERROR")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not found. LSTM model will be skipped.")

# ── SHAP Interpretability ─────────────────────────────────────────────────────
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not found. Interpretability section will be skipped.")


# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

class Config:
    """
    Central configuration class.
    Change these values to run the project on a different stock or time range.
    """
    # Stock settings
    TICKER        = "AAPL"          # Primary stock ticker
    BENCHMARK     = "^GSPC"         # S&P 500 as market benchmark
    START_DATE    = "2019-01-01"    # 5+ years of history
    END_DATE      = datetime.today().strftime("%Y-%m-%d")

    # Model settings
    TEST_SIZE     = 0.20            # 20% of data reserved for testing
    RANDOM_STATE  = 42
    CV_SPLITS     = 5               # Time-series cross-validation folds

    # LSTM settings
    SEQUENCE_LEN  = 60              # Look-back window (60 trading days ≈ 3 months)
    LSTM_EPOCHS   = 60
    LSTM_BATCH    = 32
    LSTM_UNITS    = 64

    # Hybrid ensemble weights (XGBoost vs LSTM)
    XGB_WEIGHT    = 0.55
    LSTM_WEIGHT   = 0.45

    # Output paths
    OUTPUT_DIR    = "outputs"
    MODEL_DIR     = "saved_models"

    # Columns always excluded from feature sets
    DROP_COLS     = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Target"]


# =============================================================================
# SECTION 2 — DATA COLLECTION
# =============================================================================

class DataCollector:
    """
    Fetches historical OHLCV data and benchmark index data using yfinance.

    Why yfinance?
    - Free, no API key required
    - Adjusted close prices account for splits and dividends
    - Supports global exchanges (NYSE, NASDAQ, NSE, etc.)
    """

    def __init__(self, config: Config):
        self.cfg = config

    def fetch(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Download OHLCV data for the given ticker."""
        logger.info(f"Fetching data for {ticker} from {start} to {end}")
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError(f"No data returned for ticker: {ticker}")

        # Flatten multi-level columns that yfinance sometimes produces
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index = pd.to_datetime(df.index)
        df.dropna(inplace=True)
        logger.info(f"  Fetched {len(df)} rows for {ticker}")
        return df

    def fetch_all(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch primary stock and benchmark index."""
        stock = self.fetch(self.cfg.TICKER, self.cfg.START_DATE, self.cfg.END_DATE)
        bench = self.fetch(self.cfg.BENCHMARK, self.cfg.START_DATE, self.cfg.END_DATE)
        return stock, bench


# =============================================================================
# SECTION 3 — FEATURE ENGINEERING
# =============================================================================

class FeatureEngineer:
    """
    Creates a rich feature set from raw OHLCV data.

    Why feature engineering matters for stock prediction:
    - Raw prices carry no predictive signal on their own
    - Technical indicators capture momentum, trend, and volatility — patterns
      that traders actually use to make decisions
    - Lag features allow the model to learn from recent history
    - Benchmark correlation captures macro market conditions

    NOTE: All features are engineered using only PAST data.
    Using future data (data leakage) would give artificially inflated accuracy.
    """

    def __init__(self, config: Config):
        self.cfg = config

    # ── Price-based features ──────────────────────────────────────────────────

    def _price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Daily returns, log returns, gaps, and price ratios."""
        df["Daily_Return"]   = df["Close"].pct_change()
        df["Log_Return"]     = np.log(df["Close"] / df["Close"].shift(1))
        df["Overnight_Gap"]  = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
        df["High_Low_Range"] = (df["High"] - df["Low"]) / df["Close"]
        df["Close_Open_Diff"] = (df["Close"] - df["Open"]) / df["Open"]
        return df

    # ── Moving averages ───────────────────────────────────────────────────────

    def _moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """SMA and EMA at multiple windows; crossover signals."""
        for w in [5, 10, 20, 50, 200]:
            df[f"SMA_{w}"]  = df["Close"].rolling(w).mean()
            df[f"EMA_{w}"]  = df["Close"].ewm(span=w, adjust=False).mean()
            df[f"SMA_Ratio_{w}"] = df["Close"] / df[f"SMA_{w}"]  # price relative to MA

        # Golden / death cross signals (price relative to long-term trend)
        df["SMA_5_20_Cross"]  = df["SMA_5"]  - df["SMA_20"]
        df["SMA_20_50_Cross"] = df["SMA_20"] - df["SMA_50"]
        df["EMA_12_26_Cross"] = df["EMA_12"] - df["EMA_26"] if "EMA_12" in df else (
            df["Close"].ewm(span=12).mean() - df["Close"].ewm(span=26).mean()
        )
        return df

    # ── Momentum indicators ───────────────────────────────────────────────────

    def _momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """RSI, MACD, and raw momentum."""
        # RSI (Relative Strength Index) — identifies overbought / oversold conditions
        if TA_AVAILABLE:
            df["RSI_14"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
            df["RSI_7"]  = ta.momentum.RSIIndicator(df["Close"], window=7).rsi()
            macd = ta.trend.MACD(df["Close"])
            df["MACD"]        = macd.macd()
            df["MACD_Signal"] = macd.macd_signal()
            df["MACD_Hist"]   = macd.macd_diff()
            df["Stoch_K"]     = ta.momentum.StochasticOscillator(
                df["High"], df["Low"], df["Close"]).stoch()
            df["Williams_R"]  = ta.momentum.WilliamsRIndicator(
                df["High"], df["Low"], df["Close"]).williams_r()
            df["ROC_10"]      = ta.momentum.ROCIndicator(df["Close"], window=10).roc()
        else:
            # Manual RSI calculation (fallback if 'ta' not installed)
            delta  = df["Close"].diff()
            gain   = delta.clip(lower=0).rolling(14).mean()
            loss   = (-delta.clip(upper=0)).rolling(14).mean()
            rs     = gain / (loss + 1e-9)
            df["RSI_14"]      = 100 - (100 / (1 + rs))
            df["MACD"]        = df["Close"].ewm(12).mean() - df["Close"].ewm(26).mean()
            df["MACD_Signal"] = df["MACD"].ewm(9).mean()
            df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

        # Raw price momentum
        for w in [5, 10, 20]:
            df[f"Momentum_{w}"] = df["Close"] - df["Close"].shift(w)

        return df

    # ── Volatility indicators ─────────────────────────────────────────────────

    def _volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bollinger Bands, ATR, and rolling volatility."""
        # Rolling volatility (annualised)
        for w in [5, 10, 20]:
            df[f"Volatility_{w}"] = df["Daily_Return"].rolling(w).std() * np.sqrt(252)

        # Bollinger Bands
        if TA_AVAILABLE:
            bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
            df["BB_Upper"] = bb.bollinger_hband()
            df["BB_Lower"] = bb.bollinger_lband()
            df["BB_Mid"]   = bb.bollinger_mavg()
            df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Mid"]
            df["BB_Pct"]   = bb.bollinger_pband()   # where is price within the band
            # ATR (Average True Range) — measures market volatility
            df["ATR_14"]   = ta.volatility.AverageTrueRange(
                df["High"], df["Low"], df["Close"], window=14).average_true_range()
        else:
            sma20 = df["Close"].rolling(20).mean()
            std20 = df["Close"].rolling(20).std()
            df["BB_Upper"] = sma20 + 2 * std20
            df["BB_Lower"] = sma20 - 2 * std20
            df["BB_Mid"]   = sma20
            df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Mid"]
            df["BB_Pct"]   = (df["Close"] - df["BB_Lower"]) / (
                df["BB_Upper"] - df["BB_Lower"] + 1e-9)
            high_low   = df["High"] - df["Low"]
            high_close = (df["High"] - df["Close"].shift(1)).abs()
            low_close  = (df["Low"]  - df["Close"].shift(1)).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df["ATR_14"] = tr.rolling(14).mean()

        return df

    # ── Volume features ───────────────────────────────────────────────────────

    def _volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume momentum and on-balance volume."""
        df["Volume_Change"] = df["Volume"].pct_change()
        df["Volume_SMA_20"] = df["Volume"].rolling(20).mean()
        df["Volume_Ratio"]  = df["Volume"] / (df["Volume_SMA_20"] + 1e-9)

        if TA_AVAILABLE:
            df["OBV"]   = ta.volume.OnBalanceVolumeIndicator(df["Close"], df["Volume"]).on_balance_volume()
            df["VWAP"]  = ta.volume.VolumeWeightedAveragePrice(
                df["High"], df["Low"], df["Close"], df["Volume"]).volume_weighted_average_price()
            df["MFI_14"] = ta.volume.MFIIndicator(
                df["High"], df["Low"], df["Close"], df["Volume"], window=14).money_flow_index()
        else:
            obv = (np.sign(df["Daily_Return"]) * df["Volume"]).cumsum()
            df["OBV"] = obv

        return df

    # ── Trend indicators ──────────────────────────────────────────────────────

    def _trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ADX trend strength and rolling statistics."""
        if TA_AVAILABLE:
            adx = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=14)
            df["ADX"]    = adx.adx()
            df["DI_Plus"]  = adx.adx_pos()
            df["DI_Minus"] = adx.adx_neg()
            df["CCI_20"]   = ta.trend.CCIIndicator(
                df["High"], df["Low"], df["Close"], window=20).cci()

        # Rolling statistics of returns
        for w in [5, 10, 20]:
            df[f"Roll_Mean_{w}"] = df["Daily_Return"].rolling(w).mean()
            df[f"Roll_Std_{w}"]  = df["Daily_Return"].rolling(w).std()
            df[f"Roll_Skew_{w}"] = df["Daily_Return"].rolling(w).skew()

        # Trend direction dummy (price above its 50-day SMA)
        if "SMA_50" in df.columns:
            df["Above_SMA50"] = (df["Close"] > df["SMA_50"]).astype(int)

        return df

    # ── Lag features ──────────────────────────────────────────────────────────

    def _lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Lag features give the model access to recent history.
        We lag returns, RSI, volume — the most informative signals.
        """
        for lag in range(1, 6):
            df[f"Return_Lag_{lag}"]  = df["Daily_Return"].shift(lag)
            df[f"Volume_Lag_{lag}"]  = df["Volume_Change"].shift(lag)
            if "RSI_14" in df.columns:
                df[f"RSI_Lag_{lag}"] = df["RSI_14"].shift(lag)

        return df

    # ── Benchmark correlation ─────────────────────────────────────────────────

    def _benchmark_features(self, df: pd.DataFrame, bench: pd.DataFrame) -> pd.DataFrame:
        """
        Add S&P 500 returns as a feature.
        A stock heavily correlated with the market will tend to move with it.
        """
        bench_ret = bench["Close"].pct_change().rename("SP500_Return")
        df = df.join(bench_ret, how="left")

        # Rolling correlation between stock and benchmark (20-day window)
        if "SP500_Return" in df.columns:
            df["Beta_20"] = (
                df["Daily_Return"].rolling(20).cov(df["SP500_Return"])
                / df["SP500_Return"].rolling(20).var()
            )

        return df

    # ── Sentiment placeholder ─────────────────────────────────────────────────

    def _sentiment_placeholder(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Placeholder architecture for future sentiment integration.

        In a production system this column would be replaced with real
        sentiment scores from news headlines (FinBERT / VADER) or
        options market implied volatility.
        """
        # Zero-filled — neutral sentiment assumption
        df["Sentiment_Score"] = 0.0
        return df

    # ── Target variable ───────────────────────────────────────────────────────

    def _create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Binary classification target:
          1 → next-day close is HIGHER than today's close  (UP)
          0 → next-day close is LOWER than today's close   (DOWN)

        We shift the close by -1 to get tomorrow's close aligned with today's features.
        This is the key prediction we care about.
        """
        df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
        return df

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def build_features(self, df: pd.DataFrame, bench: pd.DataFrame) -> pd.DataFrame:
        """Run the full feature engineering pipeline."""
        logger.info("Building feature set...")
        df = df.copy()
        df = self._price_features(df)
        df = self._moving_averages(df)
        df = self._momentum(df)
        df = self._volatility(df)
        df = self._volume_features(df)
        df = self._trend_features(df)
        df = self._lag_features(df)
        df = self._benchmark_features(df, bench)
        df = self._sentiment_placeholder(df)
        df = self._create_target(df)

        # Drop the last row (target is NaN — no "tomorrow" for the final day)
        df.dropna(inplace=True)
        logger.info(f"  Feature matrix shape: {df.shape}")
        return df


# =============================================================================
# SECTION 4 — DATA SPLITTING (Time-Series Aware)
# =============================================================================

class TimeSeriesSplitter:
    """
    Handles proper train / test splitting for time-series data.

    WHY NOT RANDOM SHUFFLE?
    ─────────────────────────────────────────────────────────────────────────
    In standard ML we shuffle data and split randomly. For time-series this
    is WRONG because:
      • It leaks future information into the training set
      • A model that "sees" Oct 15 while training on Oct 10 will look great
        on paper but fail completely in live trading
      • This is one of the most common mistakes in finance ML

    The correct approach: train on the EARLIEST data, test on the LATEST.
    """

    def __init__(self, config: Config):
        self.cfg = config

    def split(self, df: pd.DataFrame) -> tuple:
        """
        Chronological train-test split.
        Returns feature matrices and target vectors.
        """
        feature_cols = [c for c in df.columns if c not in self.cfg.DROP_COLS]

        X = df[feature_cols]
        y = df["Target"]

        split_idx = int(len(df) * (1 - self.cfg.TEST_SIZE))

        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Keep dates for plotting
        dates_test = df.index[split_idx:]

        logger.info(f"  Train: {len(X_train)} rows | Test: {len(X_test)} rows")
        logger.info(f"  Train period: {X_train.index[0].date()} → {X_train.index[-1].date()}")
        logger.info(f"  Test  period: {X_test.index[0].date()}  → {X_test.index[-1].date()}")

        return X_train, X_test, y_train, y_test, feature_cols, dates_test


# =============================================================================
# SECTION 5 — MODEL TRAINING
# =============================================================================

class ModelTrainer:
    """
    Trains all models: Logistic Regression, Random Forest, XGBoost, LSTM,
    and the final Hybrid ensemble.
    """

    def __init__(self, config: Config):
        self.cfg     = config
        self.scaler  = RobustScaler()  # RobustScaler is less sensitive to outliers
        self.models  = {}
        self.results = {}

    # ── Scale features ────────────────────────────────────────────────────────

    def _scale(self, X_train, X_test):
        """Fit scaler on train, transform both sets. Never fit on test."""
        X_tr = self.scaler.fit_transform(X_train)
        X_te = self.scaler.transform(X_test)
        return X_tr, X_te

    # ── Logistic Regression ───────────────────────────────────────────────────

    def train_logistic(self, X_train, X_test, y_train, y_test):
        """
        Baseline linear model.
        Fast, interpretable, but assumes linear decision boundary —
        unlikely to capture complex market patterns.
        """
        logger.info("Training Logistic Regression...")
        X_tr, X_te = self._scale(X_train, X_test)
        clf = LogisticRegression(
            max_iter=1000, C=0.1,
            random_state=self.cfg.RANDOM_STATE
        )
        clf.fit(X_tr, y_train)
        proba = clf.predict_proba(X_te)[:, 1]
        preds = (proba >= 0.5).astype(int)
        self.models["Logistic"] = clf
        return self._evaluate("Logistic Regression", y_test, preds, proba)

    # ── Random Forest ─────────────────────────────────────────────────────────

    def train_random_forest(self, X_train, X_test, y_train, y_test):
        """
        Ensemble of decision trees.
        Handles non-linear relationships and is robust to outliers,
        but can be slow and prone to overfitting with too many trees.
        """
        logger.info("Training Random Forest...")
        clf = RandomForestClassifier(
            n_estimators=300, max_depth=8,
            min_samples_leaf=10, n_jobs=-1,
            random_state=self.cfg.RANDOM_STATE
        )
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)
        self.models["RandomForest"] = clf
        return self._evaluate("Random Forest", y_test, preds, proba)

    # ── XGBoost ───────────────────────────────────────────────────────────────

    def train_xgboost(self, X_train, X_test, y_train, y_test):
        """
        Gradient-boosted trees — often the best performer on tabular finance data.

        WHY XGBOOST WORKS WELL FOR FINANCE:
        • Handles missing values natively
        • Built-in regularisation prevents overfitting
        • Captures complex non-linear feature interactions
        • Feature importance tells us which indicators matter most
        • Fast training with parallelised tree building
        """
        logger.info("Training XGBoost...")
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        clf = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,   # handles class imbalance
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=self.cfg.RANDOM_STATE,
            n_jobs=-1
        )
        clf.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        proba = clf.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)
        self.models["XGBoost"] = clf
        return self._evaluate("XGBoost", y_test, preds, proba)

    # ── LSTM ──────────────────────────────────────────────────────────────────

    def _build_sequences(self, X_scaled, y, seq_len):
        """
        Reshape tabular data into 3D sequences for LSTM input.
        Shape: (samples, time_steps, features)

        Each sample is a window of `seq_len` consecutive days of features.
        The label for that sample is the target of the LAST day in the window.
        """
        Xs, ys = [], []
        for i in range(seq_len, len(X_scaled)):
            Xs.append(X_scaled[i - seq_len: i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    def _build_lstm_model(self, n_features: int) -> "tf.keras.Model":
        """
        Bidirectional LSTM architecture.

        Architecture rationale:
        • Bidirectional LSTM: reads the sequence forwards AND backwards,
          capturing patterns in both directions of the time window
        • BatchNormalization: stabilises training, reduces internal covariate shift
        • Dropout: regularisation to prevent overfitting on financial noise
        • Dense(64) + Dense(1): final classification layers
        """
        model = Sequential([
            Input(shape=(self.cfg.SEQUENCE_LEN, n_features)),

            Bidirectional(LSTM(self.cfg.LSTM_UNITS, return_sequences=True)),
            BatchNormalization(),
            Dropout(0.3),

            Bidirectional(LSTM(self.cfg.LSTM_UNITS // 2, return_sequences=False)),
            BatchNormalization(),
            Dropout(0.3),

            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid")   # binary classification output
        ])

        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def train_lstm(self, X_train_raw, X_test_raw, y_train, y_test):
        """
        Train a Bidirectional LSTM on sequence data.

        WHY LSTM FOR STOCK DATA:
        • LSTM (Long Short-Term Memory) was designed for sequential data
        • It can learn long-term dependencies — e.g., a pattern that unfolds
          over many trading days
        • The "gates" (input, forget, output) allow it to remember or
          forget information selectively

        WEAKNESS: LSTMs are computationally expensive, need large datasets,
        and can overfit on noisy financial data without careful regularisation.
        """
        if not TF_AVAILABLE:
            logger.warning("TensorFlow unavailable. Skipping LSTM.")
            return None, None, None

        logger.info("Training LSTM...")

        # Scale BEFORE building sequences (still fit only on train)
        scaler_lstm = RobustScaler()
        X_tr_sc = scaler_lstm.fit_transform(X_train_raw)
        X_te_sc = scaler_lstm.transform(X_test_raw)

        y_tr_arr = y_train.values
        y_te_arr = y_test.values

        seq = self.cfg.SEQUENCE_LEN
        X_tr_seq, y_tr_seq = self._build_sequences(X_tr_sc, y_tr_arr, seq)
        X_te_seq, y_te_seq = self._build_sequences(X_te_sc, y_te_arr, seq)

        if len(X_tr_seq) == 0:
            logger.error("Not enough data to build LSTM sequences.")
            return None, None, None

        model = self._build_lstm_model(X_tr_seq.shape[2])

        os.makedirs(self.cfg.MODEL_DIR, exist_ok=True)
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
        ]

        history = model.fit(
            X_tr_seq, y_tr_seq,
            validation_split=0.15,
            epochs=self.cfg.LSTM_EPOCHS,
            batch_size=self.cfg.LSTM_BATCH,
            callbacks=callbacks,
            verbose=0
        )

        proba = model.predict(X_te_seq, verbose=0).flatten()
        preds = (proba >= 0.5).astype(int)

        # y_te_seq is aligned with the sequence offsets
        result = self._evaluate("LSTM", y_te_seq, preds, proba)

        self.models["LSTM"] = (model, scaler_lstm)
        return result, history, (X_te_seq, y_te_seq)

    # ── Hybrid Ensemble ───────────────────────────────────────────────────────

    def train_hybrid(self, xgb_proba, lstm_proba, y_test_xgb, y_test_lstm=None):
        """
        Weighted average ensemble of XGBoost and LSTM probabilities.

        WHY HYBRID MODELS ARE MORE ROBUST:
        • XGBoost and LSTM capture DIFFERENT types of patterns:
            - XGBoost: tabular feature interactions (technical indicator combinations)
            - LSTM: temporal sequential patterns (how momentum evolves over time)
        • When their errors are uncorrelated, combining them reduces variance
        • This is the same principle used in fund-of-funds portfolio construction

        Two options (configurable):
          A) Weighted average (simple, transparent)
          B) Meta-classifier: train a second-level model on XGB + LSTM outputs
        """
        logger.info("Building Hybrid Ensemble...")

        if lstm_proba is None:
            # Fall back to XGBoost-only if LSTM wasn't trained
            logger.warning("LSTM unavailable. Hybrid = XGBoost only.")
            hybrid_proba = xgb_proba
            y_ref        = y_test_xgb
        else:
            # Align lengths (LSTM sequence offset removes first `seq_len` rows)
            n = min(len(xgb_proba), len(lstm_proba))
            xgb_p  = xgb_proba[-n:]
            lstm_p = lstm_proba
            y_ref  = y_test_lstm if y_test_lstm is not None else y_test_xgb[-n:]

            hybrid_proba = (
                self.cfg.XGB_WEIGHT  * xgb_p +
                self.cfg.LSTM_WEIGHT * lstm_p
            )

        preds = (hybrid_proba >= 0.5).astype(int)
        return self._evaluate("Hybrid (XGB + LSTM)", y_ref, preds, hybrid_proba), hybrid_proba

    # ── Evaluation utility ────────────────────────────────────────────────────

    def _evaluate(self, name: str, y_true, y_pred, y_proba) -> dict:
        """
        Compute a comprehensive set of classification metrics.

        METRICS EXPLAINED:
        • Accuracy:   % of correct predictions (misleading for imbalanced classes)
        • Precision:  of all predicted UPs, how many were actually UP?
        • Recall:     of all actual UPs, how many did we catch?
        • F1-score:   harmonic mean of precision and recall (balanced measure)
        • ROC-AUC:    area under the ROC curve (1.0 = perfect, 0.5 = random)
        • Directional accuracy: same as accuracy here, but named explicitly
        """
        result = {
            "name":       name,
            "accuracy":   accuracy_score(y_true, y_pred),
            "precision":  precision_score(y_true, y_pred, zero_division=0),
            "recall":     recall_score(y_true, y_pred, zero_division=0),
            "f1":         f1_score(y_true, y_pred, zero_division=0),
            "roc_auc":    roc_auc_score(y_true, y_proba),
            "confusion":  confusion_matrix(y_true, y_pred),
            "y_true":     np.array(y_true),
            "y_pred":     np.array(y_pred),
            "y_proba":    np.array(y_proba),
        }

        logger.info(
            f"  [{name:25s}]  Acc={result['accuracy']:.3f}  "
            f"F1={result['f1']:.3f}  AUC={result['roc_auc']:.3f}"
        )

        self.results[name] = result
        return result


# =============================================================================
# SECTION 6 — BACKTESTING & STRATEGY SIMULATION
# =============================================================================

class Backtester:
    """
    Simulates a simple trading strategy based on model predictions.

    Strategy logic:
    • If model predicts UP  (1) → buy at today's close, sell at tomorrow's close
    • If model predicts DOWN (0) → hold cash (no trade)
    • Compare cumulative return vs. buy-and-hold benchmark

    IMPORTANT DISCLAIMER:
    This is a simplified simulation for portfolio/educational purposes.
    It does NOT account for: transaction costs, slippage, market impact,
    short-selling restrictions, or position sizing.
    """

    @staticmethod
    def simulate(prices: pd.Series, y_pred: np.ndarray, name: str) -> pd.Series:
        """
        Returns a Series of cumulative strategy returns.
        `prices` must be aligned with `y_pred`.
        """
        daily_ret = prices.pct_change().fillna(0).values
        n = min(len(daily_ret), len(y_pred))
        daily_ret = daily_ret[-n:]
        preds     = y_pred[-n:]

        # Strategy return: earn market return only on days we predicted UP
        strategy_ret  = np.where(preds == 1, daily_ret, 0.0)
        cumulative_ret = np.cumprod(1 + strategy_ret)
        buyhold_ret    = np.cumprod(1 + daily_ret)

        return pd.Series(cumulative_ret, name=f"{name} Strategy"), \
               pd.Series(buyhold_ret,   name="Buy & Hold")

    @staticmethod
    def sharpe(returns: np.ndarray, risk_free: float = 0.04) -> float:
        """Annualised Sharpe Ratio."""
        daily_rf  = risk_free / 252
        excess    = returns - daily_rf
        if excess.std() == 0:
            return 0.0
        return np.sqrt(252) * excess.mean() / excess.std()

    @staticmethod
    def max_drawdown(cum_returns: np.ndarray) -> float:
        """Maximum percentage drawdown from peak to trough."""
        peak = np.maximum.accumulate(cum_returns)
        dd   = (cum_returns - peak) / peak
        return dd.min()


# =============================================================================
# SECTION 7 — VISUALISATIONS
# =============================================================================

class Visualiser:
    """
    Generates all plots and saves them to the outputs directory.
    Uses Matplotlib for static publication-quality charts,
    and Plotly for interactive finance-style charts where available.
    """

    def __init__(self, config: Config):
        self.cfg = config
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        sns.set_theme(style="darkgrid", palette="muted")

    def _save(self, fig, name: str):
        path = os.path.join(self.cfg.OUTPUT_DIR, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Saved: {path}")

    # ── 1. Stock price history ────────────────────────────────────────────────

    def plot_price_history(self, df: pd.DataFrame):
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        fig.suptitle(f"{self.cfg.TICKER} — Price & Volume History", fontsize=15, fontweight="bold")

        axes[0].plot(df.index, df["Close"], color="#1f77b4", linewidth=1.2, label="Close")
        if "SMA_20" in df.columns:
            axes[0].plot(df.index, df["SMA_20"], color="orange", linewidth=0.8, label="SMA 20", alpha=0.8)
        if "SMA_50" in df.columns:
            axes[0].plot(df.index, df["SMA_50"], color="red",    linewidth=0.8, label="SMA 50", alpha=0.8)
        axes[0].set_ylabel("Price (USD)")
        axes[0].legend(fontsize=9)
        axes[0].set_title("Closing Price with Moving Averages")

        axes[1].bar(df.index, df["Volume"] / 1e6, color="#aec7e8", alpha=0.7)
        axes[1].set_ylabel("Volume (M)")
        axes[1].set_title("Daily Volume")

        if "Daily_Return" in df.columns:
            ret = df["Daily_Return"] * 100
            axes[2].bar(df.index, ret, color=np.where(ret >= 0, "#2ca02c", "#d62728"), alpha=0.7, width=1)
            axes[2].axhline(0, color="black", linewidth=0.5)
            axes[2].set_ylabel("Daily Return (%)")
            axes[2].set_title("Daily Returns")

        plt.tight_layout()
        self._save(fig, "01_price_history")

    # ── 2. Candlestick chart (Plotly) ─────────────────────────────────────────

    def plot_candlestick(self, df: pd.DataFrame):
        if not PLOTLY_AVAILABLE:
            return

        recent = df.tail(120)  # last 6 months
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.7, 0.3], vertical_spacing=0.03)

        fig.add_trace(go.Candlestick(
            x=recent.index, open=recent["Open"], high=recent["High"],
            low=recent["Low"], close=recent["Close"], name="OHLC"
        ), row=1, col=1)

        if "BB_Upper" in recent.columns:
            fig.add_trace(go.Scatter(x=recent.index, y=recent["BB_Upper"],
                line=dict(color="rgba(255,165,0,0.5)", width=1), name="BB Upper"), row=1, col=1)
            fig.add_trace(go.Scatter(x=recent.index, y=recent["BB_Lower"],
                line=dict(color="rgba(255,165,0,0.5)", width=1), name="BB Lower",
                fill="tonexty", fillcolor="rgba(255,165,0,0.05)"), row=1, col=1)

        colors = ["#2ca02c" if r >= 0 else "#d62728" for r in recent["Daily_Return"].fillna(0)]
        fig.add_trace(go.Bar(
            x=recent.index, y=recent["Volume"] / 1e6,
            marker_color=colors, name="Volume (M)"
        ), row=2, col=1)

        fig.update_layout(
            title=f"{self.cfg.TICKER} — Interactive Candlestick (Last 6 Months)",
            xaxis_rangeslider_visible=False, height=600, template="plotly_dark"
        )
        path = os.path.join(self.cfg.OUTPUT_DIR, "02_candlestick.html")
        fig.write_html(path)
        logger.info(f"  Saved interactive chart: {path}")

    # ── 3. Correlation heatmap ────────────────────────────────────────────────

    def plot_correlation_heatmap(self, df: pd.DataFrame, feature_cols: list):
        top_features = [c for c in feature_cols if c in df.columns][:25]
        corr = df[top_features + ["Target"]].corr()

        fig, ax = plt.subplots(figsize=(16, 14))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=False, fmt=".2f",
                    cmap="RdYlGn", center=0, ax=ax,
                    linewidths=0.3, cbar_kws={"shrink": 0.8})
        ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
        plt.tight_layout()
        self._save(fig, "03_correlation_heatmap")

    # ── 4. Technical indicators ───────────────────────────────────────────────

    def plot_technical_indicators(self, df: pd.DataFrame):
        recent = df.tail(252)  # last year
        fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
        fig.suptitle(f"{self.cfg.TICKER} — Technical Indicators (Last 1 Year)", fontsize=14, fontweight="bold")

        # Price + Bollinger Bands
        axes[0].plot(recent.index, recent["Close"], label="Close", color="#1f77b4")
        if "BB_Upper" in recent.columns:
            axes[0].fill_between(recent.index, recent["BB_Upper"], recent["BB_Lower"],
                                 alpha=0.15, color="orange", label="Bollinger Bands")
        axes[0].legend(fontsize=9); axes[0].set_ylabel("Price")

        # RSI
        if "RSI_14" in recent.columns:
            axes[1].plot(recent.index, recent["RSI_14"], color="purple", linewidth=1)
            axes[1].axhline(70, color="red",   linestyle="--", linewidth=0.8, label="Overbought (70)")
            axes[1].axhline(30, color="green", linestyle="--", linewidth=0.8, label="Oversold (30)")
            axes[1].legend(fontsize=9); axes[1].set_ylabel("RSI"); axes[1].set_ylim(0, 100)

        # MACD
        if "MACD" in recent.columns:
            axes[2].plot(recent.index, recent["MACD"],        color="#1f77b4", label="MACD",   linewidth=1)
            axes[2].plot(recent.index, recent["MACD_Signal"], color="orange",  label="Signal", linewidth=1)
            if "MACD_Hist" in recent.columns:
                colors = np.where(recent["MACD_Hist"] >= 0, "#2ca02c", "#d62728")
                axes[2].bar(recent.index, recent["MACD_Hist"], color=colors, alpha=0.5)
            axes[2].axhline(0, color="black", linewidth=0.5)
            axes[2].legend(fontsize=9); axes[2].set_ylabel("MACD")

        # Volume
        vol_colors = np.where(recent["Daily_Return"].fillna(0) >= 0, "#2ca02c", "#d62728")
        axes[3].bar(recent.index, recent["Volume"] / 1e6, color=vol_colors, alpha=0.7)
        axes[3].set_ylabel("Volume (M)")

        plt.tight_layout()
        self._save(fig, "04_technical_indicators")

    # ── 5. ROC curves ─────────────────────────────────────────────────────────

    def plot_roc_curves(self, results: dict):
        fig, ax = plt.subplots(figsize=(8, 7))
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        for (name, res), color in zip(results.items(), colors):
            fpr, tpr, _ = roc_curve(res["y_true"], res["y_proba"])
            ax.plot(fpr, tpr, label=f"{name} (AUC = {res['roc_auc']:.3f})", color=color, linewidth=2)

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10); ax.grid(True, alpha=0.4)
        self._save(fig, "05_roc_curves")

    # ── 6. Feature importance ─────────────────────────────────────────────────

    def plot_feature_importance(self, model, feature_cols: list, top_n: int = 25):
        if not hasattr(model, "feature_importances_"):
            return
        imp = pd.Series(model.feature_importances_, index=feature_cols).nlargest(top_n)

        fig, ax = plt.subplots(figsize=(10, 8))
        imp.sort_values().plot.barh(ax=ax, color="#1f77b4", edgecolor="white")
        ax.set_title(f"XGBoost — Top {top_n} Feature Importances", fontsize=13, fontweight="bold")
        ax.set_xlabel("Importance Score")
        plt.tight_layout()
        self._save(fig, "06_feature_importance")

    # ── 7. LSTM training loss ─────────────────────────────────────────────────

    def plot_lstm_history(self, history):
        if history is None:
            return
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("LSTM Training History", fontsize=13, fontweight="bold")

        axes[0].plot(history.history["loss"],     label="Train Loss")
        axes[0].plot(history.history["val_loss"], label="Val Loss")
        axes[0].set_title("Loss"); axes[0].legend(); axes[0].set_xlabel("Epoch")

        axes[1].plot(history.history["accuracy"],     label="Train Acc")
        axes[1].plot(history.history["val_accuracy"], label="Val Acc")
        axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].set_xlabel("Epoch")

        plt.tight_layout()
        self._save(fig, "07_lstm_training_history")

    # ── 8. Model comparison ───────────────────────────────────────────────────

    def plot_model_comparison(self, results: dict):
        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        names   = list(results.keys())
        values  = {m: [results[n][m] for n in names] for m in metrics}

        x     = np.arange(len(metrics))
        width = 0.14
        fig, ax = plt.subplots(figsize=(14, 6))

        for i, name in enumerate(names):
            vals = [values[m][i] for m in metrics]
            ax.bar(x + i * width, vals, width, label=name, alpha=0.85)

        ax.set_xticks(x + width * (len(names) - 1) / 2)
        ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.4)
        plt.tight_layout()
        self._save(fig, "08_model_comparison")

    # ── 9. Cumulative returns ─────────────────────────────────────────────────

    def plot_cumulative_returns(self, strategy_series: list, buyhold: pd.Series):
        fig, ax = plt.subplots(figsize=(13, 6))
        for s in strategy_series:
            ax.plot(s.values, label=s.name, linewidth=1.5)
        ax.plot(buyhold.values, label="Buy & Hold", linestyle="--",
                color="black", linewidth=1.5)
        ax.axhline(1.0, color="gray", linewidth=0.5, linestyle=":")
        ax.set_xlabel("Trading Days (Test Period)")
        ax.set_ylabel("Cumulative Return (1 = starting capital)")
        ax.set_title("Strategy Backtesting — Cumulative Returns", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.4)
        plt.tight_layout()
        self._save(fig, "09_cumulative_returns")

    # ── 10. Confusion matrices ────────────────────────────────────────────────

    def plot_confusion_matrices(self, results: dict):
        n = len(results)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        if n == 1:
            axes = [axes]

        for ax, (name, res) in zip(axes, results.items()):
            sns.heatmap(res["confusion"], annot=True, fmt="d", cmap="Blues",
                        xticklabels=["DOWN", "UP"], yticklabels=["DOWN", "UP"], ax=ax)
            ax.set_title(name, fontsize=10, fontweight="bold")
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

        fig.suptitle("Confusion Matrices", fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._save(fig, "10_confusion_matrices")

    # ── 11. Prediction confidence distribution ────────────────────────────────

    def plot_prediction_confidence(self, results: dict):
        fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 4))
        if len(results) == 1:
            axes = [axes]

        for ax, (name, res) in zip(axes, results.items()):
            up_conf   = res["y_proba"][res["y_true"] == 1]
            down_conf = res["y_proba"][res["y_true"] == 0]
            ax.hist(up_conf,   bins=25, alpha=0.6, label="Actual UP",   color="#2ca02c")
            ax.hist(down_conf, bins=25, alpha=0.6, label="Actual DOWN", color="#d62728")
            ax.axvline(0.5, color="black", linestyle="--", linewidth=0.8)
            ax.set_title(name, fontsize=10)
            ax.legend(fontsize=8)
            ax.set_xlabel("Predicted Probability")

        fig.suptitle("Prediction Confidence Distribution", fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._save(fig, "11_prediction_confidence")


# =============================================================================
# SECTION 8 — SHAP INTERPRETABILITY
# =============================================================================

class ShapAnalyser:
    """
    Uses SHAP (SHapley Additive exPlanations) to explain XGBoost predictions.

    WHY SHAP?
    • Black-box models are hard to trust in finance — regulators and
      risk managers need to know WHY a model made a decision
    • SHAP assigns each feature a "contribution" to each individual prediction
    • Derived from cooperative game theory (Shapley values)
    • Shows both global importance AND per-prediction attribution
    """

    def __init__(self, config: Config):
        self.cfg = config

    def analyse(self, model, X_test: pd.DataFrame, feature_cols: list):
        if not SHAP_AVAILABLE:
            logger.warning("SHAP unavailable. Skipping interpretability section.")
            return

        logger.info("Running SHAP analysis...")
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)

        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # SHAP summary bar plot (global importance)
        fig1, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_cols,
                          plot_type="bar", show=False, max_display=20)
        plt.title("SHAP Feature Importance (Global)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig1.savefig(os.path.join(self.cfg.OUTPUT_DIR, "12_shap_bar.png"),
                     dpi=150, bbox_inches="tight")
        plt.close(fig1)

        # SHAP beeswarm plot (direction of effect)
        fig2, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_cols,
                          show=False, max_display=20)
        plt.title("SHAP Beeswarm — Feature Direction & Magnitude", fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig2.savefig(os.path.join(self.cfg.OUTPUT_DIR, "13_shap_beeswarm.png"),
                     dpi=150, bbox_inches="tight")
        plt.close(fig2)

        logger.info("  SHAP analysis complete.")


# =============================================================================
# SECTION 9 — BONUS: VOLATILITY REGIME DETECTION
# =============================================================================

class VolatilityRegimeDetector:
    """
    Classifies market into HIGH and LOW volatility regimes using
    a rolling realised volatility threshold.

    WHY IT MATTERS:
    • Models trained on all market conditions often underperform in specific regimes
    • In high-volatility regimes (crashes, earnings surprises) even good models
      can perform worse because market behaviour becomes more random
    • Regime-aware trading can apply different risk rules in different conditions
    """

    @staticmethod
    def detect(df: pd.DataFrame, window: int = 20, threshold_pct: float = 75) -> pd.Series:
        """
        Returns a Series: 1 = high volatility regime, 0 = low volatility regime.
        Threshold is the Nth percentile of rolling volatility.
        """
        vol = df["Daily_Return"].rolling(window).std() * np.sqrt(252)
        threshold = vol.quantile(threshold_pct / 100)
        regime = (vol > threshold).astype(int)
        regime.name = "Volatility_Regime"
        return regime


# =============================================================================
# SECTION 10 — BONUS: MONTE CARLO SIMULATION
# =============================================================================

class MonteCarloSimulator:
    """
    Simulates possible future stock price paths using the
    Geometric Brownian Motion (GBM) model.

    Used in quantitative finance for:
    • Option pricing (Black-Scholes framework)
    • Value-at-Risk (VaR) estimation
    • Stress testing

    NOTE: GBM assumes returns are normally distributed and independent —
    both assumptions are violated in real markets. This is a simplified
    illustration only.
    """

    def __init__(self, n_simulations: int = 500, n_days: int = 30):
        self.n_sims = n_simulations
        self.n_days = n_days

    def simulate(self, last_price: float, mu: float, sigma: float) -> np.ndarray:
        """
        Returns array of shape (n_days, n_simulations).
        mu    = annualised drift (mean log return)
        sigma = annualised volatility
        """
        dt   = 1 / 252
        paths = np.zeros((self.n_days + 1, self.n_sims))
        paths[0] = last_price

        for t in range(1, self.n_days + 1):
            z = np.random.standard_normal(self.n_sims)
            paths[t] = paths[t - 1] * np.exp(
                (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
            )
        return paths

    def plot(self, paths: np.ndarray, ticker: str, output_dir: str):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(paths[:, :100], alpha=0.1, color="#1f77b4", linewidth=0.8)
        ax.plot(np.percentile(paths, 50, axis=1), color="orange", linewidth=2, label="Median")
        ax.plot(np.percentile(paths, 5,  axis=1), color="red",    linewidth=1.5,
                linestyle="--", label="5th Percentile (VaR)")
        ax.plot(np.percentile(paths, 95, axis=1), color="green",  linewidth=1.5,
                linestyle="--", label="95th Percentile")
        ax.set_title(f"{ticker} — Monte Carlo Simulation ({paths.shape[1]} paths, {paths.shape[0]-1} days)",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Trading Days Ahead")
        ax.set_ylabel("Simulated Price (USD)")
        ax.legend(fontsize=10)
        plt.tight_layout()
        path = os.path.join(output_dir, "14_monte_carlo.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Saved: {path}")


# =============================================================================
# SECTION 11 — REPORTING
# =============================================================================

class Reporter:
    """Prints formatted evaluation tables and exports predictions to CSV."""

    @staticmethod
    def print_summary_table(results: dict):
        """Print a clean comparison table of all model metrics."""
        header = f"{'Model':<28} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9}"
        sep    = "─" * 80
        print(f"\n{sep}")
        print("  MODEL PERFORMANCE COMPARISON")
        print(sep)
        print(header)
        print(sep)
        for name, res in results.items():
            print(
                f"  {name:<26} "
                f"{res['accuracy']:>9.4f} "
                f"{res['precision']:>10.4f} "
                f"{res['recall']:>8.4f} "
                f"{res['f1']:>8.4f} "
                f"{res['roc_auc']:>9.4f}"
            )
        print(sep)

    @staticmethod
    def export_predictions(dates, y_true, y_pred, y_proba, name: str, output_dir: str):
        """Save prediction results to CSV."""
        n = min(len(dates), len(y_true), len(y_pred), len(y_proba))
        df = pd.DataFrame({
            "Date":             dates[-n:],
            "Actual":           y_true[-n:],
            "Predicted":        y_pred[-n:],
            "Prob_UP":          y_proba[-n:],
            "Signal":           ["BUY" if p == 1 else "HOLD" for p in y_pred[-n:]],
            "Correct":          (y_true[-n:] == y_pred[-n:]).astype(int)
        })
        path = os.path.join(output_dir, f"predictions_{name.replace(' ', '_')}.csv")
        df.to_csv(path, index=False)
        logger.info(f"  Exported predictions: {path}")
        return df


# =============================================================================
# SECTION 12 — FINAL ANALYSIS
# =============================================================================

FINAL_ANALYSIS = """
================================================================================
  FINAL ANALYSIS — REFLECTIONS ON STOCK PREDICTION WITH ML
================================================================================

1. WHY STOCK PREDICTION IS DIFFICULT
─────────────────────────────────────
  • Markets are highly non-stationary: the patterns that worked last year
    may not work this year. Market participants adapt.
  • The Efficient Market Hypothesis (EMH) argues that all public information
    is already priced in. If a pattern is discovered and widely exploited,
    it quickly disappears (arbitrage).
  • Financial data has a very low signal-to-noise ratio. Daily returns are
    dominated by random noise; genuine signal is weak and fleeting.
  • Regime changes (COVID crash, interest rate cycles) cause model drift.

2. LIMITATIONS OF ARIMA
─────────────────────────
  • ARIMA assumes linear relationships and stationarity.
  • Real stock prices are non-linear, non-stationary, and fat-tailed.
  • ARIMA cannot incorporate exogenous features like technical indicators,
    sentiment, or market regime — all crucial in finance.
  • Useful for short-horizon volatility forecasting (GARCH), less so for
    direction prediction.

3. WHY XGBOOST WORKS WELL FOR TABULAR FINANCE DATA
────────────────────────────────────────────────────
  • Gradient boosting builds an ensemble of trees that sequentially correct
    each other's errors — powerful for capturing complex feature interactions.
  • Built-in L1/L2 regularisation prevents overfitting.
  • Handles missing values natively (common in financial datasets).
  • Feature importance is transparent — critical for compliance and risk management.
  • Fast training: can be retrained daily on new data with minimal overhead.

4. STRENGTHS AND WEAKNESSES OF LSTM
─────────────────────────────────────
  STRENGTHS:
  • Designed specifically for sequential data — learns temporal patterns
    across a look-back window of many days.
  • Can capture long-range dependencies (e.g., seasonal patterns).
  • Flexible input: can combine price, volume, and derived features.

  WEAKNESSES:
  • Computationally expensive to train and tune.
  • Requires large amounts of data to generalise well.
  • Acts as a black box — hard to interpret what it "learned".
  • Sensitive to hyperparameters (sequence length, architecture depth).
  • Often overfits to recent market regimes.

5. WHY ENSEMBLE MODELS ARE MORE ROBUST
───────────────────────────────────────
  • Bias-variance decomposition: combining models with different error patterns
    reduces overall variance without increasing bias.
  • XGBoost captures CROSS-SECTIONAL feature patterns;
    LSTM captures TEMPORAL sequential patterns.
  • Their prediction errors are partially uncorrelated — when one is wrong,
    the other may still be right, and averaging smooths this out.
  • This mirrors diversification in portfolio construction: a basket of
    uncorrelated assets has lower risk than any single asset.

6. OVERFITTING DANGERS IN FINANCE ML
───────────────────────────────────────
  • Backtest overfitting is pervasive: with 50 features and 10 years of data,
    a lucky random model can look fantastic in sample.
  • Walk-forward validation is essential to simulate real deployment.
  • Never optimise hyperparameters on the test set.
  • The "deflated Sharpe ratio" (Lopez de Prado) corrects for multiple testing.
  • Transaction costs and slippage frequently eliminate apparent alpha.

7. FUTURE IMPROVEMENTS
───────────────────────
  • Add NLP sentiment features from financial news (FinBERT).
  • Incorporate options market data (implied volatility surface, put/call ratio).
  • Use alternative data: satellite imagery, credit card transactions, web traffic.
  • Implement adaptive regime detection (Hidden Markov Models).
  • Apply Temporal Fusion Transformer (TFT) — state-of-the-art for time series.
  • Portfolio-level prediction instead of single stock.
  • Live trading integration via Alpaca / Interactive Brokers API.

================================================================================
"""


# =============================================================================
# SECTION 13 — MAIN PIPELINE ORCHESTRATION
# =============================================================================

def main():
    """
    Orchestrates the full end-to-end pipeline:
    1. Load config and set seeds
    2. Collect data
    3. Engineer features
    4. Split data (time-series aware)
    5. Train all models
    6. Backtest strategies
    7. Generate all visualisations
    8. SHAP interpretability
    9. Monte Carlo simulation
    10. Print reports and export CSVs
    """
    print("=" * 80)
    print("  HYBRID STOCK MARKET PREDICTION SYSTEM")
    print("  XGBoost + LSTM Ensemble — Next-Day Direction Forecasting")
    print("=" * 80)

    # ── Setup ──────────────────────────────────────────────────────────────────
    cfg = Config()
    np.random.seed(cfg.RANDOM_STATE)
    if TF_AVAILABLE:
        tf.random.set_seed(cfg.RANDOM_STATE)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(cfg.MODEL_DIR,  exist_ok=True)

    # ── 1. Data collection ─────────────────────────────────────────────────────
    logger.info("STEP 1: Collecting market data...")
    collector   = DataCollector(cfg)
    stock_raw, bench_raw = collector.fetch_all()

    # ── 2. Feature engineering ─────────────────────────────────────────────────
    logger.info("STEP 2: Engineering features...")
    engineer = FeatureEngineer(cfg)
    df = engineer.build_features(stock_raw, bench_raw)

    # ── 3. Data splitting ──────────────────────────────────────────────────────
    logger.info("STEP 3: Splitting data (time-series aware, no leakage)...")
    splitter = TimeSeriesSplitter(cfg)
    X_train, X_test, y_train, y_test, feature_cols, dates_test = splitter.split(df)

    # ── 4. Model training ──────────────────────────────────────────────────────
    logger.info("STEP 4: Training models...")
    trainer = ModelTrainer(cfg)

    res_lr  = trainer.train_logistic(X_train, X_test, y_train, y_test)
    res_rf  = trainer.train_random_forest(X_train, X_test, y_train, y_test)
    res_xgb = trainer.train_xgboost(X_train, X_test, y_train, y_test)

    lstm_result, lstm_history, lstm_test_data = trainer.train_lstm(
        X_train, X_test, y_train, y_test)

    # ── 5. Hybrid ensemble ─────────────────────────────────────────────────────
    lstm_proba    = lstm_test_data[1] if lstm_test_data is not None else None
    lstm_y        = lstm_test_data[1] if lstm_test_data is not None else None
    lstm_proba_arr = (lstm_result["y_proba"] if lstm_result is not None else None)

    res_hybrid, hybrid_proba = trainer.train_hybrid(
        xgb_proba  = res_xgb["y_proba"],
        lstm_proba = lstm_proba_arr,
        y_test_xgb = y_test.values,
        y_test_lstm = (lstm_test_data[1] if lstm_test_data is not None else None)
    )

    # Collect all results for comparison
    all_results = {
        "Logistic Regression": res_lr,
        "Random Forest":       res_rf,
        "XGBoost":             res_xgb,
    }
    if lstm_result is not None:
        all_results["LSTM"] = lstm_result
    all_results["Hybrid (XGB+LSTM)"] = res_hybrid

    # ── 6. Backtesting ─────────────────────────────────────────────────────────
    logger.info("STEP 5: Running strategy backtest...")
    backtester = Backtester()
    test_prices = df["Close"].iloc[-len(y_test):]

    strategy_curves = []
    bh_curve = None
    for name, res in [("XGBoost", res_xgb), ("Hybrid", res_hybrid)]:
        strat, bh = backtester.simulate(test_prices, res["y_pred"], name)
        strategy_curves.append(strat)
        bh_curve = bh

    # ── 7. Visualisations ──────────────────────────────────────────────────────
    logger.info("STEP 6: Generating visualisations...")
    vis = Visualiser(cfg)
    vis.plot_price_history(df)
    vis.plot_candlestick(stock_raw)
    vis.plot_correlation_heatmap(df, feature_cols)
    vis.plot_technical_indicators(df)
    vis.plot_roc_curves(all_results)
    vis.plot_feature_importance(trainer.models["XGBoost"], feature_cols)
    vis.plot_lstm_history(lstm_history)
    vis.plot_model_comparison(all_results)
    if strategy_curves and bh_curve is not None:
        vis.plot_cumulative_returns(strategy_curves, bh_curve)
    vis.plot_confusion_matrices(all_results)
    vis.plot_prediction_confidence(all_results)

    # ── 8. SHAP interpretability ───────────────────────────────────────────────
    logger.info("STEP 7: Running SHAP interpretability...")
    shap_analyser = ShapAnalyser(cfg)
    shap_analyser.analyse(trainer.models["XGBoost"], X_test, feature_cols)

    # ── 9. Volatility regime detection ────────────────────────────────────────
    logger.info("STEP 8: Detecting volatility regimes...")
    regimes = VolatilityRegimeDetector.detect(df)
    high_vol_days = regimes.sum()
    low_vol_days  = (regimes == 0).sum()
    print(f"\n  Volatility Regimes: HIGH={high_vol_days} days | LOW={low_vol_days} days")

    # ── 10. Monte Carlo simulation ────────────────────────────────────────────
    logger.info("STEP 9: Running Monte Carlo simulation...")
    mc = MonteCarloSimulator(n_simulations=500, n_days=30)
    log_ret = df["Log_Return"].dropna()
    mu_annual    = log_ret.mean() * 252
    sigma_annual = log_ret.std()  * np.sqrt(252)
    paths = mc.simulate(df["Close"].iloc[-1], mu_annual, sigma_annual)
    mc.plot(paths, cfg.TICKER, cfg.OUTPUT_DIR)

    # ── 11. Reporting ──────────────────────────────────────────────────────────
    logger.info("STEP 10: Generating reports...")
    Reporter.print_summary_table(all_results)

    reporter = Reporter()
    reporter.export_predictions(
        dates_test, res_hybrid["y_true"], res_hybrid["y_pred"],
        hybrid_proba, "Hybrid", cfg.OUTPUT_DIR
    )

    # Risk metrics for hybrid strategy
    if strategy_curves:
        daily_rets = np.diff(strategy_curves[0].values) / strategy_curves[0].values[:-1]
        sharpe = Backtester.sharpe(daily_rets)
        mdd    = Backtester.max_drawdown(strategy_curves[0].values)
        print(f"\n  Hybrid Strategy Risk Metrics:")
        print(f"    Sharpe Ratio (annualised) : {sharpe:.3f}")
        print(f"    Maximum Drawdown           : {mdd * 100:.2f}%")

    # ── 12. Final analysis ─────────────────────────────────────────────────────
    print(FINAL_ANALYSIS)

    print(f"\n  All outputs saved to: ./{cfg.OUTPUT_DIR}/")
    print("  Project complete.\n")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()

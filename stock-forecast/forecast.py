import numpy as np
import pandas as pd
import yfinance as yf
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from pmdarima import auto_arima
import warnings

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configuration
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = Path("stock-forecast")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
YEARS_BACK = 5
MIN_DOLLAR_VOLUME = 5e6
MIN_OBS = 200
LOOKBACK = 60
RF_RATE = 0.05  # Risk-free rate (5% annual)

def _hash_df(df: pd.DataFrame) -> str:
    """SHA-1 hash of DataFrame for change detection."""
    return hashlib.sha1(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

def _cache_path(ticker: str) -> Path:
    """Cache file path for ticker."""
    return CACHE_DIR / f"{ticker.upper()}.parquet"

def _download_ticker(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.DataFrame]:
    """Download from Yahoo Finance with proper error handling."""
    try:
        df = yf.Ticker(ticker).history(
            start=start, 
            end=end, 
            auto_adjust=False
        )
        if df.empty:
            return None
        df["DollarVolume"] = df["Close"] * df["Volume"]
        return df
    except Exception as e:
        logging.warning(f"Download failed for {ticker}: {e}")
        return None

def load_prices(tickers: List[str], years_back: int = YEARS_BACK) -> Dict[str, pd.DataFrame]:
    """Load prices with cache, liquidity filtering, and deduplication."""
    end = pd.Timestamp.utcnow().normalize()  # UTC for consistency
    start = end - pd.DateOffset(years=years_back)
    
    # Clean tickers
    tickers = pd.unique([t.upper().replace(".", "-") for t in tickers])
    data = {}
    
    for ticker in tickers:
        p = _cache_path(ticker)
        df = None
        
        # Check cache
        if p.exists():
            df = pd.read_parquet(p)
            if df.index.max() < end - timedelta(days=1):
                df = None  # Cache outdated
        
        # Download if needed
        if df is None:
            df = _download_ticker(ticker, start, end)
            if df is None:
                continue
            # Save to cache if new or changed
            if not p.exists() or (p.exists() and _hash_df(df) != _hash_df(pd.read_parquet(p))):
                df.to_parquet(p, compression="zstd")
        
        # Liquidity filters
        if len(df) < MIN_OBS:
            continue
        if df["DollarVolume"].mean() < MIN_DOLLAR_VOLUME:
            continue
            
        data[ticker] = df
    
    return data

def calculate_wilder_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Wilder's RSI (using EMA)."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators with aligned NaN handling."""
    df = df.copy()
    df['SMA_20'] = df['Adj Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Adj Close'].rolling(window=50).mean()
    df['RSI'] = calculate_wilder_rsi(df['Adj Close'])
    df['Returns'] = df['Adj Close'].pct_change()
    
    # Align NaN values
    df = df.dropna()
    return df

def lstm_forecast(data: np.ndarray, steps: int = 30) -> Optional[float]:
    """Simplified LSTM to reduce overfitting."""
    if len(data) < LOOKBACK + steps:
        return None
    
    try:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data.reshape(-1, 1))
        
        X, y = [], []
        for i in range(LOOKBACK, len(scaled_data) - steps):
            X.append(scaled_data[i-LOOKBACK:i, 0])
            y.append(scaled_data[i+steps-1, 0])
        
        if len(X) < 10:
            return None
            
        # Split for validation
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        X_train = np.array(X_train).reshape(-1, LOOKBACK, 1)
        X_val = np.array(X_val).reshape(-1, LOOKBACK, 1)
        y_train, y_val = np.array(y_train), np.array(y_val)
        
        # Simplified architecture
        model = Sequential([
            LSTM(32, input_shape=(LOOKBACK, 1)),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(0.001), loss='mse')
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50, 
            batch_size=32, 
            verbose=0,
            callbacks=[EarlyStopping(monitor='val_loss', patience=10)]
        )
        
        last_sequence = scaled_data[-LOOKBACK:].reshape(1, LOOKBACK, 1)
        prediction = model.predict(last_sequence, verbose=0)
        prediction = scaler.inverse_transform(prediction)
        
        return float(prediction[0, 0])
    except Exception as e:
        logging.error(f"LSTM failed", exc_info=True)
        return None

def arima_forecast(data: pd.Series, steps: int = 30) -> Optional[float]:
    """Auto-ARIMA with pmdarima."""
    try:
        data = data.dropna()
        if len(data) < 100:
            return None
            
        model = auto_arima(
            data, 
            seasonal=True, 
            m=12,
            max_p=3, max_q=3, max_P=2, max_Q=2,
            d=None, D=1, 
            stepwise=True, 
            suppress_warnings=True,
            error_action='ignore'
        )
        
        forecast = model.predict(steps)
        return float(forecast[-1])
    except Exception as e:
        logging.error(f"ARIMA failed", exc_info=True)
        return None

def walk_forward_validation(prices: pd.Series, model_fn, train_window: int = 756, horizon: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """Walk-forward validation for out-of-sample testing."""
    preds, actuals = [], []
    
    for end_ix in range(train_window, len(prices) - horizon, horizon):
        train = prices.iloc[end_ix - train_window:end_ix]
        test = prices.iloc[end_ix:end_ix + horizon]
        
        yhat = model_fn(train.values if hasattr(train, 'values') else train, steps=horizon)
        
        if yhat is not None:
            preds.append(yhat)
            actuals.append(test.iloc[-1])
    
    return np.array(preds), np.array(actuals)

def analyze_single_stock(ticker: str, df: pd.DataFrame) -> Optional[Dict]:
    """Analyze a single stock (for parallel processing)."""
    logging.info(f"Analyzing {ticker}...")
    
    df = calculate_technical_indicators(df)
    
    # Use Adj Close for price analysis
    prices = df['Adj Close']
    if len(prices) < 252:
        return None
        
    current_price = float(prices.iloc[-1])
    
    # Forecasts
    lstm_pred = lstm_forecast(prices.values, 30)
    arima_pred = arima_forecast(prices, 30)
    
    if lstm_pred and arima_pred:
        # Walk-forward validation for weights
        lstm_preds, actuals = walk_forward_validation(prices, lstm_forecast)
        arima_preds, _ = walk_forward_validation(prices, arima_forecast)
        
        # Calculate optimal weights based on historical performance
        if len(lstm_preds) > 0 and len(arima_preds) > 0:
            lstm_mape = np.mean(np.abs((actuals - lstm_preds) / actuals))
            arima_mape = np.mean(np.abs((actuals - arima_preds) / actuals))
            
            # Inverse MAPE weighting
            total_inv_mape = (1/lstm_mape) + (1/arima_mape)
            lstm_weight = (1/lstm_mape) / total_inv_mape
            arima_weight = (1/arima_mape) / total_inv_mape
        else:
            lstm_weight, arima_weight = 0.6, 0.4
        
        # Weighted forecast
        forecast_price = lstm_weight * lstm_pred + arima_weight * arima_pred
        expected_return = ((forecast_price - current_price) / current_price) * 100
        
        # Calculate volatility
        returns = df['Returns']
        volatility = returns.std() * np.sqrt(252) * 100
        
        # Momentum score
        momentum_score = 0
        if prices.iloc[-1] > df['SMA_20'].iloc[-1]:
            momentum_score += 1
        if prices.iloc[-1] > df['SMA_50'].iloc[-1]:
            momentum_score += 1
        rsi_value = df['RSI'].iloc[-1]
        if 30 < rsi_value < 70:
            momentum_score += 1
        
        # Sharpe ratio with risk-free rate
        excess_return = expected_return - RF_RATE
        sharpe = excess_return / volatility if volatility > 0 else 0
        
        # Dollar volume for liquidity check
        avg_dollar_volume = df['DollarVolume'].tail(20).mean()
        
        return {
            'ticker': ticker,
            'current_price': round(current_price, 2),
            'forecast_price': round(forecast_price, 2),
            'expected_return': round(expected_return, 2),
            'volatility': round(volatility, 2),
            'sharpe_ratio': round(sharpe, 2),
            'momentum_score': momentum_score,
            'rsi': round(float(rsi_value), 2) if not pd.isna(rsi_value) else 50,
            'avg_dollar_volume': round(avg_dollar_volume / 1e6, 2),
            'lstm_weight': round(lstm_weight, 3),
            'arima_weight': round(arima_weight, 3),
            'recommendation': 'BUY' if excess_return > 10 and momentum_score >= 2 else 'HOLD'
        }
    
    return None

def analyze_stocks_parallel(raw_data: Dict[str, pd.DataFrame]) -> List[Dict]:
    """Analyze stocks in parallel."""
    results = []
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(analyze_single_stock, ticker, df): ticker 
            for ticker, df in raw_data.items()
        }
        
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)
    
    return sorted(results, key=lambda x: x['sharpe_ratio'], reverse=True)

def save_results(results: List[Dict]):
    """Save results to JSON."""
    output = {
        'last_updated': datetime.now().isoformat(),
        'forecast_horizon': '30 days',
        'risk_free_rate': RF_RATE,
        'stocks': results[:20]  # Top 20
    }
    
    output_path = OUTPUT_DIR / 'data.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    logging.info(f"Results saved: {len(results)} stocks analyzed")

if __name__ == "__main__":
    # Stock universe
    UNIVERSE = [
        "MSFT", "AAPL", "GOOGL", "AMZN", "META", "NVDA", "TSLA", 
        "BRK-B", "JPM", "V", "UNH", "MA", "HD", "PG", "DIS",
        "ADBE", "CRM", "NFLX", "CMCSA", "XOM", "VZ", "INTC",
        "CSCO", "PFE", "CVX", "WMT", "ABT", "BAC", "NKE", "TMO",
        "AVGO", "LLY", "ORCL", "MRK", "COST", "MDT", "PEP", "AZN"
    ]
    
    # Load data
    logging.info(f"Loading data for {len(UNIVERSE)} tickers...")
    raw_data = load_prices(UNIVERSE)
    logging.info(f"Loaded {len(raw_data)} liquid stocks")
    
    # Analyze in parallel
    results = analyze_stocks_parallel(raw_data)
    
    # Save
    save_results(results)
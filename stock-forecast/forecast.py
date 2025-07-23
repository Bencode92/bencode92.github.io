import numpy as np
import pandas as pd
import yfinance as yf
import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings("ignore")

# Configuration
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
YEARS_BACK = 5
MIN_DOLLAR_VOLUME = 5e6
MIN_OBS = 200
LOOKBACK = 60

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
        print(f"[WARN] Download failed for {ticker}: {e}")
        return None

def load_prices(tickers: List[str], years_back: int = YEARS_BACK) -> Dict[str, pd.DataFrame]:
    """Load prices with cache, liquidity filtering, and deduplication."""
    end = pd.Timestamp.now().normalize()
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

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators."""
    df = df.copy()
    df['SMA_20'] = df['Adj Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Adj Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Adj Close'])
    df['Returns'] = df['Adj Close'].pct_change()
    return df

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def lstm_forecast(data: np.ndarray, steps: int = 30) -> Optional[float]:
    """LSTM forecast with production-ready error handling."""
    if len(data) < LOOKBACK + steps:
        return None
    
    try:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data.reshape(-1, 1))
        
        X, y = [], []
        for i in range(LOOKBACK, len(scaled_data) - steps):
            X.append(scaled_data[i-LOOKBACK:i, 0])
            y.append(scaled_data[i+steps-1, 0])
        
        if len(X) < 10:  # Need minimum samples
            return None
            
        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(LOOKBACK, 1)),
            Dropout(0.2),
            LSTM(100, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(0.001), loss='mse')
        model.fit(X, y, epochs=50, batch_size=32, verbose=0, 
                  callbacks=[EarlyStopping(patience=10)])
        
        last_sequence = scaled_data[-LOOKBACK:].reshape(1, LOOKBACK, 1)
        prediction = model.predict(last_sequence, verbose=0)
        prediction = scaler.inverse_transform(prediction)
        
        return float(prediction[0, 0])
    except Exception as e:
        print(f"[WARN] LSTM failed: {e}")
        return None

def arima_forecast(data: pd.Series, steps: int = 30) -> Optional[float]:
    """ARIMA forecast with error handling."""
    try:
        # Remove NaN values
        data = data.dropna()
        if len(data) < 100:
            return None
            
        model = SARIMAX(data, order=(2,1,2), seasonal_order=(1,1,1,12))
        fitted = model.fit(disp=False)
        forecast = fitted.forecast(steps=steps)
        return float(forecast.iloc[-1])
    except Exception as e:
        print(f"[WARN] ARIMA failed: {e}")
        return None

def analyze_stocks(raw_data: Dict[str, pd.DataFrame]) -> List[Dict]:
    """Analyze stocks with production data."""
    results = []
    
    for ticker, df in raw_data.items():
        print(f"Analyzing {ticker}...")
        
        df = calculate_technical_indicators(df)
        
        # Use Adj Close for price analysis
        prices = df['Adj Close'].dropna()
        if len(prices) < 252:
            continue
            
        current_price = float(prices.iloc[-1])
        
        # Forecasts
        lstm_pred = lstm_forecast(prices.values, 30)
        arima_pred = arima_forecast(prices, 30)
        
        if lstm_pred and arima_pred:
            # Weighted average
            forecast_price = 0.6 * lstm_pred + 0.4 * arima_pred
            expected_return = ((forecast_price - current_price) / current_price) * 100
            
            # Calculate volatility
            returns = df['Returns'].dropna()
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
            
            # Risk-adjusted return
            sharpe = expected_return / volatility if volatility > 0 else 0
            
            # Dollar volume for liquidity check
            avg_dollar_volume = df['DollarVolume'].tail(20).mean()
            
            results.append({
                'ticker': ticker,
                'current_price': round(current_price, 2),
                'forecast_price': round(forecast_price, 2),
                'expected_return': round(expected_return, 2),
                'volatility': round(volatility, 2),
                'sharpe_ratio': round(sharpe, 2),
                'momentum_score': momentum_score,
                'rsi': round(float(rsi_value), 2) if not pd.isna(rsi_value) else 50,
                'avg_dollar_volume': round(avg_dollar_volume / 1e6, 2),  # In millions
                'recommendation': 'BUY' if expected_return > 10 and momentum_score >= 2 else 'HOLD'
            })
    
    return sorted(results, key=lambda x: x['sharpe_ratio'], reverse=True)

def save_results(results: List[Dict]):
    """Save results to JSON."""
    output = {
        'last_updated': datetime.now().isoformat(),
        'forecast_horizon': '30 days',
        'stocks': results[:20]  # Top 20
    }
    
    with open('stock-forecast/data.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved: {len(results)} stocks analyzed")

if __name__ == "__main__":
    # Stock universe
    UNIVERSE = [
        "MSFT", "AAPL", "GOOGL", "AMZN", "META", "NVDA", "TSLA", 
        "BRK-B", "JPM", "V", "UNH", "MA", "HD", "PG", "DIS",
        "ADBE", "CRM", "NFLX", "CMCSA", "XOM", "VZ", "INTC",
        "CSCO", "PFE", "CVX", "WMT", "ABT", "BAC", "NKE", "TMO",
        "AVGO", "LLY", "ORCL", "MRK", "COST", "MDT", "PEP", "AZN"
    ]
    
    # Load data with production-ready loader
    print(f"Loading data for {len(UNIVERSE)} tickers...")
    raw_data = load_prices(UNIVERSE)
    print(f"Loaded {len(raw_data)} liquid stocks")
    
    # Analyze
    results = analyze_stocks(raw_data)
    
    # Save
    save_results(results)
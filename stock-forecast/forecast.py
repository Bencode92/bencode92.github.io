import numpy as np
import pandas as pd
import yfinance as yf
import json
from datetime import datetime
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
TOP_STOCKS = 30
LOOKBACK = 60

def get_stock_data(ticker, period="2y"):
    """Récupère les données historiques"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        return df
    except:
        return None

def calculate_technical_indicators(df):
    """Calcule indicateurs techniques"""
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['Returns'] = df['Close'].pct_change()
    return df

def calculate_rsi(prices, period=14):
    """Calcule le RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def lstm_forecast(data, steps=30):
    """Prévision LSTM améliorée"""
    if len(data) < LOOKBACK + steps:
        return None
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
    X, y = [], []
    for i in range(LOOKBACK, len(scaled_data) - steps):
        X.append(scaled_data[i-LOOKBACK:i, 0])
        y.append(scaled_data[i+steps-1, 0])
    
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
    
    last_sequence = scaled_data[-LOOKBACK:]
    last_sequence = last_sequence.reshape(1, LOOKBACK, 1)
    prediction = model.predict(last_sequence)
    prediction = scaler.inverse_transform(prediction)
    
    return float(prediction[0, 0])

def arima_forecast(data, steps=30):
    """Prévision ARIMA"""
    try:
        model = SARIMAX(data, order=(2,1,2), seasonal_order=(1,1,1,12))
        fitted = model.fit(disp=False)
        forecast = fitted.forecast(steps=steps)
        return float(forecast.iloc[-1])
    except:
        return None

def analyze_stocks(tickers):
    """Analyse complète des actions"""
    results = []
    
    for ticker in tickers:
        print(f"Analyse de {ticker}...")
        df = get_stock_data(ticker)
        
        if df is None or len(df) < 252:
            continue
            
        df = calculate_technical_indicators(df)
        current_price = float(df['Close'].iloc[-1])
        
        lstm_pred = lstm_forecast(df['Close'].values, 30)
        arima_pred = arima_forecast(df['Close'].values, 30)
        
        if lstm_pred and arima_pred:
            forecast_price = 0.6 * lstm_pred + 0.4 * arima_pred
            expected_return = ((forecast_price - current_price) / current_price) * 100
            
            volatility = df['Returns'].std() * np.sqrt(252) * 100
            
            momentum_score = 0
            if df['Close'].iloc[-1] > df['SMA_20'].iloc[-1]:
                momentum_score += 1
            if df['Close'].iloc[-1] > df['SMA_50'].iloc[-1]:
                momentum_score += 1
            if df['RSI'].iloc[-1] > 30 and df['RSI'].iloc[-1] < 70:
                momentum_score += 1
                
            sharpe = expected_return / volatility if volatility > 0 else 0
            
            results.append({
                'ticker': ticker,
                'current_price': round(current_price, 2),
                'forecast_price': round(forecast_price, 2),
                'expected_return': round(expected_return, 2),
                'volatility': round(volatility, 2),
                'sharpe_ratio': round(sharpe, 2),
                'momentum_score': momentum_score,
                'rsi': round(float(df['RSI'].iloc[-1]), 2),
                'recommendation': 'BUY' if expected_return > 10 and momentum_score >= 2 else 'HOLD'
            })
    
    return sorted(results, key=lambda x: x['sharpe_ratio'], reverse=True)

def save_results(results):
    """Sauvegarde les résultats en JSON"""
    output = {
        'last_updated': datetime.now().isoformat(),
        'forecast_horizon': '30 days',
        'stocks': results[:20]
    }
    
    with open('data.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Résultats sauvegardés: {len(results)} actions analysées")

if __name__ == "__main__":
    tickers = [
        "MSFT", "AAPL", "GOOGL", "AMZN", "META", "NVDA", "TSLA", 
        "BRK-B", "JPM", "V", "UNH", "MA", "HD", "PG", "DIS",
        "ADBE", "CRM", "NFLX", "CMCSA", "XOM", "VZ", "INTC",
        "CSCO", "PFE", "CVX", "WMT", "ABT", "BAC", "NKE", "TMO"
    ]
    
    results = analyze_stocks(tickers)
    save_results(results)
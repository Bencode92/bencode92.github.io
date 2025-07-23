# Stock Forecast Dashboard

Dashboard de prévisions boursières utilisant LSTM et ARIMA.

## 🚀 Installation

### Installer les dépendances Python
```bash
pip install yfinance pandas numpy tensorflow scikit-learn statsmodels
```

## 📊 Utilisation

### Générer les prévisions
```bash
python forecast.py
```

### Voir le dashboard
Accéder à: https://bencode92.github.io/stock-forecast/

## 🔧 Configuration

Modifier dans `forecast.py`:
- `TOP_STOCKS`: Nombre d'actions à analyser
- `LOOKBACK`: Fenêtre historique pour LSTM
- `tickers`: Liste des actions

## 📈 Métriques

- **Expected Return**: Rendement prévu sur 30 jours
- **Sharpe Ratio**: Rendement ajusté au risque
- **Momentum Score**: Indicateur technique (0-3)
- **RSI**: Relative Strength Index

## ⚠️ Avertissement

Ce système est à des fins éducatives uniquement. Ne pas utiliser pour des décisions d'investissement réelles sans analyse approfondie.
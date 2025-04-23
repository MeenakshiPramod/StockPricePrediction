# Stock Price Prediction Dashboard 📈

A Streamlit-based web application for predicting stock prices using various deep learning architectures. Features multiple models, technical indicators, and interactive visualizations.

![App_Screenshot](https://github.com/user-attachments/assets/4fd71829-9f33-4081-a4e8-6cdefa4b968a)
![Analyzed_app_screenshot](https://github.com/user-attachments/assets/85f11953-4ae0-4fbd-9bf3-f2f1e4af6d7c)

## Features ✨

- **Multiple Model Architectures**
  - LSTM, Bidirectional LSTM (BiLSTM), GRU, and CNN-BiLSTM
  - Side-by-side model comparison
- **Comprehensive Technical Analysis**
  - Moving Averages, RSI, Bollinger Bands, MACD, Volume Indicators
  - 15+ derived financial features
- **Advanced Visualization**
  - Interactive Plotly charts with historical data and predictions
  - 5-day price forecast visualization
- **Model Evaluation**
  - MAE, RMSE, R², and MAPE metrics
  - Confidence score based on price volatility
  - Mean actual price reference

## Installation ⚙️

1. Clone the repository:
```bash
git clone https://github.com/MeenakshiPramod/StockPricePrediction.git
cd StockPricePrediction
```

2. Run the command for virtual environment creation
```bash
python -m venv venv
pip install -r requirements.txt
```

3. Run the streamlit app
```bash
streamlit run app.py
```

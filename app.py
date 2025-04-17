import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, GRU, Conv1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Set page config
st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("Stock Price Prediction Dashboard")
st.markdown("""
This dashboard allows you to predict stock prices using various deep learning models.
Select a stock ticker, choose your preferred models, and adjust parameters to get started.
""")

# Confidence score calculation function
def calculate_confidence_score(y_true, y_pred):
    """
    Calculate improved confidence score based on prediction intervals and residual analysis
    Returns score between 0-100 where higher is better
    """
    residuals = y_true - y_pred
    std_residuals = np.std(residuals)
    mean_absolute_residual = np.mean(np.abs(residuals))
    mean_true = np.mean(y_true)
    
    # Calculate coverage of 95% prediction interval
    lower_bound = y_pred - 1.96*std_residuals
    upper_bound = y_pred + 1.96*std_residuals
    coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
    
    # Combine metrics into confidence score (40% weight to accuracy, 60% to coverage)
    confidence_score = 100 * (0.4*(1 - mean_absolute_residual/mean_true) + 0.6*coverage)
    return np.clip(confidence_score, 0, 100)

# Sidebar for user inputs
st.sidebar.header("Model Configuration")

# Stock ticker input
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL")

# Date range selection
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
with col2:
    end_date = st.date_input("End Date", datetime.now())

# Model selection
st.sidebar.subheader("Select Models")
models = {
    "LSTM": st.sidebar.checkbox("LSTM", value=True),
    "BiLSTM": st.sidebar.checkbox("BiLSTM", value=True),
    "GRU": st.sidebar.checkbox("GRU", value=True),
    "CNN-BiLSTM": st.sidebar.checkbox("CNN-BiLSTM", value=True)
}

# Model parameters
st.sidebar.subheader("Model Parameters")
sequence_length = st.sidebar.slider("Sequence Length", 30, 120, 60)
split_ratio = st.sidebar.slider("Train/Test Split Ratio", 0.6, 0.9, 0.8, 0.05)
epochs = st.sidebar.slider("Number of Epochs", 10, 100, 20)

# Fixed future prediction period (5 days)
future_days = 5

# Technical indicators selection
st.sidebar.subheader("Technical Indicators")
indicators = {
    "Moving Averages": st.sidebar.checkbox("Moving Averages", value=True),
    "RSI": st.sidebar.checkbox("RSI", value=True),
    "Bollinger Bands": st.sidebar.checkbox("Bollinger Bands", value=True),
    "MACD": st.sidebar.checkbox("MACD", value=True),
    "Volume Indicators": st.sidebar.checkbox("Volume Indicators", value=True)
}

def fetch_data(ticker, start_date, end_date):
    """Fetch stock data from Yahoo Finance"""
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def add_technical_indicators(df):
    """Add technical indicators to the dataframe"""
    # Ensure multi-level columns are flattened
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        
    if indicators["Moving Averages"]:
        df['MA_7'] = df['Close'].rolling(window=7).mean()
        df['MA_21'] = df['Close'].rolling(window=21).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        df['MA_200'] = df['Close'].rolling(window=200).mean()

    if indicators["RSI"]:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

    if indicators["Bollinger Bands"]:
        df['Bollinger_Mid'] = df['Close'].rolling(window=20).mean()
        df['Bollinger_Upper'] = df['Bollinger_Mid'] + 2 * df['Close'].rolling(window=20).std()
        df['Bollinger_Lower'] = df['Bollinger_Mid'] - 2 * df['Close'].rolling(window=20).std()

    if indicators["MACD"]:
        df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    if indicators["Volume Indicators"]:
        df['Volume_MA'] = df['Volume'].rolling(window=10).mean()

    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
    df['High_Low_Spread'] = (df['High'] - df['Low']) / df['Close'] * 100
    df['Open_Close_Spread'] = (df['Close'] - df['Open']) / df['Open'] * 100

    df.dropna(inplace=True)
    return df

def preprocess_data(df, sequence_length, split_ratio):
    """Preprocess data for model training"""
    split_index = int(len(df) * split_ratio)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train_df)
    scaled_test = scaler.transform(test_df)

    def create_sequences(data):
        X, y = [], []
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(data[i, df.columns.get_loc('Close')])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(scaled_train)
    X_test, y_test = create_sequences(scaled_test)

    return X_train, X_test, y_train, y_test, scaler, train_df, test_df

def build_lstm_model(input_shape):
    """Build LSTM model"""
    model = Sequential([
        LSTM(50, activation='tanh', input_shape=input_shape),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mae')
    return model

def build_bilstm_model(input_shape):
    """Build BiLSTM model"""
    model = Sequential([
        Bidirectional(LSTM(50, activation='tanh'), input_shape=input_shape),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mae')
    return model

def build_gru_model(input_shape):
    """Build GRU model"""
    model = Sequential([
        GRU(50, activation='tanh', input_shape=input_shape),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mae')
    return model

def build_cnn_bilstm_model(input_shape):
    """Build CNN-BiLSTM model"""
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        Bidirectional(LSTM(50, activation='tanh')),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mae')
    return model

def evaluate_model(model, X_test, y_test, scaler, df):
    """Evaluate model performance with improved confidence scoring"""
    y_pred_scaled = model.predict(X_test).flatten()

    dummy_array = np.zeros((len(y_pred_scaled), X_test.shape[2]))
    dummy_array[:, df.columns.get_loc('Close')] = y_pred_scaled
    y_pred_actual = scaler.inverse_transform(dummy_array)[:, df.columns.get_loc('Close')]

    dummy_array[:, df.columns.get_loc('Close')] = y_test
    y_test_actual = scaler.inverse_transform(dummy_array)[:, df.columns.get_loc('Close')]

    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    r2 = r2_score(y_test_actual, y_pred_actual)
    mape = np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100
    
    # Calculate improved confidence score
    confidence_score = calculate_confidence_score(y_test_actual, y_pred_actual)

    return mae, rmse, r2, mape, confidence_score, y_test_actual, y_pred_actual

def predict_future(model, last_sequence, scaler, df, days=5):
    """Predict future stock prices"""
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(days):
        # Reshape for prediction
        current_sequence_reshaped = current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1])
        # Predict next value
        next_pred_scaled = model.predict(current_sequence_reshaped).flatten()[0]
        
        # Create dummy row to inverse transform
        dummy_array = np.zeros((1, current_sequence.shape[1]))
        dummy_array[0, df.columns.get_loc('Close')] = next_pred_scaled
        next_pred = scaler.inverse_transform(dummy_array)[0, df.columns.get_loc('Close')]
        
        future_predictions.append(next_pred)
        
        # Create new row for the prediction
        new_row = current_sequence[-1].copy()
        # Update the close price with our prediction
        new_row[df.columns.get_loc('Close')] = next_pred_scaled
        
        # Remove first row and add new prediction at the end
        current_sequence = np.vstack((current_sequence[1:], new_row))
        
    return future_predictions

def plot_results(df, y_test_actual, y_pred_actual, future_dates, future_predictions, model_name):
    """Plot actual vs predicted values with prediction intervals"""
    # Calculate prediction intervals
    residuals = y_test_actual - y_pred_actual
    std_residuals = np.std(residuals)
    upper_bound = y_pred_actual + 1.96*std_residuals
    lower_bound = y_pred_actual - 1.96*std_residuals
    
    fig = go.Figure()
    
    # Add prediction interval
    fig.add_trace(go.Scatter(
        x=df.index[-len(y_pred_actual):],
        y=upper_bound,
        fill=None,
        mode='lines',
        line_color='rgba(255,165,0,0.3)',
        name='Upper Bound',
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index[-len(y_pred_actual):],
        y=lower_bound,
        fill='tonexty',
        mode='lines',
        line_color='rgba(255,165,0,0.3)',
        name='95% Prediction Interval'
    ))
    
    # Add actual values
    fig.add_trace(go.Scatter(
        x=df.index[-len(y_test_actual):],
        y=y_test_actual,
        name='Actual',
        line=dict(color='blue')
    ))
    
    # Add predicted values
    fig.add_trace(go.Scatter(
        x=df.index[-len(y_pred_actual):],
        y=y_pred_actual,
        name='Predicted',
        line=dict(color='red')
    ))
    
    # Add future predictions if available
    if len(future_predictions) > 0:
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_predictions,
            name='5-Day Forecast',
            line=dict(color='green', dash='dash')
        ))
    
    fig.update_layout(
        title=f'{model_name} Predictions with Confidence Intervals',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_dark',
        xaxis=dict(range=[df.index[-30], future_dates[-1]] if len(future_predictions) > 0 
                  else [df.index[-30], df.index[-1]])
    )
    
    return fig

# Main execution
if st.sidebar.button("Start Analysis"):
    with st.spinner("Fetching data and training models..."):
        # Fetch data
        df = fetch_data(ticker, start_date, end_date)
        
        # Display raw data
        st.subheader("Stock Data")
        st.dataframe(df.head())
        
        # Add technical indicators
        df = add_technical_indicators(df)
        
        # Preprocess data
        X_train, X_test, y_train, y_test, scaler, train_df, test_df = preprocess_data(df, sequence_length, split_ratio)
        
        # Create future dates for prediction
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=future_days)
        
        # Display information about evaluation metrics
        st.subheader("Evaluation Metrics Explanation")
        metrics_explanation = """
        - **MAE (Mean Absolute Error)**: Average of absolute differences between predicted and actual values. Lower is better.
        - **RMSE (Root Mean Square Error)**: Square root of the average of squared differences between predicted and actual values. More sensitive to outliers, lower is better.
        - **R² (R-squared)**: Proportion of variance in the dependent variable that is predictable from the independent variables. Closer to 1 is better.
        - **MAPE (Mean Absolute Percentage Error)**: Average of percentage differences between predicted and actual values. Lower is better.
        - **Confidence Score**: Combines prediction accuracy (40%) and interval reliability (60%). Scores near 100 indicate both accurate predictions and correct prediction ranges.
        """
        st.markdown(metrics_explanation)
        
        # Train and evaluate selected models
        results = {}
        for model_name, is_selected in models.items():
            if is_selected:
                st.subheader(f"Training {model_name}...")
                
                # Build and train model
                if model_name == "LSTM":
                    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
                elif model_name == "BiLSTM":
                    model = build_bilstm_model((X_train.shape[1], X_train.shape[2]))
                elif model_name == "GRU":
                    model = build_gru_model((X_train.shape[1], X_train.shape[2]))
                else:  # CNN-BiLSTM
                    model = build_cnn_bilstm_model((X_train.shape[1], X_train.shape[2]))
                
                # Train model
                history = model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=32,
                    validation_split=0.1,
                    verbose=0
                )
                
                # Evaluate model
                mae, rmse, r2, mape, confidence_score, y_test_actual, y_pred_actual = evaluate_model(
                    model, X_test, y_test, scaler, df
                )
                
                # Predict future values (5 days)
                last_sequence = X_test[-1]
                future_predictions = predict_future(model, last_sequence, scaler, df, future_days)
                
                # Store results
                results[model_name] = {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'mape': mape,
                    'confidence_score': confidence_score,
                    'y_test_actual': y_test_actual,
                    'y_pred_actual': y_pred_actual,
                    'future_predictions': future_predictions
                }
                
                # Display metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("MAE", f"${mae:.2f}")
                with col2:
                    st.metric("RMSE", f"${rmse:.2f}")
                with col3:
                    st.metric("R²", f"{r2:.3f}")
                with col4:
                    st.metric("MAPE", f"{mape:.2f}%")
                with col5:
                    st.metric("Confidence", f"{confidence_score:.2f}%")

                # Plot results
                fig = plot_results(df, y_test_actual, y_pred_actual, future_dates, future_predictions, model_name)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display table with 5-day predictions
                st.subheader(f"{model_name} 5-Day Price Forecast")
                forecast_df = pd.DataFrame({
                    'Date': future_dates.date,
                    'Predicted Close Price': [f"${price:.2f}" for price in future_predictions],
                    'Change': [f"{((future_predictions[i]/future_predictions[i-1])-1)*100:.2f}%" if i > 0 
                               else f"{((future_predictions[i]/df['Close'].iloc[-1])-1)*100:.2f}%" 
                               for i in range(len(future_predictions))]
                })
                st.table(forecast_df)
        
        # Compare models
        if len(results) > 1:
            st.subheader("Model Comparison")
            comparison_df = pd.DataFrame({
                'Model': list(results.keys()),
                'MAE': [results[model]['mae'] for model in results],
                'RMSE': [results[model]['rmse'] for model in results],
                'R²': [results[model]['r2'] for model in results],
                'MAPE': [results[model]['mape'] for model in results],
                'Confidence Score': [f"{results[model]['confidence_score']:.2f}%" for model in results]
            })
            st.dataframe(comparison_df)
            
            # Display all models' future predictions in one table
            st.subheader("All Models 5-Day Forecast Comparison")
            
            comparison_data = {'Date': future_dates.date}
            for model_name in results:
                comparison_data[f"{model_name} Prediction"] = [f"${price:.2f}" for price in results[model_name]['future_predictions']]
            
            forecast_comparison = pd.DataFrame(comparison_data)
            st.table(forecast_comparison)
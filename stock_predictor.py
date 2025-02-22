import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 🔹 Step 1: Download Stock Data
stock_symbol = "AAPL"  # Change to any stock symbol (e.g., TSLA, GOOG)
df = yf.download(stock_symbol, start="2015-01-01", end="2024-01-01")

# 🔹 Step 2: Prepare Data for LSTM
df = df[['Close']]  # Use only closing prices
scaler = MinMaxScaler(feature_range=(0,1))  # Normalize data
df_scaled = scaler.fit_transform(df)

# 🔹 Step 3: Create Sequences for LSTM (60-day lookback)
def create_sequences(data, time_steps=60):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps])
    return np.array(X), np.array(y)

time_steps = 60
X, y = create_sequences(df_scaled, time_steps)

# 🔹 Step 4: Split into Training & Testing Data
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# 🔹 Step 5: Build the LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_steps, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# 🔹 Step 6: Train the Model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# 🔹 Step 7: Make Predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)  # Convert back to original scale
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# 🔹 Step 8: Visualize the Results
plt.figure(figsize=(12, 6))
plt.plot(df.index[train_size + time_steps:], y_test_actual, label="Actual Prices", color='blue')
plt.plot(df.index[train_size + time_steps:], predictions, label="Predicted Prices", color='red')
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.title(f"Stock Price Prediction for {stock_symbol}")
plt.show()

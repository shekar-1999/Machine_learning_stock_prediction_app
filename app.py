import os
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Set Streamlit Page Configuration
st.set_page_config(page_title="Stock Market Predictor", layout="wide")

# Load Trained Model (with error handling)
model_path = 'stock_model.keras'
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    st.error("🚨 Model file not found! Please train the model first.")
    st.stop()

# Streamlit App UI
st.title('📈 Stock Market Predictor')

# Get User Input for Stock Symbol
stock = st.text_input('Enter Stock Symbol', 'GOOG')

# Fetch Stock Data
start = '2012-01-01'
end = '2022-12-31'
data = yf.download(stock, start, end)

if data.empty:
    st.error("🚨 No stock data found. Please check the stock symbol and try again.")
    st.stop()

st.subheader('📊 Stock Data')
st.write(data)

# Splitting Data
train_size = int(len(data) * 0.80)
data_train = data['Close'][:train_size]
data_test = data['Close'][train_size:]

# Prepare for Prediction
scaler = MinMaxScaler(feature_range=(0,1))
data_train_scale = scaler.fit_transform(np.array(data_train).reshape(-1,1))

past_100_days = data_train[-100:]
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scale = scaler.transform(np.array(data_test).reshape(-1,1))

# Calculate Moving Averages
st.subheader('📈 Price vs Moving Averages')

fig1 = plt.figure(figsize=(10,5))
ma_50_days = data['Close'].rolling(50).mean()
ma_100_days = data['Close'].rolling(100).mean()
ma_200_days = data['Close'].rolling(200).mean()

plt.plot(data['Close'], 'g', label="Actual Price")
plt.plot(ma_50_days, 'r', label="MA 50 Days")
plt.plot(ma_100_days, 'b', label="MA 100 Days")
plt.plot(ma_200_days, 'y', label="MA 200 Days")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig1)

# Prepare Data for Prediction
x_test, y_test = [], []
for i in range(100, len(data_test_scale)):
    x_test.append(data_test_scale[i-100:i])
    y_test.append(data_test_scale[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Make Predictions
y_pred = model.predict(x_test)

# Reverse Scaling
scale_factor = 1 / scaler.scale_[0]
y_pred = y_pred * scale_factor
y_test = y_test * scale_factor

# Plot Predictions vs Actual Prices
st.subheader('📊 Original Price vs Predicted Price')

fig2 = plt.figure(figsize=(10,5))
plt.plot(y_test, 'g', label='Original Price')
plt.plot(y_pred, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
st.pyplot(fig2)

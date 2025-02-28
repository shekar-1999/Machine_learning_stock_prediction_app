#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

# Download Stock Data
start = '2012-01-01'
end = '2022-12-21'
stock = 'GOOG'

data = yf.download(stock, start, end)
data.reset_index(inplace=True)

# Drop missing values
data.dropna(inplace=True)

# Split Data
train_size = int(len(data) * 0.80)
data_train = data['Close'][:train_size]
data_test = data['Close'][train_size:]

# Scale Data
scaler = MinMaxScaler(feature_range=(0,1))
data_train_scale = scaler.fit_transform(np.array(data_train).reshape(-1,1))

# Prepare Training Data
x_train, y_train = [], []
for i in range(100, len(data_train_scale)):
    x_train.append(data_train_scale[i-100:i])
    y_train.append(data_train_scale[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Build LSTM Model
model = Sequential([
    LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)),
    Dropout(0.2),
    
    LSTM(units=60, activation='relu', return_sequences=True),
    Dropout(0.3),

    LSTM(units=80, activation='relu', return_sequences=True),
    Dropout(0.4),

    LSTM(units=120, activation='relu'),
    Dropout(0.5),

    Dense(units=1)
])

# Compile and Train Model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=1)

# Save Model
model.save('stock_model.keras')

print("✅ Model training completed and saved successfully!")

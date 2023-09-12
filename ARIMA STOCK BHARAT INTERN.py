#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Generate sample data
np.random.seed(42)
date_range = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
stock_prices = np.cumsum(np.random.randn(len(date_range))) + 100

# Create a DataFrame
data = pd.DataFrame({'Date': date_range, 'Price': stock_prices})
data.set_index('Date', inplace=True)

# Split data into training and testing sets
train_size = int(0.8 * len(data))
train, test = data[:train_size], data[train_size:]

# ARIMA model
order = (5, 1, 0)  # Replace with your chosen ARIMA order
model = ARIMA(train['Price'], order=order)
fitted_model = model.fit()

# Forecasting
forecast_steps = len(test)
forecast = fitted_model.forecast(steps=forecast_steps, alpha=0.05)

# Create forecast index
forecast_index = pd.date_range(start=test.index[0], periods=forecast_steps, freq='D')

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(train.index, train['Price'], label='Train')
plt.plot(test.index, test['Price'], label='Test')
plt.plot(forecast_index, forecast, label='Forecast', color='red')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock Price Prediction using ARIMA')
plt.show()


# In[ ]:





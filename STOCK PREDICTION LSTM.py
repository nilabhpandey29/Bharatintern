#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error

# Generate sample data
np.random.seed(42)
date_range = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
stock_prices = np.cumsum(np.random.randn(len(date_range))) + 100

# Create a DataFrame
data = pd.DataFrame({'Date': date_range, 'Price': stock_prices})
data.set_index('Date', inplace=True)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Split data into training and testing sets
train_size = int(0.8 * len(data))
train, test = scaled_data[:train_size], scaled_data[train_size:]

# Create sequences for LSTM training
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 30  # Length of input sequences
X_train, y_train = create_sequences(train, seq_length)
X_test, y_test = create_sequences(test, seq_length)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=50, activation='relu'))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Make predictions
predicted = model.predict(X_test)
predicted = scaler.inverse_transform(predicted)
actual = scaler.inverse_transform(y_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(actual, predicted))
print(f'Root Mean Squared Error (RMSE): {rmse}')

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(data.index[train_size+seq_length:], actual, label='Actual', color='blue')
plt.plot(data.index[train_size+seq_length:], predicted, label='Predicted', color='red')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock Price Prediction using LSTM')
plt.show()


# In[11]:


import numpy as np
print(np.__version__)


# In[4]:


conda install numpy=<desired_version>


# In[5]:


conda install numpy=<desired_version>


# In[6]:


conda list --show-channel-urls


# In[7]:


conda install numpy=<desired_version>


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





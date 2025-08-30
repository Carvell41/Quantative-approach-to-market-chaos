import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = yf.download('AAPL', start='2025-04-29', end='2025-06-29', interval='1d')

open = data['Open']
close = data['Close']
data['Return'] = (data['Close'] - data['Open'])/data['Open']
#print(data.columns)

data['Lag1'] = data['Return'].shift(1)
data['Lag2'] = data['Return'].shift(2)

data['Simulated'] = data['Close']['AAPL'].shift(1) * (1 + data['Lag1'])

condition = (data['Lag1'] < 0) & (data['Lag2'] < 0)
day3_returns = data.loc[condition,'Return']



plt.title('APPL Stock price')
#plt.plot(data['simulated'])
data['MA10'] = data['Close'].rolling(window=10).mean()
data['MA5'] = data['Close'].rolling(window=5).mean()
data['EMA5'] = data['Close'].ewm(span=5,adjust=False).mean()
plt.plot(data['EMA5'], label='EMA 5', linestyle=':')

plt.plot(data['MA5'], label='MA 5', linestyle='--')
#plt.plot(data['MA10'], label='MA 10', linestyle='--')
#plt.plot(data['Open'], color='red')
#plt.plot(data['Simulated'], label='Simulation')
plt.plot(data['Close'], color='Black', label='Stock price')
#plt.plot(returns, color='blue')
plt.ylabel("Price ($)")
plt.xlabel("Date")

plt.legend()

plt.show()
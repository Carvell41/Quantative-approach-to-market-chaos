import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA


data = yf.download("AAPL", start="2024-08-04", end="2025-08-04")["Close"].dropna()


data = data.asfreq("B")
data = data.interpolate()


model = ARIMA(data, order=(1,1,1))
res = model.fit()

print(res.summary())


forecast = res.get_forecast(steps=30)
mean = forecast.predicted_mean
conf_int = forecast.conf_int()

# 4. Plot
plt.figure(figsize=(12,6))
plt.plot(data.index, data, label="Observed")
plt.plot(mean.index, mean, label="Forecast", color="orange")
plt.fill_between(mean.index, conf_int.iloc[:,0], conf_int.iloc[:,1],
                 color="blue", alpha=0.2)
plt.legend()
plt.show()
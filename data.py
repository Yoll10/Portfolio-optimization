import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

us_tickers = ['NVDA', 'TSLA', 'GOOGL', 'MSFT', 'AAPL', 'AMZN', 'META']
fr_tickers = ['OR.PA', 'MC.PA', 'RMS.PA']
all_tickers = us_tickers + fr_tickers

raw_data = yf.download(all_tickers + ['EURUSD=X'], start="2021-01-01", end="2026-01-01",auto_adjust=True)['Close']

fx_rate = raw_data['EURUSD=X']
prices = raw_data.drop(columns=['EURUSD=X'])

for ticker in fr_tickers:
    prices[ticker] = prices[ticker] * fx_rate


prices = prices.dropna()
log_returns = np.log(prices / prices.shift(1)).dropna()

annual_returns = log_returns.mean() * 252
cov_matrix = log_returns.cov() * 252

plt.figure(figsize=(12, 10))
sns.heatmap(log_returns.corr(), cmap='coolwarm', annot=True, fmt=".2f")
plt.title('Correlation Matrix (All prices converted to USD)')
plt.show()

print("Avereage yearly returns (en USD) :")
print(annual_returns.sort_values(ascending=False))
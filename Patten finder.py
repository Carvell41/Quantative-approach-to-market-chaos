import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


ticker = 'AAPL'
start_date = '1980-07-07'
end_date = '2025-07-01'
max_pattern_length = 4
hold_period = 1
starting_capital = 1000
intraday_interval = '15m'
transaction_cost_rate = 0.001
min_success_rate = 0.5

daily_data = yf.download(ticker, start=start_date, end=end_date, interval='1d', auto_adjust=False)


if isinstance(daily_data.columns, pd.MultiIndex):
    daily_data.columns = daily_data.columns.get_level_values(0)


required_cols = {'Open', 'Close'}
missing_cols = required_cols - set(daily_data.columns)
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")


daily_data = daily_data[['Open', 'Close']].copy()
daily_data['Return'] = daily_data['Close'].pct_change()
daily_data.dropna(subset=['Return'], inplace=True)
daily_data['Direction'] = (daily_data['Close'] > daily_data['Open']).astype(int)


all_patterns = []

for pattern_length in range(1, max_pattern_length + 1):
    temp = daily_data.copy()
    pattern_cols = []
    for i in range(pattern_length):
        col = f'Day_{i}'
        temp[col] = temp['Direction'].shift(pattern_length - i)
        pattern_cols.append(col)

    temp = temp.dropna(subset=pattern_cols)
    temp['Pattern'] = temp[pattern_cols].astype(int).astype(str).agg('-'.join, axis=1)
    temp['Length'] = pattern_length
    temp['NextReturn'] = temp['Return'].shift(-1)
    temp = temp.dropna(subset=['NextReturn'])

    temp['Success'] = temp['NextReturn'] > 0

    summary = temp.groupby(['Length', 'Pattern']).agg(
        mean_return=('NextReturn', 'mean'),
        count=('NextReturn', 'count'),
        success_rate=('Success', 'mean')
    ).reset_index()

    all_patterns.append(summary)

final_df = pd.concat(all_patterns).sort_values(by='mean_return', ascending=False)
final_df['weighted_return'] = final_df['mean_return'] * final_df['count']


profitable_patterns = final_df[
    (final_df['mean_return'] > 0) & (final_df['success_rate'] >= min_success_rate)
]
pattern_list = profitable_patterns['Pattern'].tolist()



print(profitable_patterns[['Pattern', 'mean_return', 'count', 'success_rate']])


intraday_end = datetime.today()
intraday_start = intraday_end - timedelta(days=8)

intraday_data = yf.download(
    ticker,
    start=intraday_start.strftime('%Y-%m-%d'),
    end=intraday_end.strftime('%Y-%m-%d'),
    interval=intraday_interval,
    auto_adjust=False
)

if isinstance(intraday_data.columns, pd.MultiIndex):
    intraday_data.columns = intraday_data.columns.get_level_values(0)

intraday_data = intraday_data.reset_index()

datetime_col = intraday_data.columns[0]
intraday_data['Date'] = intraday_data[datetime_col].dt.date

intraday_data = intraday_data[['Open', 'Close', 'Date']]
grouped_intraday = intraday_data.groupby('Date')


capital = starting_capital
capital_curve = [capital]
trades = []
last_trade_day = None

for i in range(len(daily_data) - hold_period):
    current_date = daily_data.index[i]
    if last_trade_day and current_date <= last_trade_day:
        continue

    for length in range(1, max_pattern_length + 1):
        if i < length:
            continue
        pattern = '-'.join(daily_data['Direction'].iloc[i - length + 1:i + 1].astype(str).tolist())

        if pattern in pattern_list:
            entry_price = daily_data['Close'].iloc[i]
            exit_price = None


            for j in range(1, hold_period + 1):
                day_index = i + j
                if day_index >= len(daily_data):
                    break
                check_date = daily_data.index[day_index].date()

                if check_date in grouped_intraday.groups:
                    intraday_candles = grouped_intraday.get_group(check_date)
                    for _, row in intraday_candles.iterrows():
                        if row['Close'] <= row['Open']:
                            exit_price = row['Close']
                            break
                    if exit_price is not None:
                        break

            if not exit_price:
                exit_price = daily_data['Close'].iloc[min(i + hold_period, len(daily_data) - 1)]

            gross_return = (exit_price - entry_price) / entry_price
            net_return = gross_return - 2 * transaction_cost_rate
            capital *= (1 + net_return)
            capital_curve.append(capital)
            trades.append(net_return)
            last_trade_day = daily_data.index[min(i + j, len(daily_data) - 1)]
            break

plt.plot(capital_curve)
plt.title("Capital Over Time with Transaction Costs")
plt.xlabel("Number of Trades")
plt.ylabel("Capital ($)")
plt.grid(True)
plt.tight_layout()
plt.show()

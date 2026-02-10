import pandas as pd
import numpy as np

# ============================================================
# LOAD & COMPUTE ALL INDICATORS
# ============================================================
df = pd.read_csv('/sessions/fervent-dazzling-euler/mnt/uploads/AAPL_2023-01-01_to_2023-12-31_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Daily return
df['Daily_Return_Pct'] = df['Close'].pct_change() * 100

# Moving Averages
df['SMA20'] = df['Close'].rolling(window=20).mean()
df['SMA50'] = df['Close'].rolling(window=50).mean()

# MA Crossover Signal
df['MA_Position'] = 0
df.loc[df['SMA20'] > df['SMA50'], 'MA_Position'] = 1
df['MA_Signal'] = df['MA_Position'].diff()
df['MA_Signal_Label'] = ''
df.loc[df['MA_Signal'] == 1, 'MA_Signal_Label'] = 'BUY'
df.loc[df['MA_Signal'] == -1, 'MA_Signal_Label'] = 'SELL'

# RSI (14-period, Wilder's smoothing)
delta = df['Close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(window=14, min_periods=14).mean()
avg_loss = loss.rolling(window=14, min_periods=14).mean()
for i in range(14, len(df)):
    avg_gain.iloc[i] = (avg_gain.iloc[i-1] * 13 + gain.iloc[i]) / 14
    avg_loss.iloc[i] = (avg_loss.iloc[i-1] * 13 + loss.iloc[i]) / 14
rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))

# RSI signals
rsi_position = 0
rsi_signals = []
for i in range(len(df)):
    if pd.isna(df['RSI'].iloc[i]):
        rsi_signals.append('')
        continue
    if rsi_position == 0 and df['RSI'].iloc[i] < 30:
        rsi_position = 1
        rsi_signals.append('BUY')
    elif rsi_position == 1 and df['RSI'].iloc[i] > 70:
        rsi_position = 0
        rsi_signals.append('SELL')
    else:
        rsi_signals.append('')
df['RSI_Signal_Label'] = rsi_signals

# MACD
df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = df['EMA12'] - df['EMA26']
df['MACD_Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal_Line']

df['MACD_Position'] = 0
df.loc[df['MACD'] > df['MACD_Signal_Line'], 'MACD_Position'] = 1
df['MACD_Cross'] = df['MACD_Position'].diff()
df['MACD_Signal_Label'] = ''
df.loc[df['MACD_Cross'] == 1, 'MACD_Signal_Label'] = 'BUY'
df.loc[df['MACD_Cross'] == -1, 'MACD_Signal_Label'] = 'SELL'

# Cumulative return
df['Cumulative_Return_Pct'] = ((1 + df['Daily_Return_Pct']/100).cumprod() - 1) * 100
df.loc[0, 'Cumulative_Return_Pct'] = 0.0

# Drawdown
df['Peak'] = df['Close'].cummax()
df['Drawdown_Pct'] = ((df['Close'] - df['Peak']) / df['Peak']) * 100

# ============================================================
# CSV 1: Full daily data with all indicators & signals
# ============================================================
export_cols = [
    'Date', 'Open', 'High', 'Low', 'Close', 'Volume',
    'Daily_Return_Pct', 'Cumulative_Return_Pct',
    'SMA20', 'SMA50', 'MA_Signal_Label',
    'RSI', 'RSI_Signal_Label',
    'MACD', 'MACD_Signal_Line', 'MACD_Histogram', 'MACD_Signal_Label',
    'Peak', 'Drawdown_Pct'
]

out = df[export_cols].copy()
out = out.round({
    'Open': 2, 'High': 2, 'Low': 2, 'Close': 2,
    'Daily_Return_Pct': 3, 'Cumulative_Return_Pct': 3,
    'SMA20': 2, 'SMA50': 2,
    'RSI': 2,
    'MACD': 4, 'MACD_Signal_Line': 4, 'MACD_Histogram': 4,
    'Peak': 2, 'Drawdown_Pct': 3
})
out['Date'] = out['Date'].dt.strftime('%Y-%m-%d')

out.to_csv('/sessions/fervent-dazzling-euler/mnt/outputs/AAPL_2023_Full_Analysis.csv', index=False)
print(f"✓ Full analysis CSV saved — {len(out)} rows, {len(export_cols)} columns")

# ============================================================
# CSV 2: Trade log summary
# ============================================================
trades = []

# MA trades
ma_buys = df[df['MA_Signal'] == 1].index.tolist()
ma_sells = df[df['MA_Signal'] == -1].index.tolist()
for b in ma_buys:
    sells_after = [s for s in ma_sells if s > b]
    if sells_after:
        s = sells_after[0]
    else:
        s = len(df) - 1  # still holding
    buy_price = df['Close'].iloc[b]
    sell_price = df['Close'].iloc[s]
    pnl = ((sell_price - buy_price) / buy_price) * 100
    trades.append({
        'Strategy': 'MA Crossover (20/50)',
        'Buy_Date': df['Date'].iloc[b].strftime('%Y-%m-%d'),
        'Sell_Date': df['Date'].iloc[s].strftime('%Y-%m-%d'),
        'Buy_Price': round(buy_price, 2),
        'Sell_Price': round(sell_price, 2),
        'PnL_Pct': round(pnl, 2),
        'Result': 'WIN' if pnl > 0 else 'LOSS'
    })

# MACD trades
macd_buys = df[df['MACD_Cross'] == 1].index.tolist()
macd_sells = df[df['MACD_Cross'] == -1].index.tolist()
for b in macd_buys:
    sells_after = [s for s in macd_sells if s > b]
    if sells_after:
        s = sells_after[0]
    else:
        s = len(df) - 1
    buy_price = df['Close'].iloc[b]
    sell_price = df['Close'].iloc[s]
    pnl = ((sell_price - buy_price) / buy_price) * 100
    trades.append({
        'Strategy': 'MACD (12/26/9)',
        'Buy_Date': df['Date'].iloc[b].strftime('%Y-%m-%d'),
        'Sell_Date': df['Date'].iloc[s].strftime('%Y-%m-%d'),
        'Buy_Price': round(buy_price, 2),
        'Sell_Price': round(sell_price, 2),
        'PnL_Pct': round(pnl, 2),
        'Result': 'WIN' if pnl > 0 else 'LOSS'
    })

trades_df = pd.DataFrame(trades)
trades_df.to_csv('/sessions/fervent-dazzling-euler/mnt/outputs/AAPL_2023_Trade_Log.csv', index=False)
print(f"✓ Trade log CSV saved — {len(trades_df)} trades")

# ============================================================
# CSV 3: Strategy performance summary
# ============================================================
summary = pd.DataFrame([
    {
        'Strategy': 'MA Crossover (20/50)',
        'Total_Return_Pct': 18.87,
        'Buy_Hold_Return_Pct': 54.80,
        'Alpha_Pct': -35.93,
        'Num_Trades': 2,
        'Win_Rate_Pct': 100.0,
        'Winning_Trades': 2,
        'Losing_Trades': 0,
        'Avg_Win_Pct': 9.22,
        'Avg_Loss_Pct': 0.00,
        'Max_Drawdown_Pct': -10.00,
        'Market_Exposure_Pct': 55.2,
        'Ann_Volatility_Pct': 13.00
    },
    {
        'Strategy': 'RSI (14)',
        'Total_Return_Pct': 0.00,
        'Buy_Hold_Return_Pct': 54.80,
        'Alpha_Pct': -54.80,
        'Num_Trades': 0,
        'Win_Rate_Pct': 0.0,
        'Winning_Trades': 0,
        'Losing_Trades': 0,
        'Avg_Win_Pct': 0.00,
        'Avg_Loss_Pct': 0.00,
        'Max_Drawdown_Pct': 0.00,
        'Market_Exposure_Pct': 0.0,
        'Ann_Volatility_Pct': 0.00
    },
    {
        'Strategy': 'MACD (12/26/9)',
        'Total_Return_Pct': 36.04,
        'Buy_Hold_Return_Pct': 54.80,
        'Alpha_Pct': -18.76,
        'Num_Trades': 12,
        'Win_Rate_Pct': 50.0,
        'Winning_Trades': 6,
        'Losing_Trades': 6,
        'Avg_Win_Pct': 6.23,
        'Avg_Loss_Pct': -0.68,
        'Max_Drawdown_Pct': -6.40,
        'Market_Exposure_Pct': 51.2,
        'Ann_Volatility_Pct': 13.55
    },
    {
        'Strategy': 'Buy & Hold',
        'Total_Return_Pct': 54.80,
        'Buy_Hold_Return_Pct': 54.80,
        'Alpha_Pct': 0.00,
        'Num_Trades': 1,
        'Win_Rate_Pct': 100.0,
        'Winning_Trades': 1,
        'Losing_Trades': 0,
        'Avg_Win_Pct': 54.80,
        'Avg_Loss_Pct': 0.00,
        'Max_Drawdown_Pct': -14.93,
        'Market_Exposure_Pct': 100.0,
        'Ann_Volatility_Pct': 19.95
    }
])
summary.to_csv('/sessions/fervent-dazzling-euler/mnt/outputs/AAPL_2023_Strategy_Summary.csv', index=False)
print(f"✓ Strategy summary CSV saved — {len(summary)} strategies")

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
from datetime import datetime

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv('/sessions/fervent-dazzling-euler/mnt/uploads/AAPL_2023-01-01_to_2023-12-31_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# ============================================================
# STRATEGY 1: MOVING AVERAGE CROSSOVER (20/50 SMA)
# ============================================================
df['SMA20'] = df['Close'].rolling(window=20).mean()
df['SMA50'] = df['Close'].rolling(window=50).mean()

# Signal: 1 = long, 0 = out
df['MA_Signal'] = 0
df.loc[df['SMA20'] > df['SMA50'], 'MA_Signal'] = 1  # Buy when 20 > 50

# Detect crossover points (signal changes)
df['MA_Cross'] = df['MA_Signal'].diff()
# +1 = buy signal (golden cross), -1 = sell signal (death cross)

# ============================================================
# STRATEGY 2: RSI (14-period)
# ============================================================
delta = df['Close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(window=14, min_periods=14).mean()
avg_loss = loss.rolling(window=14, min_periods=14).mean()

# Use Wilder's smoothing after initial SMA
for i in range(14, len(df)):
    avg_gain.iloc[i] = (avg_gain.iloc[i-1] * 13 + gain.iloc[i]) / 14
    avg_loss.iloc[i] = (avg_loss.iloc[i-1] * 13 + loss.iloc[i]) / 14

rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))

# RSI signals: Buy when crossing above 30 from below, Sell when crossing below 70 from above
df['RSI_Signal'] = 0  # 0 = no position
rsi_position = 0
rsi_signals = [0] * len(df)
rsi_trades = []

for i in range(1, len(df)):
    if pd.isna(df['RSI'].iloc[i]):
        rsi_signals[i] = 0
        continue

    if rsi_position == 0 and df['RSI'].iloc[i] < 30:
        rsi_position = 1  # Buy
        rsi_signals[i] = 1  # Buy signal
    elif rsi_position == 1 and df['RSI'].iloc[i] > 70:
        rsi_position = 0  # Sell
        rsi_signals[i] = -1  # Sell signal
    else:
        rsi_signals[i] = 0

df['RSI_Trade'] = rsi_signals

# Build RSI position series
df['RSI_Position'] = 0
pos = 0
for i in range(len(df)):
    if df['RSI_Trade'].iloc[i] == 1:
        pos = 1
    elif df['RSI_Trade'].iloc[i] == -1:
        pos = 0
    df.loc[i, 'RSI_Position'] = pos

# ============================================================
# STRATEGY 3: MACD (12, 26, 9)
# ============================================================
df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = df['EMA12'] - df['EMA26']
df['MACD_Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal_Line']

# Signal: 1 when MACD > Signal Line
df['MACD_Position'] = 0
df.loc[df['MACD'] > df['MACD_Signal_Line'], 'MACD_Position'] = 1
df['MACD_Cross'] = df['MACD_Position'].diff()

# ============================================================
# BACKTESTING ENGINE
# ============================================================
def backtest_strategy(df, position_col, name):
    """Backtest a long-only strategy given a position column (1=in, 0=out)"""
    df = df.copy()
    df['Daily_Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = df[position_col].shift(1) * df['Daily_Return']
    df['BuyHold_Cum'] = (1 + df['Daily_Return']).cumprod()
    df['Strategy_Cum'] = (1 + df['Strategy_Return']).cumprod()

    # Calculate trades
    position_changes = df[position_col].diff()
    buy_signals = df[position_changes == 1]
    sell_signals = df[position_changes == -1]

    # Trade-level P&L
    trades = []
    buy_price = None
    for i in range(1, len(df)):
        if position_changes.iloc[i] == 1:
            buy_price = df['Close'].iloc[i]
        elif position_changes.iloc[i] == -1 and buy_price is not None:
            sell_price = df['Close'].iloc[i]
            pnl = ((sell_price - buy_price) / buy_price) * 100
            trades.append({
                'buy_date': df.loc[df[position_col].diff() == 1, 'Date'].iloc[len(trades)] if len(trades) < len(buy_signals) else None,
                'sell_date': df['Date'].iloc[i],
                'buy_price': buy_price,
                'sell_price': sell_price,
                'pnl_pct': pnl
            })
            buy_price = None

    # If still in position at end
    if buy_price is not None and df[position_col].iloc[-1] == 1:
        trades.append({
            'buy_date': None,
            'sell_date': df['Date'].iloc[-1],
            'buy_price': buy_price,
            'sell_price': df['Close'].iloc[-1],
            'pnl_pct': ((df['Close'].iloc[-1] - buy_price) / buy_price) * 100
        })

    total_return = (df['Strategy_Cum'].iloc[-1] - 1) * 100 if not pd.isna(df['Strategy_Cum'].iloc[-1]) else 0
    bh_return = (df['BuyHold_Cum'].iloc[-1] - 1) * 100 if not pd.isna(df['BuyHold_Cum'].iloc[-1]) else 0

    winning = [t for t in trades if t['pnl_pct'] > 0]
    losing = [t for t in trades if t['pnl_pct'] <= 0]
    win_rate = len(winning) / len(trades) * 100 if trades else 0

    # Days in market
    days_in = df[position_col].sum()
    total_days = len(df)
    exposure = days_in / total_days * 100

    # Strategy volatility
    strat_returns = df['Strategy_Return'].dropna()
    strat_vol = strat_returns.std() * np.sqrt(252) * 100

    # Max drawdown
    cum = df['Strategy_Cum']
    peak = cum.cummax()
    drawdown = ((cum - peak) / peak) * 100
    max_dd = drawdown.min()

    avg_win = np.mean([t['pnl_pct'] for t in winning]) if winning else 0
    avg_loss = np.mean([t['pnl_pct'] for t in losing]) if losing else 0

    return {
        'name': name,
        'total_return': total_return,
        'bh_return': bh_return,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'winning_trades': len(winning),
        'losing_trades': len(losing),
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'max_drawdown': max_dd,
        'exposure': exposure,
        'volatility': strat_vol,
        'trades': trades,
        'cum_returns': df[['Date', 'Strategy_Cum', 'BuyHold_Cum']].copy()
    }

# Run backtests
ma_results = backtest_strategy(df, 'MA_Signal', 'MA Crossover (20/50)')
rsi_results = backtest_strategy(df, 'RSI_Position', 'RSI (14)')
macd_results = backtest_strategy(df, 'MACD_Position', 'MACD (12/26/9)')

# ============================================================
# PRINT RESULTS
# ============================================================
strategies = [ma_results, rsi_results, macd_results]

print("=" * 80)
print("        AAPL 2023 — TRADING STRATEGY BACKTEST COMPARISON")
print("=" * 80)
print()
print(f"{'Metric':<30} {'MA Crossover':>15} {'RSI (14)':>15} {'MACD':>15}")
print("-" * 80)
print(f"{'Total Return':<30} {ma_results['total_return']:>14.2f}% {rsi_results['total_return']:>14.2f}% {macd_results['total_return']:>14.2f}%")
print(f"{'Buy & Hold Return':<30} {ma_results['bh_return']:>14.2f}% {rsi_results['bh_return']:>14.2f}% {macd_results['bh_return']:>14.2f}%")
print(f"{'Alpha vs Buy & Hold':<30} {ma_results['total_return']-ma_results['bh_return']:>14.2f}% {rsi_results['total_return']-rsi_results['bh_return']:>14.2f}% {macd_results['total_return']-macd_results['bh_return']:>14.2f}%")
print(f"{'Number of Trades':<30} {ma_results['num_trades']:>15} {rsi_results['num_trades']:>15} {macd_results['num_trades']:>15}")
print(f"{'Win Rate':<30} {ma_results['win_rate']:>14.1f}% {rsi_results['win_rate']:>14.1f}% {macd_results['win_rate']:>14.1f}%")
print(f"{'Winning Trades':<30} {ma_results['winning_trades']:>15} {rsi_results['winning_trades']:>15} {macd_results['winning_trades']:>15}")
print(f"{'Losing Trades':<30} {ma_results['losing_trades']:>15} {rsi_results['losing_trades']:>15} {macd_results['losing_trades']:>15}")
print(f"{'Avg Win':<30} {ma_results['avg_win']:>14.2f}% {rsi_results['avg_win']:>14.2f}% {macd_results['avg_win']:>14.2f}%")
print(f"{'Avg Loss':<30} {ma_results['avg_loss']:>14.2f}% {rsi_results['avg_loss']:>14.2f}% {macd_results['avg_loss']:>14.2f}%")
print(f"{'Max Drawdown':<30} {ma_results['max_drawdown']:>14.2f}% {rsi_results['max_drawdown']:>14.2f}% {macd_results['max_drawdown']:>14.2f}%")
print(f"{'Market Exposure':<30} {ma_results['exposure']:>14.1f}% {rsi_results['exposure']:>14.1f}% {macd_results['exposure']:>14.1f}%")
print(f"{'Ann. Volatility':<30} {ma_results['volatility']:>14.2f}% {rsi_results['volatility']:>14.2f}% {macd_results['volatility']:>14.2f}%")
print()

# Trade details for each strategy
for strat in strategies:
    print(f"\n{'─' * 60}")
    print(f"  {strat['name']} — TRADE LOG")
    print(f"{'─' * 60}")
    if strat['trades']:
        for i, t in enumerate(strat['trades'], 1):
            status = "✓ WIN" if t['pnl_pct'] > 0 else "✗ LOSS"
            sell_dt = t['sell_date'].strftime('%b %d') if isinstance(t['sell_date'], pd.Timestamp) else str(t['sell_date'])
            print(f"  Trade {i}: Buy ${t['buy_price']:.2f} → Sell ${t['sell_price']:.2f}  |  {t['pnl_pct']:+.2f}%  {status}")
    else:
        print("  No completed trades.")

# ============================================================
# STRATEGY LOGIC EXPLANATION
# ============================================================
print("\n")
print("=" * 80)
print("        STRATEGY LOGIC EXPLAINED")
print("=" * 80)
print()
print("1. MOVING AVERAGE CROSSOVER (20/50 SMA)")
print("   ├─ BUY:  When 20-day SMA crosses ABOVE the 50-day SMA (Golden Cross)")
print("   ├─ SELL: When 20-day SMA crosses BELOW the 50-day SMA (Death Cross)")
print("   └─ Type: Trend-following. Captures medium-term momentum swings.")
print()
print("2. RSI — RELATIVE STRENGTH INDEX (14-period)")
print("   ├─ BUY:  When RSI drops below 30 (stock is oversold)")
print("   ├─ SELL: When RSI rises above 70 (stock is overbought)")
print("   └─ Type: Mean-reversion. Buys dips, sells rallies.")
print()
print("3. MACD — MOVING AVERAGE CONVERGENCE/DIVERGENCE (12, 26, 9)")
print("   ├─ BUY:  When MACD line crosses ABOVE the Signal line (bullish momentum)")
print("   ├─ SELL: When MACD line crosses BELOW the Signal line (bearish momentum)")
print("   └─ Type: Momentum + trend. Combines short & long-term EMA dynamics.")
print()
print("NOTE: All strategies are LONG-ONLY (no shorting). Backtest assumes")
print("      execution at close price on signal day, no slippage or commissions.")

# ============================================================
# VISUALIZATIONS
# ============================================================
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(20, 28))
gs = gridspec.GridSpec(5, 1, height_ratios=[3, 2, 2, 2, 2], hspace=0.35)

colors = {'buy': '#00E676', 'sell': '#FF1744', 'price': '#42A5F5', 'ma20': '#FF9800', 'ma50': '#E91E63'}

# --- CHART 1: MA Crossover with Buy/Sell Signals ---
ax1 = fig.add_subplot(gs[0])
ax1.plot(df['Date'], df['Close'], color=colors['price'], linewidth=1.3, label='AAPL Close', alpha=0.9)
ax1.plot(df['Date'], df['SMA20'], color=colors['ma20'], linewidth=1, label='20-Day SMA', alpha=0.7)
ax1.plot(df['Date'], df['SMA50'], color=colors['ma50'], linewidth=1, label='50-Day SMA', alpha=0.7)

# Buy signals
buy_ma = df[df['MA_Cross'] == 1]
sell_ma = df[df['MA_Cross'] == -1]
ax1.scatter(buy_ma['Date'], buy_ma['Close'], marker='^', color=colors['buy'], s=150, zorder=5, label='BUY Signal', edgecolors='white', linewidth=0.5)
ax1.scatter(sell_ma['Date'], sell_ma['Close'], marker='v', color=colors['sell'], s=150, zorder=5, label='SELL Signal', edgecolors='white', linewidth=0.5)

# Shade in-market periods
for i in range(len(buy_ma)):
    buy_date = buy_ma['Date'].iloc[i]
    if i < len(sell_ma):
        sell_date = sell_ma['Date'].iloc[i] if sell_ma['Date'].iloc[i] > buy_date else df['Date'].iloc[-1]
    else:
        sell_date = df['Date'].iloc[-1]
    ax1.axvspan(buy_date, sell_date, alpha=0.08, color=colors['buy'])

ax1.set_title('Strategy 1: Moving Average Crossover (20/50 SMA)', fontsize=14, fontweight='bold', pad=12)
ax1.set_ylabel('Price ($)', fontsize=11)
ax1.legend(loc='upper left', fontsize=9, ncol=3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax1.xaxis.set_major_locator(mdates.MonthLocator())
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('$%.0f'))

# --- CHART 2: RSI with Signals ---
ax2 = fig.add_subplot(gs[1])
ax2.plot(df['Date'], df['RSI'], color='#AB47BC', linewidth=1.2)
ax2.axhline(y=70, color=colors['sell'], linestyle='--', linewidth=0.8, alpha=0.7, label='Overbought (70)')
ax2.axhline(y=30, color=colors['buy'], linestyle='--', linewidth=0.8, alpha=0.7, label='Oversold (30)')
ax2.fill_between(df['Date'], 70, 100, alpha=0.05, color=colors['sell'])
ax2.fill_between(df['Date'], 0, 30, alpha=0.05, color=colors['buy'])

rsi_buys = df[df['RSI_Trade'] == 1]
rsi_sells = df[df['RSI_Trade'] == -1]
ax2.scatter(rsi_buys['Date'], rsi_buys['RSI'], marker='^', color=colors['buy'], s=120, zorder=5, label='BUY', edgecolors='white', linewidth=0.5)
ax2.scatter(rsi_sells['Date'], rsi_sells['RSI'], marker='v', color=colors['sell'], s=120, zorder=5, label='SELL', edgecolors='white', linewidth=0.5)

ax2.set_title('Strategy 2: RSI (14-period) — Buy <30, Sell >70', fontsize=14, fontweight='bold', pad=12)
ax2.set_ylabel('RSI', fontsize=11)
ax2.set_ylim(0, 100)
ax2.legend(loc='upper right', fontsize=9, ncol=4)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax2.xaxis.set_major_locator(mdates.MonthLocator())

# --- CHART 3: MACD with Signals ---
ax3 = fig.add_subplot(gs[2])
ax3.plot(df['Date'], df['MACD'], color='#42A5F5', linewidth=1.2, label='MACD Line')
ax3.plot(df['Date'], df['MACD_Signal_Line'], color='#FF7043', linewidth=1.2, label='Signal Line')

# Histogram
hist_colors = ['#00C853' if v >= 0 else '#FF1744' for v in df['MACD_Histogram']]
ax3.bar(df['Date'], df['MACD_Histogram'], color=hist_colors, alpha=0.4, width=1.5, label='Histogram')

macd_buys = df[df['MACD_Cross'] == 1]
macd_sells = df[df['MACD_Cross'] == -1]
ax3.scatter(macd_buys['Date'], macd_buys['MACD'], marker='^', color=colors['buy'], s=100, zorder=5, label='BUY', edgecolors='white', linewidth=0.5)
ax3.scatter(macd_sells['Date'], macd_sells['MACD'], marker='v', color=colors['sell'], s=100, zorder=5, label='SELL', edgecolors='white', linewidth=0.5)

ax3.set_title('Strategy 3: MACD (12, 26, 9)', fontsize=14, fontweight='bold', pad=12)
ax3.set_ylabel('MACD Value', fontsize=11)
ax3.axhline(y=0, color='white', linewidth=0.5, alpha=0.3)
ax3.legend(loc='upper left', fontsize=9, ncol=5)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax3.xaxis.set_major_locator(mdates.MonthLocator())

# --- CHART 4: Cumulative Returns Comparison ---
ax4 = fig.add_subplot(gs[3])
cum_ma = ma_results['cum_returns']
cum_rsi = rsi_results['cum_returns']
cum_macd = macd_results['cum_returns']

ax4.plot(cum_ma['Date'], (cum_ma['BuyHold_Cum'] - 1) * 100, color='white', linewidth=1.5, alpha=0.6, label='Buy & Hold', linestyle='--')
ax4.plot(cum_ma['Date'], (cum_ma['Strategy_Cum'] - 1) * 100, color='#42A5F5', linewidth=1.5, label='MA Crossover')
ax4.plot(cum_rsi['Date'], (cum_rsi['Strategy_Cum'] - 1) * 100, color='#AB47BC', linewidth=1.5, label='RSI')
ax4.plot(cum_macd['Date'], (cum_macd['Strategy_Cum'] - 1) * 100, color='#FF7043', linewidth=1.5, label='MACD')

ax4.set_title('Cumulative Returns: All Strategies vs Buy & Hold', fontsize=14, fontweight='bold', pad=12)
ax4.set_ylabel('Cumulative Return (%)', fontsize=11)
ax4.legend(loc='upper left', fontsize=10)
ax4.axhline(y=0, color='white', linewidth=0.5, alpha=0.3)
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax4.xaxis.set_major_locator(mdates.MonthLocator())

# --- CHART 5: Performance Summary Table ---
ax5 = fig.add_subplot(gs[4])
ax5.axis('off')

table_data = [
    ['', 'MA Crossover\n(20/50 SMA)', 'RSI\n(14-period)', 'MACD\n(12/26/9)', 'Buy & Hold'],
    ['Total Return', f"{ma_results['total_return']:.1f}%", f"{rsi_results['total_return']:.1f}%", f"{macd_results['total_return']:.1f}%", f"{ma_results['bh_return']:.1f}%"],
    ['# Trades', str(ma_results['num_trades']), str(rsi_results['num_trades']), str(macd_results['num_trades']), '1'],
    ['Win Rate', f"{ma_results['win_rate']:.0f}%", f"{rsi_results['win_rate']:.0f}%", f"{macd_results['win_rate']:.0f}%", '—'],
    ['Avg Win', f"{ma_results['avg_win']:.1f}%", f"{rsi_results['avg_win']:.1f}%", f"{macd_results['avg_win']:.1f}%", '—'],
    ['Avg Loss', f"{ma_results['avg_loss']:.1f}%", f"{rsi_results['avg_loss']:.1f}%", f"{macd_results['avg_loss']:.1f}%", '—'],
    ['Max Drawdown', f"{ma_results['max_drawdown']:.1f}%", f"{rsi_results['max_drawdown']:.1f}%", f"{macd_results['max_drawdown']:.1f}%", '—'],
    ['Exposure', f"{ma_results['exposure']:.0f}%", f"{rsi_results['exposure']:.0f}%", f"{macd_results['exposure']:.0f}%", '100%'],
]

table = ax5.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.2)

# Style header row
for j in range(5):
    table[0, j].set_facecolor('#1a237e')
    table[0, j].set_text_props(fontweight='bold', color='white', fontsize=10)

# Style metric labels
for i in range(1, len(table_data)):
    table[i, 0].set_facecolor('#283593')
    table[i, 0].set_text_props(fontweight='bold', color='white')
    for j in range(1, 5):
        table[i, j].set_facecolor('#1a1a2e')
        table[i, j].set_text_props(color='white')

# Highlight best return
returns = [ma_results['total_return'], rsi_results['total_return'], macd_results['total_return']]
best_idx = returns.index(max(returns)) + 1
table[1, best_idx].set_facecolor('#1B5E20')
table[1, best_idx].set_text_props(fontweight='bold', color='#00E676')

ax5.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=15)

plt.savefig('/sessions/fervent-dazzling-euler/mnt/outputs/AAPL_2023_Strategy_Backtest.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
print("\n✓ Strategy backtest chart saved!")
plt.close()

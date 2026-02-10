import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
from datetime import datetime

# Load data
df = pd.read_csv('/sessions/fervent-dazzling-euler/mnt/uploads/AAPL_2023-01-01_to_2023-12-31_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# ============================================================
# KEY METRICS
# ============================================================
open_price = df['Open'].iloc[0]
close_price = df['Close'].iloc[-1]
ytd_return = ((close_price - open_price) / open_price) * 100

year_high = df['High'].max()
year_high_date = df.loc[df['High'].idxmax(), 'Date']
year_low = df['Low'].min()
year_low_date = df.loc[df['Low'].idxmin(), 'Date']

avg_volume = df['Volume'].mean()
max_volume = df['Volume'].max()
max_vol_date = df.loc[df['Volume'].idxmax(), 'Date']

# Daily returns
df['Daily_Return'] = df['Close'].pct_change() * 100
avg_daily_return = df['Daily_Return'].mean()
std_daily_return = df['Daily_Return'].std()

# Best and worst days
best_day_idx = df['Daily_Return'].idxmax()
worst_day_idx = df['Daily_Return'].idxmin()
best_day = df.loc[best_day_idx]
worst_day = df.loc[worst_day_idx]

# Annualized volatility
ann_volatility = std_daily_return * np.sqrt(252)

# Monthly returns
df['Month'] = df['Date'].dt.to_period('M')
monthly = df.groupby('Month').agg(
    Open=('Open', 'first'),
    Close=('Close', 'last'),
    High=('High', 'max'),
    Low=('Low', 'min'),
    Volume=('Volume', 'sum')
).reset_index()
monthly['Return'] = ((monthly['Close'] - monthly['Open']) / monthly['Open']) * 100

# Moving averages
df['MA20'] = df['Close'].rolling(window=20).mean()
df['MA50'] = df['Close'].rolling(window=50).mean()
df['MA200'] = df['Close'].rolling(window=200).mean()

# Quarterly returns
df['Quarter'] = df['Date'].dt.to_period('Q')
quarterly = df.groupby('Quarter').agg(
    Open=('Open', 'first'),
    Close=('Close', 'last'),
).reset_index()
quarterly['Return'] = ((quarterly['Close'] - quarterly['Open']) / quarterly['Open']) * 100

# Max drawdown
df['Cummax'] = df['Close'].cummax()
df['Drawdown'] = (df['Close'] - df['Cummax']) / df['Cummax'] * 100
max_drawdown = df['Drawdown'].min()
max_dd_date = df.loc[df['Drawdown'].idxmin(), 'Date']

# Sharpe ratio (assuming 0 risk-free rate for simplicity)
sharpe = (avg_daily_return / std_daily_return) * np.sqrt(252) if std_daily_return != 0 else 0

print("=" * 65)
print("         AAPL STOCK ANALYSIS — FULL YEAR 2023")
print("=" * 65)
print()
print("PRICE PERFORMANCE")
print(f"  Opening Price (Jan 3):     ${open_price:.2f}")
print(f"  Closing Price (Dec 29):    ${close_price:.2f}")
print(f"  YTD Return:                {ytd_return:+.2f}%")
print(f"  52-Week High:              ${year_high:.2f}  ({year_high_date.strftime('%b %d')})")
print(f"  52-Week Low:               ${year_low:.2f}  ({year_low_date.strftime('%b %d')})")
print(f"  Price Range:               ${year_high - year_low:.2f}")
print()
print("RISK METRICS")
print(f"  Avg Daily Return:          {avg_daily_return:+.3f}%")
print(f"  Daily Volatility (σ):      {std_daily_return:.3f}%")
print(f"  Annualized Volatility:     {ann_volatility:.2f}%")
print(f"  Sharpe Ratio (ann.):       {sharpe:.2f}")
print(f"  Max Drawdown:              {max_drawdown:.2f}%  ({max_dd_date.strftime('%b %d')})")
print()
print("NOTABLE DAYS")
print(f"  Best Day:   {best_day['Date'].strftime('%b %d')}  {best_day['Daily_Return']:+.2f}%  (Close: ${best_day['Close']:.2f})")
print(f"  Worst Day:  {worst_day['Date'].strftime('%b %d')}  {worst_day['Daily_Return']:+.2f}%  (Close: ${worst_day['Close']:.2f})")
print()
print("VOLUME")
print(f"  Avg Daily Volume:          {avg_volume/1e6:.1f}M")
print(f"  Max Volume:                {max_volume/1e6:.1f}M  ({max_vol_date.strftime('%b %d')})")
print()
print("QUARTERLY RETURNS")
for _, row in quarterly.iterrows():
    print(f"  {row['Quarter']}:  {row['Return']:+.2f}%")
print()
print("MONTHLY RETURNS")
for _, row in monthly.iterrows():
    bar = "█" * int(abs(row['Return']) / 0.5) if row['Return'] >= 0 else "░" * int(abs(row['Return']) / 0.5)
    sign = "+" if row['Return'] >= 0 else "-"
    print(f"  {row['Month']}:  {row['Return']:+6.2f}%  {'▲' if row['Return'] >= 0 else '▼'} {bar}")

# ============================================================
# VISUALIZATIONS
# ============================================================
sns.set_theme(style="darkgrid")
fig = plt.figure(figsize=(18, 22))

# Color palette
up_color = '#00C853'
down_color = '#FF1744'
accent = '#2196F3'
ma20_color = '#FF9800'
ma50_color = '#E91E63'
ma200_color = '#9C27B0'

# --- Chart 1: Price with Moving Averages ---
ax1 = fig.add_subplot(4, 1, 1)
ax1.plot(df['Date'], df['Close'], color=accent, linewidth=1.5, label='Close Price', alpha=0.9)
ax1.plot(df['Date'], df['MA20'], color=ma20_color, linewidth=1, label='20-Day MA', alpha=0.7)
ax1.plot(df['Date'], df['MA50'], color=ma50_color, linewidth=1, label='50-Day MA', alpha=0.7)
ax1.fill_between(df['Date'], df['Low'], df['High'], alpha=0.1, color=accent)
ax1.set_title('AAPL Price & Moving Averages — 2023', fontsize=14, fontweight='bold', pad=12)
ax1.set_ylabel('Price ($)', fontsize=11)
ax1.legend(loc='upper left', fontsize=9)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax1.xaxis.set_major_locator(mdates.MonthLocator())
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('$%.0f'))

# Annotate high/low
ax1.annotate(f'High: ${year_high:.2f}', xy=(year_high_date, year_high),
             xytext=(year_high_date, year_high + 4), fontsize=8, ha='center',
             arrowprops=dict(arrowstyle='->', color='green'), color='green', fontweight='bold')
ax1.annotate(f'Low: ${year_low:.2f}', xy=(year_low_date, year_low),
             xytext=(year_low_date, year_low - 6), fontsize=8, ha='center',
             arrowprops=dict(arrowstyle='->', color='red'), color='red', fontweight='bold')

# --- Chart 2: Volume ---
ax2 = fig.add_subplot(4, 1, 2)
colors_vol = [up_color if df['Daily_Return'].iloc[i] >= 0 else down_color for i in range(len(df))]
ax2.bar(df['Date'], df['Volume'] / 1e6, color=colors_vol, alpha=0.7, width=1.5)
ax2.axhline(y=avg_volume/1e6, color='white', linestyle='--', linewidth=1, alpha=0.5, label=f'Avg: {avg_volume/1e6:.0f}M')
ax2.set_title('Daily Trading Volume', fontsize=14, fontweight='bold', pad=12)
ax2.set_ylabel('Volume (Millions)', fontsize=11)
ax2.legend(fontsize=9)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax2.xaxis.set_major_locator(mdates.MonthLocator())

# --- Chart 3: Daily Returns Distribution ---
ax3 = fig.add_subplot(4, 1, 3)
returns_clean = df['Daily_Return'].dropna()
n, bins, patches = ax3.hist(returns_clean, bins=50, alpha=0.75, edgecolor='white', linewidth=0.5)
for i, patch in enumerate(patches):
    if bins[i] >= 0:
        patch.set_facecolor(up_color)
    else:
        patch.set_facecolor(down_color)
ax3.axvline(x=0, color='white', linewidth=1, alpha=0.5)
ax3.axvline(x=avg_daily_return, color='yellow', linewidth=1.5, linestyle='--', alpha=0.8, label=f'Mean: {avg_daily_return:+.3f}%')
ax3.set_title('Distribution of Daily Returns', fontsize=14, fontweight='bold', pad=12)
ax3.set_xlabel('Daily Return (%)', fontsize=11)
ax3.set_ylabel('Frequency', fontsize=11)
ax3.legend(fontsize=9)

# --- Chart 4: Monthly Returns Bar Chart ---
ax4 = fig.add_subplot(4, 1, 4)
month_labels = [str(m) for m in monthly['Month']]
colors_monthly = [up_color if r >= 0 else down_color for r in monthly['Return']]
bars = ax4.bar(month_labels, monthly['Return'], color=colors_monthly, alpha=0.85, edgecolor='white', linewidth=0.5)
ax4.axhline(y=0, color='white', linewidth=0.5, alpha=0.5)
for bar, val in zip(bars, monthly['Return']):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.3 if val >= 0 else -0.8),
             f'{val:+.1f}%', ha='center', va='bottom' if val >= 0 else 'top', fontsize=8, fontweight='bold')
ax4.set_title('Monthly Returns', fontsize=14, fontweight='bold', pad=12)
ax4.set_ylabel('Return (%)', fontsize=11)
ax4.tick_params(axis='x', rotation=45)

plt.tight_layout(pad=3.0)
plt.savefig('/sessions/fervent-dazzling-euler/mnt/outputs/AAPL_2023_Analysis.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
print("\n✓ Chart saved to AAPL_2023_Analysis.png")
plt.close()

# ============================================================
# SECOND CHART: Drawdown + Cumulative Returns
# ============================================================
fig2, (ax5, ax6) = plt.subplots(2, 1, figsize=(18, 10))

# Cumulative returns
df['Cum_Return'] = ((1 + df['Daily_Return']/100).cumprod() - 1) * 100
df['Cum_Return'].iloc[0] = 0
ax5.fill_between(df['Date'], 0, df['Cum_Return'],
                  where=df['Cum_Return'] >= 0, alpha=0.3, color=up_color)
ax5.fill_between(df['Date'], 0, df['Cum_Return'],
                  where=df['Cum_Return'] < 0, alpha=0.3, color=down_color)
ax5.plot(df['Date'], df['Cum_Return'], color=accent, linewidth=1.5)
ax5.axhline(y=0, color='white', linewidth=0.5, alpha=0.5)
ax5.set_title('Cumulative Return — 2023', fontsize=14, fontweight='bold', pad=12)
ax5.set_ylabel('Cumulative Return (%)', fontsize=11)
ax5.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax5.xaxis.set_major_locator(mdates.MonthLocator())

# Drawdown
ax6.fill_between(df['Date'], df['Drawdown'], 0, alpha=0.4, color=down_color)
ax6.plot(df['Date'], df['Drawdown'], color=down_color, linewidth=1)
ax6.set_title('Drawdown from Peak', fontsize=14, fontweight='bold', pad=12)
ax6.set_ylabel('Drawdown (%)', fontsize=11)
ax6.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax6.xaxis.set_major_locator(mdates.MonthLocator())

plt.tight_layout(pad=3.0)
plt.savefig('/sessions/fervent-dazzling-euler/mnt/outputs/AAPL_2023_Drawdown.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
print("✓ Drawdown chart saved to AAPL_2023_Drawdown.png")
plt.close()

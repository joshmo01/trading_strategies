---
name: stock-turning-point-analysis
description: Run a complete Stock Turning Point Analysis — detect peaks/troughs using 6 mathematical methods + ensemble consensus, backtest all strategies, compute P&L, generate charts and Excel reports.
---

# Stock Turning Point Analysis

Analyze any stock's price history to detect turning points (peaks and troughs) using 6 mathematically distinct methods, backtest trading strategies based on those turning points, and generate comprehensive P&L reports with professional charts.

## When to Use

Use this skill when the user wants to:
- Analyze a stock's turning points / peaks and troughs
- Find the best entry/exit points for a stock
- Compare turning point detection methods
- Backtest turning point trading strategies
- Generate P&L reports for stock analysis

## Inputs

**Required:**
- **Ticker symbol** (e.g., AAPL, MSFT, QQQ) OR a CSV file path with Date and Close columns

**Optional (with smart defaults):**
- **Period**: How far back to analyze. Default: `1y`. Options: `6mo`, `1y`, `2y`, `5y`, or specific dates.
- **Starting Capital**: Default: `$100,000`
- **Output Directory**: Default: current working directory

## Workflow

### Step 1: Gather Parameters

Use AskUserQuestion to collect:

```
questions:
  - question: "Which stock ticker do you want to analyze? (or provide a CSV file path)"
    header: "Ticker"
    options:
      - label: "Enter ticker symbol"
        description: "Download data via yfinance (e.g., AAPL, MSFT, QQQ, SPY)"
      - label: "Use CSV file"
        description: "Provide path to a CSV file with Date and Close columns"
    multiSelect: false
  - question: "What time period should we analyze?"
    header: "Period"
    options:
      - label: "1 year (Recommended)"
        description: "Last 12 months of trading data"
      - label: "2 years"
        description: "Last 24 months for longer-term patterns"
      - label: "6 months"
        description: "Recent 6 months for short-term analysis"
      - label: "Custom dates"
        description: "Specify exact start and end dates"
    multiSelect: false
```

### Step 2: Install Dependencies

Run the following to ensure all required packages are installed:

```bash
pip install yfinance PyWavelets ruptures openpyxl scipy matplotlib pandas numpy -q
```

### Step 3: Run the Analysis Engine

The analysis engine is located at: `/Users/mohanjoshi/Documents/trading_strategies/stock_tp_engine.py`

Run it using Python:

```python
import sys
sys.path.insert(0, '/Users/mohanjoshi/Documents/trading_strategies')
from stock_tp_engine import run_full_analysis

results = run_full_analysis(
    ticker='<TICKER>',           # or csv_path='<PATH>'
    period='<PERIOD>',           # '1y', '2y', '6mo', etc.
    capital=<CAPITAL>,           # default 100000.0
    output_dir='<OUTPUT_DIR>',   # where to save files
)
```

If the user provides a CSV file instead of a ticker, use `csv_path=` parameter instead of `ticker=`.
If the user provides specific dates, use `start_date='YYYY-MM-DD'` and `end_date='YYYY-MM-DD'`.

### Step 4: Present Results

After the engine completes, present the user with:

1. **Summary table** — Strategy comparison (Return, Trades, Win Rate, Sharpe, Max Drawdown)
2. **Best method** — Which detection method scored highest
3. **Consensus turning points** — The high-confidence peaks and troughs
4. **Output files** — List all generated files:
   - `{TICKER}_Turning_Points.png` — Multi-panel chart with all 6 methods + ensemble
   - `{TICKER}_Cumulative_Returns.png` — Cumulative returns comparison
   - `{TICKER}_TurningPoint_PnL.xlsx` — Excel workbook (Summary + Trade Log + Daily P&L per strategy)
   - `{TICKER}_Strategy_Summary.csv` — Strategy comparison CSV
   - `{TICKER}_Trade_Log.csv` — All trades across all strategies

## Detection Methods

The engine implements these 7 mathematically distinct methods:

1. **Second Derivative (Gaussian-smoothed, sigma=8)** — Calculus-based inflection detection
2. **Ramer-Douglas-Peucker (epsilon=0.02)** — Information-theoretic curve simplification
3. **Wavelet Multi-Resolution (db4, level 4)** — Daubechies wavelet denoising
4. **PELT Change Point (penalty=1.5)** — Penalized regime detection on log returns
5. **Bry-Boschan Algorithm** — NBER gold-standard cycle dating with alternation enforcement
6. **Menger Curvature** — Geometric sharpness measure for sharp reversals
7. **Ivan Scherman VIX Divergence** — 2023 World Cup Champion strategy (75% win rate, 3:1 R/R), adapted with Realized Volatility and ATR as VIX proxies. Two variants: (a) Realized Vol 14d, (b) ATR 14d.

Plus **Ensemble Consensus** — Gaussian-weighted proximity voting across methods 1-6 (threshold >= 2.0)

## Trading Logic

**Methods 1-6 + Ensemble:**
- **BUY** at detected troughs (local minima)
- **SELL** at detected peaks (local maxima)
- Strict buy-sell alternation enforced

**Method 7 (Scherman) — different exit rules:**
- **BUY** when price makes lower low + volatility proxy makes lower high (divergence) + price within Bollinger Bands
- **SELL (profit)**: Close > 30-day SMA
- **SELL (stop)**: Close < lower Bollinger Band (2 sigma below SMA)

All strategies: Long-only, no shorting, execution at close price, no slippage or commissions

## Notes

- The engine handles data normalization, edge cases, and method-specific parameter tuning automatically
- If yfinance download fails, suggest using a CSV file instead
- If any individual method produces 0 signals, that's reported normally (not an error)
- All charts use dark theme with professional formatting
- Excel workbook includes color-coded wins (green) and losses (red)

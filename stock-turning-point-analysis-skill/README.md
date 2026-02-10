# Stock Turning Point Analysis Skill

A comprehensive skill for Claude Desktop that analyzes stock turning points using 7 different mathematical methods and generates detailed P&L reports.

## Installation

1. **Copy the skill to Claude Desktop:**
   - Extract this zip file
   - Copy the entire `stock-turning-point-analysis-skill` folder to:
     - **Mac/Linux**: `~/.claude/skills/`
     - **Windows**: `%USERPROFILE%\.claude\skills\`

2. **Install Python dependencies:**
   ```bash
   pip install yfinance PyWavelets ruptures openpyxl scipy matplotlib pandas numpy
   ```

3. **Update the path in SKILL.md:**
   - Open `SKILL.md`
   - Find line 70: `sys.path.insert(0, '/Users/mohanjoshi/Documents/trading_strategies')`
   - Change it to the directory where you placed `stock_tp_engine.py`
   - Or move `stock_tp_engine.py` to a permanent location and update the path

## Usage

Once installed, you can use this skill in Claude Desktop by asking:
- "Analyze AAPL turning points for the last year"
- "Run turning point analysis on QQQ"
- "Backtest turning point strategies for MSFT"

## What It Does

This skill:
- Detects turning points (peaks and troughs) using 7 mathematical methods
- Backtests trading strategies for each method
- Generates P&L reports with win rates, Sharpe ratios, and max drawdown
- Creates professional charts and Excel workbooks
- Compares all methods and recommends the best performer

## Detection Methods

1. Second Derivative (Gaussian-smoothed)
2. Ramer-Douglas-Peucker
3. Wavelet Multi-Resolution
4. PELT Change Point Detection
5. Bry-Boschan Algorithm
6. Menger Curvature
7. Ivan Scherman VIX Divergence Strategy
8. Ensemble Consensus (weighted voting)

## Output Files

The skill generates:
- `{TICKER}_Turning_Points.png` - Multi-panel chart
- `{TICKER}_Cumulative_Returns.png` - Strategy comparison
- `{TICKER}_TurningPoint_PnL.xlsx` - Full Excel workbook
- `{TICKER}_Strategy_Summary.csv` - Performance metrics
- `{TICKER}_Trade_Log.csv` - All trades

## Requirements

- Python 3.7+
- yfinance, PyWavelets, ruptures, openpyxl, scipy, matplotlib, pandas, numpy

## Author

Created by Mohan Joshi

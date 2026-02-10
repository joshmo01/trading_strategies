# Trading Strategies Project

**Author:** Mohan Joshi
**Location:** `/Users/mohanjoshi/Documents/trading_strategies/`
**No git repo initialized.**

---

## What Was Built

### 1. Stock Turning Point Analysis Engine (`stock_tp_engine.py`)

The core deliverable — a consolidated ~64KB Python module that detects price turning points using 7 mathematical methods, backtests trading strategies, computes P&L, generates charts, and produces Excel reports.

**Entry point:**
```python
from stock_tp_engine import run_full_analysis
results = run_full_analysis(ticker='AAPL', period='1y', capital=100000)
```

**Detection methods implemented:**
1. Second Derivative of Gaussian-Smoothed Price (sigma=8)
2. Ramer-Douglas-Peucker Piecewise Linearization (epsilon=0.02)
3. Wavelet Multi-Resolution Analysis (Daubechies db4, level 4)
4. PELT Change Point Detection (penalty=1.5)
5. Bry-Boschan Algorithm (NBER-style cycle dating)
6. Menger Curvature Analysis
7. Ivan Scherman VIX Divergence (2023 World Cup Champion strategy, adapted with Realized Vol + ATR as VIX proxies)
8. Ensemble Consensus (Gaussian-weighted proximity voting across methods 1-6)

**Trading logic:** Buy at troughs, sell at peaks, long-only, strict alternation. Scherman strategy has its own divergence-based entry and Bollinger Band exit rules.

**Outputs per run:**
- `{TICKER}_Turning_Points.png` — multi-panel chart (all methods + ensemble)
- `{TICKER}_Cumulative_Returns.png` — cumulative returns comparison
- `{TICKER}_TurningPoint_PnL.xlsx` — Excel workbook (Summary + Trade Log + Daily P&L, color-coded)
- `{TICKER}_Strategy_Summary.csv` — performance metrics
- `{TICKER}_Trade_Log.csv` — all trades across all strategies

**Dependencies:** yfinance, PyWavelets, ruptures, openpyxl, scipy, matplotlib, pandas, numpy

---

### 2. Claude Code Skill (`stock-turning-point-analysis-skill/`)

A packaged skill for Claude Code that wraps the engine into an interactive workflow. Installed at `~/.claude/skills/`.

**Files:**
- `SKILL.md` — skill definition with workflow steps (gather params, install deps, run engine, present results)
- `stock_tp_engine.py` — copy of the engine
- `requirements.txt` — Python dependencies
- `README.md` — installation instructions

**Invocation:** User can say "Analyze AAPL turning points" or use the `/stock-turning-point-analysis` skill. The skill prompts for ticker, period, and capital, then runs the analysis.

**Also packaged as:** `stock-tp-skill.zip` for distribution.

**Hardcoded path caveat:** `SKILL.md` line 76 contains `sys.path.insert(0, '/Users/mohanjoshi/Documents/trading_strategies')`. This must be updated if the engine is moved to a different location or if someone else installs the skill.

---

### 3. Supporting Scripts (created during development, Feb 8-9 2025)

These were intermediate scripts built before consolidation into `stock_tp_engine.py`:

| Script | Purpose |
|---|---|
| `turning_points.py` | Original 6-method turning point detection (AAPL 2023) |
| `analyze_aapl.py` | AAPL 2023 analysis with key metrics and charts |
| `backtest_strategies.py` | Strategy backtesting (MA crossover, RSI, Bollinger, MACD, turning points) |
| `build_pnl.py` | P&L computation and Excel workbook generation |
| `scherman_strategy.py` | Ivan Scherman VIX Divergence strategy adapted for individual stocks |
| `tp_pnl.py` | P&L specifically for the 6 turning point methods + ensemble |
| `export_csv.py` | CSV export with indicators (MA, RSI, Bollinger, MACD, signals) |

These scripts reference session paths (`/sessions/fervent-dazzling-euler/...`) from the Claude.ai environment where they were originally developed.

---

### 4. Generated Outputs (Feb 8-10 2025)

**In project root:**
- `AAPL_2023_Analysis.png`, `AAPL_2023_Drawdown.png` — analysis charts
- `AAPL_2023_Strategy_Backtest.png`, `AAPL_2023_Turning_Points.png` — backtest visualizations
- `AAPL_2023_Scherman_Strategy.png` — Scherman strategy chart
- `AAPL_2023_PnL_Analysis.xlsx`, `AAPL_2023_Scherman_PnL.xlsx`, `AAPL_2023_TurningPoint_PnL.xlsx` — Excel reports
- `AAPL_2023_Full_Analysis.csv`, `AAPL_2023_Trade_Log.csv`, `AAPL_2023_Strategy_Summary.csv` — CSV data

**In `tp_output/`:**
- `AAPL_Turning_Points.png`, `AAPL_Cumulative_Returns.png` — charts from engine run
- `AAPL_TurningPoint_PnL.xlsx` — Excel workbook
- `AAPL_Strategy_Summary.csv`, `AAPL_Trade_Log.csv` — CSV outputs

---

## Older Work (Apr-May 2025)

The project directory also contains earlier iterations of stock analysis tools built before the Claude Code sessions:

- **Indicator scripts:** `enhanced_ta_indicators1.x.py` (versions 1.0-1.8), `stock-indicators-analysis1.x.py`, `TA1-5.py`
- **Trendline analysis:** `trendline_channels_code1.x.py`, interactive visualizations (HTML, TSX, SVG)
- **Peak/trough detection:** `find_stock_extrema1.x.py`, `improved_peaks_troughs.py`, `simple_peaks_troughs.py`
- **Reversal detection:** `reversal_detection.py`, `weekly_reversal_detector.py`, `etf_reversal_tutorial.py`
- **Backtesting:** `backtester.py`, `multi_strategy_backtest.py`, `positions_backtest.py`
- **ML approaches:** `GMM_HMM.py`, `HMM_regimes.py`, `colab_to_script_swing_predict1.x.py`
- **Data files:** SPY, QQQ, AAPL CSVs and Excel files spanning 2019-2025
- **Reference docs:** `trading strategies.rtf`, `strategy.rtf`, `sideways.rtf`, `reversal logic.docx`

---

## Project Structure Notes

- No git repo — consider initializing one
- Python virtual environment exists at `myenv/`
- Node/React app at `my-app/` (likely for interactive visualizations)
- `TurningPoint_analysis/` directory contains earlier turning point work (Nov 2023)
- `.python-version` set to Python version in use

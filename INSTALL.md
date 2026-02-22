# Installation Guide

## Prerequisites

- Python 3.9+
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) installed and authenticated
- Git

---

## 1. Clone the Repository

```bash
git clone https://github.com/joshmo01/trading_strategies.git
cd trading_strategies
```

---

## 2. Create a Virtual Environment

```bash
python3 -m venv myenv
source myenv/bin/activate        # macOS / Linux
# myenv\Scripts\activate         # Windows
```

---

## 3. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Dependency overview:**

| Package | Purpose |
|---|---|
| `numpy`, `pandas`, `scipy` | Core data and signal processing |
| `PyWavelets` | Wavelet multi-resolution turning point detection |
| `ruptures` | PELT change point detection |
| `yfinance` | Download OHLCV market data |
| `matplotlib` | Chart generation |
| `openpyxl` | Excel report generation |
| `claude-agent-sdk` | Agent SDK client (`trading_agent.py`) |
| `typer`, `rich` | CLI interface for the skill wrapper |

---

## 4. Install the Claude Code Skill

Copy the skill to Claude Code's skills directory so it can be invoked by `trading_agent.py`:

```bash
mkdir -p ~/.claude/skills/stock-turning-point-analysis
cp stock-turning-point-analysis-skill/SKILL.md ~/.claude/skills/stock-turning-point-analysis/
cp stock-turning-point-analysis-skill/stock_tp_engine.py ~/.claude/skills/stock-turning-point-analysis/
```

---

## 5. Set Your Anthropic API Key

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

Add this to your shell profile (`~/.zshrc` or `~/.bashrc`) to persist it across sessions.

---

## 6. Run the Analysis

### Via the Agent SDK client

```bash
python trading_agent.py --ticker AAPL --period 1y --capital 100000
python trading_agent.py --ticker TSLA --period 2y --capital 50000
python trading_agent.py --ticker QQQ  --period 6mo
```

### Via the engine directly

```python
from stock-turning-point-analysis-skill.stock_tp_engine import run_full_analysis

results = run_full_analysis(ticker='AAPL', period='1y', capital=100_000)
```

### Via Claude Code skill

In Claude Code, say: *"Analyze AAPL turning points over 1 year with $100,000 capital"*

---

## 7. Outputs

Each run produces the following files in the current directory:

| File | Description |
|---|---|
| `{TICKER}_Turning_Points.png` | Multi-panel chart — all 7 methods + ensemble |
| `{TICKER}_Cumulative_Returns.png` | Cumulative returns comparison across strategies |
| `{TICKER}_TurningPoint_PnL.xlsx` | Excel workbook — Summary, Trade Log, Daily P&L |
| `{TICKER}_Strategy_Summary.csv` | Performance metrics per strategy |
| `{TICKER}_Trade_Log.csv` | Full trade-by-trade log |

---

## Troubleshooting

**`ModuleNotFoundError: claude_agent_sdk`**
Ensure `claude-agent-sdk` is installed and you are using the correct virtual environment:
```bash
pip install claude-agent-sdk
```

**`CLINotFoundError`**
The Claude Code CLI must be installed and on your PATH. Verify with:
```bash
claude --version
```

**`yfinance` data download fails**
Check your internet connection. For private/delisted tickers, provide a CSV file with `Date` and `Close` columns instead.

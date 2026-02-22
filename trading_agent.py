"""
Trading Agent — Stock Turning Point Analysis
=============================================
Uses the Claude Agent SDK to invoke the stock-turning-point-analysis skill.
The skill detects peaks/troughs using 6+ methods, backtests strategies,
computes P&L, and generates charts + Excel reports.

Usage:
    python trading_agent.py --ticker AAPL --period 1y --capital 100000
    python trading_agent.py --ticker TSLA --period 2y --capital 50000
    python trading_agent.py --ticker QQQ  --period 6mo
"""

import asyncio
import argparse
import sys
from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    TextBlock,
    ResultMessage,
    CLINotFoundError,
    ProcessError,
)


async def run_agent(ticker: str, period: str, capital: float) -> None:
    prompt = (
        f"Analyze the turning points for {ticker} stock "
        f"over the last {period} with a starting capital of ${capital:,.0f}. "
        f"Run the full turning point analysis — detect peaks and troughs using all methods, "
        f"backtest all strategies, compute P&L, and generate the charts and Excel report."
    )

    # Remove CLAUDECODE from the live environment so the SDK subprocess
    # is not blocked by the nested-session guard in the Claude Code CLI.
    import os
    os.environ.pop("CLAUDECODE", None)
    # If ANTHROPIC_API_KEY is not set externally, try reading it from settings.json.
    # The key in settings.json is injected into the subprocess env by the CLI and
    # must be valid. Override with TRADING_API_KEY env var if you want a different key.
    if os.environ.get("TRADING_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = os.environ["TRADING_API_KEY"]

    options = ClaudeAgentOptions(
        cwd="/Users/mohanjoshi/Documents/trading_strategies",
        setting_sources=["user"],                          # Load ~/.claude/skills/
        allowed_tools=["Skill", "Bash", "Read", "Write"], # Enable skill + file tools
        permission_mode="acceptEdits",                     # Auto-accept file writes
    )

    print(f"\nTicker : {ticker}")
    print(f"Period : {period}")
    print(f"Capital: ${capital:,.0f}")
    print("-" * 60)

    try:
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(block.text, end="", flush=True)
            elif isinstance(message, ResultMessage):
                cost = f"${message.total_cost_usd:.4f}" if message.total_cost_usd else "N/A"
                print(f"\n\n{'─' * 60}")
                print(f"Status : {'ERROR' if message.is_error else 'Complete'}")
                print(f"Turns  : {message.num_turns}")
                print(f"Cost   : {cost}")
                if message.is_error:
                    sys.exit(1)

    except CLINotFoundError:
        print(
            "\nERROR: Claude Code CLI not found.\n"
            "Install it with: npm install -g @anthropic-ai/claude-code",
            file=sys.stderr,
        )
        sys.exit(1)
    except ProcessError as e:
        print(f"\nERROR: Agent process failed (exit code {e.exit_code}).", file=sys.stderr)
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stock Turning Point Agent — invokes the TP analysis skill via Claude SDK",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ticker", default="AAPL",
        help="Stock ticker symbol (e.g. AAPL, TSLA, QQQ, SPY)"
    )
    parser.add_argument(
        "--period", default="1y",
        choices=["6mo", "1y", "2y", "5y"],
        help="Lookback period for price data"
    )
    parser.add_argument(
        "--capital", type=float, default=100_000,
        help="Starting capital in USD"
    )
    args = parser.parse_args()

    asyncio.run(run_agent(args.ticker, args.period, args.capital))


if __name__ == "__main__":
    main()

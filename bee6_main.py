"""
CLI entry point for bee6.

Usage:
  python bee6_main.py backtest [--update-data] [--use-wfo-params]
  python bee6_main.py wfo [--update-data] [--mode rolling|anchored] [--score balanced|defensive|return_only]
  python bee6_main.py download [--symbol ETHUSDT] [--interval 1h]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from bee6_binance import update_csv_cache
from bee6_data import load_klines, prepare_indicators
from bee6_params import (
    BINANCE_INTERVAL,
    BINANCE_MARKET,
    BINANCE_START_DATE,
    BINANCE_SYMBOL,
    CSV_PATH,
    DEFAULT_PARAMS,
    INITIAL_CAPITAL,
    LIVE_DAYS,
    OPT_DAYS,
    load_params,
    save_params,
)
from bee6_stats import (
    breakdown_by_regime,
    breakdown_by_side,
    breakdown_by_strategy,
    compute_stats,
    wfo_summary,
)
from bee6_strategy import Bee6Strategy
from bee6_wfo import walk_forward_optimization


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _load_or_update(
    update: bool,
    symbol: str = BINANCE_SYMBOL,
    interval: str = BINANCE_INTERVAL,
    csv_path: str = CSV_PATH,
) -> pd.DataFrame:
    if update or not Path(csv_path).exists():
        update_csv_cache(
            csv_path,
            symbol=symbol,
            interval=interval,
            start_date=BINANCE_START_DATE,
            market=BINANCE_MARKET,
            verbose=True,
        )
    return prepare_indicators(load_klines(csv_path))


def _save_outputs(
    prefix: str,
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    stats: dict,
    windows: pd.DataFrame | None = None,
) -> None:
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    stamp = _timestamp()
    if not trades.empty:
        trades.to_csv(out_dir / f"{prefix}_trades_{stamp}.csv", index=False)
    if not equity.empty:
        equity.to_csv(out_dir / f"{prefix}_equity_{stamp}.csv", index=False)
    if stats:
        pd.Series(stats).to_json(out_dir / f"{prefix}_stats_{stamp}.json", indent=2, date_format="iso")
    if windows is not None and not windows.empty:
        windows.to_csv(out_dir / f"{prefix}_windows_{stamp}.csv", index=False)
    print(f"[output] Results saved to results/{prefix}_*_{stamp}.*")


def cmd_download(args) -> None:
    symbol = args.symbol or BINANCE_SYMBOL
    interval = args.interval or BINANCE_INTERVAL
    csv_path = f"{symbol.lower()}_{interval}.csv"
    update_csv_cache(csv_path, symbol=symbol, interval=interval, start_date=BINANCE_START_DATE, verbose=True)


def cmd_backtest(args) -> None:
    params = load_params() if args.use_wfo_params else dict(DEFAULT_PARAMS)
    df = _load_or_update(args.update_data)
    trades, equity, final_cap = Bee6Strategy(params).run(df, args.initial_capital)
    stats = compute_stats(trades, equity, args.initial_capital)
    print(f"\nFinal capital: {final_cap:,.2f} USD")

    if not trades.empty:
        print("\nBy strategy:")
        print(breakdown_by_strategy(trades).to_string(index=False))
        print("\nBy regime:")
        print(breakdown_by_regime(trades).to_string(index=False))
        print("\nBy side:")
        print(breakdown_by_side(trades).to_string(index=False))

    _save_outputs("backtest", trades, equity, stats)


def cmd_wfo(args) -> None:
    df = _load_or_update(args.update_data)
    trades, equity, windows, final_cap, stopped, best_params = walk_forward_optimization(
        df=df,
        interval=BINANCE_INTERVAL,
        score_mode=args.score,
        wfo_mode=args.mode,
        verbose=True,
        opt_days=args.opt_days,
        live_days=args.live_days,
        initial_capital=args.initial_capital,
    )

    if stopped:
        print("[WFO] Stopped early.")

    if best_params:
        save_params(best_params)

    stats = compute_stats(trades, equity, args.initial_capital, label="WFO live periods")
    summary = wfo_summary(windows)
    print(f"\nWFO summary: {summary}")

    _save_outputs("wfo", trades, equity, stats, windows)


def main() -> None:
    parser = argparse.ArgumentParser(description="bee6 trading system")
    sub = parser.add_subparsers(dest="command")

    # download
    dl = sub.add_parser("download", help="Download candle data from Binance")
    dl.add_argument("--symbol", default=BINANCE_SYMBOL)
    dl.add_argument("--interval", default=BINANCE_INTERVAL)

    # backtest
    bt = sub.add_parser("backtest", help="Run single backtest")
    bt.add_argument("--update-data", action="store_true")
    bt.add_argument("--use-wfo-params", action="store_true")
    bt.add_argument("--initial-capital", type=float, default=INITIAL_CAPITAL)

    # wfo
    wfo = sub.add_parser("wfo", help="Walk-forward optimization")
    wfo.add_argument("--update-data", action="store_true")
    wfo.add_argument("--mode", choices=["rolling", "anchored"], default="rolling")
    wfo.add_argument("--score", choices=["balanced", "defensive", "return_only"], default="balanced")
    wfo.add_argument("--opt-days", type=int, default=OPT_DAYS)
    wfo.add_argument("--live-days", type=int, default=LIVE_DAYS)
    wfo.add_argument("--initial-capital", type=float, default=INITIAL_CAPITAL)

    args = parser.parse_args()

    if args.command == "download":
        cmd_download(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "wfo":
        cmd_wfo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

"""
Statistics, reports and breakdowns for bee6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bee6_params import INITIAL_CAPITAL


def compute_stats(
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
    label: str = "Bee6",
    print_output: bool = True,
) -> dict:
    if trades is None or trades.empty or equity is None or equity.empty:
        return {}

    equity = equity.dropna(subset=["time", "equity"]).reset_index(drop=True)
    final_capital = float(equity["equity"].iloc[-1])
    net_profit_usd = final_capital - initial_capital
    net_return_pct = (final_capital / initial_capital - 1.0) * 100.0
    n_trades = len(trades)

    wins_df = trades[trades["pnl"] > 0]
    losses_df = trades[trades["pnl"] <= 0]
    winrate = len(wins_df) / n_trades * 100.0 if n_trades else 0.0
    gross_profit = wins_df["pnl"].sum()
    gross_loss = losses_df["pnl"].sum()
    pf = gross_profit / abs(gross_loss) if gross_loss < 0 else float("nan")

    eq = pd.to_numeric(equity["equity"], errors="coerce").to_numpy()
    run_max = np.maximum.accumulate(eq)
    dd_arr = (eq - run_max) / np.where(run_max == 0, np.nan, run_max)
    max_dd = np.nanmin(dd_arr) * 100.0

    trade_returns = pd.to_numeric(trades["net_ret"], errors="coerce").dropna()
    if len(trade_returns) > 1:
        std = trade_returns.std(ddof=1)
        sharpe = trade_returns.mean() / std * np.sqrt(len(trade_returns)) if std > 0 else float("nan")
        downside = trade_returns[trade_returns < 0]
        down_std = downside.std(ddof=1) if len(downside) > 1 else float("nan")
        sortino = (
            trade_returns.mean() / down_std * np.sqrt(len(trade_returns))
            if down_std and down_std > 0
            else float("nan")
        )
    else:
        sharpe = sortino = float("nan")

    avg_win = wins_df["pnl"].mean() if not wins_df.empty else 0.0
    avg_loss = losses_df["pnl"].mean() if not losses_df.empty else 0.0
    expectancy = (winrate / 100.0) * avg_win + (1.0 - winrate / 100.0) * avg_loss
    risk_reward = avg_win / abs(avg_loss) if avg_loss < 0 else float("nan")

    r_mult = pd.to_numeric(trades.get("r_multiple", pd.Series(dtype="float64")), errors="coerce")
    avg_r = r_mult.mean() if len(r_mult.dropna()) else float("nan")

    consistency_pct = float("nan")
    period_return_std_pct = float("nan")
    stability_score = float("nan")
    if "exit_time" in trades.columns:
        tmp = trades.copy()
        tmp["exit_time"] = pd.to_datetime(tmp["exit_time"], utc=True, errors="coerce")
        tmp = tmp.dropna(subset=["exit_time"]).set_index("exit_time")
        monthly = tmp["pnl"].resample("ME").sum() / initial_capital * 100.0
        monthly = monthly[monthly != 0]
        if len(monthly) > 0:
            consistency_pct = (monthly > 0).mean() * 100.0
        if len(monthly) > 1:
            period_return_std_pct = monthly.std(ddof=1)
            stability_score = monthly.mean() / period_return_std_pct if period_return_std_pct > 0 else float("nan")

    t0 = pd.Timestamp(equity["time"].iloc[0])
    t1 = pd.Timestamp(equity["time"].iloc[-1])
    days = (t1 - t0).total_seconds() / 86400.0
    cagr = (
        ((final_capital / initial_capital) ** (1.0 / (days / 365.0)) - 1.0) * 100.0
        if days > 0
        else float("nan")
    )

    stats = {
        "label": label,
        "n_trades": n_trades,
        "initial_capital": initial_capital,
        "final_capital": final_capital,
        "net_profit_usd": net_profit_usd,
        "net_return_pct": net_return_pct,
        "cagr_pct": cagr,
        "winrate_pct": winrate,
        "profit_factor": pf,
        "max_drawdown_pct": max_dd,
        "return_drawdown_ratio": net_return_pct / abs(max_dd) if max_dd < 0 else float("nan"),
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "risk_reward_ratio": risk_reward,
        "expectancy_usd": expectancy,
        "avg_r_multiple": avg_r,
        "fee_total_usd": trades["fee_usd"].sum() if "fee_usd" in trades.columns else 0.0,
        "slippage_total_usd": trades["slippage_usd"].sum() if "slippage_usd" in trades.columns else 0.0,
        "consistency_pct": consistency_pct,
        "period_return_std_pct": period_return_std_pct,
        "stability_score": stability_score,
        "period_start": t0,
        "period_end": t1,
        "period_days": days,
    }

    if print_output:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        print(f"  Trades : {n_trades}")
        print(f"  Return : {net_return_pct:.2f}%  (CAGR {cagr:.1f}%)")
        print(f"  Max DD : {max_dd:.2f}%")
        print(f"  PF     : {pf:.2f}  |  Win%: {winrate:.1f}%  |  Avg-R: {avg_r:.2f}")
        print(f"  Sharpe : {sharpe:.2f}  |  Sortino: {sortino:.2f}")
        print(f"  Consis : {consistency_pct:.1f}% months positive")
        print(f"{'='*60}")
    return stats


def breakdown_by_strategy(trades: pd.DataFrame) -> pd.DataFrame:
    return _breakdown(trades, "strategy")


def breakdown_by_regime(trades: pd.DataFrame) -> pd.DataFrame:
    return _breakdown(trades, "regime")


def breakdown_by_side(trades: pd.DataFrame) -> pd.DataFrame:
    return _breakdown(trades, "side")


def _breakdown(trades: pd.DataFrame, key: str) -> pd.DataFrame:
    if trades is None or trades.empty or key not in trades.columns:
        return pd.DataFrame()
    grp = trades.groupby(key)
    win_rate = grp["pnl"].apply(lambda s: (s > 0).mean() * 100.0)
    return pd.DataFrame({
        key: grp["pnl"].count().index,
        "n_trades": grp["pnl"].count().values,
        "win_rate": win_rate.values,
        "avg_pnl": grp["pnl"].mean().values,
        "total_pnl": grp["pnl"].sum().values,
        "avg_r": grp["r_multiple"].mean().values if "r_multiple" in trades.columns else np.nan,
    })


def breakdown_by_period(trades: pd.DataFrame, freq: str = "ME") -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame()
    df = trades.copy()
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["exit_time"]).set_index("exit_time")
    grp = df.resample(freq)
    win_rate = grp["pnl"].apply(lambda s: (s > 0).mean() * 100.0 if len(s) else np.nan)
    res = pd.DataFrame({
        "n_trades": grp["pnl"].count(),
        "win_rate": win_rate,
        "total_pnl": grp["pnl"].sum(),
        "avg_pnl": grp["pnl"].mean(),
    })
    return res[res["n_trades"] > 0].reset_index().rename(columns={"exit_time": "period"})


def wfo_summary(windows_df: pd.DataFrame) -> dict:
    if windows_df is None or windows_df.empty:
        return {}
    live_ret = windows_df["live_return_pct"]
    return {
        "n_windows": len(windows_df),
        "pct_profitable_windows": (live_ret > 0).mean() * 100.0,
        "avg_live_return_pct": live_ret.mean(),
        "median_live_return_pct": live_ret.median(),
        "worst_window_pct": live_ret.min(),
        "best_window_pct": live_ret.max(),
        "avg_score": windows_df["best_score"].mean() if "best_score" in windows_df.columns else float("nan"),
    }

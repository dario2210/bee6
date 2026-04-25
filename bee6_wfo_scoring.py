"""
Walk-forward scoring for bee6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def score_params(
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    final_capital: float,
    initial_capital: float,
    mode: str = "balanced",
    min_trades: int = 20,
) -> float:
    if trades is None or trades.empty or equity is None or equity.empty:
        return -9999.0

    n = len(trades)
    ret_pct = (final_capital / initial_capital - 1.0) * 100.0
    wins = trades[trades["pnl"] > 0]["pnl"].sum()
    losses = trades[trades["pnl"] <= 0]["pnl"].sum()
    pf = wins / abs(losses) if losses < 0 else 0.0

    eq = pd.to_numeric(equity["equity"], errors="coerce").to_numpy()
    run_max = np.maximum.accumulate(eq)
    dd = (eq - run_max) / np.where(run_max == 0, np.nan, run_max)
    max_dd_pct = np.nanmin(dd) * 100.0

    r = pd.to_numeric(trades.get("r_multiple", pd.Series(dtype="float64")), errors="coerce").dropna()
    avg_r = r.mean() if len(r) else 0.0

    pf_penalty = max(0.0, (1.3 - pf) * 20.0) if pf < 1.3 else 0.0
    dd_penalty = max(0.0, abs(max_dd_pct) - 20.0) * 3.0
    trade_penalty = max(0, min_trades - n) * 0.65

    smoothness = 0.0
    if len(eq) > 3:
        changes = pd.Series(eq).pct_change().dropna()
        smoothness = -changes.std(ddof=1) * 100.0 if len(changes) > 1 else 0.0

    # Consistency bonus: monthly positive months
    consistency_bonus = 0.0
    if "exit_time" in trades.columns and not trades.empty:
        tmp = trades.copy()
        tmp["exit_time"] = pd.to_datetime(tmp["exit_time"], utc=True, errors="coerce")
        tmp = tmp.dropna(subset=["exit_time"]).set_index("exit_time")
        monthly = tmp["pnl"].resample("ME").sum()
        monthly = monthly[monthly != 0]
        if len(monthly) >= 2:
            pos_ratio = (monthly > 0).mean()
            consistency_bonus = pos_ratio * 10.0

    if mode == "defensive":
        return (
            ret_pct * 0.4
            + min(pf, 3.0) * 8.0
            + avg_r * 4.0
            + smoothness
            + consistency_bonus
            - dd_penalty * 2.0
            - pf_penalty
            - trade_penalty
        )
    if mode == "return_only":
        return ret_pct - pf_penalty - dd_penalty - trade_penalty
    # balanced (default)
    return (
        ret_pct
        + min(pf, 3.0) * 5.0
        + avg_r * 4.0
        + smoothness
        + consistency_bonus
        - dd_penalty
        - pf_penalty
        - trade_penalty
    )

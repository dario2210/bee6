"""
Walk-Forward Optimization for bee6.

Two modes:
  rolling  – fixed-size IS window slides forward (classic WFO)
  anchored – IS window grows from a fixed start (expanding window)
"""

from __future__ import annotations

from itertools import product
from typing import Callable, Optional

import pandas as pd

from bee6_binance import wfo_bars
from bee6_params import (
    ADX_RANGE_GRID,
    ADX_TREND_GRID,
    BB_WIDTH_COMPRESSION_GRID,
    BB_WIDTH_RANGE_GRID,
    BINANCE_INTERVAL,
    DEFAULT_PARAMS,
    DONCHIAN_LEN_GRID,
    INITIAL_CAPITAL,
    LIVE_DAYS,
    MACD_MODE_GRID,
    OPT_DAYS,
    RISK_PCT_GRID,
    TREND_RSI_MAX_GRID,
    TREND_RSI_MIN_GRID,
    VOLUME_SPIKE_GRID,
    WILLIAMS_R_OB_GRID,
    WILLIAMS_R_OS_GRID,
)
from bee6_strategy import Bee6Strategy
from bee6_wfo_scoring import score_params


def _clean_grid(values, fallback, caster):
    source = fallback if values is None or len(values) == 0 else values
    cleaned = []
    for v in source:
        cv = caster(v)
        if cv not in cleaned:
            cleaned.append(cv)
    return cleaned


def build_grids(grid_overrides: dict | None = None) -> dict:
    go = grid_overrides or {}
    return {
        "adx_trend_threshold":      _clean_grid(go.get("adx_trend_threshold"),      ADX_TREND_GRID,              float),
        "adx_range_threshold":      _clean_grid(go.get("adx_range_threshold"),      ADX_RANGE_GRID,              float),
        "bb_width_range_max":       _clean_grid(go.get("bb_width_range_max"),       BB_WIDTH_RANGE_GRID,         float),
        "bb_width_compression_max": _clean_grid(go.get("bb_width_compression_max"), BB_WIDTH_COMPRESSION_GRID,   float),
        "volume_spike_mult":        _clean_grid(go.get("volume_spike_mult"),        VOLUME_SPIKE_GRID,           float),
        "donchian_len":             _clean_grid(go.get("donchian_len"),             DONCHIAN_LEN_GRID,           int),
        "risk_pct":                 _clean_grid(go.get("risk_pct"),                 RISK_PCT_GRID,               float),
        "trend_rsi_min":            _clean_grid(go.get("trend_rsi_min"),            TREND_RSI_MIN_GRID,          float),
        "trend_rsi_max":            _clean_grid(go.get("trend_rsi_max"),            TREND_RSI_MAX_GRID,          float),
        "macd_mode":                _clean_grid(go.get("macd_mode"),                MACD_MODE_GRID,              str),
        "range_williams_r_short":   _clean_grid(go.get("range_williams_r_short"),   WILLIAMS_R_OB_GRID,          float),
        "range_williams_r_long":    _clean_grid(go.get("range_williams_r_long"),    WILLIAMS_R_OS_GRID,          float),
    }


def walk_forward_optimization(
    df: pd.DataFrame,
    interval: str = BINANCE_INTERVAL,
    score_mode: str = "balanced",
    wfo_mode: str = "rolling",          # "rolling" | "anchored"
    verbose: bool = True,
    on_window_done: Optional[Callable] = None,
    on_combo_progress: Optional[Callable] = None,
    should_stop: Optional[Callable] = None,
    opt_days: int = OPT_DAYS,
    live_days: int = LIVE_DAYS,
    initial_capital: float = INITIAL_CAPITAL,
    base_params: Optional[dict] = None,
    grid_overrides: Optional[dict] = None,
):
    """
    Run walk-forward optimization.

    Returns:
        trades_df, equity_df, windows_df, final_capital, stopped, latest_best_params
    """
    opt_bars, live_bars = wfo_bars(interval, opt_days, live_days)
    base = dict(DEFAULT_PARAMS)
    if base_params:
        base.update(base_params)

    grids = build_grids(grid_overrides)
    grid_names = list(grids)
    grid_values = [grids[name] for name in grid_names]
    combo_total = 1
    for gv in grid_values:
        combo_total *= max(1, len(gv))

    n = len(df)
    total_windows = max(0, (n - opt_bars) // live_bars)

    if verbose:
        print(
            f"[WFO] mode={wfo_mode} | candles={n} | ~{total_windows} windows | "
            f"opt={opt_days}d | live={live_days}d | combos/window={combo_total}"
        )

    start = 0
    window_id = 0
    current_capital = float(initial_capital)
    all_live_trades: list[pd.DataFrame] = []
    global_equity: pd.DataFrame | None = None
    window_stats: list[dict] = []
    stopped = False
    latest_best_params = None

    while start + opt_bars + live_bars <= n:
        if should_stop is not None and should_stop():
            stopped = True
            break

        # In anchored mode the IS window always starts at 0
        is_start = 0 if wfo_mode == "anchored" else start
        opt_slice = df.iloc[is_start : start + opt_bars].copy()
        live_slice = df.iloc[start + opt_bars : start + opt_bars + live_bars].copy()

        best_score = -1e12
        best_params = None
        best_opt_cap = initial_capital

        if on_combo_progress:
            on_combo_progress(window_id, total_windows, 0, combo_total)

        for combo_idx, values in enumerate(product(*grid_values), start=1):
            if should_stop is not None and should_stop():
                stopped = True
                break

            params = dict(base)
            params.update(dict(zip(grid_names, values)))

            # skip invalid RSI range
            if params["trend_rsi_min"] >= params["trend_rsi_max"]:
                continue

            trades, equity, final_cap = Bee6Strategy(params).run(opt_slice, initial_capital)
            score = score_params(trades, equity, final_cap, initial_capital, score_mode)
            if score > best_score:
                best_score = score
                best_params = params
                best_opt_cap = final_cap

            if on_combo_progress and (
                combo_idx == combo_total or combo_idx % max(1, combo_total // 20) == 0
            ):
                on_combo_progress(window_id, total_windows, combo_idx, combo_total)

        if stopped or best_params is None:
            break

        live_trades, live_equity, live_cap = Bee6Strategy(best_params).run(live_slice, current_capital)
        live_return_pct = (live_cap / current_capital - 1.0) * 100.0 if current_capital else 0.0
        current_capital = live_cap
        latest_best_params = dict(best_params)

        if not live_trades.empty:
            lt = live_trades.copy()
            lt["window_id"] = window_id
            all_live_trades.append(lt)

        le = live_equity.copy()
        le["window_id"] = window_id
        global_equity = le if global_equity is None else pd.concat([global_equity, le], ignore_index=True)

        row = {
            "window_id": window_id,
            "wfo_mode": wfo_mode,
            "opt_start": opt_slice["time"].iloc[0],
            "opt_end": opt_slice["time"].iloc[-1],
            "live_start": live_slice["time"].iloc[0],
            "live_end": live_slice["time"].iloc[-1],
            "best_score": best_score,
            "opt_final_capital": best_opt_cap,
            "live_final_capital": live_cap,
            "live_return_pct": live_return_pct,
            "n_trades_opt": 0,
            "n_trades_live": len(live_trades),
        }
        for name in grid_names:
            row[f"best_{name}"] = best_params[name]
        window_stats.append(row)

        if on_window_done:
            on_window_done(row)
        if verbose:
            print(
                f"[WFO] window={window_id:03d} | score={best_score:.2f} | "
                f"live_ret={live_return_pct:+.2f}% | trades={len(live_trades)} | "
                f"capital={current_capital:,.0f}"
            )

        start += live_bars
        window_id += 1

    trades_df = (
        pd.concat(all_live_trades, ignore_index=True) if all_live_trades else pd.DataFrame()
    )
    equity_df = (
        global_equity if global_equity is not None
        else pd.DataFrame(columns=["time", "equity"])
    )
    windows_df = pd.DataFrame(window_stats)
    return trades_df, equity_df, windows_df, current_capital, stopped, latest_best_params


def get_latest_best_params(path: str = "bee6_wfo_best_params.json") -> dict:
    from bee6_params import load_params
    return load_params()

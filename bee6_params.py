"""
Central configuration for bee6.

bee6 is a multi-indicator regime-switching strategy for DEX use:
TREND     -> Trend Pullback (EMA + ADX + RSI)
RANGE     -> Mean Reversion (BB + RSI + Williams %R)
BREAKOUT  -> Donchian Breakout (volume + BB compression)
MACD_CROSS -> MACD momentum signals
"""

from __future__ import annotations

import json
import os

# Data
CSV_PATH = "ethusdt_1h.csv"
INITIAL_CAPITAL = 10_000.0

BINANCE_SYMBOL = "ETHUSDT"
BINANCE_INTERVAL = "1h"
BINANCE_MARKET = "spot"
BINANCE_START_DATE = "2021-01-01"

# Costs (DEX: higher fees, no slippage on limit orders)
FEE_RATE = 0.0005      # 0.05% per side (Uniswap v3 / typical DEX)
SLIPPAGE_BPS = 3.0     # market order slippage estimate
SPREAD_BPS = 2.0

# Indicator lengths
EMA_FAST_LEN = 50
EMA_SLOW_LEN = 200
ADX_LEN = 14
ATR_LEN = 14
RSI_LEN = 14
BB_LEN = 20
BB_STD = 2.0
VOLUME_SMA_LEN = 20
DONCHIAN_DEFAULT = 24
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
STOCH_RSI_LEN = 14
STOCH_RSI_K = 3
STOCH_RSI_D = 3
WILLIAMS_R_LEN = 14
OBV_EMA_LEN = 21

# WFO windows
OPT_DAYS = 90
LIVE_DAYS = 30

# Risk controls
MAX_DRAWDOWN_PCT = 15.0
MAX_DAILY_LOSS_PCT = 3.0
MAX_WEEKLY_LOSS_PCT = 6.0
PAUSE_AFTER_LOSSES = 4
DISABLE_AFTER_LOSSES = 6
PAUSE_HOURS = 24

# WFO grids
ADX_TREND_GRID = [25.0, 28.0, 32.0]
ADX_RANGE_GRID = [18.0, 20.0]
BB_WIDTH_RANGE_GRID = [0.05, 0.065]
BB_WIDTH_COMPRESSION_GRID = [0.025, 0.035]
VOLUME_SPIKE_GRID = [1.35, 1.6, 1.9]
DONCHIAN_LEN_GRID = [24, 48]
RISK_PCT_GRID = [0.005, 0.0075, 0.01]
TREND_RSI_MIN_GRID = [38.0, 42.0]
TREND_RSI_MAX_GRID = [55.0, 60.0]
MACD_MODE_GRID = ["histogram", "signal_cross"]
WILLIAMS_R_OB_GRID = [-20.0, -25.0]
WILLIAMS_R_OS_GRID = [-75.0, -80.0]

DEFAULT_PARAMS = {
    # direction
    "trade_direction": "both",
    # sizing
    "risk_pct": 0.0075,
    "max_notional_pct": 1.0,
    # costs
    "fee_rate": FEE_RATE,
    "slippage_bps": SLIPPAGE_BPS,
    "spread_bps": SPREAD_BPS,
    # regime thresholds
    "adx_trend_threshold": 25.0,
    "adx_range_threshold": 20.0,
    "ema_flatness_max": 0.006,
    "bb_width_range_max": 0.055,
    "bb_width_compression_max": 0.035,
    "volume_spike_mult": 1.5,
    "donchian_len": DONCHIAN_DEFAULT,
    # trend strategy
    "trend_rsi_min": 40.0,
    "trend_rsi_max": 55.0,
    "trend_pullback_pct": 0.004,
    "trend_stop_atr_mult": 0.5,
    "trend_target_r": 2.0,
    "trend_trailing_atr_mult": 1.4,
    # range strategy
    "range_rsi_long": 30.0,
    "range_rsi_short": 70.0,
    "range_williams_r_long": -80.0,
    "range_williams_r_short": -20.0,
    "range_stop_atr_mult": 1.2,
    "range_target": "middle",
    "range_require_williams": True,
    # breakout strategy
    "breakout_stop_atr_mult": 1.0,
    "breakout_target_r": 3.0,
    "breakout_trail_trigger_r": 1.5,
    "breakout_trailing_atr_mult": 1.6,
    # MACD strategy
    "macd_mode": "histogram",          # "histogram" | "signal_cross"
    "macd_stop_atr_mult": 1.2,
    "macd_target_r": 2.5,
    "macd_trailing_atr_mult": 1.5,
    "macd_require_obv_confirm": True,
    # candlestick filter
    "use_candle_filter": True,
    # risk gates
    "max_bars_in_trade": 96,
    "max_drawdown_pct": MAX_DRAWDOWN_PCT,
    "max_daily_loss_pct": MAX_DAILY_LOSS_PCT,
    "max_weekly_loss_pct": MAX_WEEKLY_LOSS_PCT,
    "pause_after_losses": PAUSE_AFTER_LOSSES,
    "disable_after_losses": DISABLE_AFTER_LOSSES,
    "pause_hours": PAUSE_HOURS,
}

WFO_BEST_PARAMS_PATH = "bee6_wfo_best_params.json"


def load_params() -> dict:
    params = dict(DEFAULT_PARAMS)
    json_path = os.path.join(os.path.dirname(__file__), WFO_BEST_PARAMS_PATH)
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                params.update({k: data[k] for k in params if k in data})
            print(f"[params] Loaded WFO params from {WFO_BEST_PARAMS_PATH}")
        except Exception as exc:
            print(f"[params] Could not load {WFO_BEST_PARAMS_PATH}: {exc!r}")
    return params


def save_params(params: dict, path: str = WFO_BEST_PARAMS_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    print(f"[params] Saved params -> {path}")

"""
Signal engine for bee6.

Regimes: TREND | RANGE | BREAKOUT | MACD | NEUTRAL
Each regime generates entry/exit signals independently.
No live exchange connection — DEX-ready output only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

Side = Literal["long", "short"]
Action = Literal["none", "open_long", "open_short", "close"]


@dataclass
class BarData:
    time: object
    open: float
    high: float
    low: float
    close: float
    volume: float
    ema50: float
    ema200: float
    ema50_flatness: float
    adx: float
    rsi: float
    atr: float
    bb_mid: float
    bb_upper: float
    bb_lower: float
    bb_width: float
    volume_sma20: float
    volume_ratio: float
    donchian_high_prev: float
    donchian_low_prev: float
    swing_low: float
    swing_high: float
    # new in bee6
    macd: float
    macd_signal: float
    macd_hist: float
    macd_hist_prev: float
    stoch_k: float
    stoch_d: float
    williams_r: float
    obv_rising: int
    bull_power: float
    bear_power: float
    # candlestick patterns
    pat_doji: int
    pat_hammer: int
    pat_shooting_star: int
    pat_bull_engulf: int
    pat_bear_engulf: int
    pat_morning_star: int
    pat_evening_star: int
    pat_bull_marubozu: int
    pat_bear_marubozu: int


@dataclass
class Signal:
    action: Action
    reason: str = ""
    side: Optional[Side] = None
    strategy: str = ""
    regime: str = "NEUTRAL"
    stop_price: float = np.nan
    target_price: float = np.nan
    tp1_price: float = np.nan
    trailing_trigger_price: float = np.nan
    trailing_distance: float = np.nan
    exit_price: Optional[float] = None
    meta: dict = field(default_factory=dict)


@dataclass
class PositionState:
    side: Side
    strategy: str
    regime: str
    entry_price: float
    entry_time: object
    qty: float
    capital_at_open: float
    stop_price: float
    target_price: float
    tp1_price: float
    trailing_trigger_price: float
    trailing_distance: float
    entry_atr: float
    risk_usd: float
    risk_per_unit: float
    entry_meta: dict = field(default_factory=dict)
    bars_in_position: int = 0
    best_price: float = np.nan
    trailing_active: bool = False


def _float(value) -> float:
    if value is None:
        return np.nan
    try:
        if np.isnan(value):
            return np.nan
    except TypeError:
        pass
    return float(value)


def _direction_allowed(params: dict, side: Side) -> bool:
    direction = str(params.get("trade_direction", "both")).lower()
    if direction == "long":
        return side == "long"
    if direction == "short":
        return side == "short"
    return True


def market_regime(bar: BarData, prev: BarData, params: dict) -> str:
    # MACD momentum gets priority when MACD crosses zero with OBV confirmation
    if not any(np.isnan([bar.macd, bar.macd_hist, bar.macd_hist_prev, bar.macd_signal])):
        hist_cross_up = bar.macd_hist > 0 and bar.macd_hist_prev <= 0
        hist_cross_dn = bar.macd_hist < 0 and bar.macd_hist_prev >= 0
        obv_ok = not params.get("macd_require_obv_confirm", True) or bar.obv_rising in (0, 1)
        if (hist_cross_up or hist_cross_dn) and obv_ok:
            return "MACD"

    # BREAKOUT: BB compression + volume spike + Donchian breach
    if not any(np.isnan([prev.bb_width, bar.volume_ratio])):
        compression = prev.bb_width <= float(params.get("bb_width_compression_max", 0.035))
        volume_spike = bar.volume_ratio >= float(params.get("volume_spike_mult", 1.5))
        don_break = (
            (not np.isnan(bar.donchian_high_prev) and bar.close > bar.donchian_high_prev)
            or (not np.isnan(bar.donchian_low_prev) and bar.close < bar.donchian_low_prev)
        )
        if compression and volume_spike and don_break:
            return "BREAKOUT"

    # TREND: ADX + EMA structure
    if not np.isnan(bar.adx):
        trend_up = bar.ema50 > bar.ema200 and bar.close > bar.ema200
        trend_dn = bar.ema50 < bar.ema200 and bar.close < bar.ema200
        if bar.adx > float(params.get("adx_trend_threshold", 25.0)) and (trend_up or trend_dn):
            return "TREND"

    # RANGE: low ADX + flat EMA + narrow BB
    if not any(np.isnan([bar.adx, bar.ema50_flatness, bar.bb_width])):
        if (
            bar.adx < float(params.get("adx_range_threshold", 20.0))
            and bar.ema50_flatness <= float(params.get("ema_flatness_max", 0.006))
            and bar.bb_width <= float(params.get("bb_width_range_max", 0.055))
        ):
            return "RANGE"

    return "NEUTRAL"


def _meta(bar: BarData, regime: str, strategy: str) -> dict:
    def r(v, n=4):
        return round(v, n) if not np.isnan(v) else np.nan

    return {
        "regime": regime,
        "strategy": strategy,
        "atr": r(bar.atr),
        "rsi": r(bar.rsi),
        "adx": r(bar.adx),
        "macd_hist": r(bar.macd_hist, 6),
        "stoch_k": r(bar.stoch_k),
        "williams_r": r(bar.williams_r),
        "obv_rising": bar.obv_rising,
        "bb_width": r(bar.bb_width, 6),
        "volume_ratio": r(bar.volume_ratio),
        "bull_power": r(bar.bull_power),
        "bear_power": r(bar.bear_power),
    }


def _build_entry(
    side: Side,
    strategy: str,
    regime: str,
    bar: BarData,
    stop: float,
    target: float,
    tp1: float,
    trailing_trigger: float,
    trailing_distance: float,
    reason: str,
) -> Signal:
    if np.isnan(stop) or np.isnan(target) or stop <= 0.0 or target <= 0.0:
        return Signal(action="none")
    risk_per_unit = abs(bar.close - stop)
    if risk_per_unit <= 0.0:
        return Signal(action="none")
    return Signal(
        action="open_long" if side == "long" else "open_short",
        reason=reason,
        side=side,
        strategy=strategy,
        regime=regime,
        stop_price=float(stop),
        target_price=float(target),
        tp1_price=float(tp1),
        trailing_trigger_price=float(trailing_trigger),
        trailing_distance=float(trailing_distance),
        meta=_meta(bar, regime, strategy),
    )


def _bullish_candle(bar: BarData, params: dict) -> bool:
    if not params.get("use_candle_filter", True):
        return True
    return bar.pat_hammer or bar.pat_bull_engulf or bar.pat_morning_star or bar.pat_bull_marubozu


def _bearish_candle(bar: BarData, params: dict) -> bool:
    if not params.get("use_candle_filter", True):
        return True
    return bar.pat_shooting_star or bar.pat_bear_engulf or bar.pat_evening_star or bar.pat_bear_marubozu


def _trend_signal(bar: BarData, prev: BarData, params: dict, regime: str) -> Signal:
    rsi_min = float(params.get("trend_rsi_min", 40.0))
    rsi_max = float(params.get("trend_rsi_max", 55.0))
    pullback_pct = float(params.get("trend_pullback_pct", 0.004))
    stop_atr = float(params.get("trend_stop_atr_mult", 0.5))
    target_r = float(params.get("trend_target_r", 2.0))
    trail_atr = float(params.get("trend_trailing_atr_mult", 1.4))

    if any(np.isnan([bar.ema50, bar.ema200, bar.rsi, bar.atr])):
        return Signal(action="none")

    long_cond = (
        _direction_allowed(params, "long")
        and bar.ema50 > bar.ema200
        and bar.close > bar.ema200
        and bar.low <= bar.ema50 * (1.0 + pullback_pct)
        and bar.close > bar.ema50
        and rsi_min <= bar.rsi <= rsi_max
        and bar.close >= prev.close
        and _bullish_candle(bar, params)
    )
    if long_cond:
        swing = bar.swing_low if not np.isnan(bar.swing_low) else bar.low
        stop = min(swing, bar.low) - stop_atr * bar.atr
        risk = bar.close - stop
        return _build_entry(
            "long", "Trend Pullback", regime, bar,
            stop, bar.close + target_r * risk,
            bar.close + risk, bar.close + risk,
            trail_atr * bar.atr, "TREND_PULLBACK_LONG",
        )

    short_cond = (
        _direction_allowed(params, "short")
        and bar.ema50 < bar.ema200
        and bar.close < bar.ema200
        and bar.high >= bar.ema50 * (1.0 - pullback_pct)
        and bar.close < bar.ema50
        and (100.0 - rsi_max) <= bar.rsi <= (100.0 - rsi_min)
        and bar.close <= prev.close
        and _bearish_candle(bar, params)
    )
    if short_cond:
        swing = bar.swing_high if not np.isnan(bar.swing_high) else bar.high
        stop = max(swing, bar.high) + stop_atr * bar.atr
        risk = stop - bar.close
        return _build_entry(
            "short", "Trend Pullback", regime, bar,
            stop, bar.close - target_r * risk,
            bar.close - risk, bar.close - risk,
            trail_atr * bar.atr, "TREND_PULLBACK_SHORT",
        )

    return Signal(action="none")


def _range_signal(bar: BarData, params: dict, regime: str) -> Signal:
    stop_atr = float(params.get("range_stop_atr_mult", 1.2))
    target_mode = str(params.get("range_target", "middle")).lower()
    req_williams = bool(params.get("range_require_williams", True))

    if any(np.isnan([bar.bb_lower, bar.bb_upper, bar.bb_mid, bar.rsi, bar.atr])):
        return Signal(action="none")

    wr_long = float(params.get("range_williams_r_long", -80.0))
    wr_short = float(params.get("range_williams_r_short", -20.0))

    long_cond = (
        _direction_allowed(params, "long")
        and bar.low <= bar.bb_lower
        and bar.rsi < float(params.get("range_rsi_long", 30.0))
        and (not req_williams or (not np.isnan(bar.williams_r) and bar.williams_r < wr_long))
    )
    if long_cond:
        stop = bar.close - stop_atr * bar.atr
        risk = bar.close - stop
        target = bar.bb_upper if target_mode == "opposite" else bar.bb_mid
        if target <= bar.close:
            target = bar.close + risk
        return _build_entry(
            "long", "Range Reversion", regime, bar,
            stop, target, target, target,
            stop_atr * bar.atr, "RANGE_BB_RSI_LONG",
        )

    short_cond = (
        _direction_allowed(params, "short")
        and bar.high >= bar.bb_upper
        and bar.rsi > float(params.get("range_rsi_short", 70.0))
        and (not req_williams or (not np.isnan(bar.williams_r) and bar.williams_r > wr_short))
    )
    if short_cond:
        stop = bar.close + stop_atr * bar.atr
        risk = stop - bar.close
        target = bar.bb_lower if target_mode == "opposite" else bar.bb_mid
        if target >= bar.close:
            target = bar.close - risk
        return _build_entry(
            "short", "Range Reversion", regime, bar,
            stop, target, target, target,
            stop_atr * bar.atr, "RANGE_BB_RSI_SHORT",
        )

    return Signal(action="none")


def _breakout_signal(bar: BarData, prev: BarData, params: dict, regime: str) -> Signal:
    if any(np.isnan([prev.bb_width, bar.volume_ratio, bar.atr])):
        return Signal(action="none")

    stop_atr = float(params.get("breakout_stop_atr_mult", 1.0))
    target_r = float(params.get("breakout_target_r", 3.0))
    trigger_r = float(params.get("breakout_trail_trigger_r", 1.5))
    trail_atr = float(params.get("breakout_trailing_atr_mult", 1.6))

    if _direction_allowed(params, "long") and not np.isnan(bar.donchian_high_prev) and bar.close > bar.donchian_high_prev:
        # Elder Ray: bull power should be positive for confirmation
        if not np.isnan(bar.bull_power) and bar.bull_power < 0:
            return Signal(action="none")
        stop = bar.close - stop_atr * bar.atr
        risk = bar.close - stop
        return _build_entry(
            "long", "Donchian Breakout", regime, bar,
            stop, bar.close + target_r * risk,
            bar.close + trigger_r * risk, bar.close + trigger_r * risk,
            trail_atr * bar.atr, "DONCHIAN_BREAKOUT_LONG",
        )

    if _direction_allowed(params, "short") and not np.isnan(bar.donchian_low_prev) and bar.close < bar.donchian_low_prev:
        if not np.isnan(bar.bear_power) and bar.bear_power > 0:
            return Signal(action="none")
        stop = bar.close + stop_atr * bar.atr
        risk = stop - bar.close
        return _build_entry(
            "short", "Donchian Breakout", regime, bar,
            stop, bar.close - target_r * risk,
            bar.close - trigger_r * risk, bar.close - trigger_r * risk,
            trail_atr * bar.atr, "DONCHIAN_BREAKOUT_SHORT",
        )

    return Signal(action="none")


def _macd_signal(bar: BarData, params: dict, regime: str) -> Signal:
    if any(np.isnan([bar.macd, bar.macd_signal, bar.macd_hist, bar.macd_hist_prev, bar.atr])):
        return Signal(action="none")

    mode = str(params.get("macd_mode", "histogram")).lower()
    stop_atr = float(params.get("macd_stop_atr_mult", 1.2))
    target_r = float(params.get("macd_target_r", 2.5))
    trail_atr = float(params.get("macd_trailing_atr_mult", 1.5))
    req_obv = bool(params.get("macd_require_obv_confirm", True))

    if mode == "signal_cross":
        long_trigger = bar.macd > bar.macd_signal and bar.macd_hist_prev < 0
        short_trigger = bar.macd < bar.macd_signal and bar.macd_hist_prev > 0
    else:  # histogram
        long_trigger = bar.macd_hist > 0 and bar.macd_hist_prev <= 0
        short_trigger = bar.macd_hist < 0 and bar.macd_hist_prev >= 0

    if long_trigger and _direction_allowed(params, "long"):
        if req_obv and bar.obv_rising == 0:
            return Signal(action="none")
        stop = bar.close - stop_atr * bar.atr
        risk = bar.close - stop
        if risk <= 0:
            return Signal(action="none")
        return _build_entry(
            "long", "MACD Cross", regime, bar,
            stop, bar.close + target_r * risk,
            bar.close + risk, bar.close + risk,
            trail_atr * bar.atr, "MACD_HIST_LONG",
        )

    if short_trigger and _direction_allowed(params, "short"):
        if req_obv and bar.obv_rising == 1:
            return Signal(action="none")
        stop = bar.close + stop_atr * bar.atr
        risk = stop - bar.close
        if risk <= 0:
            return Signal(action="none")
        return _build_entry(
            "short", "MACD Cross", regime, bar,
            stop, bar.close - target_r * risk,
            bar.close - risk, bar.close - risk,
            trail_atr * bar.atr, "MACD_HIST_SHORT",
        )

    return Signal(action="none")


def generate_entry_signal(bar: BarData, prev_bar: BarData, params: dict, position) -> Signal:
    if position is not None:
        return Signal(action="none")
    if any(np.isnan([bar.close, bar.atr])):
        return Signal(action="none")

    regime = market_regime(bar, prev_bar, params)

    if regime == "MACD":
        return _macd_signal(bar, params, regime)
    if regime == "BREAKOUT":
        return _breakout_signal(bar, prev_bar, params, regime)
    if regime == "TREND":
        return _trend_signal(bar, prev_bar, params, regime)
    if regime == "RANGE":
        return _range_signal(bar, params, regime)
    return Signal(action="none", regime=regime)


def generate_exit_signal(bar: BarData, prev_bar: BarData, params: dict, position: PositionState) -> Signal:
    position.bars_in_position += 1
    max_bars = int(params.get("max_bars_in_trade", 0) or 0)

    if position.side == "long":
        position.best_price = max(
            position.best_price if not np.isnan(position.best_price) else position.entry_price,
            bar.high,
        )
        if not position.trailing_active and bar.high >= position.trailing_trigger_price:
            position.trailing_active = True
        if position.trailing_active and not np.isnan(position.trailing_distance):
            position.stop_price = max(position.stop_price, position.best_price - position.trailing_distance)
        if bar.low <= position.stop_price:
            return Signal(action="close", reason="STOP_OR_TRAIL", exit_price=position.stop_price)
        if bar.high >= position.target_price:
            return Signal(action="close", reason="TAKE_PROFIT", exit_price=position.target_price)
    else:
        position.best_price = min(
            position.best_price if not np.isnan(position.best_price) else position.entry_price,
            bar.low,
        )
        if not position.trailing_active and bar.low <= position.trailing_trigger_price:
            position.trailing_active = True
        if position.trailing_active and not np.isnan(position.trailing_distance):
            position.stop_price = min(position.stop_price, position.best_price + position.trailing_distance)
        if bar.high >= position.stop_price:
            return Signal(action="close", reason="STOP_OR_TRAIL", exit_price=position.stop_price)
        if bar.low <= position.target_price:
            return Signal(action="close", reason="TAKE_PROFIT", exit_price=position.target_price)

    if max_bars > 0 and position.bars_in_position >= max_bars:
        return Signal(action="close", reason="TIME_STOP", exit_price=bar.close)

    return Signal(action="none")


def apply_slippage(price: float, side: Side, action: str, slippage_bps: float = 0.0, spread_bps: float = 0.0) -> float:
    total_bps = slippage_bps + spread_bps / 2.0
    if total_bps == 0.0:
        return float(price)
    factor = total_bps / 10_000.0
    opening = action == "open"
    if (side == "long" and opening) or (side == "short" and not opening):
        return float(price) * (1.0 + factor)
    return float(price) * (1.0 - factor)


def compute_trade_close(entry_price: float, exit_price: float, side: Side, qty: float, fee_rate: float) -> dict:
    gross_pnl = (exit_price - entry_price) * qty if side == "long" else (entry_price - exit_price) * qty
    entry_notional = abs(entry_price * qty)
    exit_notional = abs(exit_price * qty)
    fee_usd = (entry_notional + exit_notional) * fee_rate
    return {
        "gross_pnl": gross_pnl,
        "fee_usd": fee_usd,
        "pnl": gross_pnl - fee_usd,
        "entry_notional": entry_notional,
        "exit_notional": exit_notional,
    }


def bar_from_row(row, params: dict) -> BarData:
    donchian_len = int(params.get("donchian_len", 24) or 24)
    return BarData(
        time=row["time"],
        open=_float(row.get("open")),
        high=_float(row.get("high")),
        low=_float(row.get("low")),
        close=_float(row.get("close")),
        volume=_float(row.get("volume")),
        ema50=_float(row.get("ema50")),
        ema200=_float(row.get("ema200")),
        ema50_flatness=_float(row.get("ema50_flatness")),
        adx=_float(row.get("adx", row.get("adx_14"))),
        rsi=_float(row.get("rsi", row.get("rsi_14"))),
        atr=_float(row.get("atr", row.get("atr_14"))),
        bb_mid=_float(row.get("bb_mid")),
        bb_upper=_float(row.get("bb_upper")),
        bb_lower=_float(row.get("bb_lower")),
        bb_width=_float(row.get("bb_width")),
        volume_sma20=_float(row.get("volume_sma20")),
        volume_ratio=_float(row.get("volume_ratio")),
        donchian_high_prev=_float(row.get(f"donchian_high_{donchian_len}_prev")),
        donchian_low_prev=_float(row.get(f"donchian_low_{donchian_len}_prev")),
        swing_low=_float(row.get("swing_low_5")),
        swing_high=_float(row.get("swing_high_5")),
        macd=_float(row.get("macd")),
        macd_signal=_float(row.get("macd_signal")),
        macd_hist=_float(row.get("macd_hist")),
        macd_hist_prev=_float(row.get("macd_hist_prev")),
        stoch_k=_float(row.get("stoch_k")),
        stoch_d=_float(row.get("stoch_d")),
        williams_r=_float(row.get("williams_r")),
        obv_rising=int(row.get("obv_rising", 0)),
        bull_power=_float(row.get("bull_power")),
        bear_power=_float(row.get("bear_power")),
        pat_doji=int(row.get("pat_doji", 0)),
        pat_hammer=int(row.get("pat_hammer", 0)),
        pat_shooting_star=int(row.get("pat_shooting_star", 0)),
        pat_bull_engulf=int(row.get("pat_bull_engulf", 0)),
        pat_bear_engulf=int(row.get("pat_bear_engulf", 0)),
        pat_morning_star=int(row.get("pat_morning_star", 0)),
        pat_evening_star=int(row.get("pat_evening_star", 0)),
        pat_bull_marubozu=int(row.get("pat_bull_marubozu", 0)),
        pat_bear_marubozu=int(row.get("pat_bear_marubozu", 0)),
    )

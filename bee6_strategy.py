"""
Backtest execution layer for bee6.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bee6_engine import (
    PositionState,
    Signal,
    apply_slippage,
    bar_from_row,
    compute_trade_close,
    generate_entry_signal,
    generate_exit_signal,
)
from bee6_params import DEFAULT_PARAMS, FEE_RATE


@dataclass
class TradeRecord:
    side: str
    regime: str
    strategy: str
    entry_time: object
    exit_time: object
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: float
    risk_usd: float
    r_multiple: float
    gross_pnl: float
    fee_usd: float
    slippage_usd: float
    pnl: float
    gross_ret: float
    net_ret: float
    reason: str
    capital_before: float
    capital_after: float
    position_notional: float
    qty: float
    entry_atr: float
    entry_rsi: float
    entry_adx: float
    entry_bb_width: float
    entry_volume_ratio: float
    entry_macd_hist: float
    entry_williams_r: float
    entry_stoch_k: float
    exit_atr: float
    exit_rsi: float
    exit_adx: float
    bars_in_trade: int


class Bee6Strategy:
    """Multi-indicator regime-switching strategy (DEX-ready, no live exchange)."""

    def __init__(self, params: dict | None = None, fee_rate: float = FEE_RATE):
        merged = dict(DEFAULT_PARAMS)
        if params:
            merged.update(params)
        self.params = merged
        self.fee_rate = float(self.params.get("fee_rate", fee_rate))
        self.slippage_bps = float(self.params.get("slippage_bps", 0.0))
        self.spread_bps = float(self.params.get("spread_bps", 0.0))
        self.position: PositionState | None = None
        self.strategy_safety: dict[str, dict] = {}
        self.daily_loss: dict[str, float] = {}
        self.weekly_loss: dict[str, float] = {}
        self.peak_capital = 0.0

    def _safety_for(self, strategy: str) -> dict:
        return self.strategy_safety.setdefault(
            strategy,
            {"losses": 0, "paused_until": None, "disabled": False},
        )

    def _strategy_blocked(self, strategy: str, now) -> bool:
        safety = self._safety_for(strategy)
        if safety["disabled"]:
            return True
        paused_until = safety.get("paused_until")
        if paused_until is not None and pd.Timestamp(now) < pd.Timestamp(paused_until):
            return True
        return False

    def _update_strategy_safety(self, strategy: str, exit_time, pnl: float) -> None:
        safety = self._safety_for(strategy)
        if pnl > 0:
            safety["losses"] = 0
            safety["paused_until"] = None
            return
        safety["losses"] += 1
        if safety["losses"] >= int(self.params.get("disable_after_losses", 6)):
            safety["disabled"] = True
            return
        if safety["losses"] >= int(self.params.get("pause_after_losses", 4)):
            safety["paused_until"] = pd.Timestamp(exit_time) + pd.Timedelta(
                hours=float(self.params.get("pause_hours", 24))
            )

    def _loss_key(self, ts, weekly: bool = False) -> str:
        stamp = pd.Timestamp(ts)
        if weekly:
            iso = stamp.isocalendar()
            return f"{iso.year}-{iso.week:02d}"
        return stamp.strftime("%Y-%m-%d")

    def _risk_gate_open(self, ts, capital: float) -> bool:
        if capital <= 0.0:
            return False
        dd_pct = (self.peak_capital - capital) / max(self.peak_capital, 1.0) * 100.0
        if dd_pct >= float(self.params.get("max_drawdown_pct", 15.0)):
            return False
        day_loss = self.daily_loss.get(self._loss_key(ts), 0.0)
        week_loss = self.weekly_loss.get(self._loss_key(ts, weekly=True), 0.0)
        if day_loss / capital * 100.0 >= float(self.params.get("max_daily_loss_pct", 3.0)):
            return False
        if week_loss / capital * 100.0 >= float(self.params.get("max_weekly_loss_pct", 6.0)):
            return False
        return True

    def _record_loss(self, ts, pnl: float) -> None:
        if pnl >= 0:
            return
        loss = abs(float(pnl))
        self.daily_loss[self._loss_key(ts)] = self.daily_loss.get(self._loss_key(ts), 0.0) + loss
        wk = self._loss_key(ts, weekly=True)
        self.weekly_loss[wk] = self.weekly_loss.get(wk, 0.0) + loss

    def _position_size(self, signal: Signal, entry_price: float, capital: float) -> tuple[float, float, float]:
        risk_pct = float(self.params.get("risk_pct", 0.0075))
        risk_usd = max(0.0, capital * risk_pct)
        risk_per_unit = abs(entry_price - signal.stop_price)
        if risk_per_unit <= 0.0 or risk_usd <= 0.0:
            return 0.0, 0.0, 0.0

        qty = risk_usd / risk_per_unit
        max_notional = capital * float(self.params.get("max_notional_pct", 1.0))
        notional = qty * entry_price
        if max_notional > 0.0 and notional > max_notional:
            qty = max_notional / entry_price
            notional = qty * entry_price
            risk_usd = risk_per_unit * qty
        return qty, risk_usd, risk_per_unit

    def _open_position(self, bar, sig: Signal, capital: float) -> None:
        side = "long" if sig.action == "open_long" else "short"
        entry_price = apply_slippage(bar.close, side, "open", self.slippage_bps, self.spread_bps)
        qty, risk_usd, _ = self._position_size(sig, entry_price, capital)
        if qty <= 0.0:
            return
        self.position = PositionState(
            side=side,
            strategy=sig.strategy,
            regime=sig.regime,
            entry_price=entry_price,
            entry_time=bar.time,
            qty=qty,
            capital_at_open=capital,
            stop_price=sig.stop_price,
            target_price=sig.target_price,
            tp1_price=sig.tp1_price,
            trailing_trigger_price=sig.trailing_trigger_price,
            trailing_distance=sig.trailing_distance,
            entry_atr=bar.atr,
            risk_usd=risk_usd,
            risk_per_unit=abs(entry_price - sig.stop_price),
            entry_meta=dict(sig.meta or {}),
            best_price=entry_price,
        )

    def _close_position(self, bar, sig: Signal, capital: float) -> tuple[TradeRecord, float]:
        pos = self.position
        raw_exit = sig.exit_price if sig.exit_price is not None else bar.close
        exit_price = apply_slippage(raw_exit, pos.side, "close", self.slippage_bps, self.spread_bps)
        result = compute_trade_close(pos.entry_price, exit_price, pos.side, pos.qty, self.fee_rate)
        pnl = result["pnl"]
        new_capital = capital + pnl
        r_multiple = pnl / pos.risk_usd if pos.risk_usd > 0.0 else np.nan
        slippage_usd = abs(exit_price - raw_exit) * pos.qty

        em = pos.entry_meta or {}
        rec = TradeRecord(
            side=pos.side,
            regime=pos.regime,
            strategy=pos.strategy,
            entry_time=pos.entry_time,
            exit_time=bar.time,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            stop_price=pos.stop_price,
            target_price=pos.target_price,
            risk_usd=pos.risk_usd,
            r_multiple=r_multiple,
            gross_pnl=result["gross_pnl"],
            fee_usd=result["fee_usd"],
            slippage_usd=slippage_usd,
            pnl=pnl,
            gross_ret=result["gross_pnl"] / pos.capital_at_open if pos.capital_at_open else 0.0,
            net_ret=pnl / pos.capital_at_open if pos.capital_at_open else 0.0,
            reason=sig.reason,
            capital_before=capital,
            capital_after=new_capital,
            position_notional=result["entry_notional"],
            qty=pos.qty,
            entry_atr=float(em.get("atr", np.nan)),
            entry_rsi=float(em.get("rsi", np.nan)),
            entry_adx=float(em.get("adx", np.nan)),
            entry_bb_width=float(em.get("bb_width", np.nan)),
            entry_volume_ratio=float(em.get("volume_ratio", np.nan)),
            entry_macd_hist=float(em.get("macd_hist", np.nan)),
            entry_williams_r=float(em.get("williams_r", np.nan)),
            entry_stoch_k=float(em.get("stoch_k", np.nan)),
            exit_atr=bar.atr,
            exit_rsi=bar.rsi,
            exit_adx=bar.adx,
            bars_in_trade=pos.bars_in_position,
        )
        self.position = None
        self._record_loss(bar.time, pnl)
        self._update_strategy_safety(pos.strategy, bar.time, pnl)
        return rec, new_capital

    def _mark_to_market(self, bar, capital: float) -> float:
        if self.position is None:
            return capital
        pos = self.position
        if pos.side == "long":
            return capital + (bar.close - pos.entry_price) * pos.qty
        return capital + (pos.entry_price - bar.close) * pos.qty

    def run(self, df: pd.DataFrame, initial_capital: float):
        capital = float(initial_capital)
        self.peak_capital = capital
        trades: list[TradeRecord] = []
        equity_curve: list[tuple] = []
        self.position = None
        self.strategy_safety = {}
        self.daily_loss = {}
        self.weekly_loss = {}

        if df is None or df.empty:
            return pd.DataFrame(), pd.DataFrame(columns=["time", "equity"]), capital

        equity_curve.append((df["time"].iloc[0], capital))

        for i in range(1, len(df)):
            bar = bar_from_row(df.iloc[i], self.params)
            prev = bar_from_row(df.iloc[i - 1], self.params)

            if np.isnan(bar.close) or np.isnan(prev.close):
                continue

            if self.position is not None:
                sig = generate_exit_signal(bar, prev, self.params, self.position)
                if sig.action == "close":
                    rec, capital = self._close_position(bar, sig, capital)
                    trades.append(rec)
                    self.peak_capital = max(self.peak_capital, capital)

            if self.position is None and self._risk_gate_open(bar.time, capital):
                sig = generate_entry_signal(bar, prev, self.params, None)
                if sig.action in ("open_long", "open_short") and not self._strategy_blocked(sig.strategy, bar.time):
                    sig.meta["raw_entry"] = bar.close
                    self._open_position(bar, sig, capital)

            mtm = self._mark_to_market(bar, capital)
            self.peak_capital = max(self.peak_capital, mtm)
            equity_curve.append((bar.time, mtm))

        if self.position is not None:
            last_bar = bar_from_row(df.iloc[-1], self.params)
            force_sig = Signal(action="close", reason="FORCE_EXIT_END", exit_price=last_bar.close)
            rec, capital = self._close_position(last_bar, force_sig, capital)
            trades.append(rec)
            equity_curve.append((last_bar.time, capital))

        cols = list(TradeRecord.__dataclass_fields__.keys())
        trades_df = pd.DataFrame([{c: getattr(t, c) for c in cols} for t in trades], columns=cols)
        equity_df = pd.DataFrame(equity_curve, columns=["time", "equity"])
        return trades_df, equity_df, capital

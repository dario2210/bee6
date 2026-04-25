"""
Public Binance OHLCV downloader for bee6.
No API key required for candle data.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import requests

BINANCE_SPOT_URL = "https://api.binance.com/api/v3/klines"
BINANCE_FUTURES_URL = "https://fapi.binance.com/fapi/v1/klines"
BINANCE_LIMIT = 1000
REQUEST_DELAY = 0.25

TF_MINUTES: dict[str, float] = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "6h": 360,
    "8h": 480,
    "12h": 720,
    "1d": 1440,
    "3d": 4320,
    "1w": 10080,
}


def get_bars_per_day(interval: str) -> float:
    minutes = TF_MINUTES.get(interval)
    if minutes is None:
        raise ValueError(f"Unknown interval '{interval}'. Available: {list(TF_MINUTES)}")
    return 1440.0 / minutes


def bars_for_days(days: int, interval: str) -> int:
    return int(round(days * get_bars_per_day(interval)))


def interval_to_ms(interval: str) -> int:
    return int(TF_MINUTES[interval] * 60 * 1000)


def fetch_klines_binance(
    symbol: str,
    interval: str,
    start_time_ms: int | None = None,
    end_time_ms: int | None = None,
    market: str = "spot",
    verbose: bool = True,
) -> pd.DataFrame:
    base_url = BINANCE_FUTURES_URL if market == "futures" else BINANCE_SPOT_URL
    if interval not in TF_MINUTES:
        raise ValueError(f"Unknown interval '{interval}'. Available: {list(TF_MINUTES)}")

    all_rows: list = []
    current_start = int(start_time_ms) if start_time_ms is not None else None
    bar_ms = interval_to_ms(interval)

    if verbose:
        start_str = (
            pd.Timestamp(current_start, unit="ms", tz="UTC").strftime("%Y-%m-%d")
            if current_start
            else "start"
        )
        end_str = (
            pd.Timestamp(end_time_ms, unit="ms", tz="UTC").strftime("%Y-%m-%d")
            if end_time_ms
            else "now"
        )
        print(f"[Binance] {symbol} {interval} | {start_str} -> {end_str} | {market}")

    while True:
        params = {"symbol": symbol, "interval": interval, "limit": BINANCE_LIMIT}
        if current_start is not None:
            params["startTime"] = current_start
        if end_time_ms is not None:
            params["endTime"] = end_time_ms

        try:
            resp = requests.get(base_url, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"[Binance] HTTP error: {exc!r}") from exc

        if not data:
            break

        all_rows.extend(data)
        last_open_time_ms = int(data[-1][0])
        if len(data) < BINANCE_LIMIT:
            break
        if end_time_ms is not None and last_open_time_ms >= end_time_ms:
            break

        current_start = last_open_time_ms + bar_ms
        time.sleep(REQUEST_DELAY)

    if not all_rows:
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(all_rows, columns=cols)
    df["open_time"] = df["open_time"].astype("int64")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    return df[["open_time", "open", "high", "low", "close", "volume"]]


def update_csv_cache(
    csv_path: str,
    symbol: str,
    interval: str,
    start_date: str,
    market: str = "spot",
    verbose: bool = True,
) -> pd.DataFrame:
    if os.path.exists(csv_path):
        if verbose:
            print(f"[Cache] Loading {csv_path}")
        df_old = pd.read_csv(csv_path)
        if "open_time" not in df_old.columns:
            if "time" in df_old.columns:
                df_old = df_old.rename(columns={"time": "open_time"})
            else:
                raise ValueError("CSV must have open_time or time column.")
        if not np.issubdtype(df_old["open_time"].dtype, np.number):
            df_old["open_time"] = (
                pd.to_datetime(df_old["open_time"], utc=True).astype("int64") // 10**6
            ).astype("int64")

        last_ms = int(df_old["open_time"].iloc[-1])
        next_start_ms = last_ms + interval_to_ms(interval)
        now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        if next_start_ms >= now_ms:
            if verbose:
                print(f"[Cache] Up to date ({len(df_old)} candles)")
            return df_old[["open_time", "open", "high", "low", "close", "volume"]].copy()

        df_new = fetch_klines_binance(
            symbol=symbol, interval=interval,
            start_time_ms=next_start_ms, market=market, verbose=verbose,
        )
        keep = ["open_time", "open", "high", "low", "close", "volume"]
        df_all = pd.concat([df_old[keep], df_new[keep]], ignore_index=True)
        df_all = df_all.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    else:
        start_ms = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
        df_all = fetch_klines_binance(
            symbol=symbol, interval=interval,
            start_time_ms=start_ms, market=market, verbose=verbose,
        )
        if df_all.empty:
            raise RuntimeError(f"No candles downloaded for {symbol} {interval}.")

    df_all.to_csv(csv_path, index=False)
    if verbose:
        print(f"[Cache] Saved {csv_path} ({len(df_all)} candles)")
    return df_all


def wfo_bars(interval: str, opt_days: int, live_days: int) -> tuple[int, int]:
    return bars_for_days(opt_days, interval), bars_for_days(live_days, interval)


def list_symbols(market: str = "spot", quote: str = "USDT") -> list[str]:
    """Return all trading symbols ending with quote currency."""
    url = (
        "https://fapi.binance.com/fapi/v1/exchangeInfo"
        if market == "futures"
        else "https://api.binance.com/api/v3/exchangeInfo"
    )
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        info = resp.json()
        return [
            s["symbol"] for s in info.get("symbols", [])
            if s["symbol"].endswith(quote) and s.get("status") == "TRADING"
        ]
    except requests.RequestException as exc:
        raise RuntimeError(f"[Binance] Cannot fetch symbols: {exc!r}") from exc

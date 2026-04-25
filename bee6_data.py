"""
OHLCV loading and indicator computation for bee6.

Indicators beyond bee5:
  - MACD (histogram + signal line)
  - Stochastic RSI (%K, %D)
  - Williams %R
  - OBV + OBV EMA
  - Elder Ray (bull/bear power)
  - Candlestick pattern flags
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bee6_params import (
    ADX_LEN, ATR_LEN, BB_LEN, BB_STD,
    DONCHIAN_LEN_GRID, EMA_FAST_LEN, EMA_SLOW_LEN,
    MACD_FAST, MACD_SIGNAL, MACD_SLOW,
    OBV_EMA_LEN, RSI_LEN, STOCH_RSI_D, STOCH_RSI_K,
    STOCH_RSI_LEN, VOLUME_SMA_LEN, WILLIAMS_R_LEN,
)


# ---------------------------------------------------------------------------
# Base indicator helpers
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, length: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").ewm(span=int(length), adjust=False).mean()


def _sma(series: pd.Series, length: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").rolling(int(length)).mean()


def _atr(df: pd.DataFrame, length: int = ATR_LEN) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / max(int(length), 1), adjust=False).mean()


def _rsi(close: pd.Series, length: int = RSI_LEN) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / max(int(length), 1), adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / max(int(length), 1), adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _adx(df: pd.DataFrame, length: int = ADX_LEN) -> tuple[pd.Series, pd.Series, pd.Series]:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr = _atr(df, length).replace(0.0, np.nan)
    alpha = 1.0 / max(int(length), 1)
    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean() / atr
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean() / atr
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)) * 100.0
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    return adx, plus_di, minus_di


# ---------------------------------------------------------------------------
# New indicators
# ---------------------------------------------------------------------------

def _macd(close: pd.Series, fast: int = MACD_FAST, slow: int = MACD_SLOW, signal: int = MACD_SIGNAL):
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _stoch_rsi(close: pd.Series, rsi_len: int = STOCH_RSI_LEN, k: int = STOCH_RSI_K, d: int = STOCH_RSI_D):
    rsi = _rsi(close, rsi_len)
    rsi_min = rsi.rolling(rsi_len).min()
    rsi_max = rsi.rolling(rsi_len).max()
    stoch = (rsi - rsi_min) / (rsi_max - rsi_min).replace(0.0, np.nan) * 100.0
    k_line = stoch.rolling(k).mean()
    d_line = k_line.rolling(d).mean()
    return k_line, d_line


def _williams_r(df: pd.DataFrame, length: int = WILLIAMS_R_LEN) -> pd.Series:
    highest_high = df["high"].rolling(length).max()
    lowest_low = df["low"].rolling(length).min()
    return (highest_high - df["close"]) / (highest_high - lowest_low).replace(0.0, np.nan) * -100.0


def _obv(df: pd.DataFrame, ema_len: int = OBV_EMA_LEN) -> tuple[pd.Series, pd.Series]:
    direction = np.sign(df["close"].diff()).fillna(0.0)
    obv = (direction * df["volume"]).cumsum()
    obv_ema = _ema(obv, ema_len)
    return obv, obv_ema


def _elder_ray(df: pd.DataFrame, ema_len: int = 13) -> tuple[pd.Series, pd.Series]:
    """Bull power = High - EMA; Bear power = Low - EMA"""
    ema = _ema(df["close"], ema_len)
    bull_power = df["high"] - ema
    bear_power = df["low"] - ema
    return bull_power, bear_power


# ---------------------------------------------------------------------------
# Candlestick patterns (single-bar and two-bar)
# ---------------------------------------------------------------------------

def _body(df: pd.DataFrame) -> pd.Series:
    return (df["close"] - df["open"]).abs()


def _upper_wick(df: pd.DataFrame) -> pd.Series:
    return df["high"] - df[["open", "close"]].max(axis=1)


def _lower_wick(df: pd.DataFrame) -> pd.Series:
    return df[["open", "close"]].min(axis=1) - df["low"]


def _candle_range(df: pd.DataFrame) -> pd.Series:
    return (df["high"] - df["low"]).replace(0.0, np.nan)


def add_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    body = _body(df)
    upper = _upper_wick(df)
    lower = _lower_wick(df)
    rng = _candle_range(df)
    body_ratio = body / rng

    # Doji: tiny body relative to range
    df["pat_doji"] = (body_ratio < 0.1).astype(int)

    # Hammer: small body, long lower wick, small upper wick (bullish reversal)
    df["pat_hammer"] = (
        (lower > 2.0 * body)
        & (upper < 0.3 * body.where(body > 0, rng * 0.1))
        & (body_ratio < 0.4)
    ).astype(int)

    # Shooting star: small body, long upper wick (bearish reversal)
    df["pat_shooting_star"] = (
        (upper > 2.0 * body)
        & (lower < 0.3 * body.where(body > 0, rng * 0.1))
        & (body_ratio < 0.4)
    ).astype(int)

    # Bullish engulfing
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    df["pat_bull_engulf"] = (
        (prev_close < prev_open)           # prev bearish
        & (df["close"] > df["open"])       # current bullish
        & (df["open"] < prev_close)        # open below prev close
        & (df["close"] > prev_open)        # close above prev open
    ).astype(int)

    # Bearish engulfing
    df["pat_bear_engulf"] = (
        (prev_close > prev_open)           # prev bullish
        & (df["close"] < df["open"])       # current bearish
        & (df["open"] > prev_close)        # open above prev close
        & (df["close"] < prev_open)        # close below prev open
    ).astype(int)

    # Morning star (3-bar): bearish + doji/small + bullish
    p2_doji = (body.shift(1) < rng.shift(1) * 0.3)
    p3_bull = df["close"] > df["open"]
    p1_bear = df["close"].shift(2) < df["open"].shift(2)
    df["pat_morning_star"] = (p1_bear & p2_doji & p3_bull).astype(int)

    # Evening star (3-bar): bullish + doji/small + bearish
    p1_bull = df["close"].shift(2) > df["open"].shift(2)
    p3_bear = df["close"] < df["open"]
    df["pat_evening_star"] = (p1_bull & p2_doji & p3_bear).astype(int)

    # Marubozu bullish: almost no wicks, bullish
    df["pat_bull_marubozu"] = (
        (body_ratio > 0.9) & (df["close"] > df["open"])
    ).astype(int)

    # Marubozu bearish
    df["pat_bear_marubozu"] = (
        (body_ratio > 0.9) & (df["close"] < df["open"])
    ).astype(int)

    return df


# ---------------------------------------------------------------------------
# Main indicator preparation
# ---------------------------------------------------------------------------

def prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Trend
    df["ema50"] = _ema(df["close"], EMA_FAST_LEN)
    df["ema200"] = _ema(df["close"], EMA_SLOW_LEN)
    df["ema50_slope_24"] = (df["ema50"] - df["ema50"].shift(24)) / df["close"].replace(0.0, np.nan)
    df["ema50_flatness"] = df["ema50_slope_24"].abs()

    # Volatility
    df["atr_14"] = _atr(df, ATR_LEN)
    df["atr"] = df["atr_14"]

    # Momentum
    df["rsi_14"] = _rsi(df["close"], RSI_LEN)
    df["rsi"] = df["rsi_14"]

    # MACD
    df["macd"], df["macd_signal"], df["macd_hist"] = _macd(df["close"])
    df["macd_hist_prev"] = df["macd_hist"].shift(1)

    # Stochastic RSI
    df["stoch_k"], df["stoch_d"] = _stoch_rsi(df["close"])

    # Williams %R
    df["williams_r"] = _williams_r(df)

    # OBV
    df["obv"], df["obv_ema"] = _obv(df)
    df["obv_rising"] = (df["obv"] > df["obv_ema"]).astype(int)

    # Elder Ray
    df["bull_power"], df["bear_power"] = _elder_ray(df)

    # ADX
    adx, plus_di, minus_di = _adx(df, ADX_LEN)
    df["adx_14"] = adx
    df["adx"] = adx
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di

    # Bollinger Bands
    df["bb_mid"] = df["close"].rolling(BB_LEN).mean()
    bb_std = df["close"].rolling(BB_LEN).std(ddof=0)
    df["bb_upper"] = df["bb_mid"] + BB_STD * bb_std
    df["bb_lower"] = df["bb_mid"] - BB_STD * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"].replace(0.0, np.nan)
    df["bb_width_rank_120"] = df["bb_width"].rolling(120).rank(pct=True)

    # Volume
    df["volume_sma20"] = df["volume"].rolling(VOLUME_SMA_LEN).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma20"].replace(0.0, np.nan)

    # Donchian channels
    donchian_lens = sorted(set([24, 48] + list(DONCHIAN_LEN_GRID)))
    for length in donchian_lens:
        high_col = f"donchian_high_{int(length)}"
        low_col = f"donchian_low_{int(length)}"
        df[high_col] = df["high"].rolling(int(length)).max()
        df[low_col] = df["low"].rolling(int(length)).min()
        df[f"{high_col}_prev"] = df[high_col].shift(1)
        df[f"{low_col}_prev"] = df[low_col].shift(1)

    # Swings
    df["swing_low_5"] = df["low"].rolling(5).min().shift(1)
    df["swing_high_5"] = df["high"].rolling(5).max().shift(1)
    df["swing_low_10"] = df["low"].rolling(10).min().shift(1)
    df["swing_high_10"] = df["high"].rolling(10).max().shift(1)

    # Candlestick patterns
    df = add_candle_patterns(df)

    return df


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_klines(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.rename(columns={c: c.lower() for c in df.columns}, inplace=True)

    if "open_time" in df.columns:
        col = df["open_time"]
        is_numeric = np.issubdtype(col.dtype, np.number)
        if is_numeric:
            unit = "ms" if col.max() > 1e12 else "s"
            df["time"] = pd.to_datetime(col, unit=unit, utc=True)
        else:
            df["time"] = pd.to_datetime(col, utc=True, errors="coerce")
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    else:
        raise ValueError("Missing open_time / time column in CSV.")

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise ValueError(f"Missing '{col}' column in CSV.")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["time", "open", "high", "low", "close", "volume"])
    return df.sort_values("time").reset_index(drop=True)


def format_ts(ts) -> str:
    if pd.isna(ts):
        return "NaT"
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%d %H:%M")

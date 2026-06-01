"""
indicators.py
-------------
Core technical indicator computations.
All functions take a pandas DataFrame with columns: O, H, L, C
Returns a new column (Series) unless stated otherwise.
"""

import pandas as pd
import numpy as np


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average. adjust=False = standard Wilder-style."""
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(period).mean()


def bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0):
    """
    Bollinger Bands (20, 2).
    Returns: bb_mid, bb_upper, bb_lower, bb_pct_b, bb_width
    """
    mid   = sma(df["C"], period)
    std   = df["C"].rolling(period).std(ddof=0)
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    pct_b = (df["C"] - lower) / (upper - lower + 1e-9)
    width = (upper - lower) / (mid + 1e-9)
    return mid, upper, lower, pct_b, width


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range (exponential).
    TR = max(H-L, |H-C_prev|, |L-C_prev|)
    """
    prev_close = df["C"].shift(1)
    tr = pd.concat([
        df["H"] - df["L"],
        (df["H"] - prev_close).abs(),
        (df["L"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI — Wilder smoothing via EMA span=period.
    Range: 0–100. <40 oversold, >70 overbought.
    """
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def stochastic(df: pd.DataFrame, period: int = 14):
    """
    Stochastic %K and %D.
    %K = 100 * (C - lowest_low) / (highest_high - lowest_low)
    %D = SMA(%K, 3)
    """
    low_min  = df["L"].rolling(period).min()
    high_max = df["H"].rolling(period).max()
    k = 100 * (df["C"] - low_min) / (high_max - low_min + 1e-9)
    d = k.rolling(3).mean()
    return k, d


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Williams %R.
    Range: -100 to 0. Below -80 = oversold.
    """
    high_max = df["H"].rolling(period).max()
    low_min  = df["L"].rolling(period).min()
    return -100 * (high_max - df["C"]) / (high_max - low_min + 1e-9)


def adx(df: pd.DataFrame, period: int = 14):
    """
    ADX, +DI, -DI.
    ADX > 25 = trending. ADX < 20 = choppy/ranging.
    """
    up   = df["H"].diff()
    down = -df["L"].diff()

    plus_dm  = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)

    atr14    = atr(df, period)
    plus_di  = plus_dm.ewm(span=period, adjust=False).mean()  / (atr14 + 1e-9) * 100
    minus_di = minus_dm.ewm(span=period, adjust=False).mean() / (atr14 + 1e-9) * 100

    dx       = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9) * 100
    adx_val  = dx.ewm(span=period, adjust=False).mean()
    return adx_val, plus_di, minus_di


def resample_ohlc(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Resample 1-min OHLC to a higher timeframe.
    df must have a 'datetime' index or column.
    freq examples: '5min', '15min', '60min'
    """
    return (
        df.set_index("datetime")[["O", "H", "L", "C"]]
        .resample(freq, label="left", closed="left")
        .agg({"O": "first", "H": "max", "L": "min", "C": "last"})
        .dropna()
        .reset_index()
    )

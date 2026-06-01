"""
pipeline.py
-----------
Feature engineering pipeline for the NIFTY BB+EMA9 bounce strategy.

Main entry point:
    records = process_day("path/to/nifty_spot09_01_2020.csv")
    df      = pd.DataFrame(records)

Each record = one signal with all 117 features + label.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from src.indicators import (
    ema, sma, bollinger_bands, atr, rsi,
    stochastic, williams_r, adx, resample_ohlc
)


# ── SIGNAL ARMING THRESHOLD ───────────────────────────────────────────────────
BB_ARM_BUFFER = 0.5   # price must touch within 0.5 pts of lower band to arm


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all 1-min indicators to a raw OHLC dataframe."""
    df = df.copy().reset_index(drop=True)
    n  = len(df)

    # EMAs
    df["ema9"]  = ema(df["C"], min(9,  n))
    df["ema21"] = ema(df["C"], min(21, n))
    df["ema50"] = ema(df["C"], min(50, n))

    # Bollinger Bands
    df["bb_mid"], df["bb_upper"], df["bb_lower"], df["bb_pct_b"], df["bb_width"] = \
        bollinger_bands(df, period=min(20, n))

    # ATR
    df["atr14"] = atr(df, period=min(14, n))

    # Lower band reach (how deep the low went below BB lower)
    df["bb_lbr"] = (df["L"] - df["bb_lower"]) / (df["atr14"] + 1e-9)

    # Oscillators
    df["rsi14"]   = rsi(df["C"], period=min(14, n))
    df["stoch_k"], df["stoch_d"] = stochastic(df, period=min(14, n))
    df["will_r"]  = williams_r(df, period=min(14, n))
    df["adx"], df["plus_di"], df["minus_di"] = adx(df, period=min(14, n))

    # EMA distances and slope
    df["ema9_vs_ema21"]  = df["ema9"]  - df["ema21"]
    df["ema21_vs_ema50"] = df["ema21"] - df["ema50"]
    df["close_vs_ema9"]  = df["C"] - df["ema9"]
    df["close_vs_ema21"] = df["C"] - df["ema21"]
    df["ema9_slope"]     = df["ema9"].diff(min(3, n - 1))

    # Candle structure
    candle_range        = df["H"] - df["L"]
    df["candle_range"]  = candle_range
    df["candle_body"]   = (df["C"] - df["O"]).abs()
    df["body_ratio"]    = df["candle_body"]  / (candle_range + 1e-9)
    df["lwr_ratio"]     = (df[["O","C"]].min(axis=1) - df["L"]) / (candle_range + 1e-9)
    df["is_bull"]       = (df["C"] >= df["O"]).astype(int)
    df["range_vs_atr"]  = candle_range / (df["atr14"] + 1e-9)
    df["close_pos"]     = (df["C"] - df["L"]) / (candle_range + 1e-9)

    # Momentum / returns
    df["ret1"]  = df["C"].pct_change(1) * 100
    df["ret3"]  = df["C"].pct_change(3) * 100
    df["ret5"]  = df["C"].pct_change(5) * 100
    df["mom10"] = df["C"] - df["C"].shift(10)
    df["vol5"]  = df["ret1"].rolling(5).std()
    df["effort"]= candle_range / (candle_range.rolling(10).mean() + 1e-9)

    # Higher high / higher low structure
    df["hh"]    = (df["H"] > df["H"].shift(3)).astype(int)
    df["hl"]    = (df["L"] > df["L"].shift(3)).astype(int)
    df["hh_hl"] = ((df["hh"] == 1) & (df["hl"] == 1)).astype(int)

    # ATR expansion flag
    df["atr_exp"] = (df["atr14"] > df["atr14"].shift(5)).astype(int)

    # Distance from recent highs/lows (ATR-normalised)
    df["d5l"]  = (df["C"] - df["L"].rolling(5).min())  / (df["atr14"] + 1e-9)
    df["d5h"]  = (df["C"] - df["H"].rolling(5).max())  / (df["atr14"] + 1e-9)
    df["d20l"] = (df["C"] - df["L"].rolling(20).min()) / (df["atr14"] + 1e-9)
    df["d20h"] = (df["C"] - df["H"].rolling(20).max()) / (df["atr14"] + 1e-9)

    return df


def _get_tf_features(tf_df: pd.DataFrame, sig_dt, entry: float,
                     atr1m: float, tgt3: float, prefix: str):
    """
    Compute higher-timeframe features for one signal.
    Uses only bars completed before the signal time.
    Returns: (feature_dict, signal_active_flag) or (None, None) on failure.
    """
    hist = tf_df[tf_df["datetime"] < sig_dt].copy()
    if len(hist) < 2:
        return None, None

    hist = _add_indicators(hist)
    i    = len(hist) - 1

    # Check if BB+EMA9 signal also fired on this TF
    armed   = False
    for k in range(max(0, i - 4), i + 1):
        bl = hist.loc[k, "bb_lower"]
        if pd.notna(bl) and hist.loc[k, "L"] <= bl + BB_ARM_BUFFER:
            armed = True
            break
    e9      = hist.loc[i, "ema9"]
    trigger = bool(pd.notna(e9) and hist.loc[i, "C"] > e9)
    sig_active = int(armed and trigger)

    def g(col, default=0):
        v = hist.loc[i, col]
        return round(float(v), 4) if pd.notna(v) else default

    ph = g("H", entry)   # prior bar high
    pl = g("L", entry)   # prior bar low

    feats = {
        f"{prefix}_ema9_vs_ema21":        g("ema9_vs_ema21"),
        f"{prefix}_ema21_vs_ema50":       g("ema21_vs_ema50"),
        f"{prefix}_close_vs_ema9":        g("close_vs_ema9"),
        f"{prefix}_ema9_slope":           g("ema9_slope"),
        f"{prefix}_bb_pct_b":             g("bb_pct_b", 0.5),
        f"{prefix}_bb_width":             g("bb_width"),
        f"{prefix}_bb_lower_breach_atr":  g("bb_lbr"),
        f"{prefix}_rsi14":                g("rsi14", 50),
        f"{prefix}_stoch_k":              g("stoch_k", 50),
        f"{prefix}_will_r":               g("will_r", -50),
        f"{prefix}_is_bullish":           int(hist.loc[i, "is_bull"]),
        f"{prefix}_body_ratio":           g("body_ratio"),
        f"{prefix}_lower_wick_ratio":     g("lwr_ratio"),
        f"{prefix}_range_vs_atr":         g("range_vs_atr", 1),
        f"{prefix}_close_pos":            g("close_pos"),
        f"{prefix}_signal_active":        sig_active,
        f"{prefix}_dist_to_prev_high_atr":round((ph - entry) / (atr1m + 1e-9), 4),
        f"{prefix}_dist_to_prev_low_atr": round((entry - pl) / (atr1m + 1e-9), 4),
        f"{prefix}_room_to_tgt3":         int(tgt3 < ph),
    }
    return feats, sig_active


def _analyse_post_signal(df: pd.DataFrame, sig_idx: int, sl: float,
                         entry: float, atr14: float, n: int, lookahead: int = 10):
    """
    Classify post-entry price action into one of 6 patterns.
    ALL features here are look-ahead — for exit management only, not live entry.
    """
    broke = False
    consec = 0
    first_ema9_bar = None
    recovered = False
    n_small = 0
    max_dd = 0.0
    breakout_bar = 0
    breakout_str = 0.0
    highs = []

    for k in range(1, lookahead + 1):
        j = sig_idx + k
        if j >= n:
            break
        ch = df.loc[j, "H"]
        cl = df.loc[j, "L"]
        cc = df.loc[j, "C"]
        co = df.loc[j, "O"]
        cr = ch - cl
        highs.append(ch)

        if cl <= sl:
            broke = True
            max_dd = max(max_dd, sl - cl)

        ema9_j = df.loc[j, "ema9"]
        if pd.notna(ema9_j) and cc > ema9_j:
            if first_ema9_bar is None:
                first_ema9_bar = k
            if not broke and k == consec + 1:
                consec = k
            if broke:
                recovered = True

        if cr < atr14 * 0.6:
            n_small += 1

        if k >= 2 and cr > atr14 * 1.2 and cc > co and breakout_bar == 0:
            breakout_bar = k
            breakout_str = cr / (atr14 + 1e-9)

    ttc = first_ema9_bar if first_ema9_bar else lookahead + 1

    if not broke and first_ema9_bar == 1:
        pattern = "A_clean"
    elif not broke and first_ema9_bar == 2:
        pattern = "A_slight_delay"
    elif broke and recovered:
        pattern = "B_wick_recover"
    elif not broke and n_small >= 2:
        pattern = "C_coil"
    elif broke and not recovered:
        pattern = "D_sl_hit"
    else:
        pattern = "E_unclear"

    mean_rng5 = np.mean([df.loc[sig_idx+k,"H"] - df.loc[sig_idx+k,"L"]
                         for k in range(1, min(6, n - sig_idx))]) if n > sig_idx + 1 else atr14
    max_h5    = max(highs[:5]) if len(highs) >= 5 else (max(highs) if highs else entry)

    return {
        "post_pattern":      pattern,
        "post_A_clean":      int(pattern == "A_clean"),
        "post_A_delay":      int(pattern == "A_slight_delay"),
        "post_B_wick":       int(pattern == "B_wick_recover"),
        "post_C_coil":       int(pattern == "C_coil"),
        "bars_to_ema9":      ttc,
        "consec_above_ema9": consec,
        "broke_sl":          int(broke),
        "broke_sl_recovered":int(recovered),
        "max_dd_atr":        round(max_dd / (atr14 + 1e-9), 4),
        "n_small_candles":   n_small,
        "range_compress":    round(mean_rng5 / (atr14 + 1e-9), 4),
        "breakout_bar":      breakout_bar,
        "breakout_str":      round(breakout_str, 4),
        "pts5bars_atr":      round((max_h5 - entry) / (atr14 + 1e-9), 4),
    }


def process_day(filepath: str) -> list:
    """
    Full feature engineering pipeline for one day of 1-min NIFTY data.

    Parameters
    ----------
    filepath : str
        Path to CSV with columns: date, time, symbol, open, high, low, close

    Returns
    -------
    list of dicts — one dict per signal, containing all 117 features + label.
    Pass to pd.DataFrame() to get a scored dataset.
    """
    df = pd.read_csv(filepath)
    df["datetime"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str)
    )
    df = df.sort_values("datetime").reset_index(drop=True)
    df = df.rename(columns={"open":"O","high":"H","low":"L","close":"C"})
    n  = len(df)
    if n < 50:
        return []

    # Resample to higher timeframes
    df5  = resample_ohlc(df, "5min")
    df15 = resample_ohlc(df, "15min")
    df60 = resample_ohlc(df, "60min")

    # Add 1-min indicators
    df = _add_indicators(df)

    # Time features
    df["hour"]   = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute
    df["mins_open"]  = (df["hour"] - 9) * 60 + df["minute"] - 15
    df["mins_close"] = (15 * 60 + 29) - (df["hour"] * 60 + df["minute"])

    def _session(r):
        t = r["hour"] * 60 + r["minute"]
        if t < 9 * 60 + 30:  return 0
        if t < 11 * 60:       return 1
        if t < 13 * 60:       return 2
        if t < 14 * 60 + 30:  return 3
        return 4
    df["session"] = df.apply(_session, axis=1)

    # Regime features (day-level, computed once)
    day_open  = df.loc[0, "O"]
    o30       = df[df["mins_open"] <= 30]
    o30_high  = o30["H"].max()
    o30_low   = o30["L"].min()
    o30_close = o30.iloc[-1]["C"] if len(o30) > 0 else day_open
    o30_dir   = int(o30_close > day_open)

    # Signal detection loop
    records = []
    armed   = False

    for i in range(10, n):
        t_mins = df.loc[i, "hour"] * 60 + df.loc[i, "minute"]

        # Only trade 9:30 AM – 2:30 PM
        if t_mins < 9 * 60 + 30 or t_mins > 14 * 60 + 30:
            armed = False
            continue

        # Arm: price touched BB lower band
        bb_lower_i = df.loc[i, "bb_lower"]
        if pd.notna(bb_lower_i) and df.loc[i, "L"] <= bb_lower_i + BB_ARM_BUFFER:
            armed = True

        # Trigger: close above EMA9 while armed
        ema9_i = df.loc[i, "ema9"]
        if not (armed and pd.notna(ema9_i) and df.loc[i, "C"] > ema9_i):
            continue

        armed  = False
        entry  = df.loc[i, "C"]
        sl     = df.loc[i, "L"]
        risk   = entry - sl
        if risk <= 0:
            continue

        tgt3   = entry + risk * 3
        atr1m  = df.loc[i, "atr14"]
        sig_dt = df.loc[i, "datetime"]

        # Label: did 1:3 target hit before SL?
        label = 0
        for j in range(i + 1, n):
            if df.loc[j, "L"] <= sl:
                break
            if df.loc[j, "H"] >= tgt3:
                label = 1
                break

        # Higher TF features
        f5,  s5  = _get_tf_features(df5,  sig_dt, entry, atr1m, tgt3, "5m")
        f15, s15 = _get_tf_features(df15, sig_dt, entry, atr1m, tgt3, "15m")
        f60, s60 = _get_tf_features(df60, sig_dt, entry, atr1m, tgt3, "1h")
        if any(x is None for x in [f5, f15, f60]):
            continue

        # Confluence scores
        tf_conf  = s5 + s15 + s60
        tr_align = (int(f5["5m_ema9_vs_ema21"]   > 0) +
                    int(f15["15m_ema9_vs_ema21"]  > 0) +
                    int(f60["1h_ema9_vs_ema21"]   > 0))

        # Post-signal features (look-ahead)
        post = _analyse_post_signal(df, i, sl, entry, atr1m, n)

        def g1(col):
            v = df.loc[i, col]
            return round(float(v), 4) if pd.notna(v) else 0

        record = {
            # Metadata (not model features)
            "date":      df.loc[i, "date"],
            "time":      df.loc[i, "time"],
            "entry":     round(entry, 2),
            "sl":        round(sl, 2),
            "tgt_1_3":   round(tgt3, 2),
            "risk_pts":  round(risk, 2),

            # 1-min BB
            "1m_bb_pct_b":  g1("bb_pct_b"),
            "1m_bb_width":  g1("bb_width"),
            "1m_bb_lbr":    g1("bb_lbr"),

            # 1-min EMA
            "1m_ema9_vs_21":    g1("ema9_vs_ema21"),
            "1m_ema21_vs_50":   g1("ema21_vs_ema50"),
            "1m_close_vs_ema9": g1("close_vs_ema9"),
            "1m_close_vs_ema21":g1("close_vs_ema21"),
            "1m_ema9_slope":    g1("ema9_slope"),

            # 1-min oscillators
            "1m_rsi14":    g1("rsi14"),
            "1m_stoch_k":  g1("stoch_k"),
            "1m_will_r":   g1("will_r"),
            "1m_mom10":    g1("mom10"),
            "1m_adx":      g1("adx"),
            "1m_plus_di":  g1("plus_di"),
            "1m_minus_di": g1("minus_di"),

            # 1-min candle structure
            "1m_body_ratio":   g1("body_ratio"),
            "1m_lwr_ratio":    g1("lwr_ratio"),
            "1m_is_bull":      int(df.loc[i, "is_bull"]),
            "1m_range_vs_atr": g1("range_vs_atr"),
            "1m_close_pos":    g1("close_pos"),
            "1m_hh_hl":        int(df.loc[i, "hh_hl"]),

            # 1-min returns
            "1m_ret1": g1("ret1"),
            "1m_ret3": g1("ret3"),
            "1m_ret5": g1("ret5"),

            # 1-min distance features
            "1m_d5l":  g1("d5l"),
            "1m_d5h":  g1("d5h"),
            "1m_d20l": g1("d20l"),
            "1m_d20h": g1("d20h"),

            # 1-min volatility
            "1m_atr14":   g1("atr14"),
            "1m_atr_exp": int(df.loc[i, "atr_exp"]),
            "1m_vol5":    g1("vol5"),
            "1m_effort":  g1("effort"),

            # Multi-TF features
            **f5, **f15, **f60,

            # Confluence
            "tf_conf":    tf_conf,
            "tr_align":   tr_align,
            "5m15m_both": int(s5 == 1 and s15 == 1),
            "15m1h_both": int(s15 == 1 and s60 == 1),
            "all3tf_sig": int(tf_conf == 3),
            "all3tf_up":  int(tr_align == 3),

            # Regime
            "price_vs_open_atr":  round((entry - day_open) / (atr1m + 1e-9), 4),
            "price_vs_open_pct":  round((entry - day_open) / (day_open + 1e-9) * 100, 4),
            "open30_dir":         o30_dir,
            "open30_rng_atr":     round((o30_high - o30_low) / (atr1m + 1e-9), 4),
            "day_range_so_far":   round((entry - df.loc[:i,"L"].min()) / (atr1m + 1e-9), 4),

            # Time
            "mins_open":  int(df.loc[i, "mins_open"]),
            "mins_close": int(df.loc[i, "mins_close"]),
            "session":    int(df.loc[i, "session"]),

            # Post-signal (look-ahead)
            **post,

            # Label
            "label": label,
        }
        records.append(record)

    return records

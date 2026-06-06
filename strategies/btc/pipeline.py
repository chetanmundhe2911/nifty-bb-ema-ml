"""
strategies/btc/pipeline.py
--------------------------
Feature engineering pipeline for BTC BB+EMA9 bounce strategy.

Key differences from NIFTY pipeline:
  - BUY  signal : price touches BB lower band → closes above EMA9
  - SELL signal : price touches BB upper band → closes below EMA9
  - RR          : 1:5 risk-reward (vs 1:3 for NIFTY)
  - Sessions    : 24/7 (no session filter)
  - Direction   : signal_direction = 1 (buy) / -1 (sell)
  - Data format : Unix timestamp CSV (Timestamp, Open, High, Low, Close, Volume)

Usage:
    from strategies.btc.pipeline import process_day_btc, process_csv_btc
    records = process_csv_btc('data/btc/raw/btcusd_1-min_data.csv',
                               start_date='2019-01-01',
                               end_date='2022-12-31')
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool, cpu_count

# ── CONFIG ────────────────────────────────────────────────────────────────────
BB_PERIOD     = 20
BB_STD        = 2.0
EMA_PERIOD    = 9
ATR_PERIOD    = 14
RSI_PERIOD    = 14
RR            = 5.0        # 1:5 risk-reward
BB_ARM_BUFFER = 0.001      # 0.1% buffer for arming (price-relative for BTC)


# ── INDICATORS ────────────────────────────────────────────────────────────────
def ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def sma(s, p):
    return s.rolling(p).mean()

def atr_fn(df, p=14):
    prev_c = df["C"].shift(1)
    tr = pd.concat([
        df["H"] - df["L"],
        (df["H"] - prev_c).abs(),
        (df["L"] - prev_c).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=p, adjust=False).mean()

def rsi_fn(s, p=14):
    d     = s.diff()
    gain  = d.clip(lower=0).ewm(span=p, adjust=False).mean()
    loss  = (-d.clip(upper=0)).ewm(span=p, adjust=False).mean()
    return 100 - 100 / (1 + gain / (loss + 1e-9))

def stoch_fn(df, p=14):
    low_min  = df["L"].rolling(p).min()
    high_max = df["H"].rolling(p).max()
    k = 100 * (df["C"] - low_min) / (high_max - low_min + 1e-9)
    return k, k.rolling(3).mean()

def williams_r_fn(df, p=14):
    high_max = df["H"].rolling(p).max()
    low_min  = df["L"].rolling(p).min()
    return -100 * (high_max - df["C"]) / (high_max - low_min + 1e-9)

def adx_fn(df, p=14):
    up   = df["H"].diff()
    down = -df["L"].diff()
    pdm  = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    mdm  = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    atr  = atr_fn(df, p)
    pdi  = pdm.ewm(span=p, adjust=False).mean() / (atr + 1e-9) * 100
    mdi  = mdm.ewm(span=p, adjust=False).mean() / (atr + 1e-9) * 100
    dx   = (pdi - mdi).abs() / (pdi + mdi + 1e-9) * 100
    return dx.ewm(span=p, adjust=False).mean(), pdi, mdi

def resample_ohlcv(df, freq):
    return (
        df.set_index("datetime")[["O","H","L","C","volume"]]
        .resample(freq, label="left", closed="left")
        .agg({"O":"first","H":"max","L":"min","C":"last","volume":"sum"})
        .dropna()
        .reset_index()
    )


def _add_indicators(df):
    """Add all technical indicators to a 1-min OHLC dataframe."""
    df = df.copy().reset_index(drop=True)
    n  = len(df)

    # EMAs
    df["ema9"]  = ema(df["C"], min(9,  n))
    df["ema21"] = ema(df["C"], min(21, n))
    df["ema50"] = ema(df["C"], min(50, n))

    # Bollinger Bands
    bm          = sma(df["C"], min(BB_PERIOD, n))
    std         = df["C"].rolling(min(BB_PERIOD, n)).std(ddof=0)
    bu          = bm + BB_STD * std
    bl          = bm - BB_STD * std
    df["bb_mid"]    = bm
    df["bb_upper"]  = bu
    df["bb_lower"]  = bl
    df["bb_width"]  = (bu - bl) / (bm + 1e-9)
    df["bb_pct_b"]  = (df["C"] - bl) / (bu - bl + 1e-9)

    # ATR
    df["atr14"]     = atr_fn(df, min(ATR_PERIOD, n))

    # Lower/upper band reach (ATR-normalised)
    df["bb_lbr"]    = (df["L"] - bl) / (df["atr14"] + 1e-9)   # negative = breach lower
    df["bb_ubr"]    = (df["H"] - bu) / (df["atr14"] + 1e-9)   # positive = breach upper

    # Oscillators
    df["rsi14"]     = rsi_fn(df["C"], min(RSI_PERIOD, n))
    df["stoch_k"], df["stoch_d"] = stoch_fn(df, min(14, n))
    df["will_r"]    = williams_r_fn(df, min(14, n))
    df["adx"], df["plus_di"], df["minus_di"] = adx_fn(df, min(14, n))

    # EMA distances
    df["ema9_vs_ema21"]  = df["ema9"]  - df["ema21"]
    df["ema21_vs_ema50"] = df["ema21"] - df["ema50"]
    df["close_vs_ema9"]  = df["C"] - df["ema9"]
    df["close_vs_ema21"] = df["C"] - df["ema21"]
    df["ema9_slope"]     = df["ema9"].diff(min(3, n-1))

    # Candle structure
    cr = df["H"] - df["L"]
    df["candle_range"]  = cr
    df["body_ratio"]    = (df["C"] - df["O"]).abs() / (cr + 1e-9)
    df["lwr_ratio"]     = (df[["O","C"]].min(axis=1) - df["L"]) / (cr + 1e-9)
    df["upr_ratio"]     = (df["H"] - df[["O","C"]].max(axis=1)) / (cr + 1e-9)
    df["is_bull"]       = (df["C"] >= df["O"]).astype(int)
    df["range_vs_atr"]  = cr / (df["atr14"] + 1e-9)
    df["close_pos"]     = (df["C"] - df["L"]) / (cr + 1e-9)
    df["effort"]        = cr / (cr.rolling(10).mean() + 1e-9)

    # Returns
    df["ret1"]  = df["C"].pct_change(1) * 100
    df["ret3"]  = df["C"].pct_change(3) * 100
    df["ret5"]  = df["C"].pct_change(5) * 100
    df["mom10"] = df["C"] - df["C"].shift(10)
    df["vol5"]  = df["ret1"].rolling(5).std()

    # Distance features (ATR-normalised)
    df["d5l"]   = (df["C"] - df["L"].rolling(5).min())  / (df["atr14"] + 1e-9)
    df["d5h"]   = (df["C"] - df["H"].rolling(5).max())  / (df["atr14"] + 1e-9)
    df["d20l"]  = (df["C"] - df["L"].rolling(20).min()) / (df["atr14"] + 1e-9)
    df["d20h"]  = (df["C"] - df["H"].rolling(20).max()) / (df["atr14"] + 1e-9)

    # HH/HL structure
    df["hh"]    = (df["H"] > df["H"].shift(3)).astype(int)
    df["hl"]    = (df["L"] > df["L"].shift(3)).astype(int)
    df["hh_hl"] = ((df["hh"]==1) & (df["hl"]==1)).astype(int)
    df["lh_ll"] = ((df["H"] < df["H"].shift(3)) & (df["L"] < df["L"].shift(3))).astype(int)

    # Volume features
    df["vol_ratio"] = df["volume"] / (df["volume"].rolling(20).mean() + 1e-9)
    df["atr_exp"]   = (df["atr14"] > df["atr14"].shift(5)).astype(int)

    return df


def _get_tf_features(tf_df, sig_dt, entry, sl, atr1m, direction, prefix):
    """
    Compute higher-timeframe features for one signal.
    direction: 1 = BUY, -1 = SELL
    """
    hist = tf_df[tf_df["datetime"] < sig_dt].copy()
    if len(hist) < 2:
        return None, None

    hist = _add_indicators(hist)
    i    = len(hist) - 1

    # Check if signal also active on this TF
    armed_buy  = False
    armed_sell = False
    for k in range(max(0, i-4), i+1):
        bl = hist.loc[k, "bb_lower"]
        bu = hist.loc[k, "bb_upper"]
        if pd.notna(bl) and hist.loc[k, "L"] <= bl * (1 + BB_ARM_BUFFER):
            armed_buy = True
        if pd.notna(bu) and hist.loc[k, "H"] >= bu * (1 - BB_ARM_BUFFER):
            armed_sell = True

    e9  = hist.loc[i, "ema9"]
    sig_active_buy  = int(armed_buy  and pd.notna(e9) and hist.loc[i, "C"] > e9)
    sig_active_sell = int(armed_sell and pd.notna(e9) and hist.loc[i, "C"] < e9)
    sig_active      = sig_active_buy if direction == 1 else sig_active_sell

    def g(col, default=0):
        v = hist.loc[i, col]
        return round(float(v), 6) if pd.notna(v) else default

    # For SELL signals flip directional features
    d = direction
    feats = {
        f"{prefix}_ema9_vs_ema21":       g("ema9_vs_ema21") * d,
        f"{prefix}_ema21_vs_ema50":      g("ema21_vs_ema50") * d,
        f"{prefix}_close_vs_ema9":       g("close_vs_ema9") * d,
        f"{prefix}_ema9_slope":          g("ema9_slope") * d,
        f"{prefix}_bb_pct_b":            g("bb_pct_b", 0.5) if d==1 else 1 - g("bb_pct_b", 0.5),
        f"{prefix}_bb_width":            g("bb_width"),
        f"{prefix}_rsi14":               g("rsi14", 50) if d==1 else 100 - g("rsi14", 50),
        f"{prefix}_stoch_k":             g("stoch_k", 50) if d==1 else 100 - g("stoch_k", 50),
        f"{prefix}_will_r":              g("will_r", -50) if d==1 else -100 - g("will_r", -50),
        f"{prefix}_is_bullish":          int(hist.loc[i, "is_bull"]) if d==1 else 1 - int(hist.loc[i, "is_bull"]),
        f"{prefix}_body_ratio":          g("body_ratio"),
        f"{prefix}_range_vs_atr":        g("range_vs_atr", 1),
        f"{prefix}_close_pos":           g("close_pos") if d==1 else 1 - g("close_pos"),
        f"{prefix}_signal_active":       sig_active,
        f"{prefix}_vol_ratio":           g("vol_ratio", 1),
        f"{prefix}_adx":                 g("adx"),
    }
    return feats, sig_active


def _analyse_post_signal(df, sig_idx, sl, entry, atr, n, direction, lookahead=10):
    """
    Classify post-entry price action.
    direction: 1 = BUY (long), -1 = SELL (short)
    """
    broke = False
    consec = 0
    first_ema9_bar = None
    recovered = False
    n_small = 0
    max_dd = 0.0
    breakout_bar = 0
    breakout_str = 0.0
    highs_lows = []

    for k in range(1, lookahead + 1):
        j = sig_idx + k
        if j >= n:
            break

        ch = df.loc[j, "H"]
        cl = df.loc[j, "L"]
        cc = df.loc[j, "C"]
        co = df.loc[j, "O"]
        cr = ch - cl
        ema9_j = df.loc[j, "ema9"]

        if direction == 1:   # BUY — long
            highs_lows.append(ch)
            if cl <= sl:
                broke = True
                max_dd = max(max_dd, sl - cl)
            if pd.notna(ema9_j) and cc > ema9_j:
                if first_ema9_bar is None: first_ema9_bar = k
                if not broke and k == consec + 1: consec = k
                if broke: recovered = True
            if cr < atr * 0.6: n_small += 1
            if k >= 2 and cr > atr * 1.2 and cc > co and breakout_bar == 0:
                breakout_bar = k
                breakout_str = cr / (atr + 1e-9)
        else:                # SELL — short
            highs_lows.append(cl)
            if ch >= sl:
                broke = True
                max_dd = max(max_dd, ch - sl)
            if pd.notna(ema9_j) and cc < ema9_j:
                if first_ema9_bar is None: first_ema9_bar = k
                if not broke and k == consec + 1: consec = k
                if broke: recovered = True
            if cr < atr * 0.6: n_small += 1
            if k >= 2 and cr > atr * 1.2 and cc < co and breakout_bar == 0:
                breakout_bar = k
                breakout_str = cr / (atr + 1e-9)

    ttc = first_ema9_bar if first_ema9_bar else lookahead + 1

    if not broke and first_ema9_bar == 1:       pat = "A_clean"
    elif not broke and first_ema9_bar == 2:     pat = "A_slight_delay"
    elif broke and recovered:                    pat = "B_wick_recover"
    elif not broke and n_small >= 2:             pat = "C_coil"
    elif broke and not recovered:                pat = "D_sl_hit"
    else:                                        pat = "E_unclear"

    mean_rng5 = np.mean([
        df.loc[sig_idx+k,"H"] - df.loc[sig_idx+k,"L"]
        for k in range(1, min(6, n - sig_idx))
    ]) if n > sig_idx + 1 else atr

    if direction == 1:
        best5 = max(highs_lows[:5]) if len(highs_lows) >= 5 else (max(highs_lows) if highs_lows else entry)
        pts5  = (best5 - entry) / (atr + 1e-9)
    else:
        best5 = min(highs_lows[:5]) if len(highs_lows) >= 5 else (min(highs_lows) if highs_lows else entry)
        pts5  = (entry - best5) / (atr + 1e-9)

    return {
        "post_pattern":       pat,
        "post_A_clean":       int(pat == "A_clean"),
        "post_A_delay":       int(pat == "A_slight_delay"),
        "post_B_wick":        int(pat == "B_wick_recover"),
        "post_C_coil":        int(pat == "C_coil"),
        "bars_to_ema9":       ttc,
        "consec_above_ema9":  consec,
        "broke_sl":           int(broke),
        "broke_sl_recovered": int(recovered),
        "max_dd_atr":         round(max_dd / (atr + 1e-9), 6),
        "n_small_candles":    n_small,
        "range_compress":     round(mean_rng5 / (atr + 1e-9), 6),
        "breakout_bar":       breakout_bar,
        "breakout_str":       round(breakout_str, 6),
        "pts5bars_atr":       round(pts5, 6),
    }


def process_chunk(df_1min, start_dt=None, end_dt=None):
    """
    Process a chunk of 1-min BTC data and extract all BUY + SELL signals.

    Parameters
    ----------
    df_1min   : pd.DataFrame — must have columns: datetime, O, H, L, C, volume
    start_dt  : str or None — filter start datetime
    end_dt    : str or None — filter end datetime

    Returns
    -------
    list of signal records (dicts) with all features + label
    """
    df = df_1min.copy().sort_values("datetime").reset_index(drop=True)

    if start_dt:
        df = df[df["datetime"] >= pd.to_datetime(start_dt)].reset_index(drop=True)
    if end_dt:
        df = df[df["datetime"] <= pd.to_datetime(end_dt)].reset_index(drop=True)

    if len(df) < 100:
        return []

    # Resample to higher TFs
    df5  = resample_ohlcv(df, "5min")
    df15 = resample_ohlcv(df, "15min")
    df60 = resample_ohlcv(df, "60min")

    # Add indicators
    df = _add_indicators(df)
    n  = len(df)

    records    = []
    buy_armed  = False
    sell_armed = False

    for i in range(BB_PERIOD + 5, n):
        sig_dt = df.loc[i, "datetime"]
        close  = df.loc[i, "C"]
        high   = df.loc[i, "H"]
        low    = df.loc[i, "L"]
        bb_l   = df.loc[i, "bb_lower"]
        bb_u   = df.loc[i, "bb_upper"]
        e9     = df.loc[i, "ema9"]
        atr1m  = df.loc[i, "atr14"]

        if pd.isna(bb_l) or pd.isna(e9) or atr1m <= 0:
            continue

        # ── ARM signals ───────────────────────────────────────────────────────
        if low  <= bb_l * (1 + BB_ARM_BUFFER): buy_armed  = True
        if high >= bb_u * (1 - BB_ARM_BUFFER): sell_armed = True

        signals_to_process = []

        # ── TRIGGER BUY ───────────────────────────────────────────────────────
        if buy_armed and close > e9:
            buy_armed = False
            entry  = close
            sl     = low
            risk   = entry - sl
            if risk > 0:
                target = entry + risk * RR
                signals_to_process.append(("BUY", 1, entry, sl, target, risk))

        # ── TRIGGER SELL ──────────────────────────────────────────────────────
        if sell_armed and close < e9:
            sell_armed = False
            entry  = close
            sl     = high
            risk   = sl - entry
            if risk > 0:
                target = entry - risk * RR
                signals_to_process.append(("SELL", -1, entry, sl, target, risk))

        for sig_type, direction, entry, sl, target, risk in signals_to_process:

            # Label: did 1:5 target hit before SL?
            label = 0
            for j in range(i + 1, min(i + 200, n)):  # look up to 200 bars forward
                if direction == 1:
                    if df.loc[j, "L"] <= sl:       break
                    if df.loc[j, "H"] >= target:   label = 1; break
                else:
                    if df.loc[j, "H"] >= sl:       break
                    if df.loc[j, "L"] <= target:   label = 1; break

            # Higher TF features
            f5,  s5  = _get_tf_features(df5,  sig_dt, entry, sl, atr1m, direction, "5m")
            f15, s15 = _get_tf_features(df15, sig_dt, entry, sl, atr1m, direction, "15m")
            f60, s60 = _get_tf_features(df60, sig_dt, entry, sl, atr1m, direction, "1h")
            if any(x is None for x in [f5, f15, f60]):
                continue

            tf_conf  = s5 + s15 + s60
            tr_align = (
                int(f5["5m_ema9_vs_ema21"]   > 0) +
                int(f15["15m_ema9_vs_ema21"]  > 0) +
                int(f60["1h_ema9_vs_ema21"]   > 0)
            )

            # Post-signal features
            post = _analyse_post_signal(df, i, sl, entry, atr1m, n, direction)

            def g1(col):
                v = df.loc[i, col]
                return round(float(v), 6) if pd.notna(v) else 0

            # Flip directional features for SELL signals
            d = direction
            record = {
                # Metadata
                "datetime":       str(sig_dt),
                "signal_type":    sig_type,
                "signal_direction": direction,
                "entry":          round(entry, 4),
                "sl":             round(sl, 4),
                "target_1_5":     round(target, 4),
                "risk_pts":       round(risk, 4),
                "risk_pct":       round(risk / entry * 100, 4),

                # 1-min BB
                "1m_bb_pct_b":    g1("bb_pct_b") if d==1 else 1 - g1("bb_pct_b"),
                "1m_bb_width":    g1("bb_width"),
                "1m_bb_lbr":      g1("bb_lbr") if d==1 else -g1("bb_ubr"),

                # 1-min EMA (flipped for SELL)
                "1m_ema9_vs_21":      g1("ema9_vs_ema21") * d,
                "1m_ema21_vs_50":     g1("ema21_vs_ema50") * d,
                "1m_close_vs_ema9":   g1("close_vs_ema9") * d,
                "1m_close_vs_ema21":  g1("close_vs_ema21") * d,
                "1m_ema9_slope":      g1("ema9_slope") * d,

                # 1-min oscillators (flipped for SELL)
                "1m_rsi14":     g1("rsi14") if d==1 else 100 - g1("rsi14"),
                "1m_stoch_k":   g1("stoch_k") if d==1 else 100 - g1("stoch_k"),
                "1m_will_r":    g1("will_r") if d==1 else -100 - g1("will_r"),
                "1m_mom10":     g1("mom10") * d,
                "1m_adx":       g1("adx"),
                "1m_plus_di":   g1("plus_di") if d==1 else g1("minus_di"),
                "1m_minus_di":  g1("minus_di") if d==1 else g1("plus_di"),
                "1m_ret1":      g1("ret1") * d,
                "1m_ret3":      g1("ret3") * d,
                "1m_ret5":      g1("ret5") * d,

                # 1-min candle (flipped for SELL)
                "1m_body_ratio":   g1("body_ratio"),
                "1m_lwr_ratio":    g1("lwr_ratio") if d==1 else g1("upr_ratio"),
                "1m_is_bull":      int(df.loc[i,"is_bull"]) if d==1 else 1 - int(df.loc[i,"is_bull"]),
                "1m_range_vs_atr": g1("range_vs_atr"),
                "1m_close_pos":    g1("close_pos") if d==1 else 1 - g1("close_pos"),
                "1m_effort":       g1("effort"),
                "1m_vol5":         g1("vol5"),
                "1m_vol_ratio":    g1("vol_ratio"),
                "1m_atr14":        g1("atr14"),
                "1m_atr_exp":      int(df.loc[i,"atr_exp"]),
                "1m_hh_hl":        int(df.loc[i,"hh_hl"]) if d==1 else int(df.loc[i,"lh_ll"]),

                # Distance features (flipped for SELL)
                "1m_d5l":   g1("d5l") if d==1 else -g1("d5h"),
                "1m_d5h":   g1("d5h") if d==1 else -g1("d5l"),
                "1m_d20l":  g1("d20l") if d==1 else -g1("d20h"),
                "1m_d20h":  g1("d20h") if d==1 else -g1("d20l"),

                # Multi-TF features
                **f5, **f15, **f60,

                # Confluence
                "tf_conf":    tf_conf,
                "tr_align":   tr_align * d,
                "all3tf_sig": int(tf_conf == 3),
                "all3tf_dir": int(abs(tr_align) == 3),

                # Signal direction (key feature)
                "signal_direction": direction,

                # Post-signal (look-ahead)
                **post,

                # Label
                "label": label,
            }
            records.append(record)

    return records


def process_csv_btc(filepath, start_date=None, end_date=None, chunk_days=30):
    """
    Process the full BTC CSV file in chunks.

    Parameters
    ----------
    filepath   : str — path to btcusd_1-min_data.csv
    start_date : str — 'YYYY-MM-DD'
    end_date   : str — 'YYYY-MM-DD'
    chunk_days : int — process N days at a time to manage memory

    Returns
    -------
    pd.DataFrame — all signals with features + labels
    """
    print(f"Loading BTC data from {filepath}...")
    df_raw = pd.read_csv(filepath)

    # Convert Unix timestamp to datetime
    df_raw["datetime"] = pd.to_datetime(df_raw["Timestamp"], unit="s")
    df_raw = df_raw.rename(columns={
        "Open":"O", "High":"H", "Low":"L", "Close":"C", "Volume":"volume"
    })
    df_raw = df_raw[["datetime","O","H","L","C","volume"]].sort_values("datetime").reset_index(drop=True)

    # Filter date range
    if start_date:
        df_raw = df_raw[df_raw["datetime"] >= pd.to_datetime(start_date)]
    if end_date:
        df_raw = df_raw[df_raw["datetime"] <= pd.to_datetime(end_date)]

    df_raw = df_raw.reset_index(drop=True)
    print(f"Rows after filter: {len(df_raw):,}")
    print(f"Date range: {df_raw['datetime'].min()} -> {df_raw['datetime'].max()}")

    # Process in chunks
    all_records = []
    dates       = df_raw["datetime"].dt.date.unique()
    total_days  = len(dates)

    print(f"Processing {total_days} days in chunks of {chunk_days}...")

    for start_idx in range(0, total_days, chunk_days):
        chunk_dates = dates[start_idx : start_idx + chunk_days]
        chunk_df    = df_raw[df_raw["datetime"].dt.date.isin(chunk_dates)].reset_index(drop=True)

        if len(chunk_df) < 100:
            continue

        records = process_chunk(chunk_df)
        all_records.extend(records)

        buys  = sum(1 for r in records if r["signal_type"] == "BUY")
        sells = sum(1 for r in records if r["signal_type"] == "SELL")
        wins  = sum(r["label"] for r in records)
        print(f"  Days {start_idx+1}-{min(start_idx+chunk_days, total_days)}: "
              f"{len(records)} signals (B:{buys} S:{sells}) | "
              f"Wins: {wins} ({wins/len(records)*100:.1f}%)" if records else
              f"  Days {start_idx+1}-{min(start_idx+chunk_days, total_days)}: 0 signals")

    if not all_records:
        print("No signals found.")
        return pd.DataFrame()

    df_out = pd.DataFrame(all_records)
    print(f"\nTotal: {len(df_out)} signals | "
          f"Wins: {df_out['label'].sum()} ({df_out['label'].mean()*100:.1f}%)")
    print(f"BUY: {(df_out['signal_type']=='BUY').sum()} | "
          f"SELL: {(df_out['signal_type']=='SELL').sum()}")
    return df_out


if __name__ == "__main__":
    import sys
    filepath   = r"C:\Users\cheta\Downloads\archive\btcusd_1-min_data.csv"
    start_date = sys.argv[1] if len(sys.argv) > 1 else "2019-01-01"
    end_date   = sys.argv[2] if len(sys.argv) > 2 else "2019-03-31"

    print(f"Testing BTC pipeline: {start_date} to {end_date}")
    df = process_csv_btc(filepath, start_date=start_date, end_date=end_date)

    if not df.empty:
        print(f"\nSample signals:")
        print(df[["datetime","signal_type","entry","sl","target_1_5",
                  "risk_pct","post_pattern","label"]].head(10).to_string(index=False))
        print(f"\nPattern breakdown:")
        for pat, grp in df.groupby("post_pattern"):
            print(f"  {pat:<20} {len(grp):>5} signals | "
                  f"Win rate {grp['label'].mean()*100:.1f}%")




def _process_chunk_wrapper(args):
    chunk_df, start_dt, end_dt = args
    return process_chunk(chunk_df, start_dt, end_dt)

def process_csv_btc_fast(filepath, start_date=None, end_date=None, chunk_days=30, n_workers=None):
    """
    Multicore version of process_csv_btc.
    Uses all available CPU cores.
    """
    if n_workers is None:
        n_workers = cpu_count()
    
    print(f"Loading BTC data from {filepath}...")
    df_raw = pd.read_csv(filepath)
    df_raw['datetime'] = pd.to_datetime(df_raw['Timestamp'], unit='s')
    df_raw = df_raw.rename(columns={'Open':'O','High':'H','Low':'L','Close':'C','Volume':'volume'})
    df_raw = df_raw[['datetime','O','H','L','C','volume']].sort_values('datetime').reset_index(drop=True)

    if start_date:
        df_raw = df_raw[df_raw['datetime'] >= pd.to_datetime(start_date)]
    if end_date:
        df_raw = df_raw[df_raw['datetime'] <= pd.to_datetime(end_date)]

    df_raw = df_raw.reset_index(drop=True)
    print(f"Rows: {len(df_raw):,} | Using {n_workers} CPU cores")

    dates      = df_raw['datetime'].dt.date.unique()
    total_days = len(dates)

    # Build chunks with 5-day overlap for indicator warmup
    chunks = []
    for start_idx in range(0, total_days, chunk_days):
        chunk_dates  = dates[max(0, start_idx-5) : start_idx + chunk_days]
        actual_start = str(dates[start_idx])
        actual_end   = str(dates[min(start_idx + chunk_days - 1, total_days - 1)])
        chunk_df     = df_raw[df_raw['datetime'].dt.date.isin(chunk_dates)].reset_index(drop=True)
        chunks.append((chunk_df, actual_start, actual_end))

    print(f"Processing {len(chunks)} chunks across {n_workers} cores...")

    with Pool(n_workers) as pool:
        results = pool.map(_process_chunk_wrapper, chunks)

    all_records = [r for chunk_result in results for r in chunk_result]

    if not all_records:
        print("No signals found.")
        return pd.DataFrame()

    df_out = pd.DataFrame(all_records)
    df_out = df_out.drop_duplicates(subset=['datetime','signal_type']).reset_index(drop=True)

    print(f"\nTotal: {len(df_out)} signals | Wins: {df_out['label'].sum()} ({df_out['label'].mean()*100:.1f}%)")
    return df_out
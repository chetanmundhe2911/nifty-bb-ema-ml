# -*- coding: utf-8 -*-
"""
strategies/btc/simple_scanner.py
---------------------------------
Simple BB+EMA9 Buy/Sell scanner with 1H regime filter.
No ML model — pure rule based.
1-min feed: Binance (accurate, no duplicates)
1H regime : Delta Exchange (previous closed candle vs EMA9)
"""

import os, sys, time, requests
import pandas as pd
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# -- CONFIG -------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = "8725357996:AAHJbUVxY6huX8SEgUFhRCYdzIW8MgqXpLg"
TELEGRAM_CHAT_ID   = "7804862044"
POLL_SECONDS       = 60
CANDLES_1M         = 300
RR                 = 3

# -- COLOURS ------------------------------------------------------------------
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def send_telegram(msg):
    try:
        requests.post(
            "https://api.telegram.org/bot{}/sendMessage".format(TELEGRAM_BOT_TOKEN),
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"},
            timeout=10
        )
    except Exception as e:
        print("Telegram error: {}".format(e))


def get_binance_candles(limit=300):
    """Fetch 1-min candles from Binance — clean, no duplicates."""
    r = requests.get(
        "https://api.binance.com/api/v3/klines",
        params={"symbol": "BTCUSDT", "interval": "1m", "limit": limit},
        timeout=10
    )
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data, columns=[
        "timestamp","open","high","low","close","volume",
        "close_time","quote_vol","trades","taker_base","taker_quote","ignore"
    ])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["O"] = df["open"].astype(float)
    df["H"] = df["high"].astype(float)
    df["L"] = df["low"].astype(float)
    df["C"] = df["close"].astype(float)
    return df[["datetime","O","H","L","C"]].reset_index(drop=True)


def get_regime():
    """Get 1H regime from Delta — previous CLOSED candle vs EMA9."""
    end   = int(time.time())
    start = end - 50 * 3600
    url   = "https://api.delta.exchange/v2/history/candles"
    params = {"resolution": "1h", "symbol": "BTCUSDT", "start": start, "end": end}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    result = r.json().get("result", [])
    if not result or len(result) < 10:
        return None, None, None
    df = pd.DataFrame(result)
    df["datetime"] = pd.to_datetime(df["time"], unit="s")
    df["C"] = df["close"].astype(float)
    df = df.sort_values("datetime").reset_index(drop=True)
    df["ema9"] = df["C"].ewm(span=9, adjust=False).mean()
    prev = df.iloc[-2]  # previous CLOSED candle
    bull = prev["C"] > prev["ema9"]
    return bull, prev["C"], prev["ema9"]


def detect_signals(df):
    """Detect BB+EMA9 BUY and SELL signals — no risk filter."""
    df = df.copy()

    # BB(20,2)
    df["bb_mid"]   = df["C"].rolling(20).mean()
    df["bb_std"]   = df["C"].rolling(20).std(ddof=0)
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]

    # EMA9
    df["ema9"] = df["C"].ewm(span=9, adjust=False).mean()

    signals    = []
    armed_buy  = False
    armed_sell = False

    for i in range(20, len(df)):
        row = df.iloc[i]

        # Arm BUY
        if row["L"] <= row["bb_lower"]:
            armed_buy = True

        # Arm SELL
        if row["H"] >= row["bb_upper"]:
            armed_sell = True

        # BUY signal
        if armed_buy and row["C"] > row["ema9"]:
            entry = row["C"]
            sl    = row["bb_lower"]
            risk  = entry - sl
            signals.append({
                "datetime": row["datetime"],
                "type":     "BUY",
                "entry":    round(entry, 2),
                "sl":       round(sl, 2),
                "target":   round(entry + risk * RR, 2),
                "risk_pct": round(risk / entry * 100, 3),
            })
            armed_buy = False

        # SELL signal
        if armed_sell and row["C"] < row["ema9"]:
            entry = row["C"]
            sl    = row["bb_upper"]
            risk  = sl - entry
            signals.append({
                "datetime": row["datetime"],
                "type":     "SELL",
                "entry":    round(entry, 2),
                "sl":       round(sl, 2),
                "target":   round(entry - risk * RR, 2),
                "risk_pct": round(risk / entry * 100, 3),
            })
            armed_sell = False

    return signals


def main():
    print("\n{}{}".format(BOLD, "="*60))
    print("  BTC BB+EMA9 SIMPLE SCANNER — BUY + SELL")
    print("{}{}".format("="*60, RESET))
    print("  1-min feed : Binance")
    print("  1H regime  : Delta Exchange (prev closed candle)")
    print("  RR         : 1:{}".format(RR))
    print("  Risk filter: NONE — all signals fire")
    print("  Poll       : every {}s".format(POLL_SECONDS))
    print("{}{}\n".format(BOLD, "="*60 + RESET))

    send_telegram(
        "<b>BTC Simple Scanner started</b>\n"
        "1-min: Binance | 1H regime: Delta\n"
        "BB+EMA9 | RR: 1:{} | All signals".format(RR)
    )

    seen_signals = set()
    buy_count    = 0
    sell_count   = 0

    print("{}Scanning... (Ctrl+C to stop){}".format(GREEN, RESET))
    print("-"*60)

    while True:
        try:
            # Get 1H regime
            bull, price_1h, ema9_1h = get_regime()
            if bull is None:
                print("\n{}Could not get 1H regime{}".format(YELLOW, RESET))
                time.sleep(POLL_SECONDS)
                continue

            regime_str = "BULL - BUY only" if bull else "BEAR - SELL only"

            # Get 1-min candles from Binance
            df = get_binance_candles(CANDLES_1M)
            if df.empty:
                time.sleep(POLL_SECONDS)
                continue

            # Detect signals
            signals = detect_signals(df)

            # Filter by regime and alert
            latest = df["datetime"].max()
            for sig in signals:
                sig_key = str(sig["datetime"]) + sig["type"]
                if sig_key in seen_signals:
                    continue

                # Regime filter
                if bull and sig["type"] == "SELL":
                    continue
                if not bull and sig["type"] == "BUY":
                    continue

                seen_signals.add(sig_key)

                # Only alert recent signals (last 5 mins)
                if sig["datetime"] < latest - pd.Timedelta(minutes=5):
                    continue

                if sig["type"] == "BUY":
                    buy_count += 1
                    msg = (
                        "<b>BTC BUY SIGNAL</b>\n\n"
                        "Time   : {}\n"
                        "Entry  : ${:,.2f}\n"
                        "SL     : ${:,.2f}\n"
                        "Target : ${:,.2f} (1:{})\n"
                        "Risk   : {:.3f}%\n"
                        "Regime : {}"
                    ).format(sig["datetime"], sig["entry"], sig["sl"],
                             sig["target"], RR, sig["risk_pct"], regime_str)
                    send_telegram(msg)
                    print("\n{}{} BUY SIGNAL {}{}".format(BOLD, GREEN, RESET, RESET))
                    print("  Time   : {}".format(sig["datetime"]))
                    print("  Entry  : {}${:,.2f}{}".format(GREEN, sig["entry"], RESET))
                    print("  SL     : ${:,.2f}".format(sig["sl"]))
                    print("  Target : ${:,.2f}".format(sig["target"]))
                    print("  Risk   : {:.3f}%".format(sig["risk_pct"]))

                else:
                    sell_count += 1
                    msg = (
                        "<b>BTC SELL SIGNAL</b>\n\n"
                        "Time   : {}\n"
                        "Entry  : ${:,.2f}\n"
                        "SL     : ${:,.2f}\n"
                        "Target : ${:,.2f} (1:{})\n"
                        "Risk   : {:.3f}%\n"
                        "Regime : {}"
                    ).format(sig["datetime"], sig["entry"], sig["sl"],
                             sig["target"], RR, sig["risk_pct"], regime_str)
                    send_telegram(msg)
                    print("\n{}{} SELL SIGNAL {}{}".format(BOLD, RED, RESET, RESET))
                    print("  Time   : {}".format(sig["datetime"]))
                    print("  Entry  : {}${:,.2f}{}".format(RED, sig["entry"], RESET))
                    print("  SL     : ${:,.2f}".format(sig["sl"]))
                    print("  Target : ${:,.2f}".format(sig["target"]))
                    print("  Risk   : {:.3f}%".format(sig["risk_pct"]))

            now   = datetime.now(timezone.utc).strftime("%H:%M:%S")
            price = float(df["C"].iloc[-1])
            print(
                "\r  [{} UTC]  BTC ${:,.0f}  |  Regime: {}  |  {}BUY:{}{}  {}SELL:{}{}  ".format(
                    now, price, regime_str,
                    GREEN, buy_count, RESET,
                    RED, sell_count, RESET
                ),
                end="", flush=True
            )

        except KeyboardInterrupt:
            print("\n\n{}Stopped.{}".format(YELLOW, RESET))
            send_telegram("BTC Simple Scanner stopped.")
            break
        except Exception as e:
            print("\n{}Error: {}{}".format(RED, e, RESET))

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()

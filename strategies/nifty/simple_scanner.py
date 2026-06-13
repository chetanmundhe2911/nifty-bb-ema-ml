# -*- coding: utf-8 -*-
"""
strategies/nifty/simple_scanner.py
------------------------------------
Simple BB+EMA9 Buy/Sell scanner for NIFTY 50.
No ML model — pure rule based.
Upstox feed — market hours only 9:15 AM - 3:30 PM IST
"""

import os, sys, time, requests
import pandas as pd
from datetime import datetime, timezone
import pytz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# -- CONFIG -------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = "8725357996:AAHJbUVxY6huX8SEgUFhRCYdzIW8MgqXpLg"
TELEGRAM_CHAT_ID   = "7804862044"
POLL_SECONDS       = 60
RR                 = 3
IST                = pytz.timezone("Asia/Kolkata")

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


def in_market_hours():
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    market_open  = now.replace(hour=9,  minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


def get_nifty_candles():
    """Fetch intraday 1-min NIFTY candles from Upstox."""
    from src.upstox_feed import get_intraday_candles
    df = get_intraday_candles()
    if df.empty:
        return pd.DataFrame()
    df = df.rename(columns={'high':'H','low':'L','close':'C','open':'O'})
    df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str))
    return df[['datetime','O','H','L','C']].reset_index(drop=True)


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
    print("  NIFTY BB+EMA9 SIMPLE SCANNER — BUY + SELL")
    print("{}{}".format("="*60, RESET))
    print("  Feed      : Upstox (1-min)")
    print("  Hours     : 9:15 AM - 3:30 PM IST Mon-Fri")
    print("  RR        : 1:{}".format(RR))
    print("  Risk filter: NONE — all signals fire")
    print("  Poll      : every {}s".format(POLL_SECONDS))
    print("{}{}\n".format(BOLD, "="*60 + RESET))

    send_telegram(
        "<b>NIFTY Simple Scanner started</b>\n"
        "BB+EMA9 | RR: 1:{} | All signals\n"
        "Hours: 9:15 AM - 3:30 PM IST".format(RR)
    )

    seen_signals = set()
    buy_count    = 0
    sell_count   = 0

    print("{}Scanning... (Ctrl+C to stop){}".format(GREEN, RESET))
    print("-"*60)

    while True:
        try:
            now_ist = datetime.now(IST)

            if not in_market_hours():
                print("\r  [{}]  Market closed — waiting...   ".format(
                    now_ist.strftime("%H:%M:%S IST")), end="", flush=True)
                time.sleep(POLL_SECONDS)
                continue

            # Get candles
            df = get_nifty_candles()
            if df.empty:
                print("\n{}No candle data{}".format(YELLOW, RESET))
                time.sleep(POLL_SECONDS)
                continue

            # Detect signals
            signals = detect_signals(df)

            # Alert new signals
            latest = df["datetime"].max()
            for sig in signals:
                sig_key = str(sig["datetime"]) + sig["type"]
                if sig_key in seen_signals:
                    continue
                seen_signals.add(sig_key)

                # Only alert recent signals (last 5 mins)
                if pd.to_datetime(sig["datetime"]) < pd.to_datetime(latest) - pd.Timedelta(minutes=5):
                    continue

                if sig["type"] == "BUY":
                    buy_count += 1
                    msg = (
                        "<b>NIFTY BUY SIGNAL</b>\n\n"
                        "Time   : {}\n"
                        "Entry  : {:.2f}\n"
                        "SL     : {:.2f}\n"
                        "Target : {:.2f} (1:{})\n"
                        "Risk   : {:.3f}%"
                    ).format(sig["datetime"], sig["entry"], sig["sl"],
                             sig["target"], RR, sig["risk_pct"])
                    send_telegram(msg)
                    print("\n{}{} NIFTY BUY SIGNAL {}{}".format(BOLD, GREEN, RESET, RESET))
                    print("  Time   : {}".format(sig["datetime"]))
                    print("  Entry  : {}${:.2f}{}".format(GREEN, sig["entry"], RESET))
                    print("  SL     : {:.2f}".format(sig["sl"]))
                    print("  Target : {:.2f}".format(sig["target"]))

                else:
                    sell_count += 1
                    msg = (
                        "<b>NIFTY SELL SIGNAL</b>\n\n"
                        "Time   : {}\n"
                        "Entry  : {:.2f}\n"
                        "SL     : {:.2f}\n"
                        "Target : {:.2f} (1:{})\n"
                        "Risk   : {:.3f}%"
                    ).format(sig["datetime"], sig["entry"], sig["sl"],
                             sig["target"], RR, sig["risk_pct"])
                    send_telegram(msg)
                    print("\n{}{} NIFTY SELL SIGNAL {}{}".format(BOLD, RED, RESET, RESET))
                    print("  Time   : {}".format(sig["datetime"]))
                    print("  Entry  : {}${:.2f}{}".format(RED, sig["entry"], RESET))
                    print("  SL     : {:.2f}".format(sig["sl"]))
                    print("  Target : {:.2f}".format(sig["target"]))

            price = float(df["C"].iloc[-1])
            print(
                "\r  [{}]  NIFTY {:.0f}  |  {}BUY:{}{}  {}SELL:{}{}  Candles:{}  ".format(
                    now_ist.strftime("%H:%M:%S IST"), price,
                    GREEN, buy_count, RESET,
                    RED, sell_count, RESET,
                    len(df)
                ),
                end="", flush=True
            )

        except KeyboardInterrupt:
            print("\n\n{}Stopped.{}".format(YELLOW, RESET))
            send_telegram("NIFTY Simple Scanner stopped.")
            break
        except Exception as e:
            print("\n{}Error: {}{}".format(RED, e, RESET))

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
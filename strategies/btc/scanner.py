# -*- coding: utf-8 -*-
"""
strategies/btc/scanner.py
-------------------------
Live BTC BB+EMA9 signal scanner using Binance REST API.
Uses 1:3 RR model (signal_model_btc_v1_rr3.pkl)
Sends Telegram alerts on BUY signals.
Poll every 60 seconds - 24/7.
"""

import os, sys, time, pickle, argparse, requests
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from strategies.btc.pipeline import process_chunk


# -- CONFIG -------------------------------------------------------------------
MODEL_PATH         = "models/btc/v1/signal_model_btc_v1_rr3.pkl"
TELEGRAM_BOT_TOKEN = "8725357996:AAHJbUVxY6huX8SEgUFhRCYdzIW8MgqXpLg"
TELEGRAM_CHAT_ID   = "7804862044"
POLL_SECONDS       = 60
CANDLES_NEEDED     = 300

# -- ARGS ---------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--feed",      default="binance", choices=["binance","delta"])
parser.add_argument("--threshold", type=float, default=0.50)
parser.add_argument("--risk",      type=float, default=0.15)
args = parser.parse_args()

FEED        = args.feed
THRESHOLD   = args.threshold
RISK_FILTER = args.risk

args = parser.parse_args()

FEED        = args.feed
THRESHOLD   = args.threshold
RISK_FILTER = args.risk

# -- FEED SELECTION -----------------------------------------------------------
if FEED == "delta":
    from strategies.btc.feeds.delta_feed import get_candles
    FEED_LABEL = "Delta Exchange"
else:
    from strategies.btc.feeds.binance_feed import get_candles
    FEED_LABEL = "Binance"

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
        print("{}Telegram error: {}{}".format(RED, e, RESET))


def score_signals(df_candles, model, feat_cols):
    if len(df_candles) < 50:
        return pd.DataFrame()
    try:
        records = process_chunk(df_candles)
    except Exception as e:
        print("{}Pipeline error: {}{}".format(RED, e, RESET))
        return pd.DataFrame()
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df[df["risk_pct"] >= RISK_FILTER].copy()
    if df.empty:
        return pd.DataFrame()

    X = pd.DataFrame(0, index=df.index, columns=feat_cols)
    for col in feat_cols:
        if col in df.columns:
            X[col] = df[col].fillna(0)

    df["prob"] = model.predict_proba(X)[:, 1].round(4)
    df["pred"] = (df["prob"] >= THRESHOLD).astype(int)
    return df


def main():
    print("\n{}{}".format(BOLD, "="*60))
    print("  BTC BB+EMA9 LIVE SCANNER  --  1:3 RR")
    print("{}{}".format("="*60, RESET))
    print("  Symbol    : BTCUSDT ({})".format(FEED_LABEL))
    print("  Timeframe : 1-min")
    print("  RR        : 1:3")
    print("  Threshold : {}".format(THRESHOLD))
    print("  Risk min  : {}%".format(RISK_FILTER))
    print("  Sessions  : 24/7")
    print("  Poll      : every {}s".format(POLL_SECONDS))
    print("{}{}\n".format(BOLD, "="*60 + RESET))

    bundle    = pickle.load(open(MODEL_PATH, "rb"))
    model     = bundle["model"]
    feat_cols = bundle["feature_cols"]
    print("Model     : {}".format(bundle["description"]))
    print("CV AUC    : {:.3f}  |  Val AUC : {:.3f}".format(
        bundle["cv_auc_mean"], bundle["val_auc"]))
    print("Features  : {}\n".format(len(feat_cols)))

    send_telegram(
        "<b>BTC Scanner started</b>\n"
        "Model: 1:3 RR | Threshold: {}\n"
        "Val AUC: {:.3f}".format(THRESHOLD, bundle["val_auc"])
    )

    seen_signals = set()
    buy_count    = 0
    skip_count   = 0

    print("{}Scanning 24/7... (Ctrl+C to stop){}".format(GREEN, RESET))
    print("-"*60)

    while True:
        try:
            df_candles = get_candles(CANDLES_NEEDED)

            # Print OHLC every 5 minutes
            now_min = datetime.utcnow().minute
            if now_min % 5 == 0:
                last = df_candles.iloc[-1]
                print(f"\n  {last['datetime']}  O:{last['O']:.2f}  H:{last['H']:.2f}  L:{last['L']:.2f}  C:{last['C']:.2f}")
            df_scored  = score_signals(df_candles, model, feat_cols)

            if not df_scored.empty:
                latest_dt = df_candles["datetime"].max()
                recent    = df_scored[
                    df_scored["datetime"] >= latest_dt - pd.Timedelta(minutes=5)
                ]

                for _, row in recent.iterrows():
                    sig_key = str(row["datetime"])
                    if sig_key in seen_signals:
                        continue
                    seen_signals.add(sig_key)

                    pred    = int(row["pred"])
                    prob    = float(row["prob"])
                    pattern = row.get("post_pattern", "unknown")
                    entry   = float(row["entry"])
                    sl      = float(row["sl"])
                    risk    = float(row["risk_pts"])
                    tgt3    = entry + (entry - sl) * 3

                    if pred == 1:
                        buy_count += 1
                        msg = (
                            "<b>BTC BUY SIGNAL</b>\n\n"
                            "Time    : {}\n"
                            "Entry   : ${:,.2f}\n"
                            "SL      : ${:,.2f}\n"
                            "Target  : ${:,.2f} (1:3)\n"
                            "Risk    : ${:,.2f} ({:.2f}%)\n"
                            "Prob    : {:.1%}\n"
                            "Pattern : {}"
                        ).format(
                            row["datetime"], entry, sl, tgt3,
                            risk, row["risk_pct"], prob, pattern
                        )
                        send_telegram(msg)
                        print("\n{}{} BUY SIGNAL {}{}".format(BOLD, GREEN, RESET, RESET))
                        print("  Time    : {}".format(row["datetime"]))
                        print("  Entry   : {}${:,.2f}{}".format(GREEN, entry, RESET))
                        print("  SL      : ${:,.2f}".format(sl))
                        print("  Target  : ${:,.2f}".format(tgt3))
                        print("  Prob    : {}{:.3f}{}".format(GREEN, prob, RESET))
                        print("  Pattern : {}".format(pattern))
                    else:
                        skip_count += 1

            now   = datetime.utcnow().strftime("%H:%M:%S")
            price = float(df_candles["C"].iloc[-1])
            print(
                "\r  [{} UTC]  BTC ${:,.0f}  |  {}BUY: {}{}  |  Skip: {}  |  Candles: {}   ".format(
                    now, price, GREEN, buy_count, RESET, skip_count, len(df_candles)
                ),
                end="", flush=True
            )

        except KeyboardInterrupt:
            print("\n\n{}Stopped.{}".format(YELLOW, RESET))
            send_telegram("BTC Scanner stopped.")
            break
        except Exception as e:
            print("\n{}Error: {}{}".format(RED, e, RESET))

        # Update price display every 10s, but only scan every 60s
        for _ in range(6):
            time.sleep(10)
            try:
                df_price = get_candles(5)
                price = float(df_price["C"].iloc[-1])
                now   = datetime.utcnow().strftime("%H:%M:%S")
                last_candle_time  = df_candles["datetime"].iloc[-1].strftime("%H:%M")
                last_candle_close = float(df_candles["C"].iloc[-1])
                print(
                    "\r  [{} UTC]  {} BTC ${:,.0f}  |  Last candle: {} @ ${:,.0f}  |  {}BUY: {}{}  |  Skip: {}  |  Candles: {}   ".format(
                        now, FEED_LABEL, price,
                        last_candle_time, last_candle_close,
                        GREEN, buy_count, RESET,
                        skip_count, len(df_candles)
                    ),
                    end="", flush=True
                )
            except:
                pass


if __name__ == "__main__":
    main()

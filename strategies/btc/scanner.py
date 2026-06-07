# -*- coding: utf-8 -*-
"""
strategies/btc/scanner.py
-------------------------
Live BTC BB+EMA9 signal scanner using Binance REST API.
Uses 1:3 RR model (signal_model_btc_v1_rr3.pkl)
Sends Telegram alerts on BUY signals.
Poll every 60 seconds - 24/7.
"""

import os, sys, time, pickle, requests
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
RISK_FILTER        = 0.15
THRESHOLD          = 0.50

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


def get_candles(limit=300):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": "1m", "limit": limit}
    r = requests.get(url, params=params, timeout=10)
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
    df["volume"] = df["volume"].astype(float)
    return df[["datetime","O","H","L","C","volume"]].reset_index(drop=True)


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
    print("  Symbol    : BTCUSDT (Binance)")
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

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()

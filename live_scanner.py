"""
live_scanner.py
---------------
Real-time NIFTY BB+EMA9 signal scanner using Upstox intraday feed.

Usage:
    python live_scanner.py

Polls Upstox every 60 seconds from 9:45 AM IST.
Prints BUY signals with probability as they are detected.
Logs all signals to logs/signals_YYYYMMDD.csv
"""

import os
import sys
import time
import pickle
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.upstox_feed import get_intraday_candles
from src.pipeline import process_day

load_dotenv()

MODEL_PATH   = "models/v1/signal_model.pkl"
LOG_DIR      = "logs"
POLL_SECONDS = 60
START_HOUR   = 9
START_MIN    = 45
END_HOUR     = 14
END_MIN      = 30

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def load_model(path: str):
    bundle    = pickle.load(open(path, "rb"))
    model     = bundle["model"]
    feat_cols = bundle["feature_cols"]
    le        = bundle["label_encoder"]
    threshold = bundle["threshold"]
    print(f"{CYAN}Model loaded — CV AUC {bundle['cv_auc_mean']:.3f} "
          f"| Threshold {threshold}{RESET}")
    return model, feat_cols, le, threshold


def score_signals(df_candles, model, feat_cols, le, threshold):
    if df_candles.empty or len(df_candles) < 25:
        return pd.DataFrame()
    tmp_path = "logs/.tmp_intraday.csv"
    os.makedirs("logs", exist_ok=True)
    df_candles.to_csv(tmp_path, index=False)
    try:
        records = process_day(tmp_path)
    except Exception as e:
        print(f"{RED}Pipeline error: {e}{RESET}")
        return pd.DataFrame()
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["post_pattern_enc"] = df["post_pattern"].apply(
        lambda p: list(le.classes_).index(p) if p in le.classes_ else 0
    )
    X          = df[feat_cols].fillna(0)
    df["prob"] = model.predict_proba(X)[:, 1].round(4)
    df["pred"] = (df["prob"] >= threshold).astype(int)
    return df


def print_signal(row, is_new=True):
    ts      = f"{row['date']} {row['time'][:5]}"
    pred    = int(row["pred"])
    prob    = float(row["prob"])
    pattern = row["post_pattern"]
    if pred == 1:
        colour = GREEN
        action = f"{BOLD}{GREEN}▲ BUY{RESET}"
    elif prob > 0.25:
        colour = YELLOW
        action = f"{YELLOW}– skip (borderline){RESET}"
    else:
        colour = RESET
        action = "  skip"
    tag = "NEW  " if is_new else "     "
    print(
        f"\n{tag}[{ts}]  "
        f"Entry {colour}{row['entry']:>9.2f}{RESET}  "
        f"SL {row['sl']:>9.2f}  "
        f"T3 {row['tgt_1_3']:>9.2f}  "
        f"Risk {row['risk_pts']:>5.1f}pt  "
        f"Pattern {pattern:<18}  "
        f"Prob {colour}{prob:.3f}{RESET}  "
        f"{action}"
    )


def save_signals(df, date_str):
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = f"{LOG_DIR}/signals_{date_str}.csv"
    cols = ["date","time","entry","sl","tgt_1_3","risk_pts","post_pattern","prob","pred"]
    if os.path.exists(log_path):
        df[cols].to_csv(log_path, mode="a", header=False, index=False)
    else:
        df[cols].to_csv(log_path, index=False)


def in_market_hours():
    now = datetime.now()
    t   = now.hour * 60 + now.minute
    return (START_HOUR * 60 + START_MIN) <= t <= (END_HOUR * 60 + END_MIN)


def wait_for_market_open():
    now   = datetime.now()
    t     = now.hour * 60 + now.minute
    start = START_HOUR * 60 + START_MIN
    if t < start:
        secs = (start - t) * 60 - now.second
        print(f"\n{CYAN}Waiting for market window "
              f"({START_HOUR:02d}:{START_MIN:02d} IST)... "
              f"{secs//60}m {secs%60}s{RESET}\n")
        time.sleep(secs)


def main():
    print(f"\n{BOLD}{'='*65}{RESET}")
    print(f"{BOLD}  NIFTY BB+EMA9 LIVE SCANNER{RESET}")
    print(f"{BOLD}{'='*65}{RESET}")
    print(f"  Strategy  : BB(20,2) lower touch → EMA9 close → BUY")
    print(f"  Target    : 1:3 Risk-Reward")
    print(f"  Window    : {START_HOUR:02d}:{START_MIN:02d} – {END_HOUR:02d}:{END_MIN:02d} IST")
    print(f"  Poll rate : every {POLL_SECONDS}s")
    print(f"{BOLD}{'='*65}{RESET}\n")

    model, feat_cols, le, threshold = load_model(MODEL_PATH)
    wait_for_market_open()

    seen_signals = set()
    date_str     = datetime.now().strftime("%Y%m%d")
    buy_count    = 0
    skip_count   = 0

    print(f"\n{CYAN}Scanning... (Ctrl+C to stop){RESET}")
    print(f"{'─'*65}")

    while in_market_hours():
        try:
            df_candles = get_intraday_candles()

            if df_candles.empty:
                time.sleep(POLL_SECONDS)
                continue

            df_scored = score_signals(df_candles, model, feat_cols, le, threshold)

            if not df_scored.empty:
                for _, row in df_scored.iterrows():
                    sig_key = (row["date"], row["time"])
                    if sig_key not in seen_signals:
                        seen_signals.add(sig_key)
                        print_signal(row, is_new=True)
                        if int(row["pred"]) == 1:
                            buy_count += 1
                        else:
                            skip_count += 1

                save_signals(df_scored, date_str)

            now_str = datetime.now().strftime("%H:%M:%S")
            candles = len(df_candles)
            print(f"\r  [{now_str}]  Candles: {candles:>3}  |  "
                  f"Signals: {len(seen_signals):>2}  |  "
                  f"{GREEN}BUY: {buy_count}{RESET}  |  "
                  f"Skip: {skip_count}   ",
                  end="", flush=True)

        except KeyboardInterrupt:
            print(f"\n\n{YELLOW}Stopped.{RESET}")
            break
        except Exception as e:
            print(f"\n{RED}Error: {e}{RESET}")

        time.sleep(POLL_SECONDS)

    print(f"\n\n{BOLD}Session complete.{RESET}")
    log_path = f"{LOG_DIR}/signals_{date_str}.csv"
    if os.path.exists(log_path):
        print(f"\nAll signals logged to: {log_path}")
        df_log = pd.read_csv(log_path)
        print(df_log[["time","entry","sl","tgt_1_3","prob","pred"]].to_string(index=False))
    else:
        print("No signals detected today.")


if __name__ == "__main__":
    main()

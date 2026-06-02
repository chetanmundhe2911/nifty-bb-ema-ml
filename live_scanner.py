"""
live_scanner.py
---------------
Real-time NIFTY BB+EMA9 signal scanner — v2 model (pre-entry only).

Two-stage filtering:
  Stage 1 — ML model scores setup at entry (pre-entry features, threshold 0.50)
  Stage 2 — Pattern rule monitors bars 1-2 after entry for exit decision

Usage:
    python live_scanner.py

Saves on market close:
    data/raw/nifty_YYYYMMDD.parquet     full day 1-min candles
    logs/signals_YYYYMMDD.csv           all signals with scores
    logs/summary_YYYYMMDD.json          session summary
"""

import os
import sys
import time
import json
import pickle
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.upstox_feed import get_intraday_candles
from src.pipeline import process_day

load_dotenv()

# ── config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = "models/v2/signal_model_v2.pkl"
LOG_DIR      = "logs"
DATA_DIR     = "data/raw"
POLL_SECONDS = 60
START_HOUR   = 9
START_MIN    = 45
END_HOUR     = 14
END_MIN      = 30

# ── colours ───────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def load_model(path):
    bundle    = pickle.load(open(path, "rb"))
    model     = bundle["model"]
    feat_cols = bundle["feature_cols"]
    threshold = bundle["threshold"]
    print(f"{CYAN}Model     : {bundle['description']}{RESET}")
    print(f"{CYAN}CV AUC    : {bundle['cv_auc_mean']:.3f}  |  "
          f"Val AUC : {bundle['val_auc']:.3f}  |  "
          f"Threshold : {threshold:.2f}{RESET}")
    print(f"{CYAN}Features  : {len(feat_cols)} (look-ahead removed: "
          f"{bundle.get('lookahead_removed', False)}){RESET}")
    return model, feat_cols, threshold


def score_signals(df_candles, model, feat_cols, threshold):
    """Run feature pipeline + model. Returns scored DataFrame."""
    if df_candles.empty or len(df_candles) < 25:
        return pd.DataFrame()

    tmp = os.path.join(LOG_DIR, ".tmp_intraday.csv")
    os.makedirs(LOG_DIR, exist_ok=True)
    df_candles.to_csv(tmp, index=False)

    try:
        records = process_day(tmp)
    except Exception as e:
        print(f"\n{RED}Pipeline error: {e}{RESET}")
        return pd.DataFrame()

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Use only pre-entry features (v2 model has no look-ahead)
    X          = df[feat_cols].fillna(0)
    df["prob"] = model.predict_proba(X)[:, 1].round(4)
    df["pred"] = (df["prob"] >= threshold).astype(int)
    return df


def get_exit_guidance(pattern: str, prob: float) -> str:
    """
    Post-entry exit rule based on pattern.
    Called AFTER entry, bar by bar.
    """
    rules = {
        "A_clean":        f"{GREEN}HOLD → target  (bar 1 confirmed above EMA9){RESET}",
        "A_slight_delay": f"{GREEN}HOLD → watch bar 2  (pause then continue){RESET}",
        "B_wick_recover": f"{RED}EXIT NOW → wicked below SL (1.8% win rate){RESET}",
        "C_coil":         f"{YELLOW}HOLD → wait for breakout candle{RESET}",
        "D_sl_hit":       f"{RED}EXIT NOW → SL broken (4.6% win rate){RESET}",
        "E_unclear":      f"{YELLOW}WATCH → classify after bar 2{RESET}",
    }
    return rules.get(pattern, f"{YELLOW}WATCH{RESET}")


def print_signal(row, is_new=True):
    ts      = f"{row['date']} {row['time'][:5]}"
    pred    = int(row["pred"])
    prob    = float(row["prob"])
    pattern = row.get("post_pattern", "unknown")

    if pred == 1:
        colour = GREEN
        action = f"{BOLD}{GREEN}▲  BUY{RESET}"
    elif prob > 0.35:
        colour = YELLOW
        action = f"{YELLOW}─  skip (borderline){RESET}"
    else:
        colour = RESET
        action = "   skip"

    tag = "NEW  " if is_new else "     "

    print(
        f"\n{tag}[{ts}]  "
        f"Entry {colour}{row['entry']:>9.2f}{RESET}  "
        f"SL {row['sl']:>9.2f}  "
        f"T3 {row['tgt_1_3']:>9.2f}  "
        f"Risk {row['risk_pts']:>5.1f}pt  "
        f"Prob {colour}{prob:.3f}{RESET}  "
        f"{action}"
    )

    # Post-entry exit guidance for BUY signals
    if pred == 1:
        guidance = get_exit_guidance(pattern, prob)
        print(f"       Pattern: {pattern:<20}  Exit rule: {guidance}")


def save_signals_csv(df, date_str):
    os.makedirs(LOG_DIR, exist_ok=True)
    path = f"{LOG_DIR}/signals_{date_str}.csv"
    cols = ["date","time","entry","sl","tgt_1_3","risk_pts",
            "post_pattern","prob","pred"]
    df[cols].to_csv(
        path,
        mode   = "a" if os.path.exists(path) else "w",
        header = not os.path.exists(path),
        index  = False
    )


def save_day_parquet(df_candles, date_str):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = f"{DATA_DIR}/nifty_{date_str}.parquet"
    df_candles.to_parquet(path, index=False, compression="snappy")
    size_kb = os.path.getsize(path) / 1024
    print(f"  Candles  -> {path}  ({size_kb:.1f} KB, "
          f"{len(df_candles)} rows)")


def save_summary_json(signals, date_str, df_candles):
    os.makedirs(LOG_DIR, exist_ok=True)
    path  = f"{LOG_DIR}/summary_{date_str}.json"
    buys  = [s for s in signals if s["pred"] == 1]
    skips = [s for s in signals if s["pred"] == 0]

    summary = {
        "date":          date_str,
        "model_version": "v2",
        "day_open":      float(df_candles.iloc[0]["open"])   if not df_candles.empty else None,
        "day_high":      float(df_candles["high"].max())     if not df_candles.empty else None,
        "day_low":       float(df_candles["low"].min())      if not df_candles.empty else None,
        "day_close":     float(df_candles.iloc[-1]["close"]) if not df_candles.empty else None,
        "total_candles": len(df_candles),
        "total_signals": len(signals),
        "buy_calls":     len(buys),
        "skip_calls":    len(skips),
        "signals": [
            {
                "time":         s["time"],
                "entry":        s["entry"],
                "sl":           s["sl"],
                "tgt_1_3":      s["tgt_1_3"],
                "risk_pts":     s["risk_pts"],
                "post_pattern": s.get("post_pattern",""),
                "prob":         s["prob"],
                "pred":         s["pred"],
                "action":       "BUY" if s["pred"] == 1 else "skip",
                "exit_rule":    get_exit_guidance(
                                    s.get("post_pattern",""), s["prob"]
                                ).replace("\033[92m","").replace("\033[91m","")
                                 .replace("\033[93m","").replace("\033[0m","")
            }
            for s in signals
        ]
    }

    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    size_kb = os.path.getsize(path) / 1024
    print(f"  Summary  -> {path}  ({size_kb:.1f} KB)")


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
        print(f"\n{CYAN}Waiting for {START_HOUR:02d}:{START_MIN:02d} IST ... "
              f"{secs//60}m {secs%60}s{RESET}\n")
        time.sleep(secs)


def main():
    print(f"\n{BOLD}{'='*65}{RESET}")
    print(f"{BOLD}  NIFTY BB+EMA9 LIVE SCANNER  —  v2{RESET}")
    print(f"{BOLD}{'='*65}{RESET}")
    print(f"  Strategy  : BB(20,2) lower touch -> EMA9 close -> BUY")
    print(f"  Target    : 1:3 Risk-Reward")
    print(f"  Window    : {START_HOUR:02d}:{START_MIN:02d} - {END_HOUR:02d}:{END_MIN:02d} IST")
    print(f"  Poll rate : every {POLL_SECONDS}s")
    print(f"  Stage 1   : ML model (pre-entry, threshold 0.50)")
    print(f"  Stage 2   : Pattern exit rule (post-entry bar 1-2)")
    print(f"{BOLD}{'='*65}{RESET}\n")

    model, feat_cols, threshold = load_model(MODEL_PATH)
    wait_for_market_open()

    seen_signals = set()
    all_signals  = []
    date_str     = datetime.now().strftime("%Y%m%d")
    df_candles   = pd.DataFrame()
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

            df_scored = score_signals(df_candles, model, feat_cols, threshold)

            if not df_scored.empty:
                for _, row in df_scored.iterrows():
                    sig_key = (row["date"], row["time"])
                    if sig_key not in seen_signals:
                        seen_signals.add(sig_key)
                        print_signal(row, is_new=True)
                        all_signals.append(row.to_dict())
                        if int(row["pred"]) == 1:
                            buy_count += 1
                        else:
                            skip_count += 1
                        save_signals_csv(
                            df_scored[df_scored.apply(
                                lambda r: (r["date"], r["time"]) == sig_key,
                                axis=1
                            )], date_str
                        )

            now_str = datetime.now().strftime("%H:%M:%S")
            print(
                f"\r  [{now_str}]  "
                f"Candles: {len(df_candles):>3}  |  "
                f"Signals: {len(seen_signals):>2}  |  "
                f"{GREEN}BUY: {buy_count}{RESET}  |  "
                f"Skip: {skip_count}   ",
                end="", flush=True
            )

        except KeyboardInterrupt:
            print(f"\n\n{YELLOW}Stopped.{RESET}")
            break
        except Exception as e:
            print(f"\n{RED}Error: {e}{RESET}")

        time.sleep(POLL_SECONDS)

    # ── end of session ────────────────────────────────────────────────────────
    print(f"\n\n{BOLD}{'─'*65}{RESET}")
    print(f"{BOLD}Session complete — saving...{RESET}")

    if not df_candles.empty:
        save_day_parquet(df_candles, date_str)

    if all_signals:
        save_summary_json(all_signals, date_str, df_candles)

    print(f"\n{BOLD}Today's signals:{RESET}")
    if os.path.exists(f"{LOG_DIR}/signals_{date_str}.csv"):
        df_log = pd.read_csv(f"{LOG_DIR}/signals_{date_str}.csv")
        print(df_log[["time","entry","sl","tgt_1_3",
                      "post_pattern","prob","pred"]].to_string(index=False))
        print(f"\n  Total   : {len(all_signals)}")
        print(f"  {GREEN}BUY     : {buy_count}{RESET}")
        print(f"  Skip    : {skip_count}")
    else:
        print("  No signals today.")

    print(f"\n{BOLD}{'='*65}{RESET}\n")


if __name__ == "__main__":
    main()

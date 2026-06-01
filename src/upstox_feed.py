"""
upstox_feed.py
--------------
Fetch live and historical 1-min NIFTY 50 candles from Upstox API v2.
"""

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN")
BASE_URL     = "https://api.upstox.com/v2"
NIFTY_KEY    = "NSE_INDEX%7CNifty%2050"

HEADERS = {
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "Accept":        "application/json"
}


def _last_trading_day(lookback: int = 10) -> str:
    """
    Scan backwards up to lookback days to find the most recent
    date that returned candle data. Handles weekends AND holidays.
    """
    d = datetime.now()
    for _ in range(lookback):
        d -= timedelta(days=1)
        if d.weekday() >= 5:
            continue
        date_str = d.strftime("%Y-%m-%d")
        url = (f"{BASE_URL}/historical-candle/{NIFTY_KEY}"
               f"/1minute/{date_str}/{date_str}")
        r = requests.get(url, headers=HEADERS)
        if r.status_code == 200:
            candles = r.json().get("data", {}).get("candles", [])
            if candles:
                print(f"  Last trading day: {date_str} ({len(candles)} candles)")
                return date_str
    raise ValueError("No trading day found in last 10 days.")


def get_historical_candles(
    instrument_key: str = NIFTY_KEY,
    interval: str = "1minute",
    from_date: str = None,
    to_date: str   = None
) -> pd.DataFrame:
    """
    Fetch historical 1-min OHLC candles from Upstox.
    Defaults to last available trading day (auto-detected).
    Returns DataFrame matching training CSV schema exactly.
    """
    if not to_date:
        to_date = _last_trading_day()
    from_date = from_date or to_date

    url = (f"{BASE_URL}/historical-candle/{instrument_key}"
           f"/{interval}/{to_date}/{from_date}")
    print(f"  URL: {url}")
    r = requests.get(url, headers=HEADERS)

    if r.status_code != 200:
        print(f"  Error {r.status_code}: {r.text}")
        return pd.DataFrame()

    data = r.json().get("data", {}).get("candles", [])
    if not data:
        print("  No candle data returned.")
        return pd.DataFrame()

    return _parse_candles(data)


def get_intraday_candles() -> pd.DataFrame:
    """
    Fetch today's intraday 1-min candles.
    Use this during market hours (9:15 AM - 3:30 PM IST).
    """
    url = f"{BASE_URL}/historical-candle/intraday/{NIFTY_KEY}/1minute"
    r   = requests.get(url, headers=HEADERS)

    if r.status_code != 200:
        print(f"  Error {r.status_code}: {r.text}")
        return pd.DataFrame()

    data = r.json().get("data", {}).get("candles", [])
    if not data:
        print("  No intraday data yet.")
        return pd.DataFrame()

    return _parse_candles(data)


def _parse_candles(data: list) -> pd.DataFrame:
    """
    Parse raw Upstox candles into clean DataFrame.
    Upstox format : [timestamp, open, high, low, close, volume, oi]
    Output format : date, time, symbol, open, high, low, close
    """
    df = pd.DataFrame(
        data,
        columns=["timestamp","open","high","low","close","volume","oi"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"]      = df["timestamp"].dt.strftime("%Y-%m-%d")
    df["time"]      = df["timestamp"].dt.strftime("%H:%M:%S")
    df["symbol"]    = "NIFTY"
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df[["date","time","symbol","open","high","low","close"]]
    print(f"  {len(df)} candles | "
          f"{df.iloc[0]['date']} {df.iloc[0]['time']} → "
          f"{df.iloc[-1]['date']} {df.iloc[-1]['time']}")
    return df


def save_to_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
    print(f"  Saved → {path}")


if __name__ == "__main__":
    print("=" * 55)
    print("UPSTOX FEED — CONNECTION TEST")
    print("=" * 55)

    print("\nTest 1 — Historical candles (last trading day)")
    df = get_historical_candles()

    if not df.empty:
        print(f"\nFirst 5 candles:")
        print(df.head().to_string(index=False))
        print(f"\nLast 5 candles:")
        print(df.tail().to_string(index=False))

        # Schema validation
        expected = ["date","time","symbol","open","high","low","close"]
        assert list(df.columns) == expected, "Column mismatch!"
        print(f"\n  ✓ Schema matches training data")
        print(f"  ✓ {len(df)} candles (full session)")

        # Save it
        date_str  = df.iloc[0]["date"].replace("-","")
        save_path = f"data/raw/nifty_spot{date_str}.csv"
        save_to_csv(df, save_path)

    print(f"\nTest 2 — Intraday (market hours only)")
    df_live = get_intraday_candles()
    if df_live.empty:
        print("  Expected — market is closed right now.")

    print("\n" + "=" * 55)
    print("Feed ready.")
    print("=" * 55)

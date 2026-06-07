"""Binance REST feed for BTC 1-min candles."""
import requests
import pandas as pd

def get_candles(limit=300):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": "1m", "limit": limit}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    df = pd.DataFrame(r.json(), columns=[
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

if __name__ == "__main__":
    df = get_candles(10)
    print(f"Binance: {len(df)} candles | Latest: {df.iloc[-1]['datetime']} | Price: {df.iloc[-1]['C']}")

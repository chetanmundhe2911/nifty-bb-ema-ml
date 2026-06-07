"""Delta Exchange REST feed for BTC 1-min candles."""
import time
import requests
import pandas as pd

def get_candles(limit=300):
    end   = int(time.time())
    start = end - limit * 60
    url   = "https://api.delta.exchange/v2/history/candles"
    params = {"resolution": "1m", "symbol": "BTCUSDT", "start": start, "end": end}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    result = r.json().get("result", [])
    if not result:
        return pd.DataFrame()
    df = pd.DataFrame(result)
    df["datetime"] = pd.to_datetime(df["time"], unit="s")
    df["O"] = df["open"].astype(float)
    df["H"] = df["high"].astype(float)
    df["L"] = df["low"].astype(float)
    df["C"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df[["datetime","O","H","L","C","volume"]].sort_values("datetime").reset_index(drop=True)

if __name__ == "__main__":
    df = get_candles(10)
    print(f"Delta: {len(df)} candles | Latest: {df.iloc[-1]['datetime']} | Price: {df.iloc[-1]['C']}")

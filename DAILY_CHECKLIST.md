# Daily Operations Checklist

## Morning Routine (run every trading day)

### 1. SSH into AWS EC2
```bash
ssh -i "C:/Users/cheta/Downloads/nifty-scanner-key.pem" ubuntu@13.62.206.194
cd nifty-bb-ema-ml && source venv/bin/activate
```

---

### 2. Check BTC Scanner (24/7)
```bash
# Is it running?
ps aux | grep btc_scanner | grep -v grep

# Latest logs
tail -10 logs/btc_scanner.log

# If NOT running — restart
/home/ubuntu/run_btc_scanner.sh
```

**Expected output:**
```
[HH:MM:SS UTC]  Binance BTC $XX,XXX  |  BUY: X  |  Skip: X  |  Candles: 300
```

---

### 3. Check NIFTY Scanner (Mon-Fri 9:45 AM - 3:15 PM IST)
```bash
# Latest logs
tail -20 logs/cron.log

# Today's signals
cat logs/signals_$(date +%Y%m%d).csv 2>/dev/null || echo "No signals file yet"

# Today's summary
cat logs/summary_$(date +%Y%m%d).json 2>/dev/null || echo "No summary yet"
```

---

### 4. Check Disk Space
```bash
df -h /
```
**Warning if > 80% used.**

---

### 5. Exit Server
```bash
exit
```

---

## Local Machine Checks

### Check Yesterday's BTC Performance
```bash
cd ~/Downloads/nifty-bb-ema-ml/nifty-bb-ema-ml

python -c "
import pickle, pandas as pd
from strategies.btc.feeds.binance_feed import get_candles
from strategies.btc.pipeline import process_chunk

df = get_candles(1440)
print(f'Candles: {len(df)} | From: {df.iloc[0][\"datetime\"]} -> {df.iloc[-1][\"datetime\"]}')

records = process_chunk(df)
df_sig  = pd.DataFrame(records)
buys    = df_sig[(df_sig['signal_type']=='BUY') & (df_sig['risk_pct']>=0.15)]

b         = pickle.load(open('models/btc/v1/signal_model_btc_v1_rr3.pkl','rb'))
feat_cols = b['feature_cols']
X = pd.DataFrame(0, index=buys.index, columns=feat_cols)
for col in feat_cols:
    if col in buys.columns:
        X[col] = buys[col].fillna(0)

buys['prob'] = b['model'].predict_proba(X)[:,1].round(3)
buys['pred'] = (buys['prob'] >= 0.50).astype(int)

print(f'BUY calls: {buys.pred.sum()} / {len(buys)}')
print(buys[['datetime','entry','sl','risk_pct','post_pattern','prob','pred']].to_string(index=False))
"
```

---

### Check Yesterday's NIFTY Performance
```bash
python -c "
import pickle, pandas as pd
from src.upstox_feed import get_historical_candles, save_to_csv
from src.pipeline import process_day
import datetime

yesterday = (datetime.date.today() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
df = get_historical_candles(from_date=yesterday, to_date=yesterday)

if df.empty:
    print('No data — market may have been closed')
else:
    save_to_csv(df, f'data/nifty/raw/nifty_{yesterday}.csv')
    b         = pickle.load(open('models/nifty/v2/signal_model.pkl','rb'))
    recs      = process_day(f'data/nifty/raw/nifty_{yesterday}.csv')
    df_sig    = pd.DataFrame(recs)
    df_sig['prob'] = b['model'].predict_proba(df_sig[b['feature_cols']].fillna(0))[:,1].round(3)
    df_sig['pred'] = (df_sig['prob'] >= 0.50).astype(int)
    print(f'Date: {yesterday}')
    print(f'BUY calls: {df_sig.pred.sum()} / {len(df_sig)}')
    print(df_sig[['time','entry','sl','tgt_1_3','post_pattern','prob','pred']].to_string(index=False))
"
```

---

## Manual Signal Tracking Log

Update this table daily:

| Date | Market | Time | Entry | SL | Target | Prob | Pattern | Result | Notes |
|------|--------|------|-------|-----|--------|------|---------|--------|-------|
| - | NIFTY | - | - | - | - | - | - | - | - |
| - | BTC | - | - | - | - | - | - | - | - |

---

## Scanner Commands Reference

### Start/Stop BTC Scanner
```bash
# Start with auto-restart
nohup /home/ubuntu/run_btc_scanner.sh >> /dev/null 2>&1 &

# Stop
pkill -f btc_scanner

# Check status
ps aux | grep btc_scanner | grep -v grep
```

### Run BTC Scanner with Different Settings
```bash
# Default (Binance, threshold 0.50)
python strategies/btc/scanner.py

# Delta Exchange feed
python strategies/btc/scanner.py --feed delta

# Higher threshold (fewer but better signals)
python strategies/btc/scanner.py --threshold 0.70

# Delta + higher threshold
python strategies/btc/scanner.py --feed delta --threshold 0.65
```

### Run NIFTY Scanner Manually
```bash
python live_scanner.py
```

---

## Model Info

| Model | File | Val AUC | Threshold | RR |
|-------|------|---------|-----------|-----|
| NIFTY v2 | models/nifty/v2/signal_model.pkl | 0.893 | 0.50 | 1:3 |
| BTC v1 1:5 | models/btc/v1/signal_model_btc_v1.pkl | 0.842 | 0.55 | 1:5 |
| BTC v1 1:3 | models/btc/v1/signal_model_btc_v1_rr3.pkl | 0.842 | 0.50 | 1:3 |

---

## AWS Infrastructure

| Component | Details |
|-----------|---------|
| EC2 Instance | i-05332134996165ad4 |
| Elastic IP | 13.62.206.194 |
| Region | eu-north-1 (Stockholm) |
| Instance type | t3.micro |
| Key file | nifty-scanner-key.pem |
| EventBridge | DISABLED (EC2 runs 24/7) |

---

## Telegram Bot

| Item | Value |
|------|-------|
| Bot | @Daily2911_stock_scanner_bot |
| Alerts | BUY signals from both NIFTY and BTC |

---

## 30-Day Observation Rules

1. **No real money** until 30 days of paper trading complete
2. **Log every signal** — entry, SL, target, result
3. **Note SL buffer** — if price wicks below SL then recovers, note it
4. **Check patterns** — A_clean signals are most reliable
5. **After 30 days** — add 1-point SL buffer rule and evaluate


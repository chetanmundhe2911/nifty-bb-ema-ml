# nifty-bb-ema-ml

Systematic intraday strategy for NIFTY 50 using Bollinger Band lower-band touch + EMA9 close as a buy signal, filtered by a calibrated XGBoost model trained on 117 engineered features.

## Strategy Logic
Price low touches BB lower band (20,2)  →  signal ARMED
Next candle closes above EMA9           →  BUY triggered
Entry = signal candle close
SL    = signal candle low
Target = Entry + Risk × 3  (1:3 RR)
## Model

| | Value |
|--|--|
| Algorithm | XGBoost (Calibrated) + Random Forest |
| Features | 117 across 1m, 5m, 15m, 1H |
| Training data | NIFTY 50 · January 2020 · 22 days · 211 signals |
| CV AUC | 0.928 |
| Holdout AUC | 0.969 |
| Out-of-sample (Jun 2024) | 2/3 BUY calls correct |

## Repository Structure
nifty-bb-ema-ml/
├── data/
│   └── raw/               # daily 1-min CSVs (gitignored)
├── models/
│   └── v1/
│       └── signal_model.pkl
├── src/
│   ├── indicators.py      # EMA, BB, ATR, RSI etc.
│   ├── pipeline.py        # feature engineering — process_day()
│   ├── train.py           # model training script
│   └── predict.py         # load model + score live signals
├── docs/
│   └── feature_reference.md
└── requirements.txt
## Versions

| Version | Date | Description |
|---------|------|-------------|
| v1 | Jun 2025 | Baseline model trained on Jan 2020. 117 features. |

## Setup

```bash
git clone https://github.com/chetanmundhe2911/nifty-bb-ema-ml.git
cd nifty-bb-ema-ml
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

## Quick start

```python
import pickle
import pandas as pd
from src.pipeline import process_day

# Load model
bundle = pickle.load(open("models/v1/signal_model.pkl", "rb"))
model  = bundle["model"]
feats  = bundle["feature_cols"]

# Score a day
signals = process_day("data/raw/nifty_spot09_01_2020.csv")
df = pd.DataFrame(signals)
df["prob"] = model.predict_proba(df[feats].fillna(0))[:, 1]
print(df[["time", "entry", "sl", "tgt_1_3", "prob"]])
```

## Roadmap

- [ ] Collect Feb–Dec 2020 data and retrain (target 1,000+ signals)
- [ ] Add walk-forward validation
- [ ] Live Upstox API feed integration
- [ ] Pre-entry only model (remove look-ahead features)
- [ ] BankNifty extension

## Disclaimer

For research and educational purposes only. Not financial advice.

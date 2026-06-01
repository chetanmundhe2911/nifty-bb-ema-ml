"""
predict.py
----------
Load the trained model and score signals from a day file.

Usage:
    python -m src.predict data/raw/nifty_spot09_01_2020.csv
"""

import sys
import pickle
import pandas as pd
from src.pipeline import process_day


MODEL_PATH = "models/v1/signal_model.pkl"


def load_model(path: str = MODEL_PATH):
    bundle = pickle.load(open(path, "rb"))
    return bundle["model"], bundle["feature_cols"], bundle["threshold"]


def score_day(filepath: str, model_path: str = MODEL_PATH) -> pd.DataFrame:
    """
    Run the full pipeline on one day CSV and return scored signals.

    Returns a DataFrame with all features + prob + pred columns.
    """
    model, feat_cols, threshold = load_model(model_path)

    records = process_day(filepath)
    if not records:
        print("No signals detected.")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Encode post_pattern for model input
    bundle = pickle.load(open(model_path, "rb"))
    le     = bundle["label_encoder"]
    df["post_pattern_enc"] = df["post_pattern"].apply(
        lambda p: list(le.classes_).index(p) if p in le.classes_ else 0
    )

    X          = df[feat_cols].fillna(0)
    df["prob"] = model.predict_proba(X)[:, 1].round(4)
    df["pred"] = (df["prob"] >= threshold).astype(int)

    display_cols = ["date","time","entry","sl","tgt_1_3","risk_pts",
                    "post_pattern","prob","pred","label"]
    print(df[display_cols].to_string(index=False))
    print(f"\nSignals: {len(df)} | BUY calls: {df['pred'].sum()} | "
          f"Actual wins: {df['label'].sum()}")
    return df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.predict <path_to_csv>")
        sys.exit(1)
    score_day(sys.argv[1])

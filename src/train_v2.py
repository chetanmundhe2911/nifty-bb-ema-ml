"""
train_v2.py
-----------
Train v2 model on full 5-year dataset (2020-2023).
Key changes from v1:
  - Trained on 10,909 signals vs 211
  - Post-signal look-ahead features REMOVED
  - Pre-entry only — honest, deployable live
  - Train 2020-2023, validate on full 2024
  - Threshold tuned on validation set
"""

import os
import pickle
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (roc_auc_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
import xgboost as xgb

# ── POST-SIGNAL LOOK-AHEAD FEATURES TO REMOVE ────────────────────────────────
LOOKAHEAD_FEATURES = [
    "post_pattern", "post_pattern_enc",
    "post_A_clean", "post_A_delay", "post_B_wick", "post_C_coil",
    "broke_sl", "broke_sl_recovered",
    "bars_to_ema9", "consec_above_ema9",
    "max_dd_atr", "n_small_candles", "range_compress",
    "breakout_bar", "breakout_str", "pts5bars_atr",
]

# ── METADATA (not features) ───────────────────────────────────────────────────
META_COLS = [
    "date", "time", "entry", "sl", "tgt_1_3", "risk_pts", "label",
    "5m_prev_high", "5m_prev_low", "15m_prev_high", "15m_prev_low",
    "1h_prev_high", "1h_prev_low",
]

REMOVE_COLS = set(LOOKAHEAD_FEATURES + META_COLS)


def load_dataset(path):
    df = pd.read_csv(path)
    print(f"Loaded: {path}")
    print(f"  Shape   : {df.shape}")
    print(f"  Wins    : {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
    print(f"  Dates   : {df['date'].min()} -> {df['date'].max()}")
    return df


def get_feature_cols(df):
    feat_cols = [c for c in df.columns if c not in REMOVE_COLS]
    removed   = [c for c in LOOKAHEAD_FEATURES if c in df.columns]
    print(f"\nFeatures after removing look-ahead : {len(feat_cols)}")
    print(f"Look-ahead features removed        : {len(removed)}")
    return feat_cols


def tune_threshold(y_true, y_prob):
    best_f1, best_t = 0, 0.50
    print(f"\n  {'Thresh':>7}  {'F1':>6}  {'Prec':>6}  {'Rec':>6}  {'BUY calls':>10}")
    print(f"  {'-'*50}")
    for t in np.arange(0.30, 0.86, 0.05):
        preds = (y_prob >= t).astype(int)
        f1    = f1_score(y_true, preds, zero_division=0)
        prec  = precision_score(y_true, preds, zero_division=0)
        rec   = recall_score(y_true, preds, zero_division=0)
        buys  = (preds==1).sum()
        print(f"  {t:>7.2f}  {f1:>6.3f}  {prec:>6.3f}  {rec:>6.3f}  {buys:>5}/{len(preds)}")
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"\n  Best threshold: {best_t:.2f}  (F1={best_f1:.3f})")
    return best_t


def evaluate(y_true, y_prob, threshold, label):
    preds = (y_prob >= threshold).astype(int)
    cm    = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    baseline = y_true.mean()
    model_wr = tp/(tp+fp) if (tp+fp) > 0 else 0
    print(f"\n{'='*55}")
    print(f"EVALUATION — {label}")
    print(f"{'='*55}")
    print(f"  AUC            : {roc_auc_score(y_true, y_prob):.3f}")
    print(f"  Precision      : {precision_score(y_true, preds, zero_division=0):.3f}")
    print(f"  Recall         : {recall_score(y_true, preds, zero_division=0):.3f}")
    print(f"  F1             : {f1_score(y_true, preds, zero_division=0):.3f}")
    print(f"  BUY calls      : {(preds==1).sum()}/{len(preds)}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Baseline WR    : {baseline*100:.1f}%")
    print(f"  Model WR       : {model_wr*100:.1f}%")
    print(f"  Lift           : {(model_wr/baseline):.2f}x baseline")


def main():
    # ── Load ──────────────────────────────────────────────────────────────────
    print("="*55)
    print("v2 TRAINING — PRE-ENTRY ONLY MODEL")
    print("="*55)

    master = load_dataset("data/datasets/signals_master.csv")
    val    = load_dataset("data/datasets/signals_2024.csv")

    feat_cols = get_feature_cols(master)

    # ── Split: train 2020-2023, validate 2024 ─────────────────────────────────
    train = master[pd.to_datetime(master["date"]).dt.year < 2024].copy()
    print(f"\nTrain (2020-2023) : {len(train)} signals | WR {train['label'].mean()*100:.1f}%")
    print(f"Val   (2024)      : {len(val)} signals   | WR {val['label'].mean()*100:.1f}%")

    X_train = train[feat_cols].fillna(0)
    y_train = train["label"]
    X_val   = val[feat_cols].fillna(0)
    y_val   = val["label"]

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("TRAINING XGBoost (500 trees, pre-entry features only)")
    print(f"{'='*55}")

    scale_pw = (y_train==0).sum() / (y_train==1).sum()
    print(f"scale_pos_weight : {scale_pw:.2f}")

    model = xgb.XGBClassifier(
        n_estimators     = 500,
        max_depth        = 5,
        learning_rate    = 0.03,
        subsample        = 0.8,
        colsample_bytree = 0.75,
        min_child_weight = 5,
        gamma            = 0.1,
        reg_alpha        = 0.1,
        reg_lambda       = 1.0,
        scale_pos_weight = scale_pw,
        objective        = "binary:logistic",
        eval_metric      = "logloss",
        random_state     = 42,
        verbosity        = 0,
    )

    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"\n5-fold CV AUC : {auc.mean():.3f} +/- {auc.std():.3f}")
    print(f"Folds         : {auc.round(3)}")

    model.fit(X_train, y_train)

    # ── Threshold tuning on 2024 ──────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("THRESHOLD TUNING ON 2024 VALIDATION SET")
    print(f"{'='*55}")
    val_probs = model.predict_proba(X_val)[:, 1]
    best_t    = tune_threshold(y_val, val_probs)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    train_probs = model.predict_proba(X_train)[:, 1]
    evaluate(y_train, train_probs, best_t, f"TRAIN 2020-2023  (t={best_t:.2f})")
    evaluate(y_val,   val_probs,   best_t, f"VAL   2024       (t={best_t:.2f})")

    # ── Feature importance ────────────────────────────────────────────────────
    fi = pd.Series(model.feature_importances_, index=feat_cols)
    fi = fi.sort_values(ascending=False)
    print(f"\n{'='*55}")
    print("TOP 20 FEATURES (pre-entry only)")
    print(f"{'='*55}")
    print(fi.head(20).round(4).to_string())

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs("models/v2", exist_ok=True)
    bundle = {
        "model":              model,
        "feature_cols":       feat_cols,
        "threshold":          best_t,
        "cv_auc_mean":        float(auc.mean()),
        "val_auc":            float(roc_auc_score(y_val, val_probs)),
        "feature_importance": fi.to_dict(),
        "train_years":        "2020-2023",
        "val_year":           "2024",
        "n_train_signals":    int(len(X_train)),
        "n_val_signals":      int(len(X_val)),
        "lookahead_removed":  True,
        "description":        "XGBoost v2 — pre-entry only — 2020-2023 train — 2024 val",
    }
    pickle.dump(bundle, open("models/v2/signal_model_v2.pkl", "wb"))

    fi.reset_index().rename(
        columns={"index":"feature", 0:"importance"}
    ).to_csv("models/v2/feature_importance_v2.csv", index=False)

    print(f"\n{'='*55}")
    print("SAVED")
    print(f"{'='*55}")
    print(f"  models/v2/signal_model_v2.pkl")
    print(f"  models/v2/feature_importance_v2.csv")
    print(f"\n  Train signals  : {len(X_train)}")
    print(f"  Val signals    : {len(X_val)}")
    print(f"  CV AUC         : {auc.mean():.3f}")
    print(f"  Val AUC        : {roc_auc_score(y_val, val_probs):.3f}")
    print(f"  Threshold      : {best_t:.2f}")
    print(f"  Look-ahead     : REMOVED")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()

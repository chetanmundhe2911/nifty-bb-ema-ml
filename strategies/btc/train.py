"""
strategies/btc/train.py
-----------------------
Train BTC BB+EMA9 XGBoost model (pre-entry only).

Train  : 2019-2021
Validate: 2022
"""

import os, pickle
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_auc_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
import xgboost as xgb

# ── LOOK-AHEAD FEATURES TO REMOVE ────────────────────────────────────────────
LOOKAHEAD = [
    "post_pattern", "post_A_clean", "post_A_delay", "post_B_wick",
    "post_C_coil", "broke_sl", "broke_sl_recovered",
    "bars_to_ema9", "consec_above_ema9", "max_dd_atr",
    "n_small_candles", "range_compress", "breakout_bar",
    "breakout_str", "pts5bars_atr",
]

META = [
    "datetime", "signal_type", "signal_direction",
    "entry", "sl", "target_1_5", "target_1_3",
    "risk_pts", "risk_pct", "label", "label_1_3", "label_1_5",
    "year",  # ← add this
]

REMOVE = set(LOOKAHEAD + META)


def tune_threshold(y_true, y_prob):
    best_f1, best_t = 0, 0.50
    print(f"\n  {'Thresh':>7}  {'F1':>6}  {'Prec':>6}  {'Rec':>6}  {'BUY calls':>10}  {'WR':>6}")
    print(f"  {'-'*55}")
    for t in np.arange(0.30, 0.86, 0.05):
        preds = (y_prob >= t).astype(int)
        f1    = f1_score(y_true, preds, zero_division=0)
        prec  = precision_score(y_true, preds, zero_division=0)
        rec   = recall_score(y_true, preds, zero_division=0)
        buys  = (preds==1).sum()
        wr    = prec * 100
        print(f"  {t:>7.2f}  {f1:>6.3f}  {prec:>6.3f}  {rec:>6.3f}  {buys:>5}/{len(preds)}  {wr:>5.1f}%")
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"\n  Best threshold: {best_t:.2f}  (F1={best_f1:.3f})")
    return best_t


def evaluate(y_true, y_prob, threshold, label):
    preds = (y_prob >= threshold).astype(int)
    cm    = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel() if cm.size==4 else (0,0,0,0)
    baseline = y_true.mean()
    model_wr = tp/(tp+fp) if (tp+fp)>0 else 0
    print(f"\n{'='*55}")
    print(f"EVALUATION — {label}")
    print(f"{'='*55}")
    print(f"  AUC         : {roc_auc_score(y_true, y_prob):.3f}")
    print(f"  Precision   : {precision_score(y_true, preds, zero_division=0):.3f}")
    print(f"  Recall      : {recall_score(y_true, preds, zero_division=0):.3f}")
    print(f"  F1          : {f1_score(y_true, preds, zero_division=0):.3f}")
    print(f"  BUY calls   : {(preds==1).sum()}/{len(preds)}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Baseline WR : {baseline*100:.1f}%")
    print(f"  Model WR    : {model_wr*100:.1f}%")
    print(f"  Lift        : {(model_wr/baseline):.2f}x baseline")


def main():
    print("="*55)
    print("BTC v1 TRAINING — PRE-ENTRY ONLY MODEL")
    print("="*55)

    df = pd.read_csv("data/btc/datasets/signals_master_btc.csv")
    print(f"Loaded: {len(df):,} signals | WR {df.label.mean()*100:.1f}%")

    # Feature columns
    feat_cols = [c for c in df.columns if c not in REMOVE]
    removed   = [c for c in LOOKAHEAD if c in df.columns]
    print(f"Features           : {len(feat_cols)}")
    print(f"Look-ahead removed : {len(removed)}")

    # Train 2019-2021 | Validate 2022
    df['year'] = pd.to_datetime(df['datetime']).dt.year
    train = df[df['year'] <= 2021].copy()
    val   = df[df['year'] == 2022].copy()

    print(f"\nTrain (2019-2021) : {len(train):,} signals | WR {train.label.mean()*100:.1f}%")
    print(f"Val   (2022)      : {len(val):,} signals   | WR {val.label.mean()*100:.1f}%")

    X_train = train[feat_cols].fillna(0)
    y_train = train["label"]
    X_val   = val[feat_cols].fillna(0)
    y_val   = val["label"]

    # Train
    print(f"\n{'='*55}")
    print("TRAINING XGBoost")
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

    # Threshold tuning
    print(f"\n{'='*55}")
    print("THRESHOLD TUNING ON 2022 VALIDATION SET")
    print(f"{'='*55}")
    val_probs = model.predict_proba(X_val)[:, 1]
    best_t    = tune_threshold(y_val, val_probs)

    # Evaluate
    train_probs = model.predict_proba(X_train)[:, 1]
    evaluate(y_train, train_probs, best_t, f"TRAIN 2019-2021  (t={best_t:.2f})")
    evaluate(y_val,   val_probs,   best_t, f"VAL   2022       (t={best_t:.2f})")

    # Feature importance
    fi = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)
    print(f"\n{'='*55}")
    print("TOP 20 FEATURES")
    print(f"{'='*55}")
    print(fi.head(20).round(4).to_string())

    # Save
    os.makedirs("models/btc/v1", exist_ok=True)
    bundle = {
        "model":           model,
        "feature_cols":    feat_cols,
        "threshold":       best_t,
        "cv_auc_mean":     float(auc.mean()),
        "val_auc":         float(roc_auc_score(y_val, val_probs)),
        "feature_importance": fi.to_dict(),
        "train_years":     "2019-2021",
        "val_year":        "2022",
        "n_train":         int(len(X_train)),
        "n_val":           int(len(X_val)),
        "lookahead_removed": True,
        "rr":              "1:5",
        "description":     "XGBoost BTC v1 — pre-entry only — 2019-2021 train — 2022 val",
    }
    pickle.dump(bundle, open("models/btc/v1/signal_model_btc_v1.pkl","wb"))
    fi.reset_index().rename(columns={"index":"feature",0:"importance"}).to_csv(
        "models/btc/v1/feature_importance_btc_v1.csv", index=False)

    print(f"\n{'='*55}")
    print("SAVED")
    print(f"{'='*55}")
    print(f"  models/btc/v1/signal_model_btc_v1.pkl")
    print(f"  Train signals : {len(X_train):,}")
    print(f"  Val signals   : {len(X_val):,}")
    print(f"  CV AUC        : {auc.mean():.3f}")
    print(f"  Val AUC       : {roc_auc_score(y_val, val_probs):.3f}")
    print(f"  Threshold     : {best_t:.2f}")
    print(f"  Look-ahead    : REMOVED")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()

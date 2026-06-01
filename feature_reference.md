# VeltrixAI — Feature Reference

**Strategy:** NIFTY 50 · BB (20,2) + EMA9 Bounce · 1:3 Risk-Reward  
**Model:** XGBoost (Calibrated) · Trained Jan 2020 · 117 features · 4 timeframes  
**Signal logic:** Price touches BB lower band → next candle closes above EMA9 → BUY

---

## Table of Contents

1. [Top 20 Most Important Features](#top-20-most-important-features)
2. [Feature Groups Overview](#feature-groups-overview)
3. [1-Minute Features](#1-minute-features-38-features)
4. [Multi-Timeframe Features](#multi-timeframe-features-63-features)
5. [Post-Signal Confirmation Features](#post-signal-confirmation-features-15-features)
6. [Regime Features](#regime-features-5-features)
7. [Time Features](#time-features-3-features)
8. [Cross-TF Confluence Features](#cross-tf-confluence-features-7-features)
9. [Signal Metadata](#signal-metadata-not-used-as-model-features)
10. [Important Notes for Live Trading](#important-notes-for-live-trading)

---

## Top 20 Most Important Features

Ranked by combined XGBoost + Random Forest importance score. Higher = more predictive.

| Rank | Feature | Group | Why It Matters |
|------|---------|-------|----------------|
| 1 | `broke_sl` | Post-signal | Whether the candle after entry broke the signal low. If yes → near-certain loss. Single most predictive feature. |
| 2 | `max_dd_atr` | Post-signal | Maximum drawdown in the 10 bars after entry, normalised by ATR. Deep drawdowns = trade failing. |
| 3 | `pts5bars_atr` | Post-signal | Points gained in 5 bars after entry, ATR-normalised. Captures early momentum — winners move immediately. |
| 4 | `post_pattern_enc` | Post-signal | Encoded post-signal candle pattern (A_clean, A_delay, B_wick, C_coil, D_sl_hit). Most predictive categorical feature. |
| 5 | `consec_above_ema9` | Post-signal | Consecutive candles after entry that close above EMA9. More consecutive = stronger hold = higher win probability. |
| 6 | `1h_close_pos` | 1H TF | Close position within the hourly candle range (0=bottom, 1=top). Bullish 1H structure supports the signal. |
| 7 | `1h_bb_pct_b` | 1H TF | 1H Bollinger Band %B position (0=lower, 1=upper). Low value = room to run upward. |
| 8 | `1m_ret_1` | 1-Min | 1-bar return % at signal candle. Momentum at trigger moment. |
| 9 | `1m_range_vs_atr` | 1-Min | Signal candle range divided by ATR14. Strong candles (>1×ATR) confirm conviction. |
| 10 | `1h_close_vs_ema9` | 1H TF | Distance of 1H close from EMA9. Positive = 1H trend supporting the 1-min signal. |
| 11 | `1m_effort` | 1-Min | Signal candle range vs average range of last 10 candles. High effort = unusual energy = genuine breakout. |
| 12 | `bars_to_ema9` | Post-signal | How many bars until the first close above EMA9 after entry. Fewer bars = cleaner setup. |
| 13 | `1h_ema9_vs_ema21` | 1H TF | 1H EMA9 minus EMA21. Positive = 1H uptrend. Confirms macro direction. |
| 14 | `tf_conf` | Confluence | Multi-TF signal confluence score (0–3). How many of 5m/15m/1H also fired BB+EMA9 signal. |
| 15 | `5m_bb_pct_b` | 5M TF | 5M Bollinger Band %B. Low = price near lower band on 5-min = oversold momentum aligning. |
| 16 | `1m_lwr_ratio` | 1-Min | Lower wick as fraction of total candle range. Large lower wick = rejection of lows = buy pressure. |
| 17 | `5m_rsi14` | 5M TF | 5M RSI14. Below 40 = momentum supportive of bounce. Above 70 = avoid. |
| 18 | `tr_align` | Confluence | Trend alignment score (0–3). How many TFs have EMA9 > EMA21. 3/3 = strong uptrend on all TFs. |
| 19 | `1m_close_pos` | 1-Min | Signal candle close position within its own range (0=bottom, 1=top). Close near top = bullish. |
| 20 | `session` | Time | Session bucket (0=open, 1=morning, 2=midday, 3=afternoon, 4=close). Best setups in sessions 1–3. |

> **Key insight:** The top 5 features are all post-signal (look-ahead). They describe what happens in the 10 minutes *after* entry — not before. This means the model is best used as an **exit manager** in live trading. For pure pre-entry filtering, rely on features ranked 6–20.

---

## Feature Groups Overview

| Group | Count | Timeframe | Purpose |
|-------|-------|-----------|---------|
| BB Indicators | 6 | 1-min | Band position, width, squeeze detection |
| EMA / Trend | 9 | 1-min | EMA alignment, slope, distance |
| Momentum / Oscillators | 9 | 1-min | RSI, Stochastic, Williams %R, ADX, MFI |
| Candle Structure | 8 | 1-min | Body, wick, range, effort |
| Multi-TF Context | 63 | 5m + 15m + 1H | Same 21 indicators × 3 higher timeframes |
| Post-Signal Confirmation | 15 | 1-min | What happens in 10 bars after entry |
| Regime | 5 | Daily | Day context — open direction, range |
| Time | 3 | Intraday | Session, minutes from open/close |
| Cross-TF Confluence | 7 | Multi | Combined TF signal and trend scores |
| **Total** | **117** | | |

---

## 1-Minute Features (38 features)

### Bollinger Band Features (6)

| Feature | Formula | Range | Interpretation |
|---------|---------|-------|----------------|
| `1m_bb_pct_b` | `(Close - BB_lower) / (BB_upper - BB_lower)` | 0–1+ | 0 = at lower band, 1 = at upper band. Signal fires when this was ≤0 recently. |
| `1m_bb_width` | `(BB_upper - BB_lower) / BB_mid` | 0–∞ | Wider = more volatile environment. Narrow = squeeze, potential breakout. |
| `1m_bb_lbr` | `(Low - BB_lower) / ATR14` | negative–0 | How far the low penetrated the lower band in ATR units. Deeper breach = more oversold. |
| `1m_bb_upper` | BB upper band price | Price | Upper band level (reference, not used as direct model input). |
| `1m_bb_lower` | BB lower band price | Price | Lower band level — the trigger zone for signal arming. |
| `1m_bb_mid` | 20-period SMA of close | Price | Middle band — acts as resistance/support midpoint. |

### EMA / Trend Features (9)

| Feature | Formula | Interpretation |
|---------|---------|----------------|
| `1m_ema9_vs_21` | `EMA9 - EMA21` | Positive = short-term uptrend on 1-min. Signal requires close > EMA9. |
| `1m_ema21_vs_50` | `EMA21 - EMA50` | Positive = medium-term 1-min trend is bullish. |
| `1m_close_vs_ema9` | `Close - EMA9` | Distance of close from EMA9. Positive at signal candle = trigger confirmed. |
| `1m_close_vs_ema21` | `Close - EMA21` | Broader trend context. |
| `1m_ema9_slope` | `EMA9[i] - EMA9[i-3]` | Positive = EMA9 rising. Steeper slope = stronger momentum. |
| `1m_ema9` | EMA(Close, 9) | The trigger EMA itself. |
| `1m_ema21` | EMA(Close, 21) | Intermediate trend reference. |
| `1m_ema50` | EMA(Close, 50) | Slow trend reference. |
| `1m_hh_hl` | `1 if High > High[3] and Low > Low[3]` | Binary. 1 = higher high AND higher low — classic uptrend structure on 1-min. |

### Momentum / Oscillator Features (9)

| Feature | Formula | Range | Interpretation |
|---------|---------|-------|----------------|
| `1m_rsi14` | RSI(14) of close | 0–100 | Below 40 = oversold, supportive of bounce. Above 70 = overbought, risky. |
| `1m_stoch_k` | `100 × (Close - Low14) / (High14 - Low14)` | 0–100 | Below 20 = oversold. Fast stochastic K line. |
| `1m_will_r` | `-100 × (High14 - Close) / (High14 - Low14)` | -100–0 | Below -80 = oversold, potential reversal zone. |
| `1m_mom10` | `Close - Close[10]` | Price diff | Raw 10-bar momentum. Positive = price rising over last 10 minutes. |
| `1m_adx` | Average Directional Index (14) | 0–100 | Above 25 = trending. Strong ADX at signal = directional move likely. |
| `1m_plus_di` | +DI from ADX calculation | 0–100 | Positive directional indicator. +DI > -DI = upward pressure dominant. |
| `1m_minus_di` | -DI from ADX calculation | 0–100 | Negative directional indicator. -DI > +DI = downward pressure dominant. |
| `1m_ret_1` | `(Close[i] / Close[i-1] - 1) × 100` | % | 1-bar return % at signal candle. Positive = signal candle closed up. |
| `1m_atr_exp` | `1 if ATR14 > ATR14[5]` | 0/1 | ATR expanding = volatility increasing. May indicate stronger follow-through. |

### Candle Structure Features (8)

| Feature | Formula | Range | Interpretation |
|---------|---------|-------|----------------|
| `1m_body_ratio` | `abs(Close - Open) / (High - Low)` | 0–1 | Close to 1 = full body candle (decisive). Close to 0 = doji (indecision). |
| `1m_lwr_ratio` | `min(Open, Close) - Low) / (High - Low)` | 0–1 | Fraction of candle that is lower wick. High lower wick = rejection of lows. |
| `1m_is_bull` | `1 if Close >= Open` | 0/1 | Binary. Signal candle must be bullish (close ≥ open) for clean A_clean pattern. |
| `1m_range_vs_atr` | `(High - Low) / ATR14` | 0–∞ | Candle range relative to recent average. >1.0 = above-average range = conviction. |
| `1m_close_pos` | `(Close - Low) / (High - Low)` | 0–1 | Where close sits within the candle. >0.6 = closed near highs = bullish. |
| `1m_effort` | `(High - Low) / rolling_mean(High-Low, 10)` | 0–∞ | Candle effort vs recent average. High effort at signal = genuine participation. |
| `1m_vol5` | `std(ret_1, 5)` | % | 5-bar return volatility. Low vol + good structure = cleaner setup. |
| `1m_ret3` / `1m_ret5` | 3-bar and 5-bar returns | % | Shorter and medium momentum into the signal. |

### Distance / Structure Features (6)

| Feature | Formula | Interpretation |
|---------|---------|----------------|
| `1m_d5l` | `(Close - Low5) / ATR14` | Distance from 5-bar low in ATR units. Low value = near recent support. |
| `1m_d5h` | `(Close - High5) / ATR14` | Distance from 5-bar high. Negative = below recent high = room to run. |
| `1m_d20l` | `(Close - Low20) / ATR14` | Distance from 20-bar low. Broader support context. |
| `1m_d20h` | `(Close - High20) / ATR14` | Distance from 20-bar high. Negative = room to previous resistance. |
| `1m_atr14` | Exponential ATR(14) | Absolute ATR value. Used to normalise other features. |
| `1m_bb_lbr` | `(Low - BB_lower) / ATR14` | How deeply price wicked below the lower band. |

---

## Multi-Timeframe Features (63 features)

The same 21 indicators are computed on three higher timeframes and prefixed accordingly. Each timeframe adds context about the larger trend and structure at the moment the 1-min signal fires.

**Prefixes:** `5m_` (5-minute), `15m_` (15-minute), `1h_` (1-hour)

### Features computed on each timeframe (21 each × 3 = 63)

| Feature Suffix | Definition |
|---------------|------------|
| `_ema9_vs_ema21` | EMA9 minus EMA21. Positive = uptrend on that TF. |
| `_ema21_vs_ema50` | EMA21 minus EMA50. Longer-term trend direction. |
| `_close_vs_ema9` | Close minus EMA9. Positive = price above short EMA. |
| `_ema9_slope` | Rate of change of EMA9 over last 3 bars. |
| `_bb_pct_b` | %B position within Bollinger Bands. Low = near lower band. |
| `_bb_width` | Band width normalised by midpoint. |
| `_bb_lower_breach_atr` | How far low breached lower band, ATR-normalised. |
| `_rsi14` | 14-period RSI. Oversold (<40) or overbought (>70). |
| `_stoch_k` | Fast stochastic %K. |
| `_will_r` | Williams %R. |
| `_is_bullish` | Binary — last completed candle is bullish. |
| `_body_ratio` | Candle body fraction of total range. |
| `_lower_wick_ratio` | Lower wick fraction — rejection of lows. |
| `_range_vs_atr` | Candle range vs ATR. |
| `_close_pos` | Close position within candle range. |
| `_signal_active` | Binary — did BB+EMA9 signal also fire on this TF? |
| `_dist_to_prev_high_atr` | Distance to prior bar high, ATR-normalised. |
| `_dist_to_prev_low_atr` | Distance to prior bar low, ATR-normalised. |
| `_room_to_tgt3` | Binary — 1 if 1:3 target is below prior TF high (room to run). |
| `_adx` | ADX trend strength on that TF. |
| `_atr14` | ATR14 on that TF (absolute volatility reference). |

### Why multi-timeframe matters

A signal that fires on 1-min but occurs when the 1H chart is overbought, near resistance, or in a downtrend has a much lower hit rate. The higher TF features capture this structural context without requiring manual inspection.

---

## Post-Signal Confirmation Features (15 features)

> **Critical warning:** These features use data from the 10 candles *after* entry. They are look-ahead features — unavailable at entry time in live trading. They describe what the trade actually does after you enter.
>
> **Use case:** Live exit management (when to cut a trade early vs hold), not pre-entry filtering.

| Feature | Definition | Interpretation |
|---------|-----------|----------------|
| `post_pattern` | Categorical: A_clean, A_slight_delay, B_wick_recover, C_coil, D_sl_hit, E_unclear | Describes the shape of price action after entry. A_clean = immediate follow-through. D_sl_hit = immediate loss. |
| `post_pattern_enc` | Integer encoding of post_pattern (0–5) | Model input version of the pattern. |
| `post_A_clean` | Binary — 1 if pattern is A_clean | Next candle immediately closes above EMA9 and holds. Highest win rate. |
| `post_A_delay` | Binary — 1 if pattern is A_slight_delay | EMA9 close occurs on bar 2 instead of bar 1. Still good. |
| `post_B_wick` | Binary — 1 if pattern is B_wick_recover | Price wicks below signal low but recovers. Borderline. |
| `post_C_coil` | Binary — 1 if pattern is C_coil | Multiple small tight candles after entry — coiling before a move. |
| `broke_sl` | Binary — 1 if any of bars 1–10 close below signal candle low | **#1 predictor.** If next candle breaks below entry candle low → exit immediately. |
| `broke_sl_recovered` | Binary — 1 if SL was broken but price recovered above EMA9 | Whipsaw pattern. Rare but can still be saved. |
| `bars_to_ema9` | Integer — how many bars until first close above EMA9 | Fewer = cleaner. If >5, setup is stalling. |
| `consec_above_ema9` | Integer — consecutive bars above EMA9 after entry | More consecutive = stronger hold = more likely to reach target. |
| `max_dd_atr` | Max drawdown in 10 bars / ATR14 | How deeply price dipped below entry in ATR units. >1.0 ATR = trade in trouble. |
| `n_small_candles` | Count of candles with range < 0.6× ATR in 10 bars | Many tiny candles = compression = indecision. |
| `range_compress` | Mean range of bars 1–5 / ATR14 | <0.7 = range contracting = coil building. |
| `breakout_bar` | Bar number of first strong breakout candle (range > 1.2× ATR, bullish) | Identifies which bar the trade accelerated. 0 = no breakout. |
| `breakout_str` | Breakout candle range / ATR14 | Strength of breakout candle. Higher = more conviction. |
| `pts5bars_atr` | `(max_high_bars_1_to_5 - entry) / ATR14` | Points gained in first 5 bars, ATR-normalised. The faster it moves up, the more likely it reaches 1:3. |

### Post-signal pattern guide

| Pattern | Description | Win Rate (training) | Action in live |
|---------|-------------|-------------------|----------------|
| `A_clean` | Next candle closes above EMA9 cleanly. Holds above. | ~85% | Hold to target |
| `A_slight_delay` | EMA9 close on bar 2, not bar 1. Brief pause then continues. | ~70% | Hold — watch bar 3 |
| `B_wick_recover` | Price wicks below entry candle low but recovers above EMA9. | ~45% | Tight stop — consider exiting |
| `C_coil` | 3+ small candles after entry. Range contracting. | ~55% | Wait for breakout candle |
| `D_sl_hit` | Next candle hits stop loss. | ~5% | Already stopped out |
| `E_unclear` | None of the above. | ~30% | Default caution |

---

## Regime Features (5 features)

These capture the broader day context at the time the signal fires.

| Feature | Definition | Interpretation |
|---------|-----------|----------------|
| `open30_dir` | `1 if close_of_bar_at_9:45 > day_open` | Whether the first 30 minutes closed bullish. Most reliable signals fire when the opening half-hour is up. |
| `open30_rng_atr` | `(open30_high - open30_low) / ATR14` | Range of the opening 30 minutes vs ATR. Wide opening range = volatile day ahead. |
| `price_vs_open_atr` | `(entry_price - day_open) / ATR14` | How far price has moved from the day open by signal time. Signals >2 ATR from open tend to be extended. |
| `day_range_so_far` | `(entry_price - day_low_so_far) / ATR14` | How much of the day's potential range has already been consumed. |
| `price_vs_open_pct` | `(entry_price - day_open) / day_open × 100` | Percentage move from day open to signal time. |

---

## Time Features (3 features)

| Feature | Definition | Range | Notes |
|---------|-----------|-------|-------|
| `mins_open` | Minutes since 9:15 AM (market open) | 0–375 | Low value = early session signal. |
| `mins_close` | Minutes until 3:29 PM (market close) | 0–375 | Low value = late session signal. Late signals (< 60 mins) have lower hit rates. |
| `session` | Bucket: 0=pre-open, 1=morning (9:15–11:00), 2=midday (11:00–13:00), 3=afternoon (13:00–14:30), 4=last-hour (14:30–15:29) | 0–4 | Sessions 1–3 have best signal quality. Session 4 has wide SLs and noisy moves. |

---

## Cross-TF Confluence Features (7 features)

Derived from combining the three TF signal_active and trend flags.

| Feature | Definition | Range | Interpretation |
|---------|-----------|-------|----------------|
| `tf_conf` | Sum of signal_active across 5m + 15m + 1H | 0–3 | 3 = BB+EMA9 signal fired on all three higher TFs too. Extremely high conviction. |
| `tr_align` | Sum of (EMA9 > EMA21) across 5m + 15m + 1H | 0–3 | 3 = all three TFs in uptrend. Strong macro tailwind. |
| `5m15m_both` | Binary — 1 if 5m and 15m both have signal_active=1 | 0/1 | Short-term multi-TF alignment. |
| `15m1h_both` | Binary — 1 if 15m and 1H both have signal_active=1 | 0/1 | Medium-term multi-TF alignment. |
| `all3tf_sig` | Binary — 1 if all three TFs have signal_active=1 | 0/1 | Rarest. Highest win rate when true. |
| `all3tf_up` | Binary — 1 if tr_align == 3 | 0/1 | All TFs in uptrend — best macro condition for long signals. |
| `pairwise_conf` | Count of TF pairs where both are bullish | 0–3 | Intermediate confluence measure. |

---

## Signal Metadata (not used as model features)

These columns exist in the dataset for reference and labelling but are not passed to the model.

| Column | Definition |
|--------|-----------|
| `date` | Trading date |
| `time` | Signal candle time (HH:MM:SS) |
| `entry` | Entry price = close of signal candle |
| `sl` | Stop loss = low of signal candle |
| `tgt_1_3` | 1:3 target = entry + (entry − sl) × 3 |
| `risk_pts` | Risk in points = entry − sl |
| `label` | Binary — 1 if price hit tgt_1_3 before sl, 0 if sl hit first |
| `post_pattern` | String version of post-signal pattern (encoded version used in model) |

---

## Important Notes for Live Trading

### Pre-entry features only (safe to use at entry time)

All features *except* the 15 post-signal features are available at the moment the signal fires and are safe for pre-entry filtering:

- All 38 one-minute features
- All 63 multi-timeframe features
- All 5 regime features
- All 3 time features
- All 7 confluence features

**Total safe pre-entry features: 116 features** (all except `post_pattern_enc` and the 14 post-signal features)

### Post-signal features (exit management only)

The 15 post-signal features are **look-ahead only**. In live trading:

- `broke_sl` → monitor in real-time; if next candle closes below signal low → exit immediately
- `consec_above_ema9` → monitor bar by bar; if price falls back below EMA9 → consider partial exit
- `post_pattern` → classify manually after 2–3 bars post entry to decide whether to hold

### Recommended live entry checklist (data-derived)

1. `open30_dir == 1` — opening 30 mins closed bullish
2. `tr_align >= 2` — at least 2 of 3 timeframes in uptrend
3. `1h_rsi14 < 70` — not overbought on 1H
4. `1h_bb_pct_b < 0.7` — room to run on 1H
5. `1m_is_bull == 1` — signal candle itself is bullish
6. `1m_range_vs_atr >= 0.8` — not a doji candle
7. Skip if `tr_align == 0` — fighting all timeframes
8. Skip if `1h_rsi14 > 85` — 1H extremely overbought

---

## Model Performance Summary

| Metric | In-sample (Jan 2020) | Holdout (Jan 28–31) | Out-of-sample (Jun 2024) |
|--------|---------------------|--------------------|-----------------------|
| AUC | 0.928 (5-fold CV) | 0.969 | — |
| Precision | 0.983 | 0.933 | 0.667 (2/3 BUY calls) |
| Recall | 0.905 | 0.875 | 1.000 (both real wins caught) |
| Baseline win rate | 30.3% | 30.4% | 25.0% |
| Model win rate | 98.3% | 93.3% | 66.7% |

> **Caveat:** Model trained on 1 month (211 signals). Minimum recommended is 6–12 months (800–1,500 signals) across multiple market regimes before live deployment.

---

*Generated by VeltrixAI · Research pipeline version 1.0 · Training data: NIFTY 50 · Jan 2020*

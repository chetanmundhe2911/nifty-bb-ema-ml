"""
build_dataset.py
----------------
Build a labelled signal dataset from local historical CSV files.

Folder structure expected:
    base_path/
        2020/ 1/ nifty_spotDD_MM_YYYY.csv
              2/ ...
        2021/ ...
        2024/ ...

Usage:
    # Build full dataset (all years)
    python -m src.build_dataset

    # Build specific year only
    python -m src.build_dataset --year 2024

    # Build specific year + month
    python -m src.build_dataset --year 2024 --month 1

Output:
    data/datasets/signals_YYYY.csv       per year
    data/datasets/signals_master.csv     all years combined
"""

import os
import glob
import argparse
import pandas as pd
from datetime import datetime

from src.pipeline import process_day

# ── path to your local data ───────────────────────────────────────────────────
BASE_PATH = r"C:\Users\cheta\Downloads\options_data\nifty_data\nifty_spot"
OUT_DIR   = "data/datasets"


def find_csv_files(base_path: str, year: int = None, month: int = None) -> list:
    """
    Find all daily CSV files under base_path.
    Optionally filter by year and/or month.
    Returns sorted list of file paths.
    """
    if year and month:
        pattern = os.path.join(base_path, str(year), str(month), "*.csv")
    elif year:
        pattern = os.path.join(base_path, str(year), "*", "*.csv")
    else:
        pattern = os.path.join(base_path, "*", "*", "*.csv")

    files = sorted(glob.glob(pattern))
    return files


def build_year_dataset(year: int, month: int = None) -> pd.DataFrame:
    """
    Process all CSV files for a given year (and optionally month).
    Returns a DataFrame with all signals + features + labels.
    """
    files = find_csv_files(BASE_PATH, year=year, month=month)
    if not files:
        print(f"  No files found for year={year} month={month}")
        return pd.DataFrame()

    label  = f"{year}" + (f" month {month}" if month else "")
    print(f"\nProcessing {label} — {len(files)} files")
    print("-" * 50)

    all_records = []
    errors      = []

    for i, fp in enumerate(files):
        fname = os.path.basename(fp)
        try:
            records = process_day(fp)
            wins    = sum(r["label"] for r in records)
            print(f"  [{i+1:>3}/{len(files)}] {fname} — "
                  f"{len(records)} signals, {wins} wins")
            all_records.extend(records)
        except Exception as e:
            print(f"  [{i+1:>3}/{len(files)}] {fname} — ERROR: {e}")
            errors.append(fname)

    if errors:
        print(f"\n  Errors on {len(errors)} files: {errors[:5]}")

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    print(f"\n  Year {year} total: {len(df)} signals | "
          f"Wins: {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
    return df


def save_dataset(df: pd.DataFrame, name: str):
    """Save dataset to CSV."""
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    size_kb = os.path.getsize(path) / 1024
    print(f"  Saved → {path}  ({size_kb:.0f} KB, {len(df)} rows)")
    return path


def print_summary(df: pd.DataFrame, label: str):
    """Print dataset statistics."""
    print(f"\n{'='*55}")
    print(f"SUMMARY — {label}")
    print(f"{'='*55}")
    print(f"  Total signals : {len(df)}")
    print(f"  Wins (1:3)    : {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
    print(f"  Date range    : {df['date'].min()} → {df['date'].max()}")
    print(f"  Trading days  : {df['date'].nunique()}")
    print(f"  Signals/day   : {len(df)/df['date'].nunique():.1f} avg")

    print(f"\n  By year:")
    df["year"] = pd.to_datetime(df["date"]).dt.year
    for yr, grp in df.groupby("year"):
        print(f"    {yr}: {len(grp):>4} signals | "
              f"Win rate {grp['label'].mean()*100:.1f}% | "
              f"{grp['date'].nunique()} days")

    print(f"\n  Post-pattern breakdown:")
    for pat, grp in df.groupby("post_pattern"):
        print(f"    {pat:<20} {len(grp):>4} signals | "
              f"Win rate {grp['label'].mean()*100:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year",  type=int, default=None)
    parser.add_argument("--month", type=int, default=None)
    parser.add_argument("--all",   action="store_true",
                        help="Build all years and combine into master dataset")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    start = datetime.now()

    # ── single year / month ───────────────────────────────────────────────────
    if args.year and not args.all:
        df = build_year_dataset(args.year, args.month)
        if not df.empty:
            tag = f"signals_{args.year}"
            if args.month:
                tag += f"_m{args.month:02d}"
            save_dataset(df, tag)
            print_summary(df, tag)

    # ── all years → master dataset ────────────────────────────────────────────
    else:
        years     = [2020, 2021, 2022, 2023, 2024]
        all_dfs   = []

        for yr in years:
            df_yr = build_year_dataset(yr)
            if not df_yr.empty:
                save_dataset(df_yr, f"signals_{yr}")
                all_dfs.append(df_yr)

        if all_dfs:
            master = pd.concat(all_dfs, ignore_index=True)
            save_dataset(master, "signals_master")
            print_summary(master, "MASTER DATASET")

    elapsed = (datetime.now() - start).seconds
    print(f"\nDone in {elapsed}s")


if __name__ == "__main__":
    main()

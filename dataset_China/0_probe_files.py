#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
0_probe_files.py
Probe / sanity-check CSV files with robust encoding attempts and column heuristics.

Usage:
  python 0_probe_files.py --charging Charging_Data.csv --price Time-of-use_Price.csv --weather Weather_Data.csv
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np


ENCODINGS_TO_TRY = ["utf-8", "utf-8-sig", "gb18030", "gbk", "cp936"]


def _try_read_csv_head(path: str, encoding: str, nrows: int = 50, sep: Optional[str] = ",") -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(
            path,
            encoding=encoding,
            nrows=nrows,
            sep=sep,
            on_bad_lines="skip",  # robust for messy lines
            low_memory=False,
        )
        return df
    except UnicodeDecodeError:
        return None
    except TypeError:
        # older pandas might not support on_bad_lines the same way
        try:
            df = pd.read_csv(path, encoding=encoding, nrows=nrows, sep=sep, error_bad_lines=False)  # type: ignore
            return df
        except Exception:
            return None
    except Exception:
        return None


def robust_read_csv_head(path: str, nrows: int = 50) -> Tuple[pd.DataFrame, str, str]:
    """
    Returns (df, encoding_used, sep_used)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    # Try separators in a small set
    seps = [",", "\t", ";", None]
    for enc in ENCODINGS_TO_TRY:
        for sep in seps:
            df = _try_read_csv_head(path, enc, nrows=nrows, sep=sep)
            if df is None:
                continue
            # Heuristic: if sep is wrong, you'll often get a single giant column.
            if df.shape[1] == 1 and sep in [",", ";", "\t"]:
                continue
            return df, enc, (sep if sep is not None else "auto")
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Failed to decode {path} with encodings={ENCODINGS_TO_TRY}")


def guess_datetime_cols(df: pd.DataFrame) -> List[str]:
    candidates = []
    for c in df.columns:
        c_low = str(c).lower()
        if any(k in c_low for k in ["time", "date", "datetime", "timestamp"]):
            candidates.append(c)
    # also include object columns that look like datetimes
    for c in df.columns:
        if c in candidates:
            continue
        if df[c].dtype == "object":
            s = df[c].astype(str).str.strip().head(20)
            if s.str.contains(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", regex=True).mean() > 0.4:
                candidates.append(c)
    return candidates


def parse_datetime_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        out = s.copy()
    else:
        ss = s.astype(str).str.strip().str.replace("\t", "", regex=False)
        # quick timezone marker check (non-capturing groups to avoid warnings)
        has_tz = ss.str.contains(r"(?:Z$)|(?:[+-]\d{2}:?\d{2}$)|(?:UTC)", case=False, regex=True).any()
        if has_tz:
            out = pd.to_datetime(ss, errors="coerce", utc=True)
            # Convert to Asia/Shanghai then drop tz to make resample-safe (naive)
            try:
                out = out.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
            except Exception:
                out = out.dt.tz_localize(None, ambiguous="NaT", nonexistent="NaT")
        else:
            out = pd.to_datetime(ss, errors="coerce")
            # If timezone-aware sneaks in, drop it
            try:
                if getattr(out.dt, "tz", None) is not None:
                    out = out.dt.tz_localize(None)
            except Exception:
                pass

        # If too many NaT, try dayfirst
        if out.notna().mean() < 0.7:
            out2 = pd.to_datetime(ss, errors="coerce", dayfirst=True)
            if out2.notna().mean() > out.notna().mean():
                out = out2
    return out


def numeric_cols(df: pd.DataFrame) -> List[str]:
    nums = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            nums.append(c)
            continue
        # try coercion for object columns
        if df[c].dtype == "object":
            coerced = pd.to_numeric(df[c].astype(str).str.strip().str.replace("\t", "", regex=False), errors="coerce")
            if coerced.notna().mean() > 0.6:
                nums.append(c)
    return nums


def summarize_file(path: str, label: str) -> None:
    print("=" * 90)
    print(f"[{label}] {path}")
    print(f"Size: {os.path.getsize(path)/1024/1024:.2f} MB")

    df, enc, sep = robust_read_csv_head(path, nrows=100)
    print(f"Read OK. encoding='{enc}', sep='{sep}', sample_shape={df.shape}")

    print("\nColumns:")
    for c in df.columns:
        print(f"  - {c}")

    print("\nDtypes (sample):")
    print(df.dtypes)

    dt_cols = guess_datetime_cols(df)
    print("\nDatetime column candidates:")
    if dt_cols:
        for c in dt_cols:
            parsed = parse_datetime_series(df[c])
            ok_rate = float(parsed.notna().mean())
            rng = (parsed.min(), parsed.max()) if ok_rate > 0 else (None, None)
            print(f"  * {c} | parsed_ok={ok_rate:.2f} | range={rng[0]} .. {rng[1]}")
    else:
        print("  (none detected)")

    num_cols = numeric_cols(df)
    print("\nNumeric column candidates:")
    if num_cols:
        for c in num_cols[:30]:
            series = df[c]
            if not pd.api.types.is_numeric_dtype(series):
                series = pd.to_numeric(series.astype(str).str.strip().str.replace("\t", "", regex=False), errors="coerce")
            print(f"  * {c} | nonnull={series.notna().mean():.2f} | min={series.min()} | max={series.max()}")
        if len(num_cols) > 30:
            print(f"  ... (+{len(num_cols)-30} more)")
    else:
        print("  (none detected)")

    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--charging", default="Charging_Data.csv")
    ap.add_argument("--price", default="Time-of-use_Price.csv")
    ap.add_argument("--weather", default="Weather_Data.csv")
    args = ap.parse_args()

    summarize_file(args.charging, "Charging_Data")
    summarize_file(args.price, "Time-of-use_Price")
    summarize_file(args.weather, "Weather_Data")


if __name__ == "__main__":
    main()

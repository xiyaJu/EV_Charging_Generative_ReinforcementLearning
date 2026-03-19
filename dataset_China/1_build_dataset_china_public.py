#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1_build_dataset_china_public.py
Build a robust 15-min scenario table from:
  1) Charging_Data.csv (main timeline, session-level)
  2) Time-of-use_Price.csv (TOU schedule or time series)
  3) Weather_Data.csv (daily weather, may contain District Name)

Output: scenario CSV with columns:
  day_id, t, datetime, price, temp, rh, precip, load_sum, lambda_public

Key design:
- Timeline is driven by Charging_Data time range (NOT intersection with weather/price).
- Missing external features are filled (ffill/bfill/0) to avoid empty output.
- Robust encoding & datetime parsing; avoid resample errors by forcing a clean DatetimeIndex.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


ENCODINGS_TO_TRY = ["utf-8", "utf-8-sig", "gb18030", "gbk", "cp936"]
FREQ = "15min"
BIN_SECONDS = 15 * 60


# ---------------------------
# Path utils (avoid /mnt/data pitfalls)
# ---------------------------
def resolve_path(path: str) -> str:
    """
    Resolve a possibly-nonexistent path by trying:
      1) as-is
      2) basename in current working directory
      3) basename in the script directory
    """
    p = Path(path)
    if p.exists():
        return str(p)

    alt = Path.cwd() / p.name
    if alt.exists():
        return str(alt)

    alt2 = Path(__file__).resolve().parent / p.name
    if alt2.exists():
        return str(alt2)

    raise FileNotFoundError(
        f"File not found: '{path}'\n"
        f"Also tried:\n"
        f"  - '{alt}'\n"
        f"  - '{alt2}'\n"
        f"Tip: please pass the real path on your machine (e.g. ./Charging_Data.csv or an absolute path)."
    )


# ---------------------------
# CSV reading (robust encoding)
# ---------------------------
def robust_read_csv(
    path: str,
    usecols: Optional[List[str]] = None,
    nrows: Optional[int] = None,
    chunksize: Optional[int] = None,
) -> Tuple[pd.io.parsers.TextFileReader | pd.DataFrame, str]:
    """
    Try common encodings. Returns (df_or_reader, encoding_used).
    """
    path = resolve_path(path)
    last_err = None
    for enc in ENCODINGS_TO_TRY:
        try:
            kwargs = dict(
                encoding=enc,
                sep=",",
                on_bad_lines="skip",
                low_memory=False,
            )
            if usecols is not None:
                kwargs["usecols"] = usecols
            if nrows is not None:
                kwargs["nrows"] = nrows
            if chunksize is not None:
                kwargs["chunksize"] = chunksize

            out = pd.read_csv(path, **kwargs)

            # If sep is wrong, often 1 column; try a few alternatives quickly (only for small files)
            if isinstance(out, pd.DataFrame) and out.shape[1] == 1 and os.path.getsize(path) < 5_000_000:
                for sep in ["\t", ";"]:
                    try:
                        kwargs["sep"] = sep
                        out2 = pd.read_csv(path, **kwargs)
                        if out2.shape[1] > 1:
                            return out2, enc
                    except Exception:
                        pass
            return out, enc
        except UnicodeDecodeError as e:
            last_err = e
            continue
        except Exception as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    raise RuntimeError(f"Failed to read: {path}")


def _clean_str_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace("\t", "", regex=False)


# ---------------------------
# Datetime parsing (robust tz + mixed formats)
# ---------------------------
def parse_datetime_series(s: pd.Series) -> pd.Series:
    """
    Robust datetime parser:
    - strips whitespace/tabs
    - handles mixed timezone strings by normalizing to Asia/Shanghai then dropping tz
    - fallback dayfirst if needed
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        out = s.copy()
    else:
        ss = _clean_str_series(s)
        has_tz = ss.str.contains(r"(?:Z$)|(?:[+-]\d{2}:?\d{2}$)|(?:UTC)", case=False, regex=True).any()
        if has_tz:
            out = pd.to_datetime(ss, errors="coerce", utc=True)
            try:
                out = out.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
            except Exception:
                out = out.dt.tz_localize(None, ambiguous="NaT", nonexistent="NaT")
        else:
            out = pd.to_datetime(ss, errors="coerce")
            try:
                if getattr(out.dt, "tz", None) is not None:
                    out = out.dt.tz_localize(None)
            except Exception:
                pass

        if out.notna().mean() < 0.7:
            out2 = pd.to_datetime(ss, errors="coerce", dayfirst=True)
            if out2.notna().mean() > out.notna().mean():
                out = out2
    return out


def find_col(columns: List[str], patterns: List[str]) -> Optional[str]:
    cols = list(columns)
    for pat in patterns:
        rgx = re.compile(pat, re.IGNORECASE)
        for c in cols:
            if rgx.search(str(c)):
                return c
    return None


# ---------------------------
# Price template builder (TOU schedule or time series)
# ---------------------------
def build_price_template(price_csv: str) -> Tuple[np.ndarray, str]:
    """
    Returns (price_96, info_string)
    Supports:
      - TOU schedule table with a "time period" column like "00:00-08:00"
      - datetime time series (will be reduced to day-template if possible)
    """
    df, enc = robust_read_csv(price_csv, nrows=None, chunksize=None)
    if isinstance(df, pd.io.parsers.TextFileReader):
        df = next(iter(df))

    cols = list(df.columns)

    # Detect time series: need both (a) parses as datetime and (b) looks like it contains real dates.
    dt_col = None
    for c in cols:
        ss = df[c].astype(str).str.strip()
        time_period_like = ss.str.contains(r"^\s*\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}", regex=True).mean() > 0.5
        if time_period_like:
            continue
        date_like = (
            ss.str.contains(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", regex=True).mean()
            + ss.str.contains(r"\b\d{8}\b", regex=True).mean()
        ) > 0.3
        parsed = parse_datetime_series(df[c])
        if parsed.notna().mean() > 0.8 and date_like:
            dt_col = c
            break

    price_col = find_col(cols, [r"price", r"electricity\s*price", r"yuan/kwh", r"kwh", r"电价", r"价格"])
    if price_col is None:
        # choose best numeric-like column
        num_scores = []
        for c in cols:
            coerced = pd.to_numeric(_clean_str_series(df[c]), errors="coerce")
            num_scores.append((coerced.notna().mean(), c))
        num_scores.sort(reverse=True)
        price_col = num_scores[0][1] if num_scores else None

    if price_col is None:
        return np.zeros(96, dtype=float), "Price: no usable price column found -> all zeros"

    if dt_col is not None:
        ddf = df[[dt_col, price_col]].copy()
        ddf[dt_col] = parse_datetime_series(ddf[dt_col])
        ddf[price_col] = pd.to_numeric(_clean_str_series(ddf[price_col]), errors="coerce")
        ddf = ddf.dropna(subset=[dt_col]).set_index(dt_col).sort_index()
        if ddf.empty:
            return np.zeros(96, dtype=float), "Price: parsed as time series but became empty -> all zeros"

        s15 = ddf[price_col].resample(FREQ).ffill()
        tod = (s15.index.hour * 60 + s15.index.minute) // 15
        template = s15.groupby(tod).median().reindex(range(96)).ffill().bfill().fillna(0.0).to_numpy()
        return template.astype(float), f"Price: time series detected (dt_col='{dt_col}', enc='{enc}') -> built 96-step template"

    # schedule table
    period_col = find_col(cols, [r"time\s*period", r"period", r"时段", r"时间段"])
    if period_col is None:
        period_col = cols[0]

    template = np.full(96, np.nan, dtype=float)

    def to_minutes(hhmm: str) -> Optional[int]:
        m = re.match(r"^\s*(\d{1,2}):(\d{2})\s*$", hhmm)
        if not m:
            return None
        h = int(m.group(1))
        mi = int(m.group(2))
        if h == 24:
            h = 0
        return h * 60 + mi

    for _, row in df.iterrows():
        p = str(row.get(period_col, "")).strip()
        val = row.get(price_col, np.nan)
        try:
            val = float(val)
        except Exception:
            val = np.nan
        if not p or "-" not in p:
            continue
        a, b = p.split("-", 1)
        a = a.strip()
        b = b.strip()
        start_m = to_minutes(a)
        end_m = to_minutes(b if b != "23:59" else "24:00")
        if start_m is None or end_m is None:
            continue
        spans = [(start_m, end_m)] if end_m >= start_m else [(start_m, 24 * 60), (0, end_m)]
        for st, en in spans:
            for t in range(96):
                m0 = t * 15
                if (m0 >= st) and (m0 < en):
                    template[t] = val

    if np.isnan(template).all():
        return np.zeros(96, dtype=float), f"Price: schedule detected but failed to parse periods -> all zeros (enc='{enc}')"

    s = pd.Series(template).ffill().bfill().fillna(0.0)
    return s.to_numpy(dtype=float), f"Price: TOU schedule detected (period_col='{period_col}', price_col='{price_col}', enc='{enc}')"


# ---------------------------
# Weather: pick temp + RH + precip (daily) and expand to 15min
# ---------------------------
def pick_weather_features(df_weather: pd.DataFrame) -> Tuple[Dict[str, pd.Series], str]:
    """
    Return daily weather series indexed by date (Timestamp midnight) for multiple features.
    Prefer exact-ish matches:
      - temp: Temperature-like column
      - rh  : Relative Humidity-like column
      - precip: Precipitation-like column
    If a feature not found -> empty series (caller will fill zeros).

    NOTE: Weather file may have multiple districts; default aggregates across districts by daily mean.
    """
    cols = list(df_weather.columns)

    date_col = find_col(cols, [r"^date$", r"date", r"日期", r"day"])
    if date_col is None:
        for c in cols:
            parsed = parse_datetime_series(df_weather[c])
            if parsed.notna().mean() > 0.8:
                date_col = c
                break
    if date_col is None:
        empty = {"temp": pd.Series(dtype=float), "rh": pd.Series(dtype=float), "precip": pd.Series(dtype=float)}
        return empty, "Weather: no usable date column -> will fill 0"

    tmp = df_weather.copy()
    sdate_raw = _clean_str_series(tmp[date_col])
    # handle yyyymmdd first
    parsed = pd.to_datetime(sdate_raw, errors="coerce", format="%Y%m%d")
    if parsed.notna().mean() < 0.8:
        parsed = pd.to_datetime(sdate_raw, errors="coerce")
    tmp["_date"] = parsed.dt.normalize()

    def get_numeric_daily_by_colname(colname: Optional[str], agg: str = "mean") -> pd.Series:
        if not colname or colname not in tmp.columns:
            return pd.Series(dtype=float)
        x = pd.to_numeric(_clean_str_series(tmp[colname]), errors="coerce")
        if x.notna().mean() < 0.2:
            return pd.Series(dtype=float)
        g = tmp.assign(_x=x).dropna(subset=["_date"]).groupby("_date")["_x"]
        if agg == "sum":
            return g.sum().sort_index()
        return g.mean().sort_index()

    # Try strongest patterns first (you said these exact names exist)
    temp_col = find_col(cols, [r"temperature", r"\btemp\b", r"气温", r"℃"])
    rh_col = find_col(cols, [r"relative\s*humidity", r"humidity", r"相对湿度", r"\brh\b"])
    precip_col = find_col(cols, [r"precipitation", r"\bprecip\b", r"rain", r"降水", r"降雨"])

    # temp and rh as daily mean; precip: default mean (safe). If you confirm it's daily total, change to sum.
    temp_s = get_numeric_daily_by_colname(temp_col, agg="mean")
    rh_s = get_numeric_daily_by_colname(rh_col, agg="mean")
    precip_s = get_numeric_daily_by_colname(precip_col, agg="mean")

    info = (
        f"Weather: temp='{temp_col}', rh='{rh_col}', precip='{precip_col}'. "
        f"Aggregated daily mean across districts; missing features filled 0."
    )

    return {"temp": temp_s, "rh": rh_s, "precip": precip_s}, info


def daily_to_15min(daily: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """Expand daily series to 15-min time axis by repeating same-day values."""
    if daily is None or len(daily) == 0:
        return pd.Series(0.0, index=idx, dtype=float)

    unique_days = pd.to_datetime(sorted(set(idx.normalize())))
    daily_on_axis = daily.reindex(unique_days).ffill().bfill()

    out = pd.Series([daily_on_axis.get(d.normalize(), np.nan) for d in idx], index=idx, dtype=float)
    return out.ffill().bfill().fillna(0.0)


# ---------------------------
# Charging: compute lambda + load_sum into 15min bins
# ---------------------------
def compute_lambda_and_load(
    charging_csv: str,
    start_col: str,
    end_col: Optional[str],
    energy_col: Optional[str],
    district_col: Optional[str],
    district_filter: Optional[str],
    chunksize: int = 200_000,
) -> Tuple[pd.Timestamp, pd.Timestamp, Dict[pd.Timestamp, int], Dict[pd.Timestamp, float], Dict[pd.Timestamp, float], Dict[str, float]]:
    """
    One-pass chunk scan to compute:
      - min_start, max_end
      - lambda_counts per 15min bin (Start Time count)
      - load partial contributions per bin
      - delta dict for full-bin contributions (difference array)
    Returns dicts keyed by bin timestamp.
    """
    usecols = [c for c in [start_col, end_col, energy_col, district_col] if c]
    reader, _enc = robust_read_csv(charging_csv, usecols=usecols, chunksize=chunksize)
    if not isinstance(reader, pd.io.parsers.TextFileReader):
        reader = [reader]  # type: ignore

    min_start = None
    max_end = None

    lambda_dict: Dict[pd.Timestamp, int] = {}
    partial_dict: Dict[pd.Timestamp, float] = {}
    delta_dict: Dict[pd.Timestamp, float] = {}

    stats = {
        "rows_total": 0,
        "rows_after_district": 0,
        "rows_valid_start": 0,
        "rows_valid_energy": 0,
        "rows_valid_for_load": 0,
        "rows_skipped_duration<=0": 0,
        "rows_missing_end": 0,
    }

    for chunk in reader:
        stats["rows_total"] += len(chunk)

        if district_filter and district_col and district_col in chunk.columns:
            chunk = chunk[chunk[district_col].astype(str).str.strip().eq(district_filter)]
        stats["rows_after_district"] += len(chunk)
        if len(chunk) == 0:
            continue

        start = parse_datetime_series(chunk[start_col])
        stats["rows_valid_start"] += int(start.notna().sum())

        if start.notna().any():
            smin = start.min()
            if min_start is None or smin < min_start:
                min_start = smin

        start_bin = start.dt.floor(FREQ)
        vc = start_bin.value_counts(dropna=True)
        for ts, cnt in vc.items():
            lambda_dict[ts] = lambda_dict.get(ts, 0) + int(cnt)

        # load: prefer energy + duration
        if energy_col is None or energy_col not in chunk.columns:
            continue

        energy = pd.to_numeric(_clean_str_series(chunk[energy_col]), errors="coerce")
        stats["rows_valid_energy"] += int(energy.notna().sum())

        if end_col is None or end_col not in chunk.columns:
            # fallback: dump energy into start bin, convert kWh/bin -> kW by /0.25h
            valid_e = start.notna() & energy.notna()
            if valid_e.any():
                ebin = energy[valid_e].astype(float).groupby(start_bin[valid_e]).sum()
                for ts, val in (ebin * 4.0).items():
                    partial_dict[ts] = partial_dict.get(ts, 0.0) + float(val)
            continue

        end = parse_datetime_series(chunk[end_col])
        stats["rows_missing_end"] += int(end.isna().sum())

        valid = start.notna() & end.notna() & energy.notna()
        if not valid.any():
            continue

        start_v = start[valid]
        end_v = end[valid]
        energy_v = energy[valid].astype(float)

        emax = end_v.max()
        if max_end is None or emax > max_end:
            max_end = emax

        dur_s = (end_v - start_v).dt.total_seconds()
        good = dur_s > 0
        stats["rows_skipped_duration<=0"] += int((~good).sum())
        if not good.any():
            continue

        start_v = start_v[good]
        end_v = end_v[good]
        energy_v = energy_v[good]
        dur_s = dur_s[good]

        stats["rows_valid_for_load"] += len(start_v)

        dur_h = dur_s / 3600.0
        power_kw = energy_v / dur_h  # avg power over session

        first_bin = start_v.dt.floor(FREQ)
        last_bin = (end_v - pd.Timedelta(nanoseconds=1)).dt.floor(FREQ)

        same = first_bin.eq(last_bin)
        if same.any():
            overlap = (end_v[same] - start_v[same]).dt.total_seconds() / BIN_SECONDS
            overlap = overlap.clip(lower=0.0, upper=1.0)
            for ts, val in (power_kw[same] * overlap).groupby(first_bin[same]).sum().items():
                partial_dict[ts] = partial_dict.get(ts, 0.0) + float(val)

        multi = ~same
        if multi.any():
            fb = first_bin[multi]
            lb = last_bin[multi]
            sv = start_v[multi]
            ev = end_v[multi]
            pw = power_kw[multi]

            first_frac = ((fb + pd.Timedelta(minutes=15)) - sv).dt.total_seconds() / BIN_SECONDS
            last_frac = (ev - lb).dt.total_seconds() / BIN_SECONDS
            first_frac = first_frac.clip(lower=0.0, upper=1.0)
            last_frac = last_frac.clip(lower=0.0, upper=1.0)

            for ts, val in (pw * first_frac).groupby(fb).sum().items():
                partial_dict[ts] = partial_dict.get(ts, 0.0) + float(val)
            for ts, val in (pw * last_frac).groupby(lb).sum().items():
                partial_dict[ts] = partial_dict.get(ts, 0.0) + float(val)

            full_start = fb + pd.Timedelta(minutes=15)
            full_end = lb - pd.Timedelta(minutes=15)
            has_full = full_start.le(full_end)
            if has_full.any():
                fs = full_start[has_full]
                fe = full_end[has_full]
                pw2 = pw[has_full]

                for ts, val in pw2.groupby(fs).sum().items():
                    delta_dict[ts] = delta_dict.get(ts, 0.0) + float(val)
                for ts, val in pw2.groupby(fe + pd.Timedelta(minutes=15)).sum().items():
                    delta_dict[ts] = delta_dict.get(ts, 0.0) - float(val)

    if min_start is None:
        raise RuntimeError("No valid Start Time parsed from Charging_Data; cannot build timeline.")
    if max_end is None:
        max_end = min_start
    if max_end < min_start:
        max_end = min_start

    return min_start, max_end, lambda_dict, partial_dict, delta_dict, stats


# ---------------------------
# Main
# ---------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--charging", default="Charging_Data.csv")
    ap.add_argument("--price", default="Time-of-use_Price.csv")
    ap.add_argument("--weather", default="Weather_Data.csv")
    ap.add_argument("--out_csv", default="scene_china_public.csv")
    ap.add_argument("--district", default=None, help="Optional: filter Charging_Data by a specific District Name")
    ap.add_argument("--chunksize", type=int, default=200_000)
    args = ap.parse_args()

    # Probe charging headers
    head_df, enc_ch = robust_read_csv(args.charging, nrows=50, chunksize=None)
    if isinstance(head_df, pd.io.parsers.TextFileReader):
        head_df = next(iter(head_df))
    cols = list(head_df.columns)

    start_col = find_col(cols, [r"^start\s*time$", r"start.*time", r"开始"])
    end_col = find_col(cols, [r"^end\s*time$", r"end.*time", r"结束"])
    energy_col = find_col(cols, [r"transaction\s*power", r"energy", r"kwh", r"电量", r"交易.*kwh"])
    district_col = find_col(cols, [r"district", r"区", r"District\s*Name"])

    if start_col is None:
        raise RuntimeError(f"Cannot find Start Time column in Charging_Data. Columns={cols}")

    print(f"[Charging_Data] encoding='{enc_ch}', start_col='{start_col}', end_col='{end_col}', energy_col='{energy_col}', district_col='{district_col}'")

    # Price
    price_96, price_info = build_price_template(args.price)
    print(price_info)

    # Weather
    wdf, enc_w = robust_read_csv(args.weather, nrows=None, chunksize=None)
    if isinstance(wdf, pd.io.parsers.TextFileReader):
        wdf = next(iter(wdf))
    weather_map, weather_info = pick_weather_features(wdf)
    print(f"[Weather] encoding='{enc_w}'. {weather_info}")

    # Charging aggregation
    min_start, max_end, lambda_dict, partial_dict, delta_dict, stats = compute_lambda_and_load(
        charging_csv=args.charging,
        start_col=start_col,
        end_col=end_col,
        energy_col=energy_col,
        district_col=district_col,
        district_filter=args.district,
        chunksize=args.chunksize,
    )
    print(f"[Charging summary] time_range: {min_start} .. {max_end}")
    print("[Charging stats] " + ", ".join([f"{k}={v}" for k, v in stats.items()]))

    # Build full time axis (by charging range)
    start_day = pd.Timestamp(min_start).normalize()
    end_exclusive = pd.Timestamp(max_end).normalize() + pd.Timedelta(days=1)
    try:
        idx = pd.date_range(start=start_day, end=end_exclusive, freq=FREQ, inclusive="left")
    except TypeError:
        idx = pd.date_range(start=start_day, end=end_exclusive - pd.Timedelta(minutes=15), freq=FREQ)

    # lambda
    lambda_s = pd.Series(lambda_dict, dtype=float).reindex(idx, fill_value=0.0)

    # load_sum from partial + difference-cumsum
    partial_s = pd.Series(partial_dict, dtype=float).reindex(idx, fill_value=0.0)
    delta_s = pd.Series(delta_dict, dtype=float).reindex(idx, fill_value=0.0)
    load_full = delta_s.cumsum()
    load_sum = (partial_s + load_full).clip(lower=0.0)

    # price expand
    tod = ((idx.hour * 60 + idx.minute) // 15).astype(int)
    price_s = pd.Series(price_96[tod], index=idx, dtype=float)

    # weather expand (daily -> 15min)
    temp_s = daily_to_15min(weather_map.get("temp", pd.Series(dtype=float)), idx)
    rh_s = daily_to_15min(weather_map.get("rh", pd.Series(dtype=float)), idx)
    precip_s = daily_to_15min(weather_map.get("precip", pd.Series(dtype=float)), idx)

    # Build scenario
    df = pd.DataFrame(
        {
            "datetime": idx,
            "price": price_s.values,
            "temp": temp_s.values,
            "rh": rh_s.values,
            "precip": precip_s.values,
            "load_sum": load_sum.values,
            "lambda_public": lambda_s.values,
        }
    )

    # day_id & t
    df["date"] = df["datetime"].dt.normalize()
    dates = pd.Index(sorted(df["date"].unique()))
    date_to_dayid = {d: i for i, d in enumerate(dates)}
    df["day_id"] = df["date"].map(date_to_dayid).astype(int)
    df["t"] = ((df["datetime"].dt.hour * 60 + df["datetime"].dt.minute) // 15).astype(int)

    df = df[["day_id", "t", "datetime", "price", "temp", "rh", "precip", "load_sum", "lambda_public"]].sort_values(["day_id", "t"])

    # Drop incomplete days
    counts = df.groupby("day_id")["t"].count()
    good_days = counts[counts == 96].index
    dropped = int((counts != 96).sum())
    if dropped > 0:
        print(f"[Drop] Dropping {dropped} incomplete day(s) (not 96 rows).")
    df = df[df["day_id"].isin(good_days)].copy()

    # Fill NaN
    for c in ["price", "temp", "rh", "precip", "load_sum", "lambda_public"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    out_path = args.out_csv
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] Wrote scenario CSV: {out_path}")
    print(f"     rows={len(df)}, days={df['day_id'].nunique()}, cols={list(df.columns)}")


if __name__ == "__main__":
    main()

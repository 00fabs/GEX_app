# ─────────────────────────────────────────────────────────────
# data_fetch.py — all iVolatility API calls
# ─────────────────────────────────────────────────────────────
import gzip
import io
import time as time_module

import ivolatility as ivol
import pandas as pd
import streamlit as st

from config import (BASE_URL, MIN_DELAY, POLL_DELAY,
                    SESSION_START, SESSION_END, STRIKE_STEP)
from greeks import bsm_vanna, bsm_charm
from config import RISK_FREE
from helpers import rate_limited_get, get_last_request_time


# ── URL search helpers ────────────────────────────────────────
def find_download_url(obj, depth=0):
    if depth > 10: return None
    if isinstance(obj, dict):
        if "urlForDownload" in obj: return obj["urlForDownload"]
        for v in obj.values():
            f = find_download_url(v, depth+1)
            if f: return f
    elif isinstance(obj, list):
        for item in obj:
            f = find_download_url(item, depth+1)
            if f: return f
    return None


def find_poll_url(obj, depth=0):
    if depth > 10: return None
    if isinstance(obj, dict):
        if "urlForDetails" in obj and obj["urlForDetails"]:
            return obj["urlForDetails"]
        for v in obj.values():
            f = find_poll_url(v, depth+1)
            if f: return f
    elif isinstance(obj, list):
        for item in obj:
            f = find_poll_url(item, depth+1)
            if f: return f
    return None


# ── Async download with polling ───────────────────────────────
def async_download(endpoint, params, headers, label="", max_polls=25):
    r = rate_limited_get(f"{BASE_URL}{endpoint}",
                         headers=headers, params=params)
    if r.status_code != 200:
        st.error(f"[{label}] HTTP {r.status_code}: {r.text[:200]}")
        return None
    body = r.json()
    url  = find_download_url(body)
    if url: return url
    poll_url = find_poll_url(body)
    if not poll_url:
        st.error(f"[{label}] No poll URL found")
        return None
    for _ in range(1, max_polls+1):
        time_module.sleep(POLL_DELAY)
        rp = rate_limited_get(poll_url, headers=headers,
                              params={"apiKey": params["apiKey"]})
        if rp.status_code == 429:
            time_module.sleep(3)
            continue
        if rp.status_code != 200:
            continue
        url = find_download_url(rp.json())
        if url: return url
    st.error(f"[{label}] Polling timed out")
    return None


def download_csv_gz(url, headers, api_key, label=""):
    r = rate_limited_get(url, headers=headers,
                         params={"apiKey": api_key})
    if r.status_code != 200:
        return None
    try:
        with gzip.open(io.BytesIO(r.content), "rt") as f:
            return pd.read_csv(f)
    except Exception:
        try:
            return pd.read_csv(io.BytesIO(r.content))
        except Exception as e:
            st.error(f"CSV parse error: {e}")
            return None


def sync_call(endpoint, params, headers, label=""):
    r = rate_limited_get(f"{BASE_URL}{endpoint}",
                         headers=headers, params=params)
    if r.status_code == 429:
        time_module.sleep(3)
        r = rate_limited_get(f"{BASE_URL}{endpoint}",
                             headers=headers, params=params)
    if r.status_code != 200:
        return None
    body    = r.json()
    records = body.get("data", []) if isinstance(body, dict) else body
    return records if records else None


# ── Step 1: Option IDs ────────────────────────────────────────
def get_option_ids(api_key, headers, prev_date_str, exp_date_str,
                   strike_min, strike_max):
    dl_url = async_download(
        "/equities/eod/option-series-on-date",
        {"symbol": "SPX", "date": prev_date_str, "apiKey": api_key},
        headers, label="series")
    if not dl_url:
        return None, None

    series_df = download_csv_gz(dl_url, headers, api_key, label="series")
    if series_df is None or series_df.empty:
        st.error("Empty option series")
        return None, None

    series_df.columns           = [c.strip().lower() for c in series_df.columns]
    series_df["expirationdate"] = pd.to_datetime(series_df["expirationdate"])
    series_df["strike"]         = pd.to_numeric(series_df["strike"],
                                                 errors="coerce")

    exp_mask    = series_df["expirationdate"] == pd.Timestamp(exp_date_str)
    strike_mask = ((series_df["strike"] >= strike_min) &
                   (series_df["strike"] <= strike_max))
    filtered    = series_df[exp_mask & strike_mask].copy()

    if filtered.empty:
        st.error(f"No contracts for expiry {exp_date_str} in "
                 f"{strike_min}–{strike_max}")
        return None, None

    cp_col = next((c for c in filtered.columns
                   if c in ["callput","call_put","type","optiontype"]), None)
    if not cp_col:
        st.error("Cannot identify call/put column")
        return None, None

    filtered["_is_spx"] = (filtered["optionsymbol"]
                           .str.strip().str.startswith("SPX "))
    filtered = (filtered
                .sort_values("_is_spx", ascending=False)
                .drop_duplicates(subset=["strike", cp_col], keep="first")
                .drop(columns=["_is_spx"])
                .reset_index(drop=True))

    return (filtered[filtered[cp_col] == "C"].sort_values("strike"),
            filtered[filtered[cp_col] == "P"].sort_values("strike"))


# ── Step 2: EOD OI ────────────────────────────────────────────
def get_eod_oi(api_key, headers, calls, puts, prev_date_str):
    oi_map = {}

    def fetch_oi(row, cp_label):
        rec = sync_call(
            "/equities/eod/single-stock-option-raw-iv",
            {"optionId": int(row["optionid"]),
             "from": prev_date_str, "to": prev_date_str,
             "apiKey": api_key},
            headers, label=f"{cp_label}{int(row['strike'])}")
        if not rec:
            return
        d      = {k.strip().lower().replace(" ", ""): v
                  for k, v in rec[0].items()}
        oi_val = (d.get("openinterest") or d.get("open_interest") or
                  d.get("oi") or d.get("openint") or 0)
        oi_map[(float(row["strike"]), cp_label)] = (
            int(float(oi_val)) if oi_val else 0)

    for _, row in calls.iterrows(): fetch_oi(row, "C")
    for _, row in puts.iterrows():  fetch_oi(row, "P")
    return oi_map


# ── Step 3a: Pilot fetch ──────────────────────────────────────
def get_session_price_range(api_key, date_str, exp_date_str,
                             rough_center, minute_type="MINUTE_1"):
    _lrt = get_last_request_time()
    ivol.setLoginParams(apiKey=api_key)
    get_intra = ivol.setMethod(
        "/equities/intraday/single-equity-option-rawiv")

    time_module.sleep(MIN_DELAY)
    try:
        df = get_intra(symbol="SPX", date=date_str,
                       expDate=exp_date_str,
                       strike=str(int(rough_center)),
                       optType="C",
                       minuteType=minute_type)
        _lrt[0] = time_module.time()
    except Exception as e:
        st.warning(f"Pilot fetch failed: {e}")
        return None, None

    if df is None or len(df) == 0:
        return None, None

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[(df["timestamp"].dt.time >= SESSION_START) &
            (df["timestamp"].dt.time <= SESSION_END)]

    if "underlyingPrice" not in df.columns or df.empty:
        return None, None

    df["underlyingPrice"] = pd.to_numeric(df["underlyingPrice"],
                                           errors="coerce")
    return df["underlyingPrice"].min(), df["underlyingPrice"].max()


# ── Step 3b: Full intraday fetch ──────────────────────────────
def get_intraday_greeks(api_key, date_str, exp_date_str,
                        strike_min, strike_max,
                        minute_type, oi_map, progress_bar):
    _lrt = get_last_request_time()
    ivol.setLoginParams(apiKey=api_key)
    get_intra = ivol.setMethod(
        "/equities/intraday/single-equity-option-rawiv")

    strikes  = list(range(int(strike_min),
                          int(strike_max) + STRIKE_STEP,
                          STRIKE_STEP))
    total    = len(strikes) * 2
    all_data = []
    count    = 0
    from datetime import datetime
    close_dt = datetime(int(date_str[:4]),
                        int(date_str[5:7]),
                        int(date_str[8:10]), 16, 0)

    for strike in strikes:
        for opt_type in ["C", "P"]:
            count += 1
            progress_bar.progress(
                count / total,
                text=f"Fetching {strike}{opt_type}  ({count}/{total})")
            try:
                time_module.sleep(MIN_DELAY)
                df = get_intra(symbol="SPX", date=date_str,
                               expDate=exp_date_str,
                               strike=str(strike),
                               optType=opt_type,
                               minuteType=minute_type)
                _lrt[0] = time_module.time()
                if df is None or len(df) == 0:
                    continue

                df = df.copy()
                df["_strike"]  = strike
                df["_optType"] = opt_type

                if "optionIv" in df.columns:
                    df["optionIv"] = pd.to_numeric(df["optionIv"],
                                                    errors="coerce")
                    df = df[df["optionIv"].notna() &
                            (df["optionIv"] > 0) &
                            (df["optionIv"] != -1)].copy()

                if len(df) == 0:
                    continue

                df["optionOI"] = df.apply(
                    lambda r: oi_map.get(
                        (float(r["_strike"]), r["_optType"]), pd.NA),
                    axis=1)
                df["timestamp"]  = pd.to_datetime(df["timestamp"])
                df["_dte_years"] = df["timestamp"].apply(
                    lambda ts: max(
                        (close_dt - ts.to_pydatetime()
                         ).total_seconds() / (252 * 6.5 * 3600),
                        1e-8))
                df["optionVanna"] = df.apply(
                    lambda r: bsm_vanna(
                        r.get("underlyingPrice", pd.NA),
                        r["_strike"], r["_dte_years"],
                        RISK_FREE, r.get("optionIv", pd.NA)), axis=1)
                df["optionCharm"] = df.apply(
                    lambda r: bsm_charm(
                        r.get("underlyingPrice", pd.NA),
                        r["_strike"], r["_dte_years"],
                        RISK_FREE, r.get("optionIv", pd.NA)), axis=1)
                all_data.append(df)
            except Exception as e:
                st.warning(f"Error {strike}{opt_type}: {e}")

    if not all_data:
        return None
    combined              = pd.concat(all_data, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])
    return combined[
        (combined["timestamp"].dt.time >= SESSION_START) &
        (combined["timestamp"].dt.time <= SESSION_END)
    ].copy()

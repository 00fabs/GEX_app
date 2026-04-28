# ─────────────────────────────────────────────────────────────
# SPX GEX / DEX Multi-Formula Analyzer — Streamlit App
# pip install streamlit requests pandas scipy ivolatility
#             streamlit-lightweight-charts
# ─────────────────────────────────────────────────────────────

import streamlit as st
import requests
import pandas as pd
import numpy as np
import ivolatility as ivol
import gzip
import io
import time as time_module
from datetime import datetime, timedelta, time, date
from scipy.stats import norm
from streamlit_lightweight_charts import renderLightweightCharts

st.set_page_config(page_title="SPX GEX/DEX Analyzer", layout="wide")
st.title("📊 SPX GEX / DEX Analyzer")

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
BASE_URL      = "https://restapi.ivolatility.com"
MIN_DELAY     = 1.05
POLL_DELAY    = 2.1
SPX_MULT      = 100
RISK_FREE     = 0.0525
SESSION_START = time(9, 30)
SESSION_END   = time(16, 0)
STRIKE_STEP   = 5          # SPX 0DTE standard step

_last_request_time = [0.0]

def rate_limited_get(url, headers, params):
    elapsed = time_module.time() - _last_request_time[0]
    if elapsed < MIN_DELAY:
        time_module.sleep(MIN_DELAY - elapsed)
    r = requests.get(url, headers=headers, params=params)
    _last_request_time[0] = time_module.time()
    return r

# ─────────────────────────────────────────────────────────────
# TIMEZONE
# ─────────────────────────────────────────────────────────────
def eat_to_et(d, t):
    eat_dt = datetime.combine(d, t)
    is_edt = 3 <= d.month <= 10
    et_dt  = eat_dt - timedelta(hours=7 if is_edt else 8)
    return et_dt, ("EDT" if is_edt else "EST")

def et_to_eat(dt_et, d):
    is_edt = 3 <= d.month <= 10
    return dt_et + timedelta(hours=7 if is_edt else 8)

def prev_trading_day(d):
    step = 3 if d.weekday() == 0 else 1
    return d - timedelta(days=step)

def session_open_et(d):
    return datetime.combine(d, time(9, 30))

def session_close_et(d):
    return datetime.combine(d, time(16, 0))

# ─────────────────────────────────────────────────────────────
# BSM VANNA & CHARM
# ─────────────────────────────────────────────────────────────
def _d1d2(S, K, T, r, sigma):
    if T <= 1e-8 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan, np.nan
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)

def bsm_vanna(S, K, T, r, sigma):
    d1, d2 = _d1d2(S, K, T, r, sigma)
    if np.isnan(d1): return np.nan
    return -norm.pdf(d1) * d2 / sigma

def bsm_charm(S, K, T, r, sigma):
    d1, d2 = _d1d2(S, K, T, r, sigma)
    if np.isnan(d1) or T <= 1e-8: return np.nan
    return -norm.pdf(d1) * (2*r*T - d2*sigma*np.sqrt(T)) / (2*T*sigma*np.sqrt(T))

# ─────────────────────────────────────────────────────────────
# API HELPERS
# ─────────────────────────────────────────────────────────────
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
        if "urlForDetails" in obj and obj["urlForDetails"]: return obj["urlForDetails"]
        for v in obj.values():
            f = find_poll_url(v, depth+1)
            if f: return f
    elif isinstance(obj, list):
        for item in obj:
            f = find_poll_url(item, depth+1)
            if f: return f
    return None

def async_download(endpoint, params, headers, label="", max_polls=25):
    r = rate_limited_get(f"{BASE_URL}{endpoint}", headers=headers, params=params)
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
        if rp.status_code == 429: time_module.sleep(3); continue
        if rp.status_code != 200: continue
        url = find_download_url(rp.json())
        if url: return url
    st.error(f"[{label}] Polling timed out")
    return None

def download_csv_gz(url, headers, api_key, label=""):
    r = rate_limited_get(url, headers=headers, params={"apiKey": api_key})
    if r.status_code != 200: return None
    try:
        with gzip.open(io.BytesIO(r.content), "rt") as f:
            return pd.read_csv(f)
    except Exception:
        try:
            return pd.read_csv(io.BytesIO(r.content))
        except Exception as e:
            st.error(f"CSV parse error: {e}"); return None

def sync_call(endpoint, params, headers, label=""):
    r = rate_limited_get(f"{BASE_URL}{endpoint}", headers=headers, params=params)
    if r.status_code == 429:
        time_module.sleep(3)
        r = rate_limited_get(f"{BASE_URL}{endpoint}", headers=headers, params=params)
    if r.status_code != 200: return None
    body    = r.json()
    records = body.get("data", []) if isinstance(body, dict) else body
    return records if records else None

# ─────────────────────────────────────────────────────────────
# STEP 1 — Option IDs
# ─────────────────────────────────────────────────────────────
def get_option_ids(api_key, headers, prev_date_str, exp_date_str,
                   strike_min, strike_max):
    dl_url = async_download(
        "/equities/eod/option-series-on-date",
        {"symbol": "SPX", "date": prev_date_str, "apiKey": api_key},
        headers, label="series"
    )
    if not dl_url: return None, None

    series_df = download_csv_gz(dl_url, headers, api_key, label="series")
    if series_df is None or series_df.empty:
        st.error("Empty option series"); return None, None

    series_df.columns           = [c.strip().lower() for c in series_df.columns]
    series_df["expirationdate"] = pd.to_datetime(series_df["expirationdate"])
    series_df["strike"]         = pd.to_numeric(series_df["strike"], errors="coerce")

    exp_mask    = series_df["expirationdate"] == pd.Timestamp(exp_date_str)
    strike_mask = (series_df["strike"] >= strike_min) & \
                  (series_df["strike"] <= strike_max)
    filtered    = series_df[exp_mask & strike_mask].copy()

    if filtered.empty:
        st.error(f"No contracts for expiry {exp_date_str} in "
                 f"{strike_min}–{strike_max}")
        return None, None

    cp_col = next((c for c in filtered.columns
                   if c in ["callput","call_put","type","optiontype"]), None)
    if not cp_col:
        st.error("Cannot identify call/put column"); return None, None

    filtered["_is_spx"] = (filtered["optionsymbol"]
                           .str.strip().str.startswith("SPX "))
    filtered = (filtered.sort_values("_is_spx", ascending=False)
                        .drop_duplicates(subset=["strike", cp_col], keep="first")
                        .drop(columns=["_is_spx"])
                        .reset_index(drop=True))

    return (filtered[filtered[cp_col] == "C"].sort_values("strike"),
            filtered[filtered[cp_col] == "P"].sort_values("strike"))

# ─────────────────────────────────────────────────────────────
# STEP 2 — EOD OI
# ─────────────────────────────────────────────────────────────
def get_eod_oi(api_key, headers, calls, puts, prev_date_str):
    oi_map = {}
    def fetch_oi(row, cp_label):
        rec = sync_call(
            "/equities/eod/single-stock-option-raw-iv",
            {"optionId": int(row["optionid"]),
             "from": prev_date_str, "to": prev_date_str,
             "apiKey": api_key},
            headers, label=f"{cp_label}{int(row['strike'])}"
        )
        if not rec: return
        d      = {k.strip().lower().replace(" ", ""): v
                  for k, v in rec[0].items()}
        oi_val = (d.get("openinterest") or d.get("open_interest") or
                  d.get("oi") or d.get("openint") or 0)
        oi_map[(float(row["strike"]), cp_label)] = \
            int(float(oi_val)) if oi_val else 0

    for _, row in calls.iterrows(): fetch_oi(row, "C")
    for _, row in puts.iterrows():  fetch_oi(row, "P")
    return oi_map

# ─────────────────────────────────────────────────────────────
# STEP 3a — Pilot fetch: establish session high/low + spot range
# Fetches ATM call only (1 strike) for the full session to read
# underlyingPrice, then returns high/low so we can build the
# final strike list before the full fetch.
# ─────────────────────────────────────────────────────────────
def get_session_price_range(api_key, date_str, exp_date_str,
                             rough_center, minute_type="MINUTE_1"):
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
        _last_request_time[0] = time_module.time()
    except Exception as e:
        st.warning(f"Pilot fetch failed: {e}"); return None, None

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
    lo = df["underlyingPrice"].min()
    hi = df["underlyingPrice"].max()
    return lo, hi

# ─────────────────────────────────────────────────────────────
# STEP 3b — Full intraday fetch (strike-by-strike, 1-min bars)
# ─────────────────────────────────────────────────────────────
def get_intraday_greeks(api_key, date_str, exp_date_str,
                        strike_min, strike_max,
                        minute_type, oi_map, progress_bar):
    ivol.setLoginParams(apiKey=api_key)
    get_intra = ivol.setMethod(
        "/equities/intraday/single-equity-option-rawiv")

    strikes  = list(range(int(strike_min),
                          int(strike_max) + STRIKE_STEP,
                          STRIKE_STEP))
    total    = len(strikes) * 2
    all_data = []
    count    = 0
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
                df = get_intra(
                    symbol="SPX", date=date_str,
                    expDate=exp_date_str,
                    strike=str(strike),
                    optType=opt_type,
                    minuteType=minute_type)
                _last_request_time[0] = time_module.time()
                if df is None or len(df) == 0: continue

                df = df.copy()
                df["_strike"]  = strike
                df["_optType"] = opt_type

                if "optionIv" in df.columns:
                    df["optionIv"] = pd.to_numeric(df["optionIv"],
                                                    errors="coerce")
                    df = df[df["optionIv"].notna() &
                            (df["optionIv"] > 0) &
                            (df["optionIv"] != -1)].copy()

                if len(df) == 0: continue

                df["optionOI"] = df.apply(
                    lambda r: oi_map.get(
                        (float(r["_strike"]), r["_optType"]), np.nan),
                    axis=1)
                df["timestamp"]  = pd.to_datetime(df["timestamp"])
                df["_dte_years"] = df["timestamp"].apply(
                    lambda ts: max(
                        (close_dt - ts.to_pydatetime()
                         ).total_seconds() / (252 * 6.5 * 3600),
                        1e-8))
                df["optionVanna"] = df.apply(
                    lambda r: bsm_vanna(
                        r.get("underlyingPrice", np.nan),
                        r["_strike"], r["_dte_years"],
                        RISK_FREE, r.get("optionIv", np.nan)), axis=1)
                df["optionCharm"] = df.apply(
                    lambda r: bsm_charm(
                        r.get("underlyingPrice", np.nan),
                        r["_strike"], r["_dte_years"],
                        RISK_FREE, r.get("optionIv", np.nan)), axis=1)
                all_data.append(df)
            except Exception as e:
                st.warning(f"Error {strike}{opt_type}: {e}")

    if not all_data: return None
    combined              = pd.concat(all_data, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])
    return combined[
        (combined["timestamp"].dt.time >= SESSION_START) &
        (combined["timestamp"].dt.time <= SESSION_END)
    ].copy()

# ─────────────────────────────────────────────────────────────
# PIVOT — wide format
# ─────────────────────────────────────────────────────────────
COL_MAP = {
    "optionDelta":    "delta",
    "optionGamma":    "gamma",
    "optionIv":       "iv",
    "optionVanna":    "vanna",
    "optionCharm":    "charm",
    "optionOI":       "oi",
    "optionVolume":   "volume",
    "optionBidPrice": "bid",
    "optionAskPrice": "ask",
}

def pivot_wide(df_all):
    def pivot_side(df, prefix):
        rename = {k: f"{prefix}_{v}"
                  for k, v in COL_MAP.items() if k in df.columns}
        df     = df.rename(columns=rename)
        base   = (["timestamp", "_strike", "underlyingPrice"]
                  if prefix == "call"
                  else ["timestamp", "_strike"])
        keep   = base + list(rename.values())
        return df[[c for c in keep if c in df.columns]].copy()

    calls_p = pivot_side(
        df_all[df_all["_optType"] == "C"].copy(), "call")
    puts_p  = pivot_side(
        df_all[df_all["_optType"] == "P"].copy(), "put")
    merged  = calls_p.merge(puts_p, on=["timestamp", "_strike"],
                             how="outer")
    merged  = merged.rename(
        columns={"_strike": "strike", "underlyingPrice": "spot"})
    return merged.sort_values(["timestamp", "strike"]).reset_index(drop=True)

# ─────────────────────────────────────────────────────────────
# FORMULA ENGINE
# ─────────────────────────────────────────────────────────────
def apply_formulas(df, spot_override, intra_date):
    out = df.copy()

    spot_col = (out["spot"].fillna(spot_override)
                if "spot" in out.columns
                else pd.Series([spot_override] * len(out),
                               index=out.index))
    out["spot_used"] = spot_col

    def col(name):
        return out.get(name,
                       pd.Series(0.0, index=out.index)).fillna(0)

    cg  = col("call_gamma");  pg  = col("put_gamma")
    coi = col("call_oi");     poi = col("put_oi")
    cd  = col("call_delta");  pd_ = col("put_delta")

    S  = spot_col
    S2 = S ** 2
    M  = SPX_MULT

    out["GEX_unsigned"]   = (cg * coi + pg * poi) * M * S2
    out["GEX_unsigned_$"] = out["GEX_unsigned"] / 1e9

    out["GEX_signed"]   = (cg * coi - pg * poi) * M * S2
    out["GEX_signed_$"] = out["GEX_signed"] / 1e9

    if "timestamp" in out.columns:
        agg_map = (out.groupby("timestamp")["GEX_signed"]
                      .sum().rename("GEX_agg_oi"))
        out = out.merge(agg_map, on="timestamp", how="left")
    else:
        out["GEX_agg_oi"] = out["GEX_signed"].sum()
    out["GEX_agg_oi_$"] = out["GEX_agg_oi"] / 1e9

    out["GEX_dealer_sp"]   = -(cg * coi + pg * poi) * M * S2
    out["GEX_dealer_sp_$"] = out["GEX_dealer_sp"] / 1e9

    out["DEX"]   = (cd * coi + pd_ * poi) * M * S
    out["DEX_$"] = out["DEX"] / 1e9

    return out

# ─────────────────────────────────────────────────────────────
# BUILD FULL MINUTE-BY-MINUTE TIME SERIES (pre-computed)
# Returns dict: { timestamp_str -> { strike -> {gex_vals, dex_val, spot} } }
# Also returns sorted list of timestamps and list of strikes
# ─────────────────────────────────────────────────────────────
def build_minute_series(wide_df, spot_override, intra_date):
    result = apply_formulas(wide_df, spot_override, intra_date)

    gex_cols = ["GEX_unsigned_$", "GEX_signed_$",
                "GEX_agg_oi_$",   "GEX_dealer_sp_$"]
    dex_col  = "DEX_$"

    # Build nested dict keyed by timestamp then strike
    ts_groups = result.groupby("timestamp")
    series = {}
    for ts, grp in ts_groups:
        ts_key = pd.Timestamp(ts).strftime("%H:%M")
        strikes_data = {}
        for _, row in grp.iterrows():
            sk = int(row["strike"])
            strikes_data[sk] = {
                "GEX_unsigned": float(row.get("GEX_unsigned_$", 0) or 0),
                "GEX_signed":   float(row.get("GEX_signed_$",   0) or 0),
                "GEX_agg_oi":   float(row.get("GEX_agg_oi_$",   0) or 0),
                "GEX_dealer_sp":float(row.get("GEX_dealer_sp_$",0) or 0),
                "DEX":          float(row.get(dex_col,           0) or 0),
                "spot":         float(row.get("spot_used",
                                              spot_override) or spot_override),
            }
        series[ts_key] = strikes_data

    sorted_ts = sorted(series.keys())

    # All strikes present across all timestamps
    all_strikes = sorted(set(
        sk for ts_data in series.values()
        for sk in ts_data.keys()))

    return series, sorted_ts, all_strikes

# ─────────────────────────────────────────────────────────────
# BUILD FULL SESSION TABLE (for data tabs)
# ─────────────────────────────────────────────────────────────
def build_session_table(wide_df, spot_override, intra_date):
    result = apply_formulas(wide_df, spot_override, intra_date)

    greek_cols   = ["call_iv","call_delta","call_gamma","call_vanna",
                    "call_charm","call_oi","call_volume",
                    "put_iv","put_delta","put_gamma","put_vanna",
                    "put_charm","put_oi","put_volume"]
    formula_cols = [c for c in result.columns if c.endswith("_$")]

    keep = (["timestamp","strike","spot_used"] +
            [c for c in greek_cols   if c in result.columns] +
            [c for c in formula_cols if c in result.columns])

    ts_df = result[[c for c in keep if c in result.columns]].copy()
    ts_df = ts_df.rename(columns={"spot_used": "spot"})
    ts_df = ts_df.sort_values(["timestamp","strike"]).reset_index(drop=True)
    ts_df.rename(
        columns={c: c.replace("_$", " ($)")
                 for c in formula_cols if c in ts_df.columns},
        inplace=True)
    return ts_df

# ─────────────────────────────────────────────────────────────
# CHART BUILDER — lightweight-charts histogram
# series_data: list of {strike, value} dicts
# spot: float — current spot price for the price line label
# title: str
# ─────────────────────────────────────────────────────────────
def build_histogram_chart(series_data, spot, title):
    """
    Renders a histogram chart using streamlit-lightweight-charts.
    X-axis = strike (treated as 'time' — we use integer strikes
    mapped to the time field as Unix-

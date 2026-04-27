# ─────────────────────────────────────────────────────────────
# SPX GEX / DEX Multi-Formula Analyzer — Streamlit App
# pip install streamlit requests pandas scipy ivolatility
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

def prev_trading_day(d):
    step = 3 if d.weekday() == 0 else 1
    return d - timedelta(days=step)

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
        rp = rate_limited_get(poll_url, headers=headers, params={"apiKey": params["apiKey"]})
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
def get_option_ids(api_key, headers, prev_date_str, exp_date_str, strike_min, strike_max):
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
    strike_mask = (series_df["strike"] >= strike_min) & (series_df["strike"] <= strike_max)
    filtered    = series_df[exp_mask & strike_mask].copy()

    if filtered.empty:
        st.error(f"No contracts for expiry {exp_date_str} in {strike_min}–{strike_max}")
        return None, None

    cp_col = next((c for c in filtered.columns
                   if c in ["callput","call_put","type","optiontype"]), None)
    if not cp_col:
        st.error("Cannot identify call/put column"); return None, None

    filtered["_is_spx"] = filtered["optionsymbol"].str.strip().str.startswith("SPX ")
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
             "from": prev_date_str, "to": prev_date_str, "apiKey": api_key},
            headers, label=f"{cp_label}{int(row['strike'])}"
        )
        if not rec: return
        d      = {k.strip().lower().replace(" ", ""): v for k, v in rec[0].items()}
        oi_val = (d.get("openinterest") or d.get("open_interest") or
                  d.get("oi") or d.get("openint") or 0)
        oi_map[(float(row["strike"]), cp_label)] = int(float(oi_val)) if oi_val else 0

    for _, row in calls.iterrows(): fetch_oi(row, "C")
    for _, row in puts.iterrows():  fetch_oi(row, "P")
    return oi_map

# ─────────────────────────────────────────────────────────────
# STEP 3 — Intraday Greeks
# ─────────────────────────────────────────────────────────────
def get_intraday_greeks(api_key, date_str, exp_date_str,
                        strike_min, strike_max, strike_step,
                        minute_type, oi_map, progress_bar):
    ivol.setLoginParams(apiKey=api_key)
    get_intra = ivol.setMethod("/equities/intraday/single-equity-option-rawiv")

    strikes  = list(range(int(strike_min), int(strike_max) + int(strike_step), int(strike_step)))
    total    = len(strikes) * 2
    all_data = []
    count    = 0
    close_dt = datetime(int(date_str[:4]), int(date_str[5:7]), int(date_str[8:10]), 16, 0)

    for strike in strikes:
        for opt_type in ["C", "P"]:
            count += 1
            progress_bar.progress(count / total,
                                  text=f"Fetching {strike}{opt_type}  ({count}/{total})")
            try:
                time_module.sleep(MIN_DELAY)
                df = get_intra(
                    symbol="SPX", date=date_str, expDate=exp_date_str,
                    strike=str(strike), optType=opt_type, minuteType=minute_type
                )
                _last_request_time[0] = time_module.time()
                if df is None or len(df) == 0: continue

                df = df.copy()
                df["_strike"]  = strike
                df["_optType"] = opt_type

                if "optionIv" in df.columns:
                    df["optionIv"] = pd.to_numeric(df["optionIv"], errors="coerce")
                    df = df[df["optionIv"].notna() &
                            (df["optionIv"] > 0) &
                            (df["optionIv"] != -1)].copy()

                if len(df) == 0: continue

                df["optionOI"]    = df.apply(
                    lambda r: oi_map.get((float(r["_strike"]), r["_optType"]), np.nan), axis=1)
                df["timestamp"]   = pd.to_datetime(df["timestamp"])
                df["_dte_years"]  = df["timestamp"].apply(
                    lambda ts: max(
                        (close_dt - ts.to_pydatetime()).total_seconds() / (252 * 6.5 * 3600),
                        1e-8))
                df["optionVanna"] = df.apply(
                    lambda r: bsm_vanna(r.get("underlyingPrice", np.nan), r["_strike"],
                                        r["_dte_years"], RISK_FREE, r.get("optionIv", np.nan)), axis=1)
                df["optionCharm"] = df.apply(
                    lambda r: bsm_charm(r.get("underlyingPrice", np.nan), r["_strike"],
                                        r["_dte_years"], RISK_FREE, r.get("optionIv", np.nan)), axis=1)
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
# PIVOT — wide format (used for both snapshot and time series)
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
    """
    Returns a wide DataFrame with one row per (timestamp, strike).
    Calls and puts columns are prefixed call_ / put_.
    spot comes from call side underlyingPrice.
    """
    def pivot_side(df, prefix):
        rename = {k: f"{prefix}_{v}" for k, v in COL_MAP.items() if k in df.columns}
        df     = df.rename(columns=rename)
        base   = (["timestamp", "_strike", "underlyingPrice"]
                  if prefix == "call" else ["timestamp", "_strike"])
        keep   = base + list(rename.values())
        return df[[c for c in keep if c in df.columns]].copy()

    calls_p = pivot_side(df_all[df_all["_optType"] == "C"].copy(), "call")
    puts_p  = pivot_side(df_all[df_all["_optType"] == "P"].copy(), "put")
    merged  = calls_p.merge(puts_p, on=["timestamp", "_strike"], how="outer")
    merged  = merged.rename(columns={"_strike": "strike", "underlyingPrice": "spot"})
    return merged.sort_values(["timestamp", "strike"]).reset_index(drop=True)

# ─────────────────────────────────────────────────────────────
# FORMULA ENGINE — operates on any wide DataFrame slice
# Returns the same DataFrame with GEX/DEX columns appended
# ─────────────────────────────────────────────────────────────
def apply_formulas(df, spot_override, intra_date):
    out = df.copy()

    spot_col = (out["spot"].fillna(spot_override)
                if "spot" in out.columns
                else pd.Series([spot_override] * len(out), index=out.index))
    out["spot_used"] = spot_col

    def col(name):
        return out.get(name, pd.Series(0.0, index=out.index)).fillna(0)

    cg   = col("call_gamma");  pg   = col("put_gamma")
    coi  = col("call_oi");     poi  = col("put_oi")
    cvol = col("call_volume"); pvol = col("put_volume")
    cd   = col("call_delta");  pd_  = col("put_delta")
    cv   = col("call_vanna");  pv   = col("put_vanna")
    cc   = col("call_charm");  pc   = col("put_charm")
    civ  = out.get("call_iv",  pd.Series(np.nan, index=out.index))
    piv  = out.get("put_iv",   pd.Series(np.nan, index=out.index))

    S  = spot_col
    S2 = S ** 2
    M  = SPX_MULT

    # Weighted OI
    vsc  = (coi / cvol.replace(0, np.nan)).fillna(1).clip(0, 10)
    vsp  = (poi / pvol.replace(0, np.nan)).fillna(1).clip(0, 10)
    cwoi = 0.7 * coi + 0.3 * (cvol * vsc)
    pwoi = 0.7 * poi + 0.3 * (pvol * vsp)

    # ATM IV proxy
    atm_iv    = civ.fillna(piv).mean()
    iv_diff_c = civ.fillna(atm_iv) - atm_iv
    iv_diff_p = piv.fillna(atm_iv) - atm_iv

    # DTE fraction per row — uses timestamp if present, else 0.5
    sess_open  = datetime.combine(intra_date, time(9, 30))
    sess_close = datetime.combine(intra_date, time(16, 0))
    total_mins = (sess_close - sess_open).seconds / 60

    if "timestamp" in out.columns:
        def _dte_frac(ts):
            rem = max((sess_close - ts.to_pydatetime()).total_seconds() / 60, 0)
            return rem / total_mins
        dte_frac = out["timestamp"].apply(_dte_frac)
    else:
        dte_frac = pd.Series(0.5, index=out.index)

    # ── GEX ──────────────────────────────────────────────────
    out["GEX1"]   = (cg*coi  - pg*poi)      * M * S2
    out["GEX1_$"] = out["GEX1"] * S / 1e9

    out["GEX2"]   = (cg*coi  + pg*poi)      * M * S2
    out["GEX2_$"] = out["GEX2"] * S / 1e9

    total_oi      = (coi + poi).replace(0, np.nan)
    skew          = (coi / total_oi).fillna(0.5)
    out["GEX3"]   = (cg*coi  - skew*pg*poi) * M * S2
    out["GEX3_$"] = out["GEX3"] * S / 1e9

    out["GEX4"]   = (cg*cwoi - pg*pwoi)     * M * S2
    out["GEX4_$"] = out["GEX4"] * S / 1e9

    out["GEX5"]   = (cg*cvol - pg*pvol)     * M * S2
    out["GEX5_$"] = out["GEX5"] * S / 1e9

    # ── DEX ──────────────────────────────────────────────────
    out["DEX1"]   = (cd*coi  - pd_*poi)     * M * S
    out["DEX1_$"] = out["DEX1"] * S / 1e9

    out["DEX2"]   = (cd*cwoi - pd_*pwoi)    * M * S
    out["DEX2_$"] = out["DEX2"] * S / 1e9

    out["DEX3"]   = (cd*cvol - pd_*pvol)    * M * S
    out["DEX3_$"] = out["DEX3"] * S / 1e9

    charm_flow    = (cc*coi  - pc*poi)  * M * dte_frac
    out["DEX4"]   = out["DEX1"] - charm_flow * S
    out["DEX4_$"] = out["DEX4"] * S / 1e9

    vanna_flow    = (cv*coi*iv_diff_c - pv*poi*iv_diff_p) * M
    out["DEX5"]   = out["DEX1"] + vanna_flow * S
    out["DEX5_$"] = out["DEX5"] * S / 1e9

    # Gamma flip per-strike (only meaningful on snapshot slices)
    g     = out["GEX1"].values
    flips = []
    for i in range(1, len(g)):
        if not (np.isnan(g[i-1]) or np.isnan(g[i])) and g[i-1] * g[i] < 0:
            flips.append(out["strike"].iloc[i])
    out["_flip"] = out["strike"].isin(flips)

    return out, flips

# ─────────────────────────────────────────────────────────────
# BUILD FULL SESSION GEX/DEX TIME SERIES
# One row per (timestamp, strike) with all formulas calculated
# ─────────────────────────────────────────────────────────────
def build_session_timeseries(wide_df, spot_override, intra_date):
    """
    Runs apply_formulas on the entire wide_df (all timestamps × all strikes).
    Returns a clean table: timestamp, strike, spot, all GEX_$, all DEX_$.
    """
    result, _ = apply_formulas(wide_df, spot_override, intra_date)

    keep = (["timestamp", "strike", "spot_used"] +
            [c for c in result.columns if c.endswith("_$")])
    ts_df = result[[c for c in keep if c in result.columns]].copy()
    ts_df = ts_df.sort_values(["timestamp", "strike"]).reset_index(drop=True)

    # Rename for display
    ts_df.rename(columns={"spot_used": "spot"}, inplace=True)
    dollar_cols = [c for c in ts_df.columns if c.endswith("_$")]
    ts_df.rename(columns={c: c.replace("_$", " ($)") for c in dollar_cols}, inplace=True)
    return ts_df

# ─────────────────────────────────────────────────────────────
# SNAPSHOT — latest bar per strike at or before requested time
# ─────────────────────────────────────────────────────────────
def get_snapshot(wide_df, target_et_full):
    snap = (wide_df[wide_df["timestamp"] <= pd.Timestamp(target_et_full)]
            .sort_values("timestamp")
            .groupby("strike").last()
            .reset_index())
    open_snap = (wide_df.sort_values("timestamp")
                 .groupby("strike").first()
                 .reset_index())
    return snap, open_snap

# ─────────────────────────────────────────────────────────────
# REGIME
# ─────────────────────────────────────────────────────────────
def interpret_regime(df, flip_strikes):
    gex1_sum = df["GEX1"].sum() if "GEX1" in df.columns else 0
    dex1_sum = df["DEX1"].sum() if "DEX1" in df.columns else 0

    regime = ("🔴 Negative Gamma — Trending / Volatile"
              if gex1_sum < 0 else
              "🟢 Positive Gamma — Mean-Reverting / Pinned")

    if gex1_sum < 0 and dex1_sum > 0:
        signal = "⚡ Bullish Move — Negative GEX + Positive DEX"
    elif gex1_sum < 0 and dex1_sum <= 0:
        signal = "⚡ Bearish Move — Negative GEX + Negative DEX"
    else:
        signal = "🔒 Pin / Fade — Positive GEX, expect mean reversion"

    flip_str = (", ".join(str(s) for s in flip_strikes)
                if flip_strikes else "None in range")
    return regime, signal, flip_str

# ─────────────────────────────────────────────────────────────
# FORMAT HELPERS
# ─────────────────────────────────────────────────────────────
def fmt_b(val):
    if pd.isna(val) or val is None: return "N/A"
    v = float(val)
    if abs(v) >= 1e9: return f"${v/1e9:.2f}B"
    if abs(v) >= 1e6: return f"${v/1e6:.2f}M"
    if abs(v) >= 1e3: return f"${v/1e3:.1f}K"
    return f"${v:.2f}"

def fmt_df_dollars(df):
    out = df.copy()
    for c in out.columns:
        if "($)" in c:
            out[c] = out[c].apply(fmt_b)
    return out

def hl_flip(row):
    flag = row.get("Flip", False)
    return (["background-color:#fef08a;color:#000"] * len(row)
            if flag else [""] * len(row))

# ─────────────────────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────────────────────
def display_results(snap_df, ts_df, regime, signal, flip_str,
                    spot_actual, date_str, et_label):

    # ── Header metrics ───────────────────────────────────────
    st.subheader(f"Snapshot — {date_str}  |  {et_label}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Regime",           regime)
    c2.metric("Signal",           signal)
    c3.metric("Gamma Flip Level", flip_str)
    c4.metric("Spot at Snapshot", f"{spot_actual:.2f}" if spot_actual else "N/A")
    st.divider()

    # ── GEX snapshot per strike ──────────────────────────────
    st.subheader("GEX Per Strike — Snapshot")
    gex_cols  = (["strike"] +
                 [c for c in snap_df.columns if c.startswith("GEX") and c.endswith("_$")] +
                 ["_flip"])
    gex_df    = snap_df[[c for c in gex_cols if c in snap_df.columns]].copy()
    gex_df.rename(columns={c: c.replace("_$", " ($)") for c in gex_df.columns
                            if c.endswith("_$")}, inplace=True)
    gex_df.rename(columns={"_flip": "Flip"}, inplace=True)
    st.dataframe(fmt_df_dollars(gex_df).style.apply(hl_flip, axis=1),
                 use_container_width=True, hide_index=True)

    # ── DEX snapshot per strike ──────────────────────────────
    st.subheader("DEX Per Strike — Snapshot")
    dex_cols = (["strike"] +
                [c for c in snap_df.columns if c.startswith("DEX") and c.endswith("_$")])
    dex_df   = snap_df[[c for c in dex_cols if c in snap_df.columns]].copy()
    dex_df.rename(columns={c: c.replace("_$", " ($)") for c in dex_df.columns
                            if c.endswith("_$")}, inplace=True)
    st.dataframe(fmt_df_dollars(dex_df), use_container_width=True, hide_index=True)

    # ── Raw Greeks snapshot ──────────────────────────────────
    with st.expander("Raw Greeks + OI + Volume — Snapshot"):
        raw_cols = ["strike", "spot_used",
                    "call_iv","call_delta","call_gamma","call_vanna","call_charm",
                    "call_oi","call_volume",
                    "put_iv","put_delta","put_gamma","put_vanna","put_charm",
                    "put_oi","put_volume"]
        raw_df = snap_df[[c for c in raw_cols if c in snap_df.columns]].copy()
        st.dataframe(raw_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Full session GEX time series ─────────────────────────
    st.subheader("GEX — Full Session Time Series (all strikes × all bars)")
    gex_ts_cols = (["timestamp", "strike", "spot"] +
                   [c for c in ts_df.columns
                    if c.startswith("GEX") and "($)" in c])
    gex_ts = ts_df[[c for c in gex_ts_cols if c in ts_df.columns]].copy()
    st.dataframe(fmt_df_dollars(gex_ts), use_container_width=True, hide_index=True)

    # ── Full session DEX time series ─────────────────────────
    st.subheader("DEX — Full Session Time Series (all strikes × all bars)")
    dex_ts_cols = (["timestamp", "strike", "spot"] +
                   [c for c in ts_df.columns
                    if c.startswith("DEX") and "($)" in c])
    dex_ts = ts_df[[c for c in dex_ts_cols if c in ts_df.columns]].copy()
    st.dataframe(fmt_df_dollars(dex_ts), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────
# INPUT FORM
# ─────────────────────────────────────────────────────────────
st.header("Parameters")

with st.form("params_form"):
    r1c1, r1c2 = st.columns(2)
    api_key     = r1c1.text_input("iVolatility API Key", type="password",
                                   placeholder="Paste your Backtest+ key")
    minute_type = r1c2.selectbox("Bar Interval",
                                  ["MINUTE_30","MINUTE_15","MINUTE_5","MINUTE_1"])

    r2c1, r2c2 = st.columns(2)
    date_input  = r2c1.date_input("Date (EAT)", value=date.today())
    time_input  = r2c2.time_input("Session Time (EAT)", value=time(17, 30))

    r3c1, r3c2, r3c3 = st.columns(3)
    center_strike = r3c1.number_input("Center Strike",      value=5580, step=5)
    num_strikes   = r3c2.number_input("Strikes Each Side",  value=5,
                                       min_value=1, max_value=20, step=1)
    strike_step   = r3c3.number_input("Strike Step",        value=5, min_value=1, step=1)

    spot_override = st.number_input(
        "Spot Override — used only if API underlyingPrice is missing",
        value=float(center_strike), step=1.0
    )
    submitted = st.form_submit_button("🚀 Fetch & Calculate", use_container_width=True)

# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────
if submitted:
    if not api_key:
        st.error("Enter your iVolatility API key"); st.stop()

    date_str      = date_input.strftime("%Y-%m-%d")
    prev_date_str = prev_trading_day(date_input).strftime("%Y-%m-%d")
    exp_date_str  = date_str
    et_dt, tz_str = eat_to_et(date_input, time_input)
    et_label      = f"{et_dt.strftime('%H:%M')} {tz_str}"
    intra_date    = date_input

    strike_min = int(center_strike) - int(num_strikes) * int(strike_step)
    strike_max = int(center_strike) + int(num_strikes) * int(strike_step)
    headers    = {"Authorization": f"Bearer {api_key}"}

    st.info(
        f"Date: {date_str}  |  OI source: {prev_date_str}  |  "
        f"Snapshot: {et_label}  |  Strikes: {strike_min}–{strike_max} "
        f"step {int(strike_step)}"
    )

    # Step 1
    with st.spinner("Step 1 / 3 — Fetching option series..."):
        calls_series, puts_series = get_option_ids(
            api_key, headers, prev_date_str, exp_date_str, strike_min, strike_max)
    if calls_series is None: st.stop()
    st.success(f"Step 1 ✅ — Calls: {len(calls_series)}  Puts: {len(puts_series)}")

    # Step 2
    with st.spinner("Step 2 / 3 — Fetching EOD OI..."):
        oi_map = get_eod_oi(api_key, headers, calls_series, puts_series, prev_date_str)
    filled = sum(1 for v in oi_map.values() if v > 0)
    st.success(f"Step 2 ✅ — OI entries: {filled}/{len(oi_map)} non-zero")

    # Step 3
    st.write("Step 3 / 3 — Fetching intraday Greeks...")
    progress = st.progress(0, text="Starting...")
    df_all = get_intraday_greeks(
        api_key, date_str, exp_date_str,
        strike_min, strike_max, strike_step,
        minute_type, oi_map, progress)
    progress.empty()
    if df_all is None:
        st.error("No intraday data returned"); st.stop()
    st.success(f"Step 3 ✅ — {len(df_all):,} intraday rows")

    # Pivot entire dataset to wide once
    with st.spinner("Building wide table..."):
        wide_df = pivot_wide(df_all)

    # Snapshot at requested time
    target_et_full = datetime.combine(intra_date, et_dt.time())
    snap_raw, _ = get_snapshot(wide_df, target_et_full)

    if snap_raw.empty:
        st.error("No bars at or before the requested time — try a later EAT time")
        st.stop()

    # Apply formulas to snapshot
    snap_calc, flip_strikes = apply_formulas(snap_raw, spot_override, intra_date)

    # Apply formulas to full session (all timestamps × all strikes)
    with st.spinner("Calculating full session GEX/DEX time series..."):
        ts_df = build_session_timeseries(wide_df, spot_override, intra_date)

    # Regime from snapshot
    regime, signal, flip_str = interpret_regime(snap_calc, flip_strikes)

    # Spot
    spot_actual = (snap_calc["spot_used"].median()
                   if "spot_used" in snap_calc.columns else spot_override)

    # Render
    display_results(snap_calc, ts_df, regime, signal, flip_str,
                    spot_actual, date_str, et_label)

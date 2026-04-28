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
    mapped to the time field as Unix-like sequential integers so
    the chart treats them as a numeric axis).
    Y-axis = GEX or DEX value in $B.
    Positive bars = green, negative bars = red.
    Spot shown as a vertical price-line marker.
    """

    # Sort by strike
    series_data = sorted(series_data, key=lambda x: x["strike"])

    if not series_data:
        st.info("No data for this timestamp.")
        return

    # Build histogram data
    # streamlit-lightweight-charts Histogram series uses {time, value, color}
    # We map strike -> sequential integer index so the x-axis is evenly spaced
    strikes = [d["strike"] for d in series_data]
    values  = [d["value"]  for d in series_data]

    hist_data = []
    for i, (sk, val) in enumerate(zip(strikes, values)):
        color = "#26a69a" if val >= 0 else "#ef5350"   # teal-green / red
        hist_data.append({
            "time":  i,          # sequential index as x position
            "value": round(val, 6),
            "color": color,
        })

    # Find spot index (closest strike)
    spot_idx = min(range(len(strikes)),
                   key=lambda i: abs(strikes[i] - spot))
    spot_strike = strikes[spot_idx]
    spot_val    = values[spot_idx]

    # Build x-axis label overrides: show strike numbers
    # We pass custom time labels via the chart options localization
    tick_marks = [{"time": i, "label": str(sk)}
                  for i, sk in enumerate(strikes)]

    chart_options = {
        "layout": {
            "background": {"type": "solid", "color": "#0e1117"},
            "textColor":  "#d1d4dc",
        },
        "grid": {
            "vertLines": {"color": "#1e2130"},
            "horzLines": {"color": "#1e2130"},
        },
        "rightPriceScale": {
            "borderColor": "#2a2e39",
            "scaleMargins": {"top": 0.15, "bottom": 0.15},
        },
        "timeScale": {
            "borderColor":       "#2a2e39",
            "barSpacing":        28,       # bar width — thick but with gap
            "minBarSpacing":     8,
            "tickMarkFormatter": None,     # we override with tickMarks below
            "fixLeftEdge":       True,
            "fixRightEdge":      True,
        },
        "crosshair": {
            "mode": 1,
            "vertLine": {
                "color":   "#9b59b6",
                "width":   1,
                "style":   2,
                "visible": True,
            },
            "horzLine": {
                "color":   "#9b59b6",
                "width":   1,
                "style":   2,
                "visible": True,
            },
        },
        "handleScroll":  True,
        "handleScale":   True,
    }

    series_options = {
        "type":    "Histogram",
        "data":    hist_data,
        "options": {
            "priceFormat": {
                "type":      "custom",
                "formatter": None,   # keep raw float
                "minMove":   0.0001,
            },
            "lastValueVisible":  False,
            "priceLineVisible":  False,
            "baseLineVisible":   True,
            "baseLineColor":     "#ffffff",
            "baseLineWidth":     1,
        },
        # Price lines: spot marker
        "priceLines": [
            {
                "price":       spot_val,
                "color":       "#f0c040",    # amber — distinct from green/red
                "lineWidth":   2,
                "lineStyle":   2,            # dashed
                "axisLabelVisible": True,
                "title":       f"Spot {spot:.1f} @ {spot_strike}",
            }
        ],
        # Markers at spot strike — vertical highlight
        "markers": [
            {
                "time":     spot_idx,
                "position": "aboveBar",
                "color":    "#f0c040",
                "shape":    "arrowDown",
                "text":     f"Spot {spot:.1f}",
            }
        ],
    }

    # Custom tick labels for x-axis (strikes)
    # streamlit-lightweight-charts passes these through chart options
    chart_options["timeScale"]["tickMarks"] = tick_marks

    st.markdown(f"**{title}**")
    renderLightweightCharts(
        [{"chart": chart_options, "series": [series_options]}],
        key=f"chart_{title.replace(' ','_')}_{id(hist_data)}",
    )

    # Strike label legend below chart
    spot_label_col, _ = st.columns([1, 3])
    spot_label_col.markdown(
        f"<span style='color:#f0c040;font-size:13px;'>"
        f"▲ Spot: **{spot:.2f}**  |  Nearest strike: **{spot_strike}**"
        f"</span>",
        unsafe_allow_html=True)

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

# ─────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "computed":        False,
        "minute_series":   {},
        "sorted_ts":       [],
        "all_strikes":     [],
        "ts_index":        0,
        "step_size":       1,
        "ts_table":        None,
        "date_str":        "",
        "spot_override":   5500.0,
        "intra_date":      None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─────────────────────────────────────────────────────────────
# GREEN RADIO STYLE
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
div[data-testid="stRadio"] label {
    cursor: pointer;
}
div[data-testid="stRadio"] div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child {
    background-color: transparent;
}
/* Selected radio dot — green */
div[data-testid="stRadio"] input[type="radio"]:checked + div > div {
    background-color: #26a69a !important;
    border-color:     #26a69a !important;
}
/* Hover */
div[data-testid="stRadio"] label:hover span {
    color: #26a69a;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# INPUT FORM
# ─────────────────────────────────────────────────────────────
st.header("Parameters")

with st.form("params_form"):
    c1, c2, c3 = st.columns(3)

    api_key     = c1.text_input(
        "iVolatility API Key", type="password",
        placeholder="Paste your Backtest+ key")

    date_input  = c2.date_input("Session Date", value=date.today())

    minute_type = c3.selectbox(
        "API Bar Resolution",
        ["MINUTE_1","MINUTE_5","MINUTE_15","MINUTE_30"],
        help="MINUTE_1 gives the finest granularity for stepping.")

    c4, c5 = st.columns(2)
    rough_center  = c4.number_input(
        "Rough ATM Strike (pilot fetch only)",
        value=5580, step=5,
        help="Used for the pilot fetch to establish session high/low. "
             "Approximate ATM is fine — the app auto-calculates the real range.")
    spot_override_input = c5.number_input(
        "Spot Override (fallback if API missing)",
        value=5580.0, step=1.0)

    submitted = st.form_submit_button(
        "🚀 Fetch & Compute Full Session", use_container_width=True)

# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE — runs once, stores everything in session_state
# ─────────────────────────────────────────────────────────────
if submitted:
    if not api_key:
        st.error("Enter your iVolatility API key"); st.stop()

    date_str      = date_input.strftime("%Y-%m-%d")
    prev_date_str = prev_trading_day(date_input).strftime("%Y-%m-%d")
    exp_date_str  = date_str
    headers       = {"Authorization": f"Bearer {api_key}"}

    st.session_state["computed"]      = False
    st.session_state["date_str"]      = date_str
    st.session_state["spot_override"] = spot_override_input
    st.session_state["intra_date"]    = date_input
    st.session_state["ts_index"]      = 0

    # ── Pilot fetch: session high/low ─────────────────────────
    with st.spinner("Pilot fetch — establishing session price range..."):
        lo, hi = get_session_price_range(
            api_key, date_str, exp_date_str,
            rough_center, minute_type)

    if lo is None or hi is None:
        st.error("Pilot fetch failed — check API key and date.")
        st.stop()

    # Strike range: floor/ceil to nearest 5, ±5 strikes buffer
    strike_min = (int(lo // STRIKE_STEP) * STRIKE_STEP
                  - 5 * STRIKE_STEP)
    strike_max = (int(hi // STRIKE_STEP) * STRIKE_STEP
                  + STRIKE_STEP                          # ceil
                  + 5 * STRIKE_STEP)

    st.info(
        f"Date: {date_str}  |  Session range: {lo:.1f} – {hi:.1f}  |  "
        f"Fetching strikes: {strike_min} – {strike_max} "
        f"(step {STRIKE_STEP})")

    # ── Step 1 ────────────────────────────────────────────────
    with st.spinner("Step 1 / 3 — Option series..."):
        calls_series, puts_series = get_option_ids(
            api_key, headers, prev_date_str, exp_date_str,
            strike_min, strike_max)
    if calls_series is None: st.stop()
    st.success(
        f"Step 1 ✅ — Calls: {len(calls_series)}  "
        f"Puts: {len(puts_series)}")

    # ── Step 2 ────────────────────────────────────────────────
    with st.spinner("Step 2 / 3 — EOD OI..."):
        oi_map = get_eod_oi(
            api_key, headers, calls_series,
            puts_series, prev_date_str)
    filled = sum(1 for v in oi_map.values() if v > 0)
    st.success(
        f"Step 2 ✅ — OI entries: {filled}/{len(oi_map)} non-zero")

    # ── Step 3 ────────────────────────────────────────────────
    st.write("Step 3 / 3 — Intraday Greeks (full session)...")
    progress = st.progress(0, text="Starting...")
    df_all = get_intraday_greeks(
        api_key, date_str, exp_date_str,
        strike_min, strike_max,
        minute_type, oi_map, progress)
    progress.empty()
    if df_all is None:
        st.error("No intraday data returned"); st.stop()
    st.success(f"Step 3 ✅ — {len(df_all):,} intraday rows")

    # ── Build wide + compute ───────────────────────────────────
    with st.spinner("Computing GEX / DEX for full session..."):
        wide_df = pivot_wide(df_all)

        minute_series, sorted_ts, all_strikes = build_minute_series(
            wide_df, spot_override_input, date_input)

        ts_table = build_session_table(
            wide_df, spot_override_input, date_input)

    st.session_state["minute_series"] = minute_series
    st.session_state["sorted_ts"]     = sorted_ts
    st.session_state["all_strikes"]   = all_strikes
    st.session_state["ts_table"]      = ts_table
    st.session_state["ts_index"]      = 0
    st.session_state["computed"]      = True
    st.success("✅ All data computed. Use the controls below to step through the session.")

# ─────────────────────────────────────────────────────────────
# VISUALIZATION SECTION — only shown after compute
# ─────────────────────────────────────────────────────────────
if st.session_state["computed"]:

    minute_series = st.session_state["minute_series"]
    sorted_ts     = st.session_state["sorted_ts"]
    all_strikes   = st.session_state["all_strikes"]
    ts_table      = st.session_state["ts_table"]
    date_str      = st.session_state["date_str"]
    spot_override = st.session_state["spot_override"]
    intra_date    = st.session_state["intra_date"]

    st.divider()
    st.header("📊 GEX / DEX Charts")

    # ── Time step controls ────────────────────────────────────
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4, ctrl_col5 = \
        st.columns([1, 1, 2, 1, 2])

    with ctrl_col1:
        if st.button("◀  Prev", use_container_width=True):
            step = st.session_state["step_size"]
            st.session_state["ts_index"] = max(
                0,
                st.session_state["ts_index"] - step)

    with ctrl_col2:
        if st.button("Next  ▶", use_container_width=True):
            step = st.session_state["step_size"]
            st.session_state["ts_index"] = min(
                len(sorted_ts) - 1,
                st.session_state["ts_index"] + step)

    with ctrl_col3:
        step_options = [1, 2, 5, 10, 15, 30]
        chosen_step  = st.selectbox(
            "Step size (bars)",
            options=step_options,
            index=step_options.index(st.session_state["step_size"])
                  if st.session_state["step_size"] in step_options
                  else 0,
            label_visibility="collapsed")
        st.session_state["step_size"] = chosen_step

    with ctrl_col4:
        # Jump to first bar
        if st.button("⏮ First", use_container_width=True):
            st.session_state["ts_index"] = 0

    with ctrl_col5:
        # Slider for direct jump
        ts_idx = st.slider(
            "Jump to bar",
            min_value=0,
            max_value=max(len(sorted_ts) - 1, 0),
            value=st.session_state["ts_index"],
            label_visibility="collapsed")
        if ts_idx != st.session_state["ts_index"]:
            st.session_state["ts_index"] = ts_idx

    current_idx = st.session_state["ts_index"]
    current_ts  = sorted_ts[current_idx]
    ts_data     = minute_series.get(current_ts, {})

    # Current spot from data
    spots_at_ts = [v["spot"] for v in ts_data.values() if v["spot"] > 0]
    current_spot = (np.mean(spots_at_ts) if spots_at_ts
                    else spot_override)

    # EAT label
    et_time = datetime.strptime(current_ts, "%H:%M").time()
    et_dt   = datetime.combine(
        intra_date if intra_date else date.today(), et_time)
    eat_dt  = et_to_eat(et_dt, intra_date if intra_date else date.today())

    st.markdown(
        f"### Bar: &nbsp; "
        f"<span style='color:#26a69a'>{current_ts} ET</span> &nbsp;|&nbsp; "
        f"<span style='color:#aaa'>{eat_dt.strftime('%H:%M')} EAT</span> &nbsp;|&nbsp; "
        f"<span style='color:#f0c040'>Spot: {current_spot:.2f}</span> &nbsp;|&nbsp; "
        f"Bar {current_idx + 1} / {len(sorted_ts)}",
        unsafe_allow_html=True)

    # ── Formula & chart-type toggles ─────────────────────────
    toggle_col1, toggle_col2 = st.columns(2)

    with toggle_col1:
        chart_type = st.radio(
            "Chart",
            ["GEX", "DEX"],
            horizontal=True,
            key="chart_type_radio")

    with toggle_col2:
        gex_formula = st.radio(
            "GEX Formula",
            ["GEX_unsigned", "GEX_signed",
             "GEX_agg_oi",   "GEX_dealer_sp"],
            horizontal=True,
            key="gex_formula_radio")

    # ── Build series for chart ────────────────────────────────
    formula_key = (gex_formula if chart_type == "GEX" else "DEX")
    chart_data  = []

    for sk in all_strikes:
        row = ts_data.get(sk, {})
        val = row.get(formula_key, 0.0)
        chart_data.append({"strike": sk, "value": val})

    chart_title = (
        f"{formula_key}  —  {date_str}  |  {current_ts} ET  |  "
        f"Spot {current_spot:.2f}"
        if chart_type == "GEX"
        else f"DEX  —  {date_str}  |  {current_ts} ET  |  "
             f"Spot {current_spot:.2f}")

    # ── Render chart ─────────────────────────────────────────
    build_histogram_chart(chart_data, current_spot, chart_title)

    st.divider()

    # ── Data tables (Greeks / GEX / DEX) ─────────────────────
    st.subheader("Full Session Data Tables")
    tab_greeks, tab_gex, tab_dex = st.tabs(["Greeks", "GEX", "DEX"])

    greek_display = ["call_iv","call_delta","call_gamma","call_vanna",
                     "call_charm","call_oi","call_volume",
                     "put_iv","put_delta","put_gamma","put_vanna",
                     "put_charm","put_oi","put_volume"]

    with tab_greeks:
        g_cols = (["timestamp","strike","spot"] +
                  [c for c in greek_display if c in ts_table.columns])
        st.dataframe(
            ts_table[[c for c in g_cols if c in ts_table.columns]],
            use_container_width=True, hide_index=True)

    with tab_gex:
        gex_ts_cols = (
            ["timestamp","strike","spot"] +
            [c for c in ts_table.columns
             if c.startswith("GEX") and "($)" in c])
        st.dataframe(
            fmt_df_dollars(
                ts_table[[c for c in gex_ts_cols
                          if c in ts_table.columns]]),
            use_container_width=True, hide_index=True)

    with tab_dex:
        dex_ts_cols = (
            ["timestamp","strike","spot"] +
            [c for c in ts_table.columns
             if c.startswith("DEX") and "($)" in c])
        st.dataframe(
            fmt_df_dollars(
                ts_table[[c for c in dex_ts_cols
                          if c in ts_table.columns]]),
            use_container_width=True, hide_index=True)

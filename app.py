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
MIN_DELAY     = 1.05   # strictly > 1s between every API request
POLL_DELAY    = 2.1
SPX_MULT      = 100
RISK_FREE     = 0.0525
SESSION_START = time(9, 30)
SESSION_END   = time(16, 0)

_last_request_time = [0.0]   # mutable container for rate limiter

def rate_limited_get(url, headers, params):
    """Enforce >= 1.05s between every outbound request."""
    elapsed = time_module.time() - _last_request_time[0]
    if elapsed < MIN_DELAY:
        time_module.sleep(MIN_DELAY - elapsed)
    r = requests.get(url, headers=headers, params=params)
    _last_request_time[0] = time_module.time()
    return r

# ─────────────────────────────────────────────────────────────
# TIMEZONE HELPERS
# ─────────────────────────────────────────────────────────────
def eat_to_et(d, t):
    eat_dt = datetime.combine(d, t)
    month  = d.month
    is_edt = 3 <= month <= 10          # rough DST window
    offset = timedelta(hours=7 if is_edt else 8)
    et_dt  = eat_dt - offset
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
    for attempt in range(1, max_polls+1):
        time_module.sleep(POLL_DELAY)
        rp = rate_limited_get(poll_url, headers=headers, params={"apiKey": params["apiKey"]})
        if rp.status_code == 429:
            time_module.sleep(3); continue
        if rp.status_code != 200: continue
        url = find_download_url(rp.json())
        if url: return url
    st.error(f"[{label}] Polling timed out after {max_polls} attempts")
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
            st.error(f"CSV parse error: {e}")
            return None

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
# STEP 1 — Option series → optionIds
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

    exp_target  = pd.Timestamp(exp_date_str)
    exp_mask    = series_df["expirationdate"] == exp_target
    strike_mask = (series_df["strike"] >= strike_min) & (series_df["strike"] <= strike_max)
    filtered    = series_df[exp_mask & strike_mask].copy()

    if filtered.empty:
        st.error(f"No contracts found for expiry {exp_date_str} in strike range {strike_min}–{strike_max}")
        return None, None

    cp_col = next((c for c in filtered.columns
                   if c in ["callput","call_put","type","optiontype"]), None)
    if not cp_col:
        st.error("Cannot identify call/put column"); return None, None

    filtered["_is_spx"] = filtered["optionsymbol"].str.strip().str.startswith("SPX ")
    filtered = filtered.sort_values("_is_spx", ascending=False)
    filtered = filtered.drop_duplicates(subset=["strike", cp_col], keep="first")
    filtered = filtered.drop(columns=["_is_spx"]).reset_index(drop=True)

    calls = filtered[filtered[cp_col] == "C"].sort_values("strike")
    puts  = filtered[filtered[cp_col] == "P"].sort_values("strike")
    return calls, puts

# ─────────────────────────────────────────────────────────────
# STEP 2 — EOD OI from prev close
# ─────────────────────────────────────────────────────────────
def get_eod_oi(api_key, headers, calls, puts, prev_date_str):
    oi_map = {}

    def fetch_oi(row, cp_label):
        oid    = int(row["optionid"])
        strike = float(row["strike"])
        rec    = sync_call(
            "/equities/eod/single-stock-option-raw-iv",
            {"optionId": oid, "from": prev_date_str,
             "to": prev_date_str, "apiKey": api_key},
            headers, label=f"{cp_label}{int(strike)}"
        )
        if not rec: return
        d = {k.strip().lower().replace(" ", ""): v for k, v in rec[0].items()}
        oi_val = (
            d.get("openinterest") or d.get("open_interest") or
            d.get("oi") or d.get("openint") or 0
        )
        oi_map[(strike, cp_label)] = int(float(oi_val)) if oi_val else 0

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

    strikes   = list(range(int(strike_min), int(strike_max) + int(strike_step), int(strike_step)))
    opt_types = ["C", "P"]
    total     = len(strikes) * len(opt_types)
    all_data  = []
    count     = 0

    close_dt = datetime(
        int(date_str[:4]), int(date_str[5:7]), int(date_str[8:10]), 16, 0
    )

    for strike in strikes:
        for opt_type in opt_types:
            count += 1
            progress_bar.progress(
                count / total,
                text=f"Fetching {strike}{opt_type}  ({count}/{total})"
            )
            try:
                # ivolatility library manages its own request;
                # sleep before to honour rate limit
                time_module.sleep(MIN_DELAY)
                df = get_intra(
                    symbol="SPX", date=date_str, expDate=exp_date_str,
                    strike=str(strike), optType=opt_type,
                    minuteType=minute_type
                )
                _last_request_time[0] = time_module.time()

                if df is None or len(df) == 0: continue

                df = df.copy()
                df["_strike"]  = strike
                df["_optType"] = opt_type

                # Drop rows where IV is -1 or NaN — Greeks are invalid
                if "optionIv" in df.columns:
                    df["optionIv"] = pd.to_numeric(df["optionIv"], errors="coerce")
                    df = df[
                        df["optionIv"].notna() &
                        (df["optionIv"] > 0) &
                        (df["optionIv"] != -1)
                    ].copy()

                if len(df) == 0: continue

                # Stamp EOD OI
                df["optionOI"] = df.apply(
                    lambda r: oi_map.get((float(r["_strike"]), r["_optType"]), np.nan),
                    axis=1
                )

                # DTE in years per timestamp
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df["_dte_years"] = df["timestamp"].apply(
                    lambda ts: max(
                        (close_dt - ts.to_pydatetime()).total_seconds() / (252 * 6.5 * 3600),
                        1e-8
                    )
                )

                # BSM Vanna & Charm from intraday IV
                df["optionVanna"] = df.apply(
                    lambda r: bsm_vanna(
                        r.get("underlyingPrice", np.nan), r["_strike"],
                        r["_dte_years"], RISK_FREE, r.get("optionIv", np.nan)
                    ), axis=1
                )
                df["optionCharm"] = df.apply(
                    lambda r: bsm_charm(
                        r.get("underlyingPrice", np.nan), r["_strike"],
                        r["_dte_years"], RISK_FREE, r.get("optionIv", np.nan)
                    ), axis=1
                )

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
# STEP 4 — Pivot wide + snapshot at requested time
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

def pivot_and_snapshot(df_all, target_et_full, intra_date):
    def pivot_side(df, prefix):
        rename = {k: f"{prefix}_{v}" for k, v in COL_MAP.items() if k in df.columns}
        df     = df.rename(columns=rename)
        base   = (["timestamp", "_strike", "underlyingPrice"]
                  if prefix == "call" else ["timestamp", "_strike"])
        keep   = base + list(rename.values())
        return df[[c for c in keep if c in df.columns]].copy()

    calls_p = pivot_side(df_all[df_all["_optType"] == "C"].copy(), "call")
    puts_p  = pivot_side(df_all[df_all["_optType"] == "P"].copy(), "put")

    merged = calls_p.merge(puts_p, on=["timestamp", "_strike"], how="outer")
    merged = merged.rename(columns={"_strike": "strike", "underlyingPrice": "spot"})
    merged = merged.sort_values(["strike", "timestamp"]).reset_index(drop=True)

    snap = (
        merged[merged["timestamp"] <= pd.Timestamp(target_et_full)]
        .sort_values("timestamp")
        .groupby("strike").last()
        .reset_index()
    )

    open_snap = (
        merged.sort_values("timestamp")
        .groupby("strike").first()
        .reset_index()
    )

    return snap, open_snap, merged

# ─────────────────────────────────────────────────────────────
# FORMULA ENGINE
# ─────────────────────────────────────────────────────────────
def calculate_formulas(df, spot_override, dte_frac_default=0.5):
    out = df.copy()

    spot_col = (out["spot"].fillna(spot_override)
                if "spot" in out.columns else
                pd.Series([spot_override] * len(out), index=out.index))
    out["spot_used"] = spot_col

    def col(name): return out.get(name, pd.Series(0.0, index=out.index)).fillna(0)

    cg   = col("call_gamma");  pg   = col("put_gamma")
    coi  = col("call_oi");     poi  = col("put_oi")
    cvol = col("call_volume"); pvol = col("put_volume")
    cd   = col("call_delta");  pd_  = col("put_delta")
    cv   = col("call_vanna");  pv   = col("put_vanna")
    cc   = col("call_charm");  pc   = col("put_charm")
    civ  = out.get("call_iv",  pd.Series(np.nan, index=out.index))
    piv  = out.get("put_iv",   pd.Series(np.nan, index=out.index))

    S    = spot_col
    S2   = S ** 2
    M    = SPX_MULT

    # Weighted OI — blend EOD OI (70%) with intraday volume (30%)
    # Volume is normalised to OI scale before blending
    vsc  = (coi / cvol.replace(0, np.nan)).fillna(1).clip(0, 10)
    vsp  = (poi / pvol.replace(0, np.nan)).fillna(1).clip(0, 10)
    cwoi = 0.7 * coi + 0.3 * (cvol * vsc)
    pwoi = 0.7 * poi + 0.3 * (pvol * vsp)

    # ATM IV proxy for vanna adjustment
    atm_iv    = civ.fillna(piv).mean()
    iv_diff_c = civ.fillna(atm_iv) - atm_iv
    iv_diff_p = piv.fillna(atm_iv) - atm_iv

    # DTE fraction (remaining session time / full session)
    dte_frac = (out["_dte_frac"].fillna(dte_frac_default)
                if "_dte_frac" in out.columns else
                pd.Series(dte_frac_default, index=out.index))

    # ── GEX ─────────────────────────────────────────────────
    # GEX-1: Standard (dealers short all)
    out["GEX1"]   = (cg*coi  - pg*poi)       * M * S2
    out["GEX1_$"] = out["GEX1"] * S / 1e9

    # GEX-2: Sign-corrected (dealers long puts — puts stabilise)
    out["GEX2"]   = (cg*coi  + pg*poi)       * M * S2
    out["GEX2_$"] = out["GEX2"] * S / 1e9

    # GEX-3: OI-skew weighted
    total_oi      = (coi + poi).replace(0, np.nan)
    skew          = (coi / total_oi).fillna(0.5)
    out["GEX3"]   = (cg*coi  - skew*pg*poi)  * M * S2
    out["GEX3_$"] = out["GEX3"] * S / 1e9

    # GEX-4: Weighted OI (OI + volume blend)
    out["GEX4"]   = (cg*cwoi - pg*pwoi)      * M * S2
    out["GEX4_$"] = out["GEX4"] * S / 1e9

    # GEX-5: Pure intraday volume
    out["GEX5"]   = (cg*cvol - pg*pvol)      * M * S2
    out["GEX5_$"] = out["GEX5"] * S / 1e9

    # ── DEX ─────────────────────────────────────────────────
    # DEX-1: Standard
    out["DEX1"]   = (cd*coi  - pd_*poi)      * M * S
    out["DEX1_$"] = out["DEX1"] * S / 1e9

    # DEX-2: Weighted OI
    out["DEX2"]   = (cd*cwoi - pd_*pwoi)     * M * S
    out["DEX2_$"] = out["DEX2"] * S / 1e9

    # DEX-3: Pure volume
    out["DEX3"]   = (cd*cvol - pd_*pvol)     * M * S
    out["DEX3_$"] = out["DEX3"] * S / 1e9

    # DEX-4: Charm-adjusted (0DTE delta decay over remaining session)
    charm_flow    = (cc*coi  - pc*poi)  * M * dte_frac
    out["DEX4"]   = out["DEX1"] - charm_flow * S
    out["DEX4_$"] = out["DEX4"] * S / 1e9

    # DEX-5: Vanna-adjusted (IV surface delta shift)
    vanna_flow    = (cv*coi*iv_diff_c - pv*poi*iv_diff_p) * M
    out["DEX5"]   = out["DEX1"] + vanna_flow * S
    out["DEX5_$"] = out["DEX5"] * S / 1e9

    # ── Gamma flip per-strike ────────────────────────────────
    g     = out["GEX1"].values
    flips = []
    for i in range(1, len(g)):
        if not (np.isnan(g[i-1]) or np.isnan(g[i])) and g[i-1] * g[i] < 0:
            flips.append(out["strike"].iloc[i])
    out["_flip"] = out["strike"].isin(flips)

    return out, flips

def rate_of_change(snap_now, snap_open):
    formula_cols = [c for c in snap_now.columns
                    if c.startswith(("GEX","DEX")) and not c.endswith(("_$","roc%","_flip"))]
    roc_parts = []
    for col in formula_cols:
        if col not in snap_open.columns: continue
        merged = snap_now[["strike", col]].merge(
            snap_open[["strike", col]].rename(columns={col: f"{col}_open"}),
            on="strike", how="left"
        )
        open_v        = merged[f"{col}_open"].replace(0, np.nan)
        roc_series    = ((merged[col] - merged[f"{col}_open"]) / open_v.abs() * 100).round(2)
        roc_series.name = f"{col}_roc%"
        roc_parts.append(roc_series)
    if roc_parts:
        return pd.concat([snap_now.reset_index(drop=True)] + roc_parts, axis=1)
    return snap_now

# ─────────────────────────────────────────────────────────────
# REGIME
# ─────────────────────────────────────────────────────────────
def interpret_regime(df, flip_strikes):
    formula_cols = [c for c in df.columns
                    if c.startswith(("GEX","DEX")) and
                    not c.endswith(("_$","roc%","_flip"))]
    net = {col: df[col].sum() for col in formula_cols}

    gex1_neg = net.get("GEX1", 0) < 0
    dex1_pos = net.get("DEX1", 0) > 0

    regime = ("🔴 Negative Gamma — Trending / Volatile"
              if gex1_neg else
              "🟢 Positive Gamma — Mean-Reverting / Pinned")

    if gex1_neg and dex1_pos:
        signal = "⚡ Bullish Move — Negative GEX + Positive DEX"
    elif gex1_neg and not dex1_pos:
        signal = "⚡ Bearish Move — Negative GEX + Negative DEX"
    else:
        signal = "🔒 Pin / Fade — Positive GEX, expect mean reversion"

    flip_str = (", ".join(str(s) for s in flip_strikes)
                if flip_strikes else "None in range")
    return regime, signal, flip_str, net

# ─────────────────────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────────────────────
def fmt_b(val):
    if pd.isna(val) or val is None: return "N/A"
    v = float(val)
    if abs(v) >= 1e9: return f"${v/1e9:.2f}B"
    if abs(v) >= 1e6: return f"${v/1e6:.2f}M"
    if abs(v) >= 1e3: return f"${v/1e3:.1f}K"
    return f"${v:.2f}"

def display_results(df, regime, signal, flip_str,
         

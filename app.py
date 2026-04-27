# ─────────────────────────────────────────────────────────────
# SPX GEX / DEX Multi-Formula Analyzer
# Streamlit app — paste into app.py
# pip install streamlit requests pandas streamlit-lightweight-charts
# ─────────────────────────────────────────────────────────────

import streamlit as st
import requests
import pandas as pd
import gzip
import io
import time

st.set_page_config(page_title="SPX GEX/DEX Analyzer", layout="wide")
st.title("SPX GEX / DEX Multi-Formula Analyzer")

# ─────────────────────────────────────────────────────────────
# SIDEBAR — INPUTS
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Parameters")
    api_key       = st.text_input("iVolatility API Key", type="password")
    date_input    = st.date_input("Date (EAT)")
    time_input    = st.time_input("Session Time (EAT)")
    center_strike = st.number_input("Center Strike", value=5580, step=5)
    num_strikes   = st.number_input("Strikes Each Side", value=5, min_value=1, max_value=20, step=1)
    run_btn       = st.button("Fetch & Calculate", type="primary")

# EAT → ET label (EAT = UTC+3, ET = UTC-5 standard / UTC-4 DST)
# We just show the converted label — API is date-based only
from datetime import datetime, timedelta
def eat_to_et_label(d, t):
    eat_dt = datetime.combine(d, t)
    et_dt  = eat_dt - timedelta(hours=8)   # EAT is +8h ahead of ET (EST)
    return et_dt.strftime("%Y-%m-%d %H:%M ET")

# ─────────────────────────────────────────────────────────────
# API HELPERS (from your working notebook)
# ─────────────────────────────────────────────────────────────
BASE_URL = "https://restapi.ivolatility.com"
DELAY      = 1.2
POLL_DELAY = 2.0

def find_url_key(obj, key, depth=0):
    if depth > 10: return None
    if isinstance(obj, dict):
        if key in obj and obj[key]: return obj[key]
        for v in obj.values():
            found = find_url_key(v, key, depth+1)
            if found: return found
    elif isinstance(obj, list):
        for item in obj:
            found = find_url_key(item, key, depth+1)
            if found: return found
    return None

def call_async_download(endpoint, params, label="", max_polls=20):
    time.sleep(DELAY)
    r = requests.get(f"{BASE_URL}{endpoint}", headers={"Authorization": f"Bearer {params['apiKey']}"}, params=params)
    if r.status_code != 200:
        st.error(f"[{label}] HTTP {r.status_code}: {r.text[:200]}")
        return None
    body = r.json()
    url  = find_url_key(body, "urlForDownload")
    if url: return url
    poll_url = find_url_key(body, "urlForDetails")
    if not poll_url:
        st.error(f"[{label}] No poll URL found")
        return None
    for attempt in range(1, max_polls + 1):
        time.sleep(POLL_DELAY)
        rp = requests.get(poll_url, headers={"Authorization": f"Bearer {params['apiKey']}"}, params={"apiKey": params["apiKey"]})
        if rp.status_code == 429: time.sleep(3); continue
        if rp.status_code != 200: continue
        url = find_url_key(rp.json(), "urlForDownload")
        if url: return url
    st.error(f"[{label}] Polling timed out")
    return None

def call_sync(endpoint, params, label=""):
    time.sleep(DELAY)
    r = requests.get(f"{BASE_URL}{endpoint}", headers={"Authorization": f"Bearer {params['apiKey']}"}, params=params)
    if r.status_code == 429:
        time.sleep(3)
        r = requests.get(f"{BASE_URL}{endpoint}", headers={"Authorization": f"Bearer {params['apiKey']}"}, params=params)
    if r.status_code != 200: return None
    body    = r.json()
    records = body.get("data", []) if isinstance(body, dict) else body
    return records if records else None

def download_csv_gz(url, api_key, label=""):
    time.sleep(DELAY)
    r = requests.get(url, headers={"Authorization": f"Bearer {api_key}"}, params={"apiKey": api_key})
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

# ─────────────────────────────────────────────────────────────
# FETCH PIPELINE
# ─────────────────────────────────────────────────────────────
def fetch_spx_data(api_key, target_date, center_strike, num_strikes):
    target_str = target_date.strftime("%Y-%m-%d")

    # Step 1 — option series
    with st.spinner("Fetching option series..."):
        dl_url = call_async_download(
            "/equities/eod/option-series-on-date",
            {"symbol": "SPX", "date": target_str, "apiKey": api_key},
            label="series"
        )
    if not dl_url:
        st.error("Failed to get option series")
        return None

    series_df = download_csv_gz(dl_url, api_key, label="series")
    if series_df is None or series_df.empty:
        st.error("Empty option series")
        return None

    series_df.columns = [c.strip().lower() for c in series_df.columns]
    series_df["expirationdate"] = pd.to_datetime(series_df["expirationdate"])
    series_df["strike"]         = pd.to_numeric(series_df["strike"], errors="coerce")

    # Nearest expiry on or after target date
    target_dt   = pd.Timestamp(target_str)
    future_exps = series_df[series_df["expirationdate"] >= target_dt]["expirationdate"].sort_values().unique()
    if len(future_exps) == 0:
        st.error("No valid expiries found")
        return None
    chosen     = future_exps[0]
    chosen_str = chosen.strftime("%Y-%m-%d")
    st.info(f"Using expiry: {chosen_str}")

    # Strike selection — 5 each side
    exp_mask       = series_df["expirationdate"] == chosen
    strikes_avail  = sorted(series_df[exp_mask]["strike"].dropna().unique())
    below          = [s for s in strikes_avail if s <= center_strike][-num_strikes:]
    above          = [s for s in strikes_avail if s >  center_strike][:num_strikes]
    selected       = below + above

    filtered = series_df[exp_mask & series_df["strike"].isin(selected)].copy()
    filtered["_is_spx"] = filtered["optionsymbol"].str.strip().str.startswith("SPX ")
    filtered = filtered.sort_values("_is_spx", ascending=False)
    filtered = filtered.drop_duplicates(subset=["strike", "callput"], keep="first")
    filtered = filtered.drop(columns=["_is_spx"]).reset_index(drop=True)

    calls = filtered[filtered["callput"] == "C"].sort_values("strike")
    puts  = filtered[filtered["callput"] == "P"].sort_values("strike")

    # Step 2 — Greeks per contract
    def fetch_greeks(row):
        oid = int(row["optionid"])
        rec = call_sync(
            "/equities/eod/single-stock-option-raw-iv",
            {"optionId": oid, "from": target_str, "to": target_str, "apiKey": api_key},
            label=f"{row['callput']}{int(row['strike'])}"
        )
        if rec:
            d = rec[0]
            d["_strike"] = row["strike"]
            d["_cp"]     = row["callput"]
            return d
        return None

    with st.spinner("Fetching Greeks for calls..."):
        call_greeks = [r for r in (fetch_greeks(row) for _, row in calls.iterrows()) if r]
    with st.spinner("Fetching Greeks for puts..."):
        put_greeks  = [r for r in (fetch_greeks(row) for _, row in puts.iterrows()) if r]

    if not call_greeks and not put_greeks:
        st.error("No Greeks returned")
        return None

    def to_df(rows, prefix):
        df = pd.DataFrame(rows)
        df.columns = [c.strip().lower() for c in df.columns]
        df["strike"] = df["_strike"]

        # Normalise column names across API variants
        for src, dst in [
            ("open interest", "oi"), ("openinterest", "oi"), ("open_interest", "oi"),
            ("rawiv", "iv"), ("impliedvolatility", "iv"),
        ]:
            if src in df.columns and "oi" not in df.columns or src in df.columns and dst not in df.columns:
                df[dst] = df[src]

        num_cols = ["iv", "delta", "gamma", "theta", "vega", "vanna", "charm", "oi"]
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = None

        df["oi"] = df["oi"].fillna(0).astype(int)
        keep = ["strike"] + [c for c in num_cols if c in df.columns]
        df = df[keep].copy()
        df.columns = ["strike"] + [f"{prefix}_{c}" for c in df.columns if c != "strike"]
        return df.sort_values("strike").reset_index(drop=True)

    calls_df = to_df(call_greeks, "call") if call_greeks else pd.DataFrame()
    puts_df  = to_df(put_greeks,  "put")  if put_greeks  else pd.DataFrame()

    if not calls_df.empty and not puts_df.empty:
        merged = calls_df.merge(puts_df, on="strike", how="outer").sort_values("strike")
    elif not calls_df.empty:
        merged = calls_df
    else:
        merged = puts_df

    merged["strike"] = merged["strike"].astype(int)
    merged["expiry"] = chosen_str
    return merged

# ─────────────────────────────────────────────────────────────
# GEX / DEX FORMULA ENGINE
# ─────────────────────────────────────────────────────────────
def calculate_all(df, spot):
    out = df.copy()

    cg  = out.get("call_gamma",  pd.Series([0.0]*len(out))).fillna(0)
    pg  = out.get("put_gamma",   pd.Series([0.0]*len(out))).fillna(0)
    coi = out.get("call_oi",     pd.Series([0.0]*len(out))).fillna(0)
    poi = out.get("put_oi",      pd.Series([0.0]*len(out))).fillna(0)
    cd  = out.get("call_delta",  pd.Series([0.0]*len(out))).fillna(0)
    pd_ = out.get("put_delta",   pd.Series([0.0]*len(out))).fillna(0)

    scale_gex = 100 * spot**2
    scale_dex = 100 * spot

    # ── GEX Variants ────────────────────────────────────────
    # GEX-1: Standard (dealers short all)
    out["GEX1_per_strike"]    = (cg*coi - pg*poi) * scale_gex
    out["GEX1_$_per_strike"]  = out["GEX1_per_strike"] * spot

    # GEX-2: Sign-corrected (dealers long puts)
    out["GEX2_per_strike"]    = (cg*coi + pg*poi) * scale_gex
    out["GEX2_$_per_strike"]  = out["GEX2_per_strike"] * spot

    # GEX-3: OI-skew weighted
    total_oi = coi + poi
    skew     = (coi / total_oi.replace(0, float("nan"))).fillna(0.5)
    out["GEX3_per_strike"]    = (cg*coi - skew*pg*poi) * scale_gex
    out["GEX3_$_per_strike"]  = out["GEX3_per_strike"] * spot

    # ── DEX Variants ────────────────────────────────────────
    # DEX-1: Standard
    out["DEX1_per_strike"]    = (cd*coi - pd_*poi) * scale_dex
    out["DEX1_$_per_strike"]  = out["DEX1_per_strike"] * spot

    # DEX-2: Charm-adjusted (0DTE) — only if charm columns exist
    has_charm = "call_charm" in out.columns and "put_charm" in out.columns
    if has_charm:
        cc = out["call_charm"].fillna(0)
        pc = out["put_charm"].fillna(0)
        # Assume 390 minutes in full session; charm decays delta over time
        # Using 390 as denominator — adjust minutes_to_expiry manually if needed
        minutes_to_expiry = 390
        charm_flow = (cc*coi - pc*poi) * 100 * (minutes_to_expiry / 390)
        out["DEX2_charm_adj_per_strike"]   = out["DEX1_per_strike"] - charm_flow * spot
        out["DEX2_charm_adj_$_per_strike"] = out["DEX2_charm_adj_per_strike"] * spot
    else:
        out["DEX2_charm_adj_per_strike"]   = None
        out["DEX2_charm_adj_$_per_strike"] = None

    # DEX-3: Vanna-adjusted — only if vanna columns exist
    has_vanna = "call_vanna" in out.columns and "put_vanna" in out.columns
    if has_vanna:
        cv = out["call_vanna"].fillna(0)
        pv = out["put_vanna"].fillna(0)
        # ATM IV proxy — use mean of call IVs as ATM reference
        atm_iv = out.get("call_iv", pd.Series([0.2]*len(out))).fillna(0.2).mean()
        iv_diff_c = out.get("call_iv", pd.Series([atm_iv]*len(out))).fillna(atm_iv) - atm_iv
        iv_diff_p = out.get("put_iv",  pd.Series([atm_iv]*len(out))).fillna(atm_iv) - atm_iv
        vanna_flow = (cv*coi*iv_diff_c - pv*poi*iv_diff_p) * 100
        out["DEX3_vanna_adj_per_strike"]   = out["DEX1_per_strike"] + vanna_flow * spot
        out["DEX3_vanna_adj_$_per_strike"] = out["DEX3_vanna_adj_per_strike"] * spot
    else:
        out["DEX3_vanna_adj_per_strike"]   = None
        out["DEX3_vanna_adj_$_per_strike"] = None

    # ── Gamma Flip Detection ─────────────────────────────────
    # Flip level = strike where GEX1 changes sign strike-to-strike
    g = out["GEX1_per_strike"].values
    flip_strikes = []
    for i in range(1, len(g)):
        if g[i-1] * g[i] < 0:   # sign change
            flip_strikes.append(out["strike"].iloc[i])
    out["_flip"] = out["strike"].isin(flip_strikes)

    return out, flip_strikes, has_charm, has_vanna

# ─────────────────────────────────────────────────────────────
# REGIME INTERPRETATION
# ─────────────────────────────────────────────────────────────
def interpret_regime(df, flip_strikes):
    net_gex1 = df["GEX1_per_strike"].sum()
    net_gex2 = df["GEX2_per_strike"].sum()
    net_dex1 = df["DEX1_per_strike"].sum()

    regime   = "🔴 Negative Gamma (Trending / Volatile)" if net_gex1 < 0 else "🟢 Positive Gamma (Mean-Reverting / Pinned)"
    dex_bias = "📈 Bullish" if net_dex1 > 0 else "📉 Bearish"

    if net_gex1 < 0 and net_dex1 > 0:
        signal = "⚡ Strong Bullish Move Signal — Negative GEX + Positive DEX"
    elif net_gex1 < 0 and net_dex1 < 0:
        signal = "⚡ Strong Bearish Move Signal — Negative GEX + Negative DEX"
    elif net_gex1 > 0:
        signal = "🔒 Pin/Fade Signal — Positive GEX regime, expect mean reversion"
    else:
        signal = "⚠️ Neutral — No strong directional bias"

    flip_str = ", ".join(str(s) for s in flip_strikes) if flip_strikes else "None identified in this range"

    return {
        "net_gex1":   net_gex1,
        "net_gex2":   net_gex2,
        "net_dex1":   net_dex1,
        "regime":     regime,
        "dex_bias":   dex_bias,
        "signal":     signal,
        "flip_level": flip_str,
    }

# ─────────────────────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────────────────────
def format_millions(val):
    if pd.isna(val) or val is None: return "N/A"
    if abs(val) >= 1_000_000:
        return f"${val/1_000_000:.2f}M"
    elif abs(val) >= 1_000:
        return f"${val/1_000:.1f}K"
    return f"${val:.2f}"

def display_results(df, interp, has_charm, has_vanna, date_label, et_label):
    st.subheader(f"Results — {date_label}  ({et_label})")

    # Regime panel
    col1, col2, col3 = st.columns(3)
    col1.metric("GEX Regime", interp["regime"])
    col2.metric("DEX Bias",   interp["dex_bias"])
    col3.metric("Gamma Flip Level", interp["flip_level"])

    st.info(interp["signal"])

    col4, col5, col6 = st.columns(3)
    col4.metric("Net GEX-1 ($)",  format_millions(interp["net_gex1"]))
    col5.metric("Net GEX-2 ($)",  format_millions(interp["net_gex2"]))
    col6.metric("Net DEX-1 ($)",  format_millions(interp["net_dex1"]))

    st.divider()

    # ── GEX Table ────────────────────────────────────────────
    st.subheader("GEX Per Strike")
    gex_cols = ["strike",
                "GEX1_per_strike", "GEX1_$_per_strike",
                "GEX2_per_strike", "GEX2_$_per_strike",
                "GEX3_per_strike", "GEX3_$_per_strike",
                "_flip"]
    gex_df = df[[c for c in gex_cols if c in df.columns]].copy()
    gex_df.rename(columns={
        "GEX1_per_strike":   "GEX1",
        "GEX1_$_per_strike": "GEX1 $",
        "GEX2_per_strike":   "GEX2",
        "GEX2_$_per_strike": "GEX2 $",
        "GEX3_per_strike":   "GEX3",
        "GEX3_$_per_strike": "GEX3 $",
        "_flip":             "Flip Zone",
    }, inplace=True)

    def highlight_flip(row):
        if row.get("Flip Zone", False):
            return ["background-color: #ffe066"] * len(row)
        return [""] * len(row)

    for col in ["GEX1 $", "GEX2 $", "GEX3 $"]:
        if col in gex_df.columns:
            gex_df[col] = gex_df[col].apply(format_millions)

    st.dataframe(
        gex_df.style.apply(highlight_flip, axis=1),
        use_container_width=True, hide_index=True
    )

    # ── DEX Table ────────────────────────────────────────────
    st.subheader("DEX Per Strike")
    dex_cols = ["strike",
                "DEX1_per_strike", "DEX1_$_per_strike"]
    if has_charm:
        dex_cols += ["DEX2_charm_adj_per_strike", "DEX2_charm_adj_$_per_strike"]
    if has_vanna:
        dex_cols += ["DEX3_vanna_adj_per_strike", "DEX3_vanna_adj_$_per_strike"]

    dex_df = df[[c for c in dex_cols if c in df.columns]].copy()
    dex_df.rename(columns={
        "DEX1_per_strike":               "DEX1",
        "DEX1_$_per_strike":             "DEX1 $",
        "DEX2_charm_adj_per_strike":     "DEX2 Charm",
        "DEX2_charm_adj_$_per_strike":   "DEX2 Charm $",
        "DEX3_vanna_adj_per_strike":     "DEX3 Vanna",
        "DEX3_vanna_adj_$_per_strike":   "DEX3 Vanna $",
    }, inplace=True)

    for col in ["DEX1 $", "DEX2 Charm $", "DEX3 Vanna $"]:
        if col in dex_df.columns:
            dex_df[col] = dex_df[col].apply(lambda x: format_millions(x) if x is not None else "N/A")

    st.dataframe(dex_df, use_container_width=True, hide_index=True)

    # ── Raw Greeks ───────────────────────────────────────────
    with st.expander("Raw Greeks"):
        raw_cols = ["strike", "call_iv", "call_delta", "call_gamma", "call_theta",
                    "call_vega", "call_oi", "put_iv", "put_delta", "put_gamma",
                    "put_theta", "put_vega", "put_oi"]
        if has_charm:
            raw_cols += ["call_charm", "put_charm"]
        if has_vanna:
            raw_cols += ["call_vanna", "put_vanna"]
        raw_df = df[[c for c in raw_cols if c in df.columns]].copy()
        st.dataframe(raw_df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────
# MAIN RUN
# ─────────────────────────────────────────────────────────────
if run_btn:
    if not api_key:
        st.error("Enter your iVolatility API key in the sidebar")
        st.stop()

    target_date   = date_input
    target_str    = target_date.strftime("%Y-%m-%d")
    et_label      = eat_to_et_label(date_input, time_input)

    # Spot price — user enters manually since we're pulling EOD data
    spot = st.sidebar.number_input("SPX Spot Price (at session time)", value=float(center_strike), step=1.0)

    raw_df = fetch_spx_data(api_key, target_date, int(center_strike), int(num_strikes))

    if raw_df is not None:
        result_df, flip_strikes, has_charm, has_vanna = calculate_all(raw_df, spot)
        interp = interpret_regime(result_df, flip_strikes)
        display_results(result_df, interp, has_charm, has_vanna,
                        date_label=target_str, et_label=et_label)

        if not has_charm:
            st.caption("ℹ️ Charm not returned by API — DEX-2 unavailable")
        if not has_vanna:
            st.caption("ℹ️ Vanna not returned by API — DEX-3 unavailable")

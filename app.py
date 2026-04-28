# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date

import streamlit.components.v1 as components

# Local imports
from utils.constants import *
from utils.helpers import (
    fmt_b, fmt_df_dollars, 
    et_to_eat, prev_trading_day,
    bsm_vanna, bsm_charm   # kept for safety even if used in api_helpers
)
from utils.api_helpers import (
    rate_limited_get,
    get_option_ids,
    get_eod_oi,
    get_session_price_range,
    get_intraday_greeks
)
from utils.data_processing import (
    pivot_wide,
    build_minute_series,
    build_session_table
)
from components.charts import build_histogram_chart

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG & TITLE
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="SPX GEX/DEX Analyzer", layout="wide")
st.title("📊 SPX GEX / DEX Analyzer")

# ─────────────────────────────────────────────────────────────
# CONSTANTS (already imported from utils.constants)
# ─────────────────────────────────────────────────────────────
# BASE_URL, MIN_DELAY, etc. are now in utils/constants.py

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
div[data-testid="stRadio"] label { cursor: pointer; }
div[data-testid="stRadio"] input[type="radio"]:checked + div > div {
    background-color: #26a69a !important;
    border-color:     #26a69a !important;
}
div[data-testid="stRadio"] label:hover span { color: #26a69a; }
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
        help="Used for the pilot fetch to establish session high/low.")
    spot_override_input = c5.number_input(
        "Spot Override (fallback if API missing)",
        value=5580.0, step=1.0)

    submitted = st.form_submit_button(
        "🚀 Fetch & Compute Full Session", use_container_width=True)

# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
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

    with st.spinner("Pilot fetch — establishing session price range..."):
        lo, hi = get_session_price_range(
            api_key, date_str, exp_date_str,
            rough_center, minute_type)

    if lo is None or hi is None:
        st.error("Pilot fetch failed — check API key and date.")
        st.stop()

    strike_min = (int(lo // STRIKE_STEP) * STRIKE_STEP - 5 * STRIKE_STEP)
    strike_max = (int(hi // STRIKE_STEP) * STRIKE_STEP + STRIKE_STEP + 5 * STRIKE_STEP)

    st.info(
        f"Date: {date_str}  |  Session range: {lo:.1f} – {hi:.1f}  |  "
        f"Fetching strikes: {strike_min} – {strike_max} "
        f"(step {STRIKE_STEP})")

    with st.spinner("Step 1 / 3 — Option series..."):
        calls_series, puts_series = get_option_ids(
            api_key, headers, prev_date_str, exp_date_str,
            strike_min, strike_max)
    if calls_series is None: st.stop()
    st.success(
        f"Step 1 ✅ — Calls: {len(calls_series)}  "
        f"Puts: {len(puts_series)}")

    with st.spinner("Step 2 / 3 — EOD OI..."):
        oi_map = get_eod_oi(
            api_key, headers, calls_series,
            puts_series, prev_date_str)
    filled = sum(1 for v in oi_map.values() if v > 0)
    st.success(
        f"Step 2 ✅ — OI entries: {filled}/{len(oi_map)} non-zero")

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
# VISUALIZATION SECTION
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

    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4, ctrl_col5 = \
        st.columns([1, 1, 2, 1, 2])

    with ctrl_col1:
        if st.button("◀  Prev", use_container_width=True):
            step = st.session_state["step_size"]
            st.session_state["ts_index"] = max(
                0, st.session_state["ts_index"] - step)

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
        if st.button("⏮ First", use_container_width=True):
            st.session_state["ts_index"] = 0

    with ctrl_col5:
        ts_idx = st.slider(
            "Jump to bar",
            min_value=0,
            max_value=max(len(sorted_ts) - 1, 0),
            value=st.session_state["ts_index"],
            label_visibility="collapsed")
        if ts_idx != st.session_state["ts_index"]:
            st.session_state["ts_index"] = ts_idx

    current_idx  = st.session_state["ts_index"]
    current_ts   = sorted_ts[current_idx]
    ts_data      = minute_series.get(current_ts, {})

    spots_at_ts  = [v["spot"] for v in ts_data.values() if v["spot"] > 0]
    current_spot = np.mean(spots_at_ts) if spots_at_ts else spot_override

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

    build_histogram_chart(chart_data, current_spot, chart_title)

    st.divider()

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

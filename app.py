# ─────────────────────────────────────────────────────────────
# app.py — UI entry point only
# All logic lives in: config, utils, greeks, formulas,
#                     data_fetch, pipeline, chart
# ─────────────────────────────────────────────────────────────
import streamlit as st
import numpy as np
from datetime import datetime, date

from config   import STRIKE_STEP
# was:  from utils import (prev_trading_day, et_to_eat, fmt_b, fmt_df_dollars)
from helpers import (prev_trading_day, et_to_eat,
                     fmt_b, fmt_df_dollars)
from data_fetch import (get_session_price_range, get_option_ids,
                        get_eod_oi, get_intraday_greeks)
from pipeline import (pivot_wide, build_minute_series,
                      build_session_table)
from chart    import build_histogram_chart

st.set_page_config(page_title="SPX GEX/DEX Analyzer", layout="wide")
st.title("📊 SPX GEX / DEX Analyzer")

# ── Radio style ───────────────────────────────────────────────
st.markdown("""
<style>
div[data-testid="stRadio"] input[type="radio"]:checked + div > div {
    background-color: #26a69a !important;
    border-color:     #26a69a !important;
}
div[data-testid="stRadio"] label:hover span { color: #26a69a; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────
def init_state():
    defaults = dict(
        computed=False, minute_series={}, sorted_ts=[],
        all_strikes=[], ts_index=0, step_size=1,
        ts_table=None, date_str="",
        spot_override=5500.0, intra_date=None)
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── Input form ────────────────────────────────────────────────
st.header("Parameters")

with st.form("params_form"):
    c1, c2, c3 = st.columns(3)
    api_key     = c1.text_input("iVolatility API Key", type="password",
                                 placeholder="Paste your Backtest+ key")
    date_input  = c2.date_input("Session Date", value=date.today())
    minute_type = c3.selectbox("API Bar Resolution",
                                ["MINUTE_1","MINUTE_5",
                                 "MINUTE_15","MINUTE_30"])

    c4, c5 = st.columns(2)
    rough_center        = c4.number_input("Rough ATM Strike",
                                           value=5580, step=5)
    spot_override_input = c5.number_input("Spot Override (fallback)",
                                           value=5580.0, step=1.0)
    submitted = st.form_submit_button(
        "🚀 Fetch & Compute Full Session", use_container_width=True)

# ── Main pipeline ─────────────────────────────────────────────
if submitted:
    if not api_key:
        st.error("Enter your iVolatility API key"); st.stop()

    date_str      = date_input.strftime("%Y-%m-%d")
    prev_date_str = prev_trading_day(date_input).strftime("%Y-%m-%d")
    exp_date_str  = date_str
    headers       = {"Authorization": f"Bearer {api_key}"}

    st.session_state.update(dict(
        computed=False, date_str=date_str,
        spot_override=spot_override_input,
        intra_date=date_input, ts_index=0))

    with st.spinner("Pilot fetch — establishing session price range..."):
        lo, hi = get_session_price_range(
            api_key, date_str, exp_date_str, rough_center, minute_type)

    if lo is None:
        st.error("Pilot fetch failed — check API key and date."); st.stop()

    strike_min = int(lo // STRIKE_STEP) * STRIKE_STEP - 5 * STRIKE_STEP
    strike_max = (int(hi // STRIKE_STEP) * STRIKE_STEP
                  + STRIKE_STEP + 5 * STRIKE_STEP)
    st.info(f"Date: {date_str}  |  Range: {lo:.1f}–{hi:.1f}  |  "
            f"Strikes: {strike_min}–{strike_max}")

    with st.spinner("Step 1/3 — Option series..."):
        calls_s, puts_s = get_option_ids(
            api_key, headers, prev_date_str, exp_date_str,
            strike_min, strike_max)
    if calls_s is None: st.stop()
    st.success(f"Step 1 ✅  Calls: {len(calls_s)}  Puts: {len(puts_s)}")

    with st.spinner("Step 2/3 — EOD OI..."):
        oi_map = get_eod_oi(api_key, headers, calls_s, puts_s, prev_date_str)
    filled = sum(1 for v in oi_map.values() if v > 0)
    st.success(f"Step 2 ✅  OI: {filled}/{len(oi_map)} non-zero")

    st.write("Step 3/3 — Intraday Greeks...")
    prog   = st.progress(0, text="Starting...")
    df_all = get_intraday_greeks(
        api_key, date_str, exp_date_str,
        strike_min, strike_max, minute_type, oi_map, prog)
    prog.empty()
    if df_all is None:
        st.error("No intraday data returned"); st.stop()
    st.success(f"Step 3 ✅  {len(df_all):,} rows")

    with st.spinner("Computing GEX / DEX..."):
        wide_df = pivot_wide(df_all)
        minute_series, sorted_ts, all_strikes = build_minute_series(
            wide_df, spot_override_input, date_input)
        ts_table = build_session_table(
            wide_df, spot_override_input, date_input)

    st.session_state.update(dict(
        minute_series=minute_series, sorted_ts=sorted_ts,
        all_strikes=all_strikes, ts_table=ts_table,
        ts_index=0, computed=True))
    st.success("✅ Done. Use controls below to step through the session.")

# ── Visualization ─────────────────────────────────────────────
if st.session_state["computed"]:
    ms           = st.session_state["minute_series"]
    sorted_ts    = st.session_state["sorted_ts"]
    all_strikes  = st.session_state["all_strikes"]
    ts_table     = st.session_state["ts_table"]
    date_str     = st.session_state["date_str"]
    spot_override= st.session_state["spot_override"]
    intra_date   = st.session_state["intra_date"]

    st.divider()
    st.header("📊 GEX / DEX Charts")

    # Controls
    cc1, cc2, cc3, cc4, cc5 = st.columns([1,1,2,1,2])
    with cc1:
        if st.button("◀ Prev", use_container_width=True):
            st.session_state["ts_index"] = max(
                0, st.session_state["ts_index"]
                   - st.session_state["step_size"])
    with cc2:
        if st.button("Next ▶", use_container_width=True):
            st.session_state["ts_index"] = min(
                len(sorted_ts)-1,
                st.session_state["ts_index"]
                + st.session_state["step_size"])
    with cc3:
        opts = [1,2,5,10,15,30]
        st.session_state["step_size"] = st.selectbox(
            "Step", opts,
            index=opts.index(st.session_state["step_size"])
                  if st.session_state["step_size"] in opts else 0,
            label_visibility="collapsed")
    with cc4:
        if st.button("⏮ First", use_container_width=True):
            st.session_state["ts_index"] = 0
    with cc5:
        jumped = st.slider("Bar", 0, max(len(sorted_ts)-1,0),
                           st.session_state["ts_index"],
                           label_visibility="collapsed")
        if jumped != st.session_state["ts_index"]:
            st.session_state["ts_index"] = jumped

    idx     = st.session_state["ts_index"]
    cur_ts  = sorted_ts[idx]
    ts_data = ms.get(cur_ts, {})

    spots   = [v["spot"] for v in ts_data.values() if v["spot"] > 0]
    cur_spot= np.mean(spots) if spots else spot_override

    et_dt   = datetime.combine(
        intra_date or date.today(),
        datetime.strptime(cur_ts, "%H:%M").time())
    eat_dt  = et_to_eat(et_dt, intra_date or date.today())

    st.markdown(
        f"### Bar: "
        f"<span style='color:#26a69a'>{cur_ts} ET</span> &nbsp;|&nbsp; "
        f"<span style='color:#aaa'>{eat_dt.strftime('%H:%M')} EAT</span>"
        f" &nbsp;|&nbsp; "
        f"<span style='color:#f0c040'>Spot: {cur_spot:.2f}</span>"
        f" &nbsp;|&nbsp; Bar {idx+1}/{len(sorted_ts)}",
        unsafe_allow_html=True)

    tc1, tc2 = st.columns(2)
    with tc1:
        chart_type = st.radio("Chart", ["GEX","DEX"],
                               horizontal=True, key="chart_type_radio")
    with tc2:
        gex_formula = st.radio(
            "GEX Formula",
            ["GEX_unsigned","GEX_signed","GEX_agg_oi","GEX_dealer_sp"],
            horizontal=True, key="gex_formula_radio")

    fkey = gex_formula if chart_type == "GEX" else "DEX"
    chart_data = [{"strike": sk,
                   "value":  ts_data.get(sk, {}).get(fkey, 0.0)}
                  for sk in all_strikes]

    title = (f"{fkey}  —  {date_str}  |  {cur_ts} ET  |  "
             f"Spot {cur_spot:.2f}")

    build_histogram_chart(chart_data, cur_spot, title)

    # ── Data tables ───────────────────────────────────────────
    st.divider()
    st.subheader("Full Session Data Tables")
    tab_g, tab_gex, tab_dex = st.tabs(["Greeks","GEX","DEX"])

    greek_cols = ["call_iv","call_delta","call_gamma","call_vanna",
                  "call_charm","call_oi","call_volume",
                  "put_iv","put_delta","put_gamma","put_vanna",
                  "put_charm","put_oi","put_volume"]

    with tab_g:
        g_cols = (["timestamp","strike","spot"] +
                  [c for c in greek_cols if c in ts_table.columns])
        st.dataframe(ts_table[[c for c in g_cols
                                if c in ts_table.columns]],
                     use_container_width=True, hide_index=True)
    with tab_gex:
        cols = (["timestamp","strike","spot"] +
                [c for c in ts_table.columns
                 if c.startswith("GEX") and "($)" in c])
        st.dataframe(fmt_df_dollars(
            ts_table[[c for c in cols if c in ts_table.columns]]),
            use_container_width=True, hide_index=True)
    with tab_dex:
        cols = (["timestamp","strike","spot"] +
                [c for c in ts_table.columns
                 if c.startswith("DEX") and "($)" in c])
        st.dataframe(fmt_df_dollars(
            ts_table[[c for c in cols if c in ts_table.columns]]),
            use_container_width=True, hide_index=True)

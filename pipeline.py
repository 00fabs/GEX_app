import pandas as pd
import numpy as np
from formulas import apply_formulas, FORMULA_COLS

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

EWMA_ALPHA = 0.2

EWMA_COLS = [
    "GEX_Call_Demand_$",
    "GEX_Put_Demand_$",
    "GEX_Reversal_$",
    "GEX_Breakout_$",
]


def pivot_wide(df_all: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-9

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
    merged  = calls_p.merge(puts_p,
                            on=["timestamp", "_strike"],
                            how="outer")
    merged  = merged.rename(columns={"_strike": "strike",
                                     "underlyingPrice": "spot"})
    merged  = merged.sort_values(
        ["timestamp", "strike"]).reset_index(drop=True)

    # ── Rolling IV stats (20-bar per strike) ─────────────────
    for side in ["call", "put"]:
        iv_col = f"{side}_iv"
        if iv_col in merged.columns:
            merged[f"{side}_iv_mean20"] = (
                merged.groupby("strike")[iv_col]
                      .transform(lambda x:
                          x.rolling(20, min_periods=1).mean()))
            merged[f"{side}_iv_std20"] = (
                merged.groupby("strike")[iv_col]
                      .transform(lambda x:
                          x.rolling(20, min_periods=1)
                           .std()
                           .fillna(eps)))
        else:
            merged[f"{side}_iv_mean20"] = 0.0
            merged[f"{side}_iv_std20"]  = eps

    # ── Session-open OI snapshot ──────────────────────────────
    for side in ["call", "put"]:
        oi_col = f"{side}_oi"
        if oi_col in merged.columns:
            merged[f"session_open_{side}_oi"] = (
                merged.groupby("strike")[oi_col]
                      .transform("first"))
        else:
            merged[f"session_open_{side}_oi"] = 0.0

    return merged


def _apply_ewma(result: pd.DataFrame) -> pd.DataFrame:
    df = result.copy()
    if "strike" not in df.columns or "timestamp" not in df.columns:
        return df

    cols_to_smooth = [c for c in EWMA_COLS if c in df.columns]
    if not cols_to_smooth:
        return df

    df = df.sort_values(["strike", "timestamp"]).reset_index(drop=True)
    for c in cols_to_smooth:
        df[c] = (df.groupby("strike")[c]
                   .transform(lambda x:
                       x.ewm(alpha=EWMA_ALPHA,
                             adjust=False).mean()))
    df = df.sort_values(["timestamp", "strike"]).reset_index(drop=True)
    return df


def _apply_spot_signals(result: pd.DataFrame) -> pd.DataFrame:
    """
    Computes GEX_Spot_Reversal and GEX_Spot_Breakout using
    cross-strike lookup per timestamp bar.

    At each bar:
      1. Find nearest call wall above spot (largest call OI above spot)
      2. Find nearest put wall below spot  (largest put OI below spot)
      3. Find ATM strike (nearest strike to spot)
      4. Read EWMA-smoothed call/put demand at ATM strike
      5. Compute wall approach weights from distance to each wall
      6. Assign spot signals to ATM strike only

    GEX_Spot_Reversal at ATM:
      put_demand_ATM  × call_wall_approach_weight  (approaching call wall → put demand = reversal down)
    + call_demand_ATM × put_wall_approach_weight   (approaching put wall → call demand = reversal up)

    GEX_Spot_Breakout at ATM:
      call_demand_ATM × call_wall_approach_weight  (approaching call wall → call demand = breakout up)
    + put_demand_ATM  × put_wall_approach_weight   (approaching put wall → put demand = breakdown)
    """
    df  = result.copy()
    eps = 1e-9

    if "timestamp" not in df.columns or "strike" not in df.columns:
        return df

    # Ensure placeholder columns exist
    if "GEX_Spot_Reversal_$" not in df.columns:
        df["GEX_Spot_Reversal_$"] = 0.0
    if "GEX_Spot_Breakout_$" not in df.columns:
        df["GEX_Spot_Breakout_$"] = 0.0

    df = df.sort_values(["timestamp", "strike"]).reset_index(drop=True)

    # ATM band for wall approach weight — 1% of spot
    # wider than proximity_wide so the signal builds
    # gradually as spot approaches the wall
    ATM_BAND_APPROACH = 0.01

    timestamps = df["timestamp"].unique()

    for ts in timestamps:
        mask      = df["timestamp"] == ts
        grp       = df[mask].copy()
        strikes   = grp["strike"].values
        spot_vals = grp["spot_used"].values if "spot_used" in grp.columns \
                    else grp["spot"].values
        spot      = float(np.nanmean(spot_vals))

        open_coi  = grp["session_open_call_oi"].values \
                    if "session_open_call_oi" in grp.columns \
                    else np.zeros(len(grp))
        open_poi  = grp["session_open_put_oi"].values \
                    if "session_open_put_oi" in grp.columns \
                    else np.zeros(len(grp))

        call_demand_vals = grp["GEX_Call_Demand_$"].values
        put_demand_vals  = grp["GEX_Put_Demand_$"].values

        # ── ATM strike index ──────────────────────────────────
        atm_idx = int(np.argmin(np.abs(strikes - spot)))

        # ── Nearest call wall above spot ──────────────────────
        above_mask    = strikes > spot
        call_oi_above = open_coi.copy()
        call_oi_above[~above_mask] = 0
        if call_oi_above.max() > 0:
            call_wall_idx    = int(np.argmax(call_oi_above))
            call_wall_strike = float(strikes[call_wall_idx])
            dist_to_call     = max(call_wall_strike - spot, eps)
        else:
            dist_to_call = np.inf

        # ── Nearest put wall below spot ───────────────────────
        below_mask   = strikes < spot
        put_oi_below = open_poi.copy()
        put_oi_below[~below_mask] = 0
        if put_oi_below.max() > 0:
            put_wall_idx    = int(np.argmax(put_oi_below))
            put_wall_strike = float(strikes[put_wall_idx])
            dist_to_put     = max(spot - put_wall_strike, eps)
        else:
            dist_to_put = np.inf

        # ── Wall approach weights ─────────────────────────────
        approach_band        = spot * ATM_BAND_APPROACH + eps
        call_approach_weight = np.exp(-dist_to_call / approach_band)
        put_approach_weight  = np.exp(-dist_to_put  / approach_band)

        # ── Demand at ATM ─────────────────────────────────────
        call_demand_atm = float(call_demand_vals[atm_idx])
        put_demand_atm  = float(put_demand_vals[atm_idx])

        # ── Spot signals ──────────────────────────────────────
        spot_reversal = (put_demand_atm  * call_approach_weight
                         + call_demand_atm * put_approach_weight)

        spot_breakout = (call_demand_atm * call_approach_weight
                         + put_demand_atm  * put_approach_weight)

        # ── Assign to ATM row only ────────────────────────────
        # All other strikes at this timestamp stay zero
        atm_global_idx = grp.index[atm_idx]
        df.loc[atm_global_idx, "GEX_Spot_Reversal_$"] = float(spot_reversal)
        df.loc[atm_global_idx, "GEX_Spot_Breakout_$"] = float(spot_breakout)

    df = df.sort_values(["timestamp", "strike"]).reset_index(drop=True)
    return df


def build_minute_series(wide_df: pd.DataFrame,
                        spot_override: float,
                        intra_date) -> tuple:

    result = apply_formulas(wide_df, spot_override, intra_date)
    result = _apply_ewma(result)
    result = _apply_spot_signals(result)

    formula_keys = [c.replace("_$", "") for c in FORMULA_COLS]

    series = {}
    for ts, grp in result.groupby("timestamp"):
        ts_key       = pd.Timestamp(ts).strftime("%H:%M")
        strikes_data = {}
        for _, row in grp.iterrows():
            sk    = int(row["strike"])
            entry = {
                "spot": float(
                    row.get("spot_used", spot_override)
                    or spot_override)
            }
            for fkey, fcol in zip(formula_keys, FORMULA_COLS):
                entry[fkey] = float(row.get(fcol, 0) or 0)
            strikes_data[sk] = entry
        series[ts_key] = strikes_data

    sorted_ts   = sorted(series.keys())
    all_strikes = sorted({sk for ts_data in series.values()
                          for sk in ts_data})
    return series, sorted_ts, all_strikes, formula_keys


def build_session_table(wide_df: pd.DataFrame,
                        spot_override: float,
                        intra_date) -> pd.DataFrame:

    result = apply_formulas(wide_df, spot_override, intra_date)
    result = _apply_ewma(result)
    result = _apply_spot_signals(result)
    result = result.reset_index(drop=True)

    greek_cols   = ["call_iv","call_delta","call_gamma",
                    "call_vanna","call_charm","call_oi",
                    "call_volume","put_iv","put_delta",
                    "put_gamma","put_vanna","put_charm",
                    "put_oi","put_volume"]
    formula_cols = [c for c in result.columns if c.endswith("_$")]
    available    = set(result.columns)

    keep = []
    for c in (["timestamp","strike","spot_used"] +
              [c for c in greek_cols   if c in available] +
              [c for c in formula_cols if c in available]):
        if c in available and c not in keep:
            keep.append(c)

    ts_df     = result[keep].copy()
    ts_df     = ts_df.rename(columns={"spot_used": "spot"})
    sort_cols = [c for c in ["timestamp","strike"]
                 if c in ts_df.columns]
    if sort_cols:
        ts_df = ts_df.sort_values(sort_cols).reset_index(drop=True)

    ts_df.rename(
        columns={c: c.replace("_$", " ($)")
                 for c in formula_cols if c in ts_df.columns},
        inplace=True)
    return ts_df

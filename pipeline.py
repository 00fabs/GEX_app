# ─────────────────────────────────────────────────────────────
# pipeline.py — pivot, session series builder, table builder
# ─────────────────────────────────────────────────────────────
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
        df = df.rename(columns=rename)
        base = (["timestamp", "_strike", "underlyingPrice"]
                if prefix == "call" else ["timestamp", "_strike"])
        keep = base + list(rename.values())
        return df[[c for c in keep if c in df.columns]].copy()

    calls_p = pivot_side(df_all[df_all["_optType"] == "C"].copy(), "call")
    puts_p  = pivot_side(df_all[df_all["_optType"] == "P"].copy(), "put")

    merged = calls_p.merge(puts_p, on=["timestamp", "_strike"], how="outer")
    merged = merged.rename(columns={"_strike": "strike",
                                    "underlyingPrice": "spot"})
    merged = merged.sort_values(["timestamp", "strike"]).reset_index(drop=True)

    # ── Rolling IV stats (20-bar per strike) ─────────────────
    for side in ["call", "put"]:
        iv_col = f"{side}_iv"
        if iv_col in merged.columns:
            merged[f"{side}_iv_mean20"] = (
                merged.groupby("strike")[iv_col]
                      .transform(lambda x: x.rolling(20, min_periods=1).mean())
            )
            merged[f"{side}_iv_std20"] = (
                merged.groupby("strike")[iv_col]
                      .transform(lambda x: x.rolling(20, min_periods=1).std())
                      .fillna(eps)
            )
        else:
            merged[f"{side}_iv_mean20"] = 0.0
            merged[f"{side}_iv_std20"]  = eps

    # ── Rolling IV slope (5-bar per strike) ───────────────────
    for side in ["call", "put"]:
        iv_col = f"{side}_iv"
        if iv_col in merged.columns:
            merged[f"{side}_iv_slope5"] = (
                merged.groupby("strike")[iv_col]
                      .transform(lambda x: x - x.shift(5).bfill())   # ← Fixed here
            )
        else:
            merged[f"{side}_iv_slope5"] = 0.0

    # ── Session-open OI snapshot ──────────────────────────────
    for side in ["call", "put"]:
        oi_col = f"{side}_oi"
        if oi_col in merged.columns:
            merged[f"session_open_{side}_oi"] = (
                merged.groupby("strike")[oi_col]
                      .transform("first")
            )
        else:
            merged[f"session_open_{side}_oi"] = 0.0

    return merged


def _apply_ewma(result: pd.DataFrame) -> pd.DataFrame:
    """Apply EWMA smoothing to selected demand signals."""
    df = result.copy()
    if "strike" not in df.columns or "timestamp" not in df.columns:
        return df

    cols_to_smooth = [c for c in EWMA_COLS if c in df.columns]
    if not cols_to_smooth:
        return df

    df = df.sort_values(["strike", "timestamp"]).reset_index(drop=True)
    for c in cols_to_smooth:
        df[c] = (df.groupby("strike")[c]
                   .transform(lambda x: x.ewm(alpha=EWMA_ALPHA, adjust=False).mean()))
    df = df.sort_values(["timestamp", "strike"]).reset_index(drop=True)
    return df


def _apply_spot_signals(result: pd.DataFrame) -> pd.DataFrame:
    """Compute spot-level reversal and breakout signals."""
    df = result.copy()
    eps = 1e-9

    if "timestamp" not in df.columns or "strike" not in df.columns:
        return df

    # Initialize columns
    for col in ["GEX_Spot_Reversal_\( ", "GEX_Spot_Breakout_ \)"]:
        if col not in df.columns:
            df[col] = 0.0

    df = df.sort_values(["timestamp", "strike"]).reset_index(drop=True)
    PROXIMITY_BAND = 0.005

    for ts in df["timestamp"].unique():
        mask = df["timestamp"] == ts
        grp = df[mask].copy()

        if len(grp) < 2:
            continue

        strikes = grp["strike"].values.astype(float)
        spot_vals = grp.get("spot_used", grp.get("spot", pd.Series([0]))).values
        spot = float(np.nanmean(spot_vals))

        def gcol(name):
            return (pd.to_numeric(grp[name], errors="coerce").fillna(0).values
                    if name in grp.columns else np.zeros(len(grp)))

        cg = gcol("call_gamma")
        pg = gcol("put_gamma")
        open_coi = gcol("session_open_call_oi")
        open_poi = gcol("session_open_put_oi")
        call_iv_slope = gcol("call_iv_slope5")
        put_iv_slope = gcol("put_iv_slope5")
        call_demand_vals = gcol("GEX_Call_Demand_$")
        put_demand_vals = gcol("GEX_Put_Demand_$")

        net_gamma = cg * open_coi - pg * open_poi

        # Find gamma flip strike
        flip_idx = None
        flip_dist = np.inf
        for i in range(len(net_gamma) - 1):
            if net_gamma[i] * net_gamma[i + 1] <= 0:
                d0 = abs(strikes[i] - spot)
                d1 = abs(strikes[i + 1] - spot)
                candidate_idx = i if d0 <= d1 else i + 1
                candidate_dist = min(d0, d1)
                if candidate_dist < flip_dist:
                    flip_dist = candidate_dist
                    flip_idx = candidate_idx

        if flip_idx is None:
            near_spot_mask = np.abs(strikes - spot) <= spot * 0.02
            if near_spot_mask.sum() > 0:
                near_indices = np.where(near_spot_mask)[0]
                near_abs_ng = np.abs(net_gamma[near_spot_mask])
                flip_idx = near_indices[int(np.argmin(near_abs_ng))]
            else:
                flip_idx = int(np.argmin(np.abs(strikes - spot)))

        flip_global_idx = grp.index[flip_idx]

        # Proximity weight
        prox_band = spot * PROXIMITY_BAND + eps
        dist = abs(spot - strikes[flip_idx])
        proximity_weight = np.exp(-dist / prox_band)

        # Signals at flip strike
        put_slope_confirm = max(float(put_iv_slope[flip_idx]), 0.0)
        call_slope_confirm = max(float(call_iv_slope[flip_idx]), 0.0)
        call_demand_flip = max(float(call_demand_vals[flip_idx]), 0.0)
        put_demand_flip = max(float(put_demand_vals[flip_idx]), 0.0)

        reversal_raw = put_demand_flip * (1.0 + put_slope_confirm)
        breakout_raw = call_demand_flip * (1.0 + call_slope_confirm)
        total_raw = reversal_raw + breakout_raw + eps

        reversal_dom = reversal_raw / total_raw
        breakout_dom = breakout_raw / total_raw

        net_reversal = max(reversal_dom - breakout_dom, 0.0)
        net_breakout = max(breakout_dom - reversal_dom, 0.0)

        spot_reversal = net_reversal * proximity_weight * total_raw
        spot_breakout = net_breakout * proximity_weight * total_raw

        df.loc[flip_global_idx, "GEX_Spot_Reversal_$"] = float(spot_reversal)
        df.loc[flip_global_idx, "GEX_Spot_Breakout_$"] = float(spot_breakout)

    return df.sort_values(["timestamp", "strike"]).reset_index(drop=True)


def build_minute_series(wide_df: pd.DataFrame,
                        spot_override: float,
                        intra_date) -> tuple:
    result = apply_formulas(wide_df, spot_override, intra_date)
    result = _apply_ewma(result)
    result = _apply_spot_signals(result)

    formula_keys = [c.replace("_$", "") for c in FORMULA_COLS]

    series = {}
    for ts, grp in result.groupby("timestamp"):
        ts_key = pd.Timestamp(ts).strftime("%H:%M")
        strikes_data = {}
        for _, row in grp.iterrows():
            sk = int(row["strike"])
            entry = {"spot": float(row.get("spot_used", spot_override) or spot_override)}
            for fkey, fcol in zip(formula_keys, FORMULA_COLS):
                entry[fkey] = float(row.get(fcol, 0) or 0)
            strikes_data[sk] = entry
        series[ts_key] = strikes_data

    sorted_ts = sorted(series.keys())
    all_strikes = sorted({sk for ts_data in series.values() for sk in ts_data})
    return series, sorted_ts, all_strikes, formula_keys


def build_session_table(wide_df: pd.DataFrame,
                        spot_override: float,
                        intra_date) -> pd.DataFrame:
    result = apply_formulas(wide_df, spot_override, intra_date)
    result = _apply_ewma(result)
    result = _apply_spot_signals(result)
    result = result.reset_index(drop=True)

    greek_cols = ["call_iv", "call_delta", "call_gamma", "call_vanna",
                  "call_charm", "call_oi", "call_volume",
                  "put_iv", "put_delta", "put_gamma", "put_vanna",
                  "put_charm", "put_oi", "put_volume"]

    formula_cols = [c for c in result.columns if c.endswith("_$")]
    available = set(result.columns)

    keep = []
    for c in (["timestamp", "strike", "spot_used"] +
              [c for c in greek_cols if c in available] +
              [c for c in formula_cols if c in available]):
        if c in available and c not in keep:
            keep.append(c)

    ts_df = result[keep].copy()
    ts_df = ts_df.rename(columns={"spot_used": "spot"})

    if ["timestamp", "strike"] == [c for c in ["timestamp", "strike"] if c in ts_df.columns]:
        ts_df = ts_df.sort_values(["timestamp", "strike"]).reset_index(drop=True)

    ts_df.rename(columns={c: c.replace("_\( ", " ( \))") 
                         for c in formula_cols if c in ts_df.columns},
                 inplace=True)
    return ts_df

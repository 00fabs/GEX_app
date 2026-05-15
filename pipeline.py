# ─────────────────────────────────────────────────────────────
# pipeline.py — Core computation pipeline
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

DEMAND_RATIO_THRESH = 2.0
REVERSAL_RATIO_THRESH = 1.5
BREAKOUT_RATIO_THRESH = 1.5
GAMMA_DENSITY_THRESH = 0.20
DELTA_ACC_THRESH = 0.20


def pivot_wide(df_all: pd.DataFrame) -> pd.DataFrame:
    """Merge calls and puts into wide format."""
    def pivot_side(df, prefix):
        rename = {k: f"{prefix}_{v}" for k, v in COL_MAP.items() if k in df.columns}
        df = df.rename(columns=rename)
        base = (["timestamp", "_strike", "underlyingPrice"] if prefix == "call"
                else ["timestamp", "_strike"])
        keep = base + list(rename.values())
        return df[[c for c in keep if c in df.columns]].copy()

    calls_p = pivot_side(df_all[df_all["_optType"] == "C"].copy(), "call")
    puts_p = pivot_side(df_all[df_all["_optType"] == "P"].copy(), "put")

    merged = calls_p.merge(puts_p, on=["timestamp", "_strike"], how="outer")
    merged = merged.rename(columns={"_strike": "strike", "underlyingPrice": "spot"})
    merged = merged.sort_values(["timestamp", "strike"]).reset_index(drop=True)
    return merged


def prepare_wide(merged: pd.DataFrame) -> pd.DataFrame:
    """Add rolling IV stats and session-open OI snapshot."""
    eps = 1e-9
    df = merged.copy()

    # Rolling IV stats
    for side in ["call", "put"]:
        iv_col = f"{side}_iv"
        if iv_col in df.columns:
            df[f"{side}_iv_mean20"] = (
                df.groupby("strike")[iv_col]
                  .transform(lambda x: x.rolling(20, min_periods=1).mean())
            )
            df[f"{side}_iv_std20"] = (
                df.groupby("strike")[iv_col]
                  .transform(lambda x: x.rolling(20, min_periods=1).std())
                  .fillna(eps)
            )
        else:
            df[f"{side}_iv_mean20"] = 0.0
            df[f"{side}_iv_std20"] = eps

    # Session-open OI snapshot
    for side in ["call", "put"]:
        oi_col = f"{side}_oi"
        if oi_col in df.columns:
            df[f"session_open_{side}_oi"] = (
                df.groupby("strike")[oi_col].transform("first")
            )
        else:
            df[f"session_open_{side}_oi"] = 0.0

    return df


def run_full_pipeline(wide_df: pd.DataFrame, spot_override: float, intra_date):
    """Full computation pipeline."""
    result = apply_formulas(wide_df, spot_override, intra_date)
    result = _apply_ewma(result)
    result = _apply_cumulative_demand(result)
    result = _apply_spot_velocity(result)      # ← New
    result = _apply_entry_signal(result)       # ← New
    return result


# ─────────────────────────────────────────────────────────────
# Internal Helpers
# ─────────────────────────────────────────────────────────────
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
                   .transform(lambda x: x.ewm(alpha=EWMA_ALPHA, adjust=False).mean()))
    return df.sort_values(["timestamp", "strike"]).reset_index(drop=True)


def _apply_cumulative_demand(result: pd.DataFrame) -> pd.DataFrame:
    df = result.copy()
    if "strike" not in df.columns or "timestamp" not in df.columns:
        return df

    df = df.sort_values(["strike", "timestamp"]).reset_index(drop=True)
    for side in ["call", "put"]:
        demand_col = f"GEX_{side.capitalize()}_Demand_$"
        cumul_col = f"_cumul_{side}_demand"
        if demand_col in df.columns:
            df[cumul_col] = df.groupby("strike")[demand_col].transform(
                lambda x: x.clip(lower=0).cumsum())
        else:
            df[cumul_col] = 0.0
    return df.sort_values(["timestamp", "strike"]).reset_index(drop=True)


def _apply_spot_velocity(result: pd.DataFrame) -> pd.DataFrame:
    """Compute spot velocity per timestamp (used by entry signal)."""
    df = result.copy()
    if "timestamp" not in df.columns or "spot_used" not in df.columns:
        df["_spot_velocity"] = 0.0
        return df

    df = df.sort_values(["timestamp", "strike"]).reset_index(drop=True)

    spot_by_ts = df.groupby("timestamp")["spot_used"].mean().sort_index()
    velocity_by_ts = spot_by_ts.diff().fillna(0.0)
    ts_to_vel = velocity_by_ts.to_dict()

    df["_spot_velocity"] = df["timestamp"].map(ts_to_vel).fillna(0.0)
    return df


def _apply_entry_signal(result: pd.DataFrame) -> pd.DataFrame:
    """Entry signal logic with wall confirmation, approach filter, and best entry strike."""
    df = result.copy()
    eps = 1e-9
    sig = "GEX_Entry_Signal_$"
    if sig not in df.columns:
        df[sig] = 0.0
    df[sig] = 0.0

    if "timestamp" not in df.columns or "strike" not in df.columns:
        return df

    df = df.sort_values(["timestamp", "strike"]).reset_index(drop=True)

    def gcol(grp, name):
        return (pd.to_numeric(grp[name], errors="coerce").fillna(0).values
                if name in grp.columns else np.zeros(len(grp)))

    for ts, grp in df.groupby("timestamp"):
        if len(grp) < 3:
            continue

        strikes = grp["strike"].values.astype(float)
        spot = float(np.nanmean(grp.get("spot_used", grp.get("spot", 0))))

        cumul_call = gcol(grp, "_cumul_call_demand")
        cumul_put = gcol(grp, "_cumul_put_demand")
        reversal_vals = gcol(grp, "GEX_Reversal_$")
        breakout_vals = gcol(grp, "GEX_Breakout_$")
        velocity = gcol(grp, "_spot_velocity")

        # Wall flags (fallback if not present)
        call_flag = gcol(grp, "_call_wall_flag")
        put_flag = gcol(grp, "_put_wall_flag") if "_put_wall_flag" in grp.columns else (1.0 - call_flag)

        reversal_candidates = []
        breakout_candidates = []

        # Scan above spot
        for i in np.where(strikes > spot)[0]:
            cp, cc = cumul_put[i], cumul_call[i]
            rev, brk = reversal_vals[i], breakout_vals[i]
            if cp > cc * DEMAND_RATIO_THRESH and rev > brk * REVERSAL_RATIO_THRESH and cp > 0:
                reversal_candidates.append((i, "reversal_down", cp))
            elif cc > cp * DEMAND_RATIO_THRESH and brk > rev * BREAKOUT_RATIO_THRESH and cc > 0:
                breakout_candidates.append((i, "breakout_up", cc))

        # Scan below spot
        for i in np.where(strikes < spot)[0]:
            cp, cc = cumul_put[i], cumul_call[i]
            rev, brk = reversal_vals[i], breakout_vals[i]
            if cc > cp * DEMAND_RATIO_THRESH and rev > brk * REVERSAL_RATIO_THRESH and cc > 0:
                reversal_candidates.append((i, "reversal_up", cc))
            elif cp > cc * DEMAND_RATIO_THRESH and brk > rev * BREAKOUT_RATIO_THRESH and cp > 0:
                breakout_candidates.append((i, "breakdown", cp))

        all_candidates = reversal_candidates + breakout_candidates
        if not all_candidates:
            continue

        # Sort by strength
        all_candidates.sort(key=lambda x: x[2], reverse=True)
        wall_idx, direction, strength = all_candidates[0]

        # Approach confirmation using velocity
        spot_vel = float(np.nanmean(velocity))
        if abs(spot_vel) > eps:
            if direction == "reversal_down" and spot_vel <= 0:
                continue
            if direction == "reversal_up" and spot_vel >= 0:
                continue
            if direction == "breakout_up" and spot_vel <= 0:
                continue
            if direction == "breakdown" and spot_vel >= 0:
                continue

        # Entry strike selection (between spot and wall)
        wall_strike = float(strikes[wall_idx])
        if direction in ("reversal_down", "breakout_up"):
            between = np.where((strikes > spot) & (strikes < wall_strike))[0]
        else:
            between = np.where((strikes < spot) & (strikes > wall_strike))[0]

        entry_idx = wall_idx
        if len(between) > 0:
            entry_idx = between[0]   # fallback - can be improved later

        entry_global_idx = grp.index[entry_idx]
        sign = 1.0 if direction in ("reversal_up", "breakout_up") else -1.0
        df.loc[entry_global_idx, sig] = sign * float(strength)

    return df.sort_values(["timestamp", "strike"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────
def build_minute_series(result: pd.DataFrame, spot_override: float, intra_date=None) -> tuple:
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


def build_session_table(result: pd.DataFrame) -> pd.DataFrame:
    df = result.reset_index(drop=True)
    greek_cols = ["call_iv","call_delta","call_gamma","call_vanna","call_charm","call_oi","call_volume",
                  "put_iv","put_delta","put_gamma","put_vanna","put_charm","put_oi","put_volume"]
    formula_cols = [c for c in df.columns if c.endswith("_$")]
    available = set(df.columns)

    keep = []
    for c in (["timestamp","strike","spot_used"] +
              [c for c in greek_cols if c in available] +
              [c for c in formula_cols if c in available]):
        if c in available and c not in keep:
            keep.append(c)

    ts_df = df[keep].copy()
    ts_df = ts_df.rename(columns={"spot_used": "spot"})
    ts_df = ts_df.sort_values(["timestamp", "strike"]).reset_index(drop=True)

    ts_df.rename(columns={c: c.replace("_\( ", " ( \))") for c in formula_cols if c in ts_df.columns},
                 inplace=True)
    return ts_df

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


def pivot_wide(df_all: pd.DataFrame) -> pd.DataFrame:
    def pivot_side(df, prefix):
        rename = {k: f"{prefix}_{v}"
                  for k, v in COL_MAP.items() if k in df.columns}
        df     = df.rename(columns=rename)
        base   = (["timestamp", "_strike", "underlyingPrice"]
                  if prefix == "call"
                  else ["timestamp", "_strike"])
        keep   = base + list(rename.values())
        return df[[c for c in keep if c in df.columns]].copy()

    calls_p = pivot_side(df_all[df_all["_optType"] == "C"].copy(), "call")
    puts_p  = pivot_side(df_all[df_all["_optType"] == "P"].copy(), "put")
    merged  = calls_p.merge(puts_p, on=["timestamp", "_strike"], how="outer")
    merged  = merged.rename(columns={"_strike": "strike",
                                     "underlyingPrice": "spot"})
    return merged.sort_values(["timestamp", "strike"]).reset_index(drop=True)


def build_minute_series(wide_df: pd.DataFrame,
                        spot_override: float,
                        intra_date) -> tuple:
    result = apply_formulas(wide_df, spot_override, intra_date)

    # Clean key names: "GEX_unsigned_$" → "GEX_unsigned"
    formula_keys = [c.replace("_$", "") for c in FORMULA_COLS]

    series = {}
    for ts, grp in result.groupby("timestamp"):
        ts_key       = pd.Timestamp(ts).strftime("%H:%M")
        strikes_data = {}
        for _, row in grp.iterrows():
            sk    = int(row["strike"])
            entry = {"spot": float(row.get("spot_used", spot_override) or spot_override)}
            for fkey, fcol in zip(formula_keys, FORMULA_COLS):
                entry[fkey] = float(row.get(fcol, 0) or 0)
            strikes_data[sk] = entry
        series[ts_key] = strikes_data

    sorted_ts   = sorted(series.keys())
    all_strikes = sorted({sk for ts_data in series.values() for sk in ts_data})
    return series, sorted_ts, all_strikes, formula_keys


def build_session_table(wide_df: pd.DataFrame,
                        spot_override: float,
                        intra_date) -> pd.DataFrame:
    result = apply_formulas(wide_df, spot_override, intra_date)

    # Ensure index is clean after apply_formulas
    result = result.reset_index(drop=True)

    greek_cols   = ["call_iv","call_delta","call_gamma","call_vanna",
                    "call_charm","call_oi","call_volume",
                    "put_iv","put_delta","put_gamma","put_vanna",
                    "put_charm","put_oi","put_volume"]
    formula_cols = [c for c in result.columns if c.endswith("_$")]

    # Only keep columns that actually exist after reset
    available = set(result.columns)

    keep = []
    for c in (["timestamp", "strike", "spot_used"] +
              [c for c in greek_cols   if c in available] +
              [c for c in formula_cols if c in available]):
        if c in available and c not in keep:
            keep.append(c)

    ts_df = result[keep].copy()
    ts_df = ts_df.rename(columns={"spot_used": "spot"})

    # Safe sort — only sort by columns that exist
    sort_cols = [c for c in ["timestamp", "strike"] if c in ts_df.columns]
    if sort_cols:
        ts_df = ts_df.sort_values(sort_cols).reset_index(drop=True)

    ts_df.rename(columns={c: c.replace("_$", " ($)")
                           for c in formula_cols if c in ts_df.columns},
                 inplace=True)
    return ts_df

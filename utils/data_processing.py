# utils/data_processing.py
import pandas as pd
from calculations import apply_formulas      # ← Fixed for your structure

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
        rename = {k: f"{prefix}_{v}" for k, v in COL_MAP.items() if k in df.columns}
        df = df.rename(columns=rename)
        base = (["timestamp", "_strike", "underlyingPrice"] if prefix == "call"
                else ["timestamp", "_strike"])
        keep = base + list(rename.values())
        return df[[c for c in keep if c in df.columns]].copy()

    calls_p = pivot_side(df_all[df_all["_optType"] == "C"].copy(), "call")
    puts_p  = pivot_side(df_all[df_all["_optType"] == "P"].copy(), "put")
    merged  = calls_p.merge(puts_p, on=["timestamp", "_strike"], how="outer")
    merged  = merged.rename(columns={"_strike": "strike", "underlyingPrice": "spot"})
    return merged.sort_values(["timestamp", "strike"]).reset_index(drop=True)


def build_minute_series(wide_df, spot_override, intra_date):
    result = apply_formulas(wide_df, spot_override, intra_date)

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
                "DEX":          float(row.get("DEX_$",           0) or 0),
                "spot":         float(row.get("spot_used", spot_override) or spot_override),
            }
        series[ts_key] = strikes_data

    sorted_ts = sorted(series.keys())
    all_strikes = sorted(set(
        sk for ts_data in series.values() for sk in ts_data.keys()))

    return series, sorted_ts, all_strikes


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
        columns={c: c.replace("_\( ", " ( \))") for c in formula_cols if c in ts_df.columns},
        inplace=True
    )
    return ts_df

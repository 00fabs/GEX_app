# utils/calculations.py
import pandas as pd
import numpy as np
from .constants import SPX_MULT

def apply_formulas(df, spot_override, intra_date):
    """Main function where GEX/DEX formulas live"""
    out = df.copy()

    spot_col = (out["spot"].fillna(spot_override)
                if "spot" in out.columns
                else pd.Series([spot_override] * len(out), index=out.index))
    out["spot_used"] = spot_col

    def col(name):
        return out.get(name, pd.Series(0.0, index=out.index)).fillna(0)

    cg  = col("call_gamma");  pg  = col("put_gamma")
    coi = col("call_oi");     poi = col("put_oi")
    cd  = col("call_delta");  pd_ = col("put_delta")

    S  = spot_col
    S2 = S ** 2
    M  = SPX_MULT

    # ================== GEX FORMULAS ==================
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

    # ================== DEX FORMULA ==================
    out["DEX"]   = (cd * coi + pd_ * poi) * M * S
    out["DEX_$"] = out["DEX"] / 1e9

    return out

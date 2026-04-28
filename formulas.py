# ─────────────────────────────────────────────────────────────
# formulas.py — GEX / DEX formula engine
# THIS IS YOUR PRIMARY WORKING FILE for adding/editing formulas
#
# To add a new formula:
#   1. Compute it inside apply_formulas() below
#   2. Add the _$ column name to FORMULA_COLS at the bottom
#   3. It will automatically appear in charts and data tables
# ─────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
from config import SPX_MULT

# ── Column registry — add new formula column names here ──────
# Any column ending in _$ is auto-detected, but listing here
# makes intent explicit and controls table column ordering.
FORMULA_COLS = [
    "GEX_unsigned_$",
    "GEX_signed_$",
    "GEX_agg_oi_$",
    "GEX_dealer_sp_$",
    "DEX_$",
    # ← add new formula columns here
]


def apply_formulas(df: pd.DataFrame, spot_override: float,
                   intra_date) -> pd.DataFrame:
    """
    Takes the wide-format merged dataframe (one row per timestamp+strike)
    and computes all GEX / DEX columns.

    All dollar-normalised outputs are in $Billions (divide raw by 1e9).

    Parameters
    ----------
    df            : wide DataFrame from pipeline.pivot_wide()
    spot_override : fallback spot price when API underlyingPrice is missing
    intra_date    : the session date (unused in math, available for future
                    time-decay formulas that need calendar date)

    Returns
    -------
    df with all formula columns appended.
    """
    out = df.copy()

    # ── Spot price ────────────────────────────────────────────
    spot_col = (out["spot"].fillna(spot_override)
                if "spot" in out.columns
                else pd.Series([spot_override] * len(out), index=out.index))
    out["spot_used"] = spot_col

    # ── Helper: safely get column or zeros ───────────────────
    def col(name):
        return out.get(name, pd.Series(0.0, index=out.index)).fillna(0)

    cg  = col("call_gamma")
    pg  = col("put_gamma")
    coi = col("call_oi")
    poi = col("put_oi")
    cd  = col("call_delta")
    pd_ = col("put_delta")

    S  = spot_col
    S2 = S ** 2
    M  = SPX_MULT

    # ── GEX: Unsigned (total gamma exposure, no sign) ────────
    # Both calls and puts treated as positive gamma sources.
    # Represents total market maker hedging pressure magnitude.
    out["GEX_unsigned"]   = (cg * coi + pg * poi) * M * S2
    out["GEX_unsigned_$"] = out["GEX_unsigned"] / 1e9

    # ── GEX: Signed (calls positive, puts negative) ──────────
    # Net dealer gamma. Positive = dealers are long gamma (stabilising).
    # Negative = dealers are short gamma (destabilising / trend-following).
    out["GEX_signed"]   = (cg * coi - pg * poi) * M * S2
    out["GEX_signed_$"] = out["GEX_signed"] / 1e9

    # ── GEX: Aggregated by timestamp (sum across all strikes) ─
    # Collapses the strike dimension: one value per minute bar.
    # Useful for seeing the net market-wide gamma flip level.
    if "timestamp" in out.columns:
        agg_map = (out.groupby("timestamp")["GEX_signed"]
                      .sum().rename("GEX_agg_oi"))
        out = out.merge(agg_map, on="timestamp", how="left")
    else:
        out["GEX_agg_oi"] = out["GEX_signed"].sum()
    out["GEX_agg_oi_$"] = out["GEX_agg_oi"] / 1e9

    # ── GEX: Dealer-signed perspective (flip of unsigned) ────
    # Dealer must hedge opposite to customer position.
    # Negative of unsigned = dealer net sell pressure.
    out["GEX_dealer_sp"]   = -(cg * coi + pg * poi) * M * S2
    out["GEX_dealer_sp_$"] = out["GEX_dealer_sp"] / 1e9

    # ── DEX: Delta Exposure ───────────────────────────────────
    # Total directional exposure dealers must hedge (shares equivalent).
    # Call delta positive, put delta is already negative from BSM.
    out["DEX"]   = (cd * coi + pd_ * poi) * M * S
    out["DEX_$"] = out["DEX"] / 1e9

    # ─────────────────────────────────────────────────────────
    # ADD NEW FORMULAS BELOW THIS LINE
    # Example template:
    #
    # out["MY_FORMULA"]   = <expression using cg, pg, coi, poi, cd, pd_, S, M>
    # out["MY_FORMULA_$"] = out["MY_FORMULA"] / 1e9
    #
    # Then add "MY_FORMULA_$" to FORMULA_COLS at the top of this file.
    # ─────────────────────────────────────────────────────────

    return out

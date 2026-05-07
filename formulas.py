# ─────────────────────────────────────────────────────────────
# formulas.py — GEX / DEX formula engine
# ─────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
from config import SPX_MULT

FORMULA_COLS = [
    "GEX_unsigned_$",
    "GEX_signed_$",
    "GEX_agg_oi_$",
    "GEX_dealer_sp_$",
    "GEX_vol_weighted_$",
    "GEX_net_oi_$",
    "DEX_$",
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

    cg   = col("call_gamma")
    pg   = col("put_gamma")
    coi  = col("call_oi")
    poi  = col("put_oi")
    cvol = col("call_volume")
    pvol = col("put_volume")
    cd   = col("call_delta")
    pd_  = col("put_delta")

    S  = spot_col
    S2 = S ** 2
    M  = SPX_MULT

    # ── GEX: Unsigned (total gamma exposure, no sign) ────────
    # Both calls and puts treated as positive gamma sources.
    # Represents total market maker hedging pressure magnitude.
    # Use this for MVC ranking — largest bar = strongest attractor.
    out["GEX_unsigned"]   = (cg * coi + pg * poi) * M * S2
    out["GEX_unsigned_$"] = out["GEX_unsigned"] / 1e9

    # ── GEX: Signed (calls positive, puts negative) ──────────
    # Net dealer gamma per strike using OI as weight.
    # Positive = call gamma dominates = ceiling / resistance.
    # Negative = put gamma dominates = floor / support.
    # Put leg gets -1 because dealer short puts require selling
    # into downward moves — opposite hedging direction to short calls.
    out["GEX_signed"]   = (cg * coi - pg * poi) * M * S2
    out["GEX_signed_$"] = out["GEX_signed"] / 1e9

    # ── GEX: Aggregated by timestamp (sum across all strikes) ─
    # Collapses the strike dimension: one scalar value per minute bar.
    # Positive aggregate = positive gamma regime = pinning expected.
    # Negative aggregate = negative gamma regime = trending expected.
    if "timestamp" in out.columns:
        agg_map = (out.groupby("timestamp")["GEX_signed"]
                      .sum().rename("GEX_agg_oi"))
        out = out.merge(agg_map, on="timestamp", how="left")
    else:
        out["GEX_agg_oi"] = out["GEX_signed"].sum()
    out["GEX_agg_oi_$"] = out["GEX_agg_oi"] / 1e9

    # ── GEX: Dealer-signed perspective (flip of unsigned) ────
    # Dealer must hedge opposite to customer position.
    # Always <= 0 by construction. Magnitude = hedging pressure.
    # Largest absolute value = strongest attractor strike.
    out["GEX_dealer_sp"]   = -(cg * coi + pg * poi) * M * S2
    out["GEX_dealer_sp_$"] = out["GEX_dealer_sp"] / 1e9

    # ── GEX: Volume-weighted signed ──────────────────────────
    # Replaces OI with intraday volume as the position weight.
    # Critical for 0DTE: prior-day OI is stale by open since
    # 0DTE contracts are opened and closed intraday. Volume
    # accumulation reflects live positioning at each bar far
    # more accurately than EOD OI.
    # Positive = call volume dominates = intraday ceiling.
    # Negative = put volume dominates = intraday floor.
    # Near zero early session (little volume traded yet) —
    # values become more meaningful as the session progresses.
    out["GEX_vol_weighted"]   = (cg * cvol - pg * pvol) * M * S2
    out["GEX_vol_weighted_$"] = out["GEX_vol_weighted"] / 1e9

    # ── GEX: Net OI signed ───────────────────────────────────
    # Uses the mean of call and put gamma at each strike as a
    # single representative gamma value, then weights by net OI
    # imbalance (call OI minus put OI).
    # Rationale: BSM gamma is theoretically identical for calls
    # and puts at the same strike (put-call parity), but iVol
    # computes them from market prices so they differ slightly —
    # averaging removes that noise and keeps the signal in the
    # OI imbalance itself.
    # Positive = call OI dominates at this strike = resistance.
    # Negative = put OI dominates at this strike = support.
    # Cleaner than GEX_signed when gamma values are noisy across
    # strikes because the regime signal comes purely from OI skew.
    gamma_avg = (cg + pg) / 2
    out["GEX_net_oi"]   = gamma_avg * (coi - poi) * M * S2
    out["GEX_net_oi_$"] = out["GEX_net_oi"] / 1e9

    # ── DEX: Delta Exposure ───────────────────────────────────
    # Total directional exposure dealers must hedge per strike.
    # Call delta positive, put delta already negative from BSM.
    # Positive DEX = calls dominate = dealers net short delta
    #   = they buy as price approaches = bullish attractor.
    # Negative DEX = puts dominate = dealers net long delta
    #   = bearish attractor / put-supported floor.
    out["DEX"]   = (cd * coi + pd_ * poi) * M * S
    out["DEX_$"] = out["DEX"] / 1e9

    # ─────────────────────────────────────────────────────────
    # ADD NEW FORMULAS BELOW THIS LINE
    # Example template:
    #
    # out["MY_FORMULA"]   = <expression using cg, pg, coi, poi,
    #                        cvol, pvol, cd, pd_, S, M>
    # out["MY_FORMULA_$"] = out["MY_FORMULA"] / 1e9
    #
    # Then add "MY_FORMULA_$" to FORMULA_COLS at the top.
    # ─────────────────────────────────────────────────────────

    return out

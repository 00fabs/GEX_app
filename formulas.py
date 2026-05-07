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
    "Dealer_GEX_Long_$",
    "Dealer_GEX_Short_$",
    "Net_Dealer_GEX_$",
    "DEX_$",
]


def apply_formulas(df: pd.DataFrame, spot_override: float,
                   intra_date) -> pd.DataFrame:
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

    # ── GEX: Unsigned ─────────────────────────────────────────
    # Total gamma exposure magnitude per strike, no sign.
    # Both calls and puts treated as positive gamma sources.
    # Largest bar = strongest attractor / pin candidate.
    out["GEX_unsigned"]   = (cg * coi + pg * poi) * M * S2
    out["GEX_unsigned_$"] = out["GEX_unsigned"] / 1e9

    # ── GEX: Signed ───────────────────────────────────────────
    # Net dealer gamma per strike using OI as weight.
    # Positive = call gamma dominates = ceiling / resistance.
    # Negative = put gamma dominates = floor / support.
    out["GEX_signed"]   = (cg * coi - pg * poi) * M * S2
    out["GEX_signed_$"] = out["GEX_signed"] / 1e9

    # ── GEX: Aggregated by timestamp ──────────────────────────
    # Sum of GEX_signed across all strikes per minute bar.
    # Positive = positive gamma regime = pinning expected.
    # Negative = negative gamma regime = trending expected.
    if "timestamp" in out.columns:
        agg_map = (out.groupby("timestamp")["GEX_signed"]
                      .sum().rename("GEX_agg_oi"))
        out = out.merge(agg_map, on="timestamp", how="left")
    else:
        out["GEX_agg_oi"] = out["GEX_signed"].sum()
    out["GEX_agg_oi_$"] = out["GEX_agg_oi"] / 1e9

    # ── GEX: Dealer-signed perspective ────────────────────────
    # Flip of unsigned. Always <= 0.
    # Magnitude = total dealer hedging pressure at each strike.
    out["GEX_dealer_sp"]   = -(cg * coi + pg * poi) * M * S2
    out["GEX_dealer_sp_$"] = out["GEX_dealer_sp"] / 1e9

    # ── GEX: Volume-weighted signed ───────────────────────────
    # Replaces OI with intraday volume as the position weight.
    # More accurate for 0DTE where prior-day OI is stale.
    # Values grow more meaningful as session volume accumulates.
    out["GEX_vol_weighted"]   = (cg * cvol - pg * pvol) * M * S2
    out["GEX_vol_weighted_$"] = out["GEX_vol_weighted"] / 1e9

    # ── GEX: Net OI signed ────────────────────────────────────
    # Average gamma × net OI imbalance per strike.
    # Averaging call and put gamma removes BSM noise.
    # Signal comes purely from OI skew at each strike.
    gamma_avg = (cg + pg) / 2
    out["GEX_net_oi"]   = gamma_avg * (coi - poi) * M * S2
    out["GEX_net_oi_$"] = out["GEX_net_oi"] / 1e9

    # ═════════════════════════════════════════════════════════
    # DEALER GAMMA FRAMEWORK
    # Convention: dealers are assumed to be short options
    # (customers buy, dealers sell). Dealer hedging direction
    # at each strike is therefore deterministic from gamma sign.
    # ═════════════════════════════════════════════════════════

    # ── Dealer GEX Long (stabilising pressure per strike) ────
    #
    # Measures how much gamma dealers are LONG at each strike,
    # i.e. situations where their hedging OPPOSES price movement.
    #
    # Mechanics:
    #   Dealer short call → long delta → sells into rallies → stabilising
    #   Dealer short put  → short delta → buys into dips   → stabilising
    #
    # Both call and put legs are additive because both represent
    # the dealer having SOLD options to customers and therefore
    # needing to hedge AGAINST the move at that strike.
    #
    # How to read the chart:
    #   Largest positive bar = strongest pin / wall / magnet level.
    #   Price is attracted toward high Dealer_GEX_Long strikes
    #   because dealer hedging flows reinforce mean-reversion there.
    #   Use these strikes as intraday support/resistance targets.
    #
    # Formula:
    #   (call_gamma × call_OI + put_gamma × put_OI) × M × S²
    #   — identical to GEX_unsigned but conceptually framed as
    #     the stabilising component of dealer activity.
    out["Dealer_GEX_Long_calls"] = cg * coi * M * S2
    out["Dealer_GEX_Long_puts"]  = pg * poi * M * S2
    out["Dealer_GEX_Long"]       = (out["Dealer_GEX_Long_calls"]
                                    + out["Dealer_GEX_Long_puts"])
    out["Dealer_GEX_Long_$"]     = out["Dealer_GEX_Long"] / 1e9

    # ── Dealer GEX Short (destabilising pressure per strike) ──
    #
    # Measures how much gamma dealers are SHORT at each strike,
    # i.e. situations where their hedging FOLLOWS price movement.
    # This happens when customers are NET SELLERS of options —
    # dealers absorb the other side and are long gamma, so they
    # hedge WITH the move, amplifying momentum.
    #
    # We cannot observe customer direction directly from OI, so
    # we estimate it via the OI skew between calls and puts:
    #
    #   OI_skew = (call_OI - put_OI) / (call_OI + put_OI + ε)
    #
    #   Range: -1 to +1
    #   +1 = pure call OI at this strike
    #   -1 = pure put OI at this strike
    #    0 = perfectly balanced
    #
    # Then:
    #   Dealer_GEX_Short = gamma_avg × total_OI × OI_skew × M × S²
    #
    # How to read the chart:
    #   Positive bar = call OI dominates = dealers lean short calls
    #     = they will SELL INTO RALLIES toward this strike
    #     = RESISTANCE / CEILING — price is repelled upward.
    #   Negative bar = put OI dominates = dealers lean short puts
    #     = they will BUY INTO DIPS toward this strike
    #     = SUPPORT / FLOOR — price is repelled downward.
    #   Near zero = balanced positioning = neutral strike.
    #   Large absolute value = strong trend amplification zone.
    eps      = 1e-9
    oi_skew  = (coi - poi) / (coi + poi + eps)
    total_oi = coi + poi
    out["Dealer_GEX_Short"]   = gamma_avg * total_oi * oi_skew * M * S2
    out["Dealer_GEX_Short_$"] = out["Dealer_GEX_Short"] / 1e9

    # ── Net Dealer GEX per strike (primary directional map) ───
    #
    # The single most important formula. Subtracts the put-side
    # dealer long gamma from the call-side dealer long gamma to
    # produce a per-strike signed directional reading.
    #
    # Formula:
    #   Net_Dealer_GEX = Dealer_GEX_Long_calls - Dealer_GEX_Long_puts
    #                  = (call_gamma × call_OI - put_gamma × put_OI) × M × S²
    #
    # How to read the chart:
    #   POSITIVE bar at a strike:
    #     Call gamma dominates → dealers sold more calls here
    #     → dealers sell as spot rallies toward this strike
    #     → acts as a CEILING / RESISTANCE
    #     → price approaching from below will face selling pressure
    #
    #   NEGATIVE bar at a strike:
    #     Put gamma dominates → dealers sold more puts here
    #     → dealers buy as spot falls toward this strike
    #     → acts as a FLOOR / SUPPORT
    #     → price approaching from above will face buying pressure
    #
    #   GAMMA FLIP LEVEL (most important single number):
    #     The strike where Net_Dealer_GEX crosses zero is the
    #     gamma flip level. It divides the chart into two regimes:
    #
    #     Spot ABOVE gamma flip:
    #       Nearby strikes have positive Net_Dealer_GEX above
    #       and negative below → dealers stabilise both ways
    #       → PINNING REGIME → expect mean-reversion, tight range
    #
    #     Spot BELOW gamma flip:
    #       Nearby strikes have negative Net_Dealer_GEX above
    #       and positive below → dealers amplify both ways
    #       → TRENDING REGIME → expect momentum, wider range
    #
    # Practical use:
    #   1. Find the zero-cross nearest to spot on the chart.
    #   2. If spot is above it → fade moves, sell breakouts.
    #   3. If spot is below it → follow moves, buy breakouts.
    #   4. The zero-cross strike IS the gamma flip level.
    out["Net_Dealer_GEX"]   = (out["Dealer_GEX_Long_calls"]
                               - out["Dealer_GEX_Long_puts"])
    out["Net_Dealer_GEX_$"] = out["Net_Dealer_GEX"] / 1e9

    # ── Upper / Lower pressure (regime indicator) ─────────────
    #
    # Sums Net_Dealer_GEX for the 5 strikes immediately above
    # and below spot separately to give a scalar regime signal
    # at each minute bar without collapsing the strike dimension.
    #
    # Stored as per-row columns so the time-series chart can
    # show regime evolution across the session.
    #
    # How to read:
    #   Upper_pressure NEGATIVE + Lower_pressure POSITIVE:
    #     Dealers sell above spot AND buy below spot
    #     → PINNING / MEAN-REVERTING regime
    #     → fade range extremes
    #
    #   Upper_pressure POSITIVE + Lower_pressure NEGATIVE:
    #     Dealers buy above spot AND sell below spot
    #     → TRENDING / DIRECTIONAL regime
    #     → follow breakouts
    #
    #   The SIGN COMBINATION is the regime signal, not magnitude.
    if "strike" in out.columns and "spot_used" in out.columns:
        def compute_pressure(group):
            spot_val    = group["spot_used"].iloc[0]
            net_gex     = group["Net_Dealer_GEX"]
            strikes_col = group["strike"]

            above_mask  = (strikes_col >  spot_val) & \
                          (strikes_col <= spot_val + 5 * 5)
            below_mask  = (strikes_col <  spot_val) & \
                          (strikes_col >= spot_val - 5 * 5)

            upper = net_gex[above_mask].sum()
            lower = net_gex[below_mask].sum()
            result = group.copy()
            result["Upper_pressure"] = upper
            result["Lower_pressure"] = lower
            return result

        if "timestamp" in out.columns:
            out = (out.groupby("timestamp", group_keys=False)
                      .apply(compute_pressure))
        else:
            spot_val    = out["spot_used"].iloc[0]
            net_gex     = out["Net_Dealer_GEX"]
            strikes_col = out["strike"]
            above_mask  = (strikes_col >  spot_val) & \
                          (strikes_col <= spot_val + 5 * 5)
            below_mask  = (strikes_col <  spot_val) & \
                          (strikes_col >= spot_val - 5 * 5)
            out["Upper_pressure"] = net_gex[above_mask].sum()
            out["Lower_pressure"] = net_gex[below_mask].sum()
    else:
        out["Upper_pressure"] = 0.0
        out["Lower_pressure"] = 0.0

    out["Upper_pressure_$"] = out["Upper_pressure"] / 1e9
    out["Lower_pressure_$"] = out["Lower_pressure"] / 1e9

    # ── DEX: Delta Exposure ───────────────────────────────────
    # Total directional delta dealers must hedge per strike.
    # Positive = calls dominate = dealers net short delta
    #   = they buy as price approaches = bullish attractor.
    # Negative = puts dominate = dealers net long delta
    #   = bearish attractor / put-supported floor.
    out["DEX"]   = (cd * coi + pd_ * poi) * M * S
    out["DEX_$"] = out["DEX"] / 1e9

    # ─────────────────────────────────────────────────────────
    # ADD NEW FORMULAS BELOW THIS LINE
    # Template:
    #   out["MY_FORMULA"]   = <expression>
    #   out["MY_FORMULA_$"] = out["MY_FORMULA"] / 1e9
    # Then add "MY_FORMULA_$" to FORMULA_COLS at the top.
    # ─────────────────────────────────────────────────────────

    return out

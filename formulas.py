# ─────────────────────────────────────────────────────────────
# formulas.py — GEX / DEX formula engine
# ─────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
from config import SPX_MULT

FORMULA_COLS = [
    "Dealer_GEX_Long_$",
    "Dealer_GEX_Short_$",
    "Net_Dealer_GEX_$",
    "Upper_pressure_$",
    "Lower_pressure_$",
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

    gamma_avg = (cg + pg) / 2

    # ── Dealer GEX Long (stabilising pressure per strike) ────
    #
    # Dealers assumed short options (customers buy, dealers sell).
    # Measures gamma dealers are LONG — hedging OPPOSES price move.
    #   Dealer short call → sells into rallies  → stabilising
    #   Dealer short put  → buys into dips      → stabilising
    # Both legs additive: both represent dealer sold to customer.
    #
    # How to read:
    #   Largest bar = strongest pin / wall / magnet level.
    #   Price is attracted to high Dealer_GEX_Long strikes.
    #   Use as intraday support/resistance targets.
    out["Dealer_GEX_Long_calls"] = cg * coi * M * S2
    out["Dealer_GEX_Long_puts"]  = pg * poi * M * S2
    out["Dealer_GEX_Long"]       = (out["Dealer_GEX_Long_calls"]
                                    + out["Dealer_GEX_Long_puts"])
    out["Dealer_GEX_Long_$"]     = out["Dealer_GEX_Long"] / 1e9

    # ── Dealer GEX Short (destabilising pressure per strike) ──
    #
    # When customers are NET SELLERS dealers are long gamma and
    # hedge WITH the move, amplifying momentum.
    # OI skew estimates which side customers are net-selling:
    #
    #   OI_skew = (call_OI - put_OI) / (call_OI + put_OI + ε)
    #   Range: -1 to +1
    #
    #   Dealer_GEX_Short = gamma_avg × total_OI × OI_skew × M × S²
    #
    # How to read:
    #   Positive = call OI dominates → dealers short calls
    #     → sell into rallies → RESISTANCE / CEILING
    #   Negative = put OI dominates → dealers short puts
    #     → buy into dips → SUPPORT / FLOOR
    #   Large absolute value = strong trend amplification zone.
    eps      = 1e-9
    oi_skew  = (coi - poi) / (coi + poi + eps)
    total_oi = coi + poi
    out["Dealer_GEX_Short"]   = gamma_avg * total_oi * oi_skew * M * S2
    out["Dealer_GEX_Short_$"] = out["Dealer_GEX_Short"] / 1e9

    # ── Net Dealer GEX per strike (primary directional map) ───
    #
    #   Net_Dealer_GEX = Dealer_GEX_Long_calls - Dealer_GEX_Long_puts
    #                  = (call_gamma × call_OI - put_gamma × put_OI) × M × S²
    #
    # How to read:
    #   POSITIVE bar → call gamma dominates → dealers sold more calls
    #     → dealers SELL as spot rallies toward this strike
    #     → CEILING / RESISTANCE
    #
    #   NEGATIVE bar → put gamma dominates → dealers sold more puts
    #     → dealers BUY as spot falls toward this strike
    #     → FLOOR / SUPPORT
    #
    #   GAMMA FLIP LEVEL = strike where Net_Dealer_GEX crosses zero
    #
    #   Spot ABOVE gamma flip:
    #     Dealers stabilise both up and down → PINNING REGIME
    #     → fade moves, sell breakouts, expect tight range
    #
    #   Spot BELOW gamma flip:
    #     Dealers amplify both up and down → TRENDING REGIME
    #     → follow moves, buy breakouts, expect wider range
    out["Net_Dealer_GEX"]   = (out["Dealer_GEX_Long_calls"]
                               - out["Dealer_GEX_Long_puts"])
    out["Net_Dealer_GEX_$"] = out["Net_Dealer_GEX"] / 1e9

    # ── Upper / Lower pressure (regime indicator) ─────────────
    #
    # Sums Net_Dealer_GEX for the 5 strikes immediately above
    # and below spot at each timestamp. Scalar regime signal
    # without collapsing the strike dimension.
    #
    # How to read (sign combination is the signal, not size):
    #   Upper NEGATIVE + Lower POSITIVE:
    #     Dealers sell above AND buy below spot
    #     → PINNING / MEAN-REVERTING → fade range extremes
    #
    #   Upper POSITIVE + Lower NEGATIVE:
    #     Dealers buy above AND sell below spot
    #     → TRENDING / DIRECTIONAL → follow breakouts
    #
    # Computed without groupby().apply() to preserve index integrity.
    NEARBY = 5 * 5   # 5 strikes × $5 step = $25 band each side

    out["Upper_pressure"] = 0.0
    out["Lower_pressure"] = 0.0

    if "strike" in out.columns and "timestamp" in out.columns:
        # Vectorised: build masks per row using aligned series
        strikes_s   = out["strike"].values
        spot_s      = out["spot_used"].values
        net_gex_s   = out["Net_Dealer_GEX"].values
        timestamps_s= out["timestamp"].values

        # Build per-timestamp lookup: ts → {strike → net_gex}
        ts_strike_gex = {}
        for ts, sk, ng in zip(timestamps_s, strikes_s, net_gex_s):
            ts_strike_gex.setdefault(ts, {})[sk] = ng

        upper_vals = np.zeros(len(out))
        lower_vals = np.zeros(len(out))

        for i, (ts, spot_val) in enumerate(zip(timestamps_s, spot_s)):
            sg = ts_strike_gex.get(ts, {})
            upper = sum(v for k, v in sg.items()
                        if spot_val < k <= spot_val + NEARBY)
            lower = sum(v for k, v in sg.items()
                        if spot_val - NEARBY <= k < spot_val)
            upper_vals[i] = upper
            lower_vals[i] = lower

        out["Upper_pressure"] = upper_vals
        out["Lower_pressure"] = lower_vals

    out["Upper_pressure_$"] = out["Upper_pressure"] / 1e9
    out["Lower_pressure_$"] = out["Lower_pressure"] / 1e9

    # ── DEX: Delta Exposure ───────────────────────────────────
    #
    # Total directional delta dealers must hedge per strike.
    #
    # How to read:
    #   Positive = calls dominate → dealers net short delta
    #     → they BUY as price approaches → bullish attractor
    #   Negative = puts dominate → dealers net long delta
    #     → bearish attractor / put-supported floor
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

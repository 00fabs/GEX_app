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

    out = df.copy().reset_index(drop=True)

    # ── Spot price ────────────────────────────────────────────
    spot_col = (out["spot"].fillna(spot_override)
                if "spot" in out.columns
                else pd.Series([spot_override] * len(out), index=out.index))
    out["spot_used"] = spot_col.values   # .values strips index alignment issues

    # ── Helper: safely get column as numpy array ─────────────
    def col(name):
        if name in out.columns:
            return pd.to_numeric(out[name], errors="coerce").fillna(0).values
        return np.zeros(len(out))

    cg   = col("call_gamma")
    pg   = col("put_gamma")
    coi  = col("call_oi")
    poi  = col("put_oi")
    cvol = col("call_volume")
    pvol = col("put_volume")
    cd   = col("call_delta")
    pd_  = col("put_delta")
    S    = spot_col.values.astype(float)
    S2   = S ** 2
    M    = SPX_MULT

    gamma_avg = (cg + pg) / 2.0

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
    #   Price attracted to high Dealer_GEX_Long strikes.
    #   Use as intraday support/resistance targets.
    dgl_calls = cg * coi * M * S2
    dgl_puts  = pg * poi * M * S2
    dgl       = dgl_calls + dgl_puts

    out["Dealer_GEX_Long_calls"] = dgl_calls
    out["Dealer_GEX_Long_puts"]  = dgl_puts
    out["Dealer_GEX_Long"]       = dgl
    out["Dealer_GEX_Long_$"]     = dgl / 1e9

    # ── Dealer GEX Short (destabilising pressure per strike) ──
    #
    # When customers NET SELL options, dealers are long gamma
    # and hedge WITH the move, amplifying momentum.
    # OI skew estimates which side customers are net-selling:
    #
    #   OI_skew = (call_OI - put_OI) / (call_OI + put_OI + ε)
    #   Range -1 to +1
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
    dgs      = gamma_avg * total_oi * oi_skew * M * S2

    out["Dealer_GEX_Short"]   = dgs
    out["Dealer_GEX_Short_$"] = dgs / 1e9

    # ── Net Dealer GEX per strike (primary directional map) ───
    #
    #   Net_Dealer_GEX = call_gamma×call_OI - put_gamma×put_OI) × M × S²
    #
    # How to read:
    #   POSITIVE bar → call gamma dominates → dealers sold more calls
    #     → SELL as spot rallies toward this strike → CEILING/RESISTANCE
    #
    #   NEGATIVE bar → put gamma dominates → dealers sold more puts
    #     → BUY as spot falls toward this strike → FLOOR/SUPPORT
    #
    #   GAMMA FLIP LEVEL = strike where Net_Dealer_GEX crosses zero
    #
    #   Spot ABOVE flip → dealers stabilise both ways → PINNING REGIME
    #     → fade moves, sell breakouts
    #   Spot BELOW flip → dealers amplify both ways  → TRENDING REGIME
    #     → follow moves, buy breakouts
    ndg = dgl_calls - dgl_puts

    out["Net_Dealer_GEX"]   = ndg
    out["Net_Dealer_GEX_$"] = ndg / 1e9

    # ── Upper / Lower pressure (regime indicator) ─────────────
    #
    # Sums Net_Dealer_GEX for the 5 strikes immediately above
    # and below spot at each timestamp bar.
    #
    # How to read (sign combination is the signal, not size):
    #   Upper NEGATIVE + Lower POSITIVE:
    #     Dealers sell above AND buy below → PINNING → fade extremes
    #
    #   Upper POSITIVE + Lower NEGATIVE:
    #     Dealers buy above AND sell below → TRENDING → follow breakouts
    NEARBY = 5 * 5   # 5 strikes × $5 step = $25 band each side

    upper_vals = np.zeros(len(out))
    lower_vals = np.zeros(len(out))

    if "strike" in out.columns and "timestamp" in out.columns:
        strikes_arr    = out["strike"].values
        spot_arr       = out["spot_used"].values
        ndg_arr        = ndg
        timestamps_arr = out["timestamp"].values

        # Build per-timestamp dict: ts → array of (strike, net_gex)
        from collections import defaultdict
        ts_data = defaultdict(list)
        for i in range(len(out)):
            ts_data[timestamps_arr[i]].append((strikes_arr[i], ndg_arr[i]))

        for i in range(len(out)):
            ts   = timestamps_arr[i]
            sv   = float(spot_arr[i])
            pairs = ts_data[ts]
            upper_vals[i] = sum(ng for sk, ng in pairs
                                if sv < sk <= sv + NEARBY)
            lower_vals[i] = sum(ng for sk, ng in pairs
                                if sv - NEARBY <= sk < sv)

    out["Upper_pressure"]   = upper_vals
    out["Lower_pressure"]   = lower_vals
    out["Upper_pressure_$"] = upper_vals / 1e9
    out["Lower_pressure_$"] = lower_vals / 1e9

    # ── DEX: Delta Exposure ───────────────────────────────────
    #
    # Total directional delta dealers must hedge per strike.
    #
    # How to read:
    #   Positive = calls dominate → dealers net short delta
    #     → BUY as price approaches → bullish attractor
    #   Negative = puts dominate → dealers net long delta
    #     → bearish attractor / put-supported floor
    dex = (cd * coi + pd_ * poi) * M * S

    out["DEX"]   = dex
    out["DEX_$"] = dex / 1e9

    # ─────────────────────────────────────────────────────────
    # ADD NEW FORMULAS BELOW THIS LINE
    # Template:
    #   arr = <numpy expression using cg, pg, coi, poi, cd, pd_, S, M>
    #   out["MY_FORMULA"]   = arr
    #   out["MY_FORMULA_$"] = arr / 1e9
    # Then add "MY_FORMULA_$" to FORMULA_COLS at the top.
    # ─────────────────────────────────────────────────────────

    return out

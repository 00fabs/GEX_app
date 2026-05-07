# ─────────────────────────────────────────────────────────────
# formulas.py — GEX / DEX formula engine
# All dynamic — no static OI used
# All formula column names start with GEX_ or DEX_
# ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from collections import defaultdict
from config import SPX_MULT

FORMULA_COLS = [
    "GEX_Gamma_IV_$",
    "GEX_Vanna_IV_$",
    "GEX_Charm_Flow_$",
    "GEX_Vol_Skew_$",
    "DEX_Dynamic_$",
    "DEX_Vanna_Flow_$",
]


def apply_formulas(df: pd.DataFrame, spot_override: float,
                   intra_date) -> pd.DataFrame:
    """
    Fully dynamic intraday GEX / DEX formulas.
    No static OI used anywhere — all signals derived from
    live Greeks, IV, and intraday volume updated every bar.

    Parameters
    ----------
    df            : wide DataFrame from pipeline.pivot_wide()
    spot_override : fallback spot price
    intra_date    : session date (available for future use)

    Returns
    -------
    df with all formula columns appended.
    """
    out = df.copy().reset_index(drop=True)

    # ── Spot price ────────────────────────────────────────────
    spot_col = (out["spot"].fillna(spot_override)
                if "spot" in out.columns
                else pd.Series([spot_override] * len(out),
                               index=out.index))
    out["spot_used"] = spot_col.values

    # ── Helper: column → clean numpy array ───────────────────
    def col(name):
        if name in out.columns:
            return pd.to_numeric(out[name],
                                 errors="coerce").fillna(0).values
        return np.zeros(len(out))

    cg    = col("call_gamma")
    pg    = col("put_gamma")
    cv    = col("call_vanna")
    pv    = col("put_vanna")
    cc    = col("call_charm")
    pc    = col("put_charm")
    cd    = col("call_delta")
    pd_   = col("put_delta")
    civ   = col("call_iv")
    piv   = col("put_iv")
    cvol  = col("call_volume")
    pvol  = col("put_volume")

    S     = out["spot_used"].values.astype(float)
    S2    = S ** 2
    M     = SPX_MULT
    eps   = 1e-9

    gamma_avg  = (cg + pg) / 2.0
    total_vol  = cvol + pvol

    # ═════════════════════════════════════════════════════════
    # GEX_Gamma_IV  —  Market-priced curvature pressure
    # ═════════════════════════════════════════════════════════
    #
    # Replaces static OI with IV as the position weight.
    # Logic: when call_gamma × call_iv is large at a strike the
    # market is simultaneously pricing high curvature AND high
    # uncertainty on the call side. This means the market expects
    # significant hedging activity there regardless of OI count.
    # IV is re-priced every minute so this is fully dynamic.
    #
    # Formula:
    #   GEX_Gamma_IV = (call_gamma × call_iv
    #                −  put_gamma  × put_iv) × M × S²
    #
    # How to read:
    #   POSITIVE bar → call gamma/IV dominates → ceiling forming
    #     → dealers sell as spot rallies toward this strike
    #     → RESISTANCE
    #   NEGATIVE bar → put gamma/IV dominates → floor forming
    #     → dealers buy as spot falls toward this strike
    #     → SUPPORT
    #   ZERO CROSS nearest spot → gamma flip level
    #     Above flip → pinning regime → fade moves
    #     Below flip → trending regime → follow breakouts
    #   Most reliable of all formulas because IV reflects live
    #   market maker inventory adjustments every minute.
    gex_gamma_iv = (cg * civ - pg * piv) * M * S2
    out["GEX_Gamma_IV"]   = gex_gamma_iv
    out["GEX_Gamma_IV_$"] = gex_gamma_iv / 1e9

    # ═════════════════════════════════════════════════════════
    # GEX_Vanna_IV  —  IV-move driven rehedging pressure
    # ═════════════════════════════════════════════════════════
    #
    # Vanna measures how delta changes when IV changes.
    # When IV moves (which it does constantly intraday) dealers
    # must rehedge delta proportional to their vanna exposure.
    # Multiplying by IV amplifies strikes where the market is
    # pricing both high vanna sensitivity AND high uncertainty —
    # those are the strikes where IV-driven rehedging will be
    # largest. Static OI completely misses this channel.
    #
    # Formula:
    #   GEX_Vanna_IV = (call_vanna × call_iv
    #                −  put_vanna  × put_iv) × M × S²
    #
    # How to read:
    #   Most powerful around events that move IV intraday:
    #   economic data, Fed speakers, unexpected news.
    #   POSITIVE → rising IV forces dealer buying → bullish support
    #   NEGATIVE → rising IV forces dealer selling → bearish resistance
    #   Use alongside GEX_Gamma_IV:
    #     Both positive at same strike → strong confirmed ceiling
    #     Both negative at same strike → strong confirmed floor
    #     Divergence → mixed signal, reduce confidence
    gex_vanna_iv = (cv * civ - pv * piv) * M * S2
    out["GEX_Vanna_IV"]   = gex_vanna_iv
    out["GEX_Vanna_IV_$"] = gex_vanna_iv / 1e9

    # ═════════════════════════════════════════════════════════
    # GEX_Charm_Flow  —  Time-decay rehedging (0DTE specific)
    # ═════════════════════════════════════════════════════════
    #
    # THE most 0DTE-specific formula in the set.
    # Charm = rate of delta decay with time (dDelta/dTime).
    # For 0DTE charm is enormous near ATM and accelerates
    # exponentially as expiration approaches.
    # Every minute that passes dealers MUST rehedge delta
    # proportional to charm regardless of spot movement.
    # This creates predictable mechanical buying/selling flows
    # purely from time passage — not directional speculation.
    # Multiplying by volume confirms which strikes are seeing
    # actual flow vs theoretical exposure.
    #
    # Formula:
    #   GEX_Charm_Flow = (call_charm × call_volume
    #                  −  put_charm  × put_volume) × M × S²
    #
    # How to read:
    #   Grows in importance as session progresses.
    #   Largest bars after 2pm are the strongest expiration magnets.
    #   POSITIVE → call charm flow dominates → time-decay buying
    #     → bullish drift toward this strike into close
    #   NEGATIVE → put charm flow dominates → time-decay selling
    #     → bearish drift toward this strike into close
    #   Near zero early session = few contracts traded yet.
    #   Diverges from GEX_Gamma_IV as day progresses = watch
    #   the divergence strikes carefully — those are where the
    #   afternoon pin battle will be fought.
    gex_charm_flow = (cc * cvol - pc * pvol) * M * S2
    out["GEX_Charm_Flow"]   = gex_charm_flow
    out["GEX_Charm_Flow_$"] = gex_charm_flow / 1e9

    # ═════════════════════════════════════════════════════════
    # GEX_Vol_Skew  —  Volume + IV skew combined pressure
    # ═════════════════════════════════════════════════════════
    #
    # Combines three live signals into one:
    #   1. gamma_avg    — curvature at this strike
    #   2. IV_skew      — which side market pays more for
    #   3. total_volume — where actual trading is happening
    #
    # IV skew per strike:
    #   IV_skew = (put_iv − call_iv) / (put_iv + call_iv + ε)
    #   +1 = pure put IV premium (fear of downside)
    #   -1 = pure call IV premium (fear of upside)
    #    0 = balanced
    #
    # Formula:
    #   GEX_Vol_Skew = gamma_avg × IV_skew × total_volume × M × S²
    #
    # How to read:
    #   POSITIVE → put IV premium + volume at this strike
    #     → market paying for downside protection here
    #     → dealers short puts → FLOOR / SUPPORT
    #   NEGATIVE → call IV premium + volume at this strike
    #     → market paying for upside protection here
    #     → dealers short calls → CEILING / RESISTANCE
    #   Zero volume early session → bars near flat, grows through day
    #   Best used from 11am onward when volume has accumulated.
    #   Compare with GEX_Charm_Flow: if both point to same strike
    #   that is your highest-confidence level of the session.
    iv_skew      = (piv - civ) / (piv + civ + eps)
    gex_vol_skew = gamma_avg * iv_skew * total_vol * M * S2
    out["GEX_Vol_Skew"]   = gex_vol_skew
    out["GEX_Vol_Skew_$"] = gex_vol_skew / 1e9

    # ═════════════════════════════════════════════════════════
    # DEX_Dynamic  —  Live delta exposure from actual flow
    # ═════════════════════════════════════════════════════════
    #
    # Standard DEX uses static OI which is stale for 0DTE.
    # DEX_Dynamic replaces OI with intraday volume so it
    # reflects today's actual directional hedging requirement
    # from contracts that have actually traded this session.
    # Put delta is already negative from BSM so the formula
    # naturally nets directional exposure.
    #
    # Formula:
    #   DEX_Dynamic = (call_delta × call_volume
    #               +  put_delta  × put_volume) × M × S
    #
    # How to read:
    #   POSITIVE → call delta flow dominates → dealers net short
    #     delta from today's trades → they BUY as price rises
    #     → bullish attractor / dynamic support
    #   NEGATIVE → put delta flow dominates → dealers net long
    #     delta → they SELL as price rises
    #     → bearish attractor / dynamic resistance
    #   The zero cross of DEX_Dynamic is a secondary flip level
    #   for delta hedging — less powerful than GEX_Gamma_IV
    #   flip but confirms the directional bias when aligned.
    dex_dynamic = (cd * cvol + pd_ * pvol) * M * S
    out["DEX_Dynamic"]   = dex_dynamic
    out["DEX_Dynamic_$"] = dex_dynamic / 1e9

    # ═════════════════════════════════════════════════════════
    # DEX_Vanna_Flow  —  IV-sensitivity weighted delta flow
    # ═════════════════════════════════════════════════════════
    #
    # Measures the delta hedging that will be TRIGGERED BY IV
    # moves rather than spot moves. When IV rises or falls,
    # dealers must rehedge delta proportional to their vanna.
    # Multiplying by volume confirms where this is actually
    # happening vs theoretical.
    #
    # Formula:
    #   DEX_Vanna_Flow = (call_vanna × call_volume
    #                  −  put_vanna  × put_volume) × M × S
    #
    # How to read:
    #   POSITIVE → call vanna flow → rising IV forces dealer
    #     delta buying at this strike → bullish IV-driven support
    #   NEGATIVE → put vanna flow → rising IV forces dealer
    #     delta selling → bearish IV-driven resistance
    #   Most useful when IV is moving intraday (check GEX_Vanna_IV
    #   alongside this). If IV is flat DEX_Vanna_Flow is secondary.
    #   When IV is spiking this becomes the primary signal because
    #   it captures the exact strikes where forced rehedging occurs.
    dex_vanna_flow = (cv * cvol - pv * pvol) * M * S
    out["DEX_Vanna_Flow"]   = dex_vanna_flow
    out["DEX_Vanna_Flow_$"] = dex_vanna_flow / 1e9

    # ─────────────────────────────────────────────────────────
    # ADD NEW FORMULAS BELOW THIS LINE
    # All inputs available as numpy arrays:
    #   cg, pg       — call/put gamma
    #   cv, pv       — call/put vanna
    #   cc, pc       — call/put charm
    #   cd, pd_      — call/put delta
    #   civ, piv     — call/put IV
    #   cvol, pvol   — call/put intraday volume
    #   S, S2        — spot, spot squared
    #   M            — multiplier (100)
    #   gamma_avg    — (cg+pg)/2
    #   total_vol    — cvol+pvol
    #   iv_skew      — (piv-civ)/(piv+civ+eps)
    #
    # Template:
    #   arr = <numpy expression>
    #   out["GEX_MY_FORMULA"]   = arr
    #   out["GEX_MY_FORMULA_$"] = arr / 1e9
    # Then add "GEX_MY_FORMULA_$" to FORMULA_COLS at the top.
    # ─────────────────────────────────────────────────────────

    return out

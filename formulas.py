import numpy as np
import pandas as pd
from config import SPX_MULT

FORMULA_COLS = [
    "GEX_Call_Wall_$",
    "GEX_Put_Wall_$",
    "GEX_Call_Demand_$",
    "GEX_Put_Demand_$",
    "GEX_Reversal_$",
    "GEX_Breakout_$",
    "GEX_Reversal_Near_$",
    "GEX_Breakout_Near_$",
    "GEX_Charm_Pin_$",
    "DEX_Call_OI_$",
    "DEX_Put_OI_$",
]


def apply_formulas(df: pd.DataFrame, spot_override: float,
                   intra_date) -> pd.DataFrame:

    out = df.copy().reset_index(drop=True)
    eps = 1e-9

    spot_col = (out["spot"].fillna(spot_override)
                if "spot" in out.columns
                else pd.Series([spot_override] * len(out),
                               index=out.index))
    out["spot_used"] = spot_col.values

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
    cbid  = col("call_bid")
    cask  = col("call_ask")
    pbid  = col("put_bid")
    pask  = col("put_ask")
    civ   = col("call_iv")
    piv   = col("put_iv")

    open_coi = col("session_open_call_oi")
    open_poi = col("session_open_put_oi")

    civ_mean = col("call_iv_mean20")
    civ_std  = col("call_iv_std20")
    piv_mean = col("put_iv_mean20")
    piv_std  = col("put_iv_std20")

    S           = out["spot_used"].values.astype(float)
    S2          = S ** 2
    M           = SPX_MULT
    strikes_arr = col("strike") if "strike" in out.columns \
                  else np.zeros(len(out))

    # ── IV Z-score and relative edge ─────────────────────────
    call_iv_z    = (civ - civ_mean) / (civ_std + eps)
    put_iv_z     = (piv - piv_mean) / (piv_std  + eps)
    EDGE_THRESH  = 0.5
    iv_call_edge = np.clip(call_iv_z - put_iv_z - EDGE_THRESH, 0, None)
    iv_put_edge  = np.clip(put_iv_z  - call_iv_z - EDGE_THRESH, 0, None)

    # ── Bid pressure ─────────────────────────────────────────
    call_mid       = (cbid + cask) / 2.0
    put_mid        = (pbid + pask) / 2.0
    call_spread    = np.abs(cask - cbid) + eps
    put_spread     = np.abs(pask - pbid) + eps
    call_bid_press = np.clip((call_mid - cbid) / call_spread, 0, 1)
    put_bid_press  = np.clip((put_mid  - pbid) / put_spread,  0, 1)

    # ── Proximity ─────────────────────────────────────────────
    atm_band_wide  = S * 0.005 + eps
    atm_band_tight = S * 0.001 + eps
    proximity_wide  = np.exp(-np.abs(strikes_arr - S) / atm_band_wide)
    proximity_tight = np.exp(-np.abs(strikes_arr - S) / atm_band_tight)

    # ── Wall magnitude — strictly OI based ───────────────────
    total_open_oi = open_coi + open_poi + eps
    max_open_oi   = total_open_oi.max() + eps
    wall_mag_norm = total_open_oi / max_open_oi

    # ── Hard wall classification ──────────────────────────────
    call_wall_flag = np.where(open_coi > open_poi * 1.5, 1.0,
                     np.where(open_poi > open_coi * 1.5, 0.0, 0.5))
    put_wall_flag  = 1.0 - call_wall_flag

    # ── GEX_Call_Wall ─────────────────────────────────────────
    gex_call_wall          = cg * open_coi * M * S2
    out["GEX_Call_Wall"]   = gex_call_wall
    out["GEX_Call_Wall_$"] = gex_call_wall / 1e9

    # ── GEX_Put_Wall ──────────────────────────────────────────
    gex_put_wall          = -(pg * open_poi * M * S2)
    out["GEX_Put_Wall"]   = gex_put_wall
    out["GEX_Put_Wall_$"] = gex_put_wall / 1e9

    # ── GEX_Call_Demand ───────────────────────────────────────
    # wall_mag_norm anchors to OI-significant strikes only
    # same formula as the last working version
    gex_call_demand          = np.abs(iv_call_edge * cv
                                      * call_bid_press
                                      * proximity_wide
                                      * wall_mag_norm
                                      * M * S2)
    out["GEX_Call_Demand"]   = gex_call_demand
    out["GEX_Call_Demand_$"] = gex_call_demand / 1e9

    # ── GEX_Put_Demand ────────────────────────────────────────
    gex_put_demand          = np.abs(iv_put_edge * pv
                                     * put_bid_press
                                     * proximity_wide
                                     * wall_mag_norm
                                     * M * S2)
    out["GEX_Put_Demand"]   = gex_put_demand
    out["GEX_Put_Demand_$"] = gex_put_demand / 1e9

    # ── GEX_Reversal (wide — structural) ─────────────────────
    wall_mag_safe = wall_mag_norm + eps
    gex_reversal  = np.abs(call_wall_flag * gex_put_demand
                           + put_wall_flag * gex_call_demand) \
                    * proximity_wide * wall_mag_safe
    out["GEX_Reversal"]   = gex_reversal
    out["GEX_Reversal_$"] = gex_reversal / 1e9

    # ── GEX_Breakout (wide — structural) ─────────────────────
    gex_breakout  = np.abs(call_wall_flag * gex_call_demand
                           + put_wall_flag * gex_put_demand) \
                    * proximity_wide * wall_mag_safe
    out["GEX_Breakout"]   = gex_breakout
    out["GEX_Breakout_$"] = gex_breakout / 1e9

    # ── GEX_Reversal_Near and GEX_Breakout_Near ───────────────
    # Computed here using raw (pre-EWMA) demand values.
    # pipeline._apply_ewma() will smooth these after the fact
    # using the same EWMA_COLS list — add them there too.
    # Using tight proximity so only fires when spot is within
    # ~7 points of the strike.
    gex_reversal_near = np.abs(call_wall_flag * gex_put_demand
                               + put_wall_flag * gex_call_demand) \
                        * proximity_tight * wall_mag_safe
    out["GEX_Reversal_Near"]   = gex_reversal_near
    out["GEX_Reversal_Near_$"] = gex_reversal_near / 1e9

    gex_breakout_near = np.abs(call_wall_flag * gex_call_demand
                               + put_wall_flag * gex_put_demand) \
                        * proximity_tight * wall_mag_safe
    out["GEX_Breakout_Near"]   = gex_breakout_near
    out["GEX_Breakout_Near_$"] = gex_breakout_near / 1e9

    # ── GEX_Charm_Pin ─────────────────────────────────────────
    gex_charm_pin          = ((cc * call_bid_press * call_wall_flag
                               - pc * put_bid_press * put_wall_flag)
                              * wall_mag_norm * proximity_wide * M * S2)
    out["GEX_Charm_Pin"]   = gex_charm_pin
    out["GEX_Charm_Pin_$"] = gex_charm_pin / 1e9

    # ── DEX_Call_OI — raw call OI (positive bars) ─────────────
    dex_call_oi          = open_coi.astype(float)
    out["DEX_Call_OI"]   = dex_call_oi
    out["DEX_Call_OI_$"] = dex_call_oi / 1e6

    # ── DEX_Put_OI — raw put OI (negative bars) ───────────────
    dex_put_oi          = -open_poi.astype(float)
    out["DEX_Put_OI"]   = dex_put_oi
    out["DEX_Put_OI_$"] = dex_put_oi / 1e6

    # ─────────────────────────────────────────────────────────
    # ADD NEW FORMULAS BELOW THIS LINE
    # Available arrays: cg, pg, cv, pv, cc, pc,
    # civ, piv, open_coi, open_poi,
    # cbid, cask, pbid, pask,
    # call_iv_z, put_iv_z, iv_call_edge, iv_put_edge,
    # call_bid_press, put_bid_press,
    # proximity_wide, proximity_tight,
    # call_wall_flag, put_wall_flag,
    # wall_mag_norm, wall_mag_safe,
    # S, S2, M, eps
    #
    # Template:
    #   arr = <numpy expression>
    #   out["GEX_MY_FORMULA"]   = arr
    #   out["GEX_MY_FORMULA_$"] = arr / 1e9
    #   Add "GEX_MY_FORMULA_$" to FORMULA_COLS at top.
    # ─────────────────────────────────────────────────────────

    return out

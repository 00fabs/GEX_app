import numpy as np
import pandas as pd
from config import SPX_MULT

FORMULA_COLS = [
    "GEX_Call_Wall_$",
    "GEX_Put_Wall_$",
    "GEX_Call_Demand_$",
    "GEX_Put_Demand_$",
    "GEX_Wall_Integrity_$",
    "GEX_Reversal_$",
    "GEX_Breakout_$",
    "GEX_Charm_Pin_$",
    "DEX_Flow_$",
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
    cd    = col("call_delta")
    pd_   = col("put_delta")
    civ   = col("call_iv")
    piv   = col("put_iv")
    coi   = col("call_oi")
    poi   = col("put_oi")
    cbid  = col("call_bid")
    cask  = col("call_ask")
    pbid  = col("put_bid")
    pask  = col("put_ask")
    cvol  = col("call_volume")
    pvol  = col("put_volume")

    civ_mean = col("call_iv_mean20")
    civ_std  = col("call_iv_std20")
    piv_mean = col("put_iv_mean20")
    piv_std  = col("put_iv_std20")

    open_coi = col("session_open_call_oi")
    open_poi = col("session_open_put_oi")

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
    iv_put_edge  = np.clip(put_iv_z - call_iv_z - EDGE_THRESH, 0, None)

    # ── Bid pressure ─────────────────────────────────────────
    call_mid       = (cbid + cask) / 2.0
    put_mid        = (pbid + pask) / 2.0
    call_spread    = np.abs(cask - cbid) + eps
    put_spread     = np.abs(pask - pbid) + eps
    call_bid_press = np.clip((call_mid - cbid) / call_spread, 0, 1)
    put_bid_press  = np.clip((put_mid  - pbid) / put_spread,  0, 1)

    # ── Proximity ─────────────────────────────────────────────
    atm_band  = S * 0.005 + eps
    proximity = np.exp(-np.abs(strikes_arr - S) / atm_band)

    # ── Wall magnitude normalization ──────────────────────────
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
    gex_call_demand          = np.abs(iv_call_edge * cv
                                      * call_bid_press
                                      * proximity
                                      * wall_mag_norm
                                      * M * S2)
    out["GEX_Call_Demand"]   = gex_call_demand
    out["GEX_Call_Demand_$"] = gex_call_demand / 1e9

    # ── GEX_Put_Demand ────────────────────────────────────────
    gex_put_demand          = np.abs(iv_put_edge * pv
                                     * put_bid_press
                                     * proximity
                                     * wall_mag_norm
                                     * M * S2)
    out["GEX_Put_Demand"]   = gex_put_demand
    out["GEX_Put_Demand_$"] = gex_put_demand / 1e9

    # ── GEX_Wall_Integrity ────────────────────────────────────
    base_integrity = (call_wall_flag * np.abs(gex_call_wall)
                      + put_wall_flag * np.abs(gex_put_wall))
    total_erosion  = gex_call_demand + gex_put_demand

    base_norm    = base_integrity / (base_integrity.max() + eps)
    erosion_norm = total_erosion  / (total_erosion.max()  + eps)

    gex_wall_integrity          = base_norm - erosion_norm
    out["GEX_Wall_Integrity"]   = gex_wall_integrity
    out["GEX_Wall_Integrity_$"] = gex_wall_integrity / 1e9

    # ── GEX_Reversal ──────────────────────────────────────────
    wall_mag_safe = wall_mag_norm + eps
    gex_reversal          = np.abs(call_wall_flag * gex_put_demand
                                   + put_wall_flag * gex_call_demand) \
                            * proximity * wall_mag_safe
    out["GEX_Reversal"]   = gex_reversal
    out["GEX_Reversal_$"] = gex_reversal / 1e9

    # ── GEX_Breakout ──────────────────────────────────────────
    gex_breakout          = np.abs(call_wall_flag * gex_call_demand
                                   + put_wall_flag * gex_put_demand) \
                            * proximity * wall_mag_safe
    out["GEX_Breakout"]   = gex_breakout
    out["GEX_Breakout_$"] = gex_breakout / 1e9

    # ── GEX_Charm_Pin ─────────────────────────────────────────
    gex_charm_pin          = ((cc * call_bid_press * call_wall_flag
                               - pc * put_bid_press * put_wall_flag)
                              * wall_mag_norm * proximity * M * S2)
    out["GEX_Charm_Pin"]   = gex_charm_pin
    out["GEX_Charm_Pin_$"] = gex_charm_pin / 1e9

    # ── DEX_Flow ──────────────────────────────────────────────
    dex_flow          = ((cd * cvol + pd_ * pvol) * M * S)
    out["DEX_Flow"]   = dex_flow
    out["DEX_Flow_$"] = dex_flow / 1e9

    # ─────────────────────────────────────────────────────────
    # ADD NEW FORMULAS BELOW THIS LINE
    # Available arrays: cg, pg, cv, pv, cc, pc, cd, pd_,
    # civ, piv, coi, poi, open_coi, open_poi,
    # cbid, cask, pbid, pask, cvol, pvol,
    # call_iv_z, put_iv_z, iv_call_edge, iv_put_edge,
    # call_bid_press, put_bid_press, proximity,
    # call_wall_flag, put_wall_flag, wall_mag_norm,
    # S, S2, M, eps
    #
    # Template:
    #   arr = <numpy expression>
    #   out["GEX_MY_FORMULA"]   = arr
    #   out["GEX_MY_FORMULA_$"] = arr / 1e9
    #   Add "GEX_MY_FORMULA_$" to FORMULA_COLS at top.
    # ─────────────────────────────────────────────────────────

    return out

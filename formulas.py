# ─────────────────────────────────────────────────────────────
# formulas.py — GEX / DEX formula engine
# ─────────────────────────────────────────────────────────────
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
    "GEX_Spot_Reversal_$",
    "GEX_Spot_Breakout_$",
    "GEX_Charm_Pin_$",
    "GEX_Approach_Zone_$",
    "GEX_Spread_Alert_$",
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

    # ── Proximity (wide only) ─────────────────────────────────
    atm_band_wide  = S * 0.005 + eps
    proximity_wide = np.exp(-np.abs(strikes_arr - S) / atm_band_wide)

    # ── Wall magnitude ────────────────────────────────────────
    total_open_oi = open_coi + open_poi + eps
    max_open_oi   = total_open_oi.max() + eps
    wall_mag_norm = total_open_oi / max_open_oi
    wall_mag_safe = wall_mag_norm + eps

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

    # ── GEX_Reversal ─────────────────────────────────────────
    gex_reversal          = np.abs(call_wall_flag * gex_put_demand
                                   + put_wall_flag * gex_call_demand) \
                            * proximity_wide * wall_mag_safe
    out["GEX_Reversal"]   = gex_reversal
    out["GEX_Reversal_$"] = gex_reversal / 1e9

    # ── GEX_Breakout ─────────────────────────────────────────
    gex_breakout          = np.abs(call_wall_flag * gex_call_demand
                                   + put_wall_flag * gex_put_demand) \
                            * proximity_wide * wall_mag_safe
    out["GEX_Breakout"]   = gex_breakout
    out["GEX_Breakout_$"] = gex_breakout / 1e9

    # ── GEX_Spot_Reversal / GEX_Spot_Breakout ────────────────
    out["GEX_Spot_Reversal"]   = 0.0
    out["GEX_Spot_Reversal_$"] = 0.0
    out["GEX_Spot_Breakout"]   = 0.0
    out["GEX_Spot_Breakout_$"] = 0.0

    # ── GEX_Charm_Pin ─────────────────────────────────────────
    gex_charm_pin          = ((cc * call_bid_press * call_wall_flag
                               - pc * put_bid_press * put_wall_flag)
                              * wall_mag_norm * proximity_wide * M * S2)
    out["GEX_Charm_Pin"]   = gex_charm_pin
    out["GEX_Charm_Pin_$"] = gex_charm_pin / 1e9

    # ─────────────────────────────────────────────────────────
    # GEX_Approach_Zone
    # ─────────────────────────────────────────────────────────
    # Identifies the exact strike where dealer hedging pressure
    # from an approaching wall becomes strong enough to reverse
    # price — typically 5-15 points before the wall strike itself.
    #
    # Logic per timestamp:
    #   1. Find the dominant wall strike: the strike with the
    #      highest GEX_Call_Wall (resistance above spot) and
    #      the highest |GEX_Put_Wall| (support below spot).
    #   2. For each wall, walk the strikes between spot and the
    #      wall, accumulating cumulative gamma pressure:
    #        cumulative_gamma = Σ (cg × open_coi + pg × open_poi)
    #        for each strike between spot and the wall.
    #   3. The approach zone boundary is the first strike where
    #      cumulative_gamma >= APPROACH_THRESH × wall_total_gamma.
    #      APPROACH_THRESH = 0.35 means 35% of the wall's total
    #      hedging demand has already been activated — this is
    #      empirically where reversal pressure overcomes momentum.
    #   4. The output value at the approach zone strike equals
    #      the wall's GEX magnitude × proximity decay from spot,
    #      so it plots as a visible bar at the correct strike.
    #
    # Positive output = call wall approach zone (resistance).
    # Negative output = put wall approach zone (support).
    # Zero everywhere else at that timestamp.
    # ─────────────────────────────────────────────────────────
    APPROACH_THRESH = 0.35

    approach_zone = np.zeros(len(out))

    if "timestamp" in out.columns:
        timestamps = out["timestamp"].values
        unique_ts  = pd.unique(timestamps)

        for ts in unique_ts:
            mask    = timestamps == ts
            idx_arr = np.where(mask)[0]

            if len(idx_arr) < 2:
                continue

            sk_ts   = strikes_arr[idx_arr]
            cg_ts   = cg[idx_arr]
            pg_ts   = pg[idx_arr]
            ocoi_ts = open_coi[idx_arr]
            opoi_ts = open_poi[idx_arr]
            s_val   = float(np.nanmean(S[idx_arr]))

            gcw_ts  = gex_call_wall[idx_arr]
            gpw_ts  = np.abs(gex_put_wall[idx_arr])

            # Total gamma pressure at each strike
            gamma_press_ts = cg_ts * ocoi_ts + pg_ts * opoi_ts

            # ── Call wall approach (resistance above spot) ────
            above_mask = sk_ts > s_val
            if above_mask.sum() >= 2:
                above_local  = np.where(above_mask)[0]
                wall_local   = above_local[np.argmax(gcw_ts[above_mask])]
                wall_gex_val = gcw_ts[wall_local]
                wall_strike  = sk_ts[wall_local]

                # Strikes between spot and wall, sorted ascending
                between_mask = (sk_ts > s_val) & (sk_ts < wall_strike)
                if between_mask.sum() > 0:
                    between_local = np.where(between_mask)[0]
                    sort_order    = np.argsort(sk_ts[between_mask])
                    between_sorted = between_local[sort_order]

                    wall_total_gamma = gamma_press_ts[wall_local] + eps
                    cumulative       = 0.0
                    zone_found       = False

                    for bl in between_sorted:
                        cumulative += gamma_press_ts[bl]
                        if cumulative >= APPROACH_THRESH * wall_total_gamma:
                            # This is the approach zone strike
                            dist_from_spot = abs(sk_ts[bl] - s_val)
                            prox_decay     = np.exp(
                                -dist_from_spot / (s_val * 0.005 + eps))
                            approach_zone[idx_arr[bl]] = (
                                wall_gex_val * prox_decay)
                            zone_found = True
                            break

                    # If no between-strikes, mark one step below wall
                    if not zone_found:
                        step_below = above_local[
                            np.argsort(sk_ts[above_mask])[0]]
                        dist_from_spot = abs(sk_ts[step_below] - s_val)
                        prox_decay     = np.exp(
                            -dist_from_spot / (s_val * 0.005 + eps))
                        approach_zone[idx_arr[step_below]] = (
                            wall_gex_val * prox_decay)

            # ── Put wall approach (support below spot) ────────
            below_mask = sk_ts < s_val
            if below_mask.sum() >= 2:
                below_local  = np.where(below_mask)[0]
                wall_local   = below_local[np.argmax(gpw_ts[below_mask])]
                wall_gex_val = gpw_ts[wall_local]
                wall_strike  = sk_ts[wall_local]

                between_mask = (sk_ts < s_val) & (sk_ts > wall_strike)
                if between_mask.sum() > 0:
                    between_local  = np.where(between_mask)[0]
                    sort_order     = np.argsort(-sk_ts[between_mask])
                    between_sorted = between_local[sort_order]

                    wall_total_gamma = gamma_press_ts[wall_local] + eps
                    cumulative       = 0.0
                    zone_found       = False

                    for bl in between_sorted:
                        cumulative += gamma_press_ts[bl]
                        if cumulative >= APPROACH_THRESH * wall_total_gamma:
                            dist_from_spot = abs(sk_ts[bl] - s_val)
                            prox_decay     = np.exp(
                                -dist_from_spot / (s_val * 0.005 + eps))
                            # Negative = put support zone
                            approach_zone[idx_arr[bl]] = -(
                                wall_gex_val * prox_decay)
                            zone_found = True
                            break

                    if not zone_found:
                        step_above = below_local[
                            np.argsort(-sk_ts[below_mask])[0]]
                        dist_from_spot = abs(sk_ts[step_above] - s_val)
                        prox_decay     = np.exp(
                            -dist_from_spot / (s_val * 0.005 + eps))
                        approach_zone[idx_arr[step_above]] = -(
                            wall_gex_val * prox_decay)

    out["GEX_Approach_Zone"]   = approach_zone
    out["GEX_Approach_Zone_$"] = approach_zone / 1e9

    # ─────────────────────────────────────────────────────────
    # GEX_Spread_Alert
    # ─────────────────────────────────────────────────────────
    # Detects market maker defensive behavior by measuring
    # bid-ask spread expansion relative to each strike's own
    # rolling baseline. When a MM widens their spread on a
    # specific strike as price approaches, they are pricing in
    # the risk of that strike becoming ATM — this is a direct
    # behavioral signal that the wall is activating.
    #
    # Construction:
    #   1. Per strike, compute a 20-bar rolling baseline spread
    #      for both calls and puts (already in pipeline as
    #      iv_mean20/iv_std20 pattern — we replicate for spread).
    #      Since pipeline does not pre-compute spread baselines,
    #      we compute them here using a grouped rolling approach.
    #   2. Spread expansion ratio:
    #        call_spread_ratio = current_spread / baseline_spread
    #        put_spread_ratio  = current_spread / baseline_spread
    #   3. Net spread alert = weighted combination:
    #        alert = (call_spread_ratio × call_wall_flag
    #               + put_spread_ratio  × put_wall_flag)
    #               × wall_mag_norm × proximity_wide
    #   4. Sign: positive when call wall is dominant (resistance
    #      activating above spot), negative when put wall dominant
    #      (support activating below spot).
    #      Sign is assigned after computing magnitude so the
    #      histogram shows which side is alerting.
    #
    # High positive bar = MM widening call spreads at resistance.
    # High negative bar = MM widening put spreads at support.
    # These bars appear AT the strike where the spread is widening,
    # which is 5-15 points before the wall — exactly where you
    # want to enter.
    # ─────────────────────────────────────────────────────────
    SPREAD_ROLL = 20

    spread_alert = np.zeros(len(out))

    if "timestamp" in out.columns and "strike" in out.columns:
        # Build temporary df for rolling grouped computation
        tmp = pd.DataFrame({
            "timestamp":   out["timestamp"].values,
            "strike":      strikes_arr,
            "call_spread": call_spread,
            "put_spread":  put_spread,
            "call_wf":     call_wall_flag,
            "put_wf":      put_wall_flag,
            "wall_mag":    wall_mag_norm,
            "prox":        proximity_wide,
        })
        tmp = tmp.sort_values(["strike", "timestamp"]).reset_index(drop=True)

        # Rolling baseline per strike (20-bar mean)
        tmp["call_spread_base"] = (
            tmp.groupby("strike")["call_spread"]
               .transform(lambda x: x.rolling(SPREAD_ROLL,
                                               min_periods=1).mean())
        )
        tmp["put_spread_base"] = (
            tmp.groupby("strike")["put_spread"]
               .transform(lambda x: x.rolling(SPREAD_ROLL,
                                               min_periods=1).mean())
        )

        # Expansion ratio — how much wider than baseline right now
        tmp["call_exp"] = (tmp["call_spread"] /
                           (tmp["call_spread_base"] + eps)).clip(0, 10)
        tmp["put_exp"]  = (tmp["put_spread"]  /
                           (tmp["put_spread_base"]  + eps)).clip(0, 10)

        # Subtract 1 so ratio=1 (no expansion) → 0 contribution
        tmp["call_exp_excess"] = (tmp["call_exp"] - 1.0).clip(0, None)
        tmp["put_exp_excess"]  = (tmp["put_exp"]  - 1.0).clip(0, None)

        # Weighted alert magnitude
        alert_mag = ((tmp["call_exp_excess"] * tmp["call_wf"] +
                      tmp["put_exp_excess"]  * tmp["put_wf"])
                     * tmp["wall_mag"] * tmp["prox"])

        # Sign: call wall above spot → positive, put wall → negative
        # Use call_wf vs put_wf dominance to assign sign
        alert_sign = np.where(tmp["call_wf"] >= tmp["put_wf"], 1.0, -1.0)
        alert_signed = alert_mag * alert_sign

        # Re-align back to out index (out was sorted by timestamp,strike
        # at reset_index; tmp was sorted by strike,timestamp)
        tmp["_alert"] = alert_signed.values
        tmp_realigned = tmp.sort_values(
            ["timestamp", "strike"]).reset_index(drop=True)
        spread_alert = tmp_realigned["_alert"].values

    out["GEX_Spread_Alert"]   = spread_alert * M * S2
    out["GEX_Spread_Alert_$"] = out["GEX_Spread_Alert"] / 1e9

    # ── DEX_Call_OI ───────────────────────────────────────────
    out["DEX_Call_OI"]   = open_coi.astype(float)
    out["DEX_Call_OI_$"] = open_coi.astype(float) / 1e6

    # ── DEX_Put_OI ────────────────────────────────────────────
    out["DEX_Put_OI"]   = -open_poi.astype(float)
    out["DEX_Put_OI_$"] = -open_poi.astype(float) / 1e6

    # ─────────────────────────────────────────────────────────
    # ADD NEW FORMULAS BELOW THIS LINE
    # Available arrays: cg, pg, cv, pv, cc, pc,
    # civ, piv, open_coi, open_poi,
    # cbid, cask, pbid, pask,
    # call_iv_z, put_iv_z, iv_call_edge, iv_put_edge,
    # call_bid_press, put_bid_press,
    # call_spread, put_spread,
    # proximity_wide, call_wall_flag, put_wall_flag,
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

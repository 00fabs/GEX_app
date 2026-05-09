# ─────────────────────────────────────────────────────────────
# formulas.py — GEX / DEX formula engine
# Dynamic intraday formulas — 0DTE SPX focused
# All formula column names start with GEX_ or DEX_
# ─────────────────────────────────────────────────────────────
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
    """
    Dynamic intraday GEX / DEX formulas for 0DTE SPX.

    Key design decisions:
      - IV Z-score relative difference (call vs put) replaces
        absolute IV Z-scores to eliminate mirror-image demand charts
      - Hard wall classification threshold replaces soft weights
        to clearly separate reversal from breakout signals
      - Proximity removed from DEX to eliminate mountain shape
      - Wall magnitude normalization anchors demand to OI strikes
      - Wall integrity uses total demand erosion not directional
      - GEX_Charm_Pin added for afternoon expiration magnets
      - Session-open wall snapshot preserved as reference columns

    Requires pipeline.py to pre-compute per strike:
        call_iv_mean20, call_iv_std20
        put_iv_mean20,  put_iv_std20
        session_open_call_oi, session_open_put_oi  (first bar OI)
    """
    out = df.copy().reset_index(drop=True)

    eps = 1e-9

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

    # ── Greeks ────────────────────────────────────────────────
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

    # ── Rolling IV stats (pre-computed in pipeline.py) ────────
    civ_mean = col("call_iv_mean20")
    civ_std  = col("call_iv_std20")
    piv_mean = col("put_iv_mean20")
    piv_std  = col("put_iv_std20")

    # ── Session-open OI snapshot (pre-computed pipeline.py) ───
    open_coi = col("session_open_call_oi")
    open_poi = col("session_open_put_oi")

    S   = out["spot_used"].values.astype(float)
    S2  = S ** 2
    M   = SPX_MULT

    strikes_arr = col("strike") if "strike" in out.columns \
                  else np.zeros(len(out))

    # ═════════════════════════════════════════════════════════
    # BUILDING BLOCKS
    # ═════════════════════════════════════════════════════════

    # ── IV Z-Score per side ───────────────────────────────────
    call_iv_z = (civ - civ_mean) / (civ_std + eps)
    put_iv_z  = (piv - piv_mean) / (piv_std  + eps)

    # ── Relative IV edge (call vs put at same strike) ─────────
    #
    # Key fix for mirror-image demand problem.
    # Instead of absolute IV Z-scores (which move together
    # because both sides are driven by the same vol surface),
    # we measure which SIDE is being bid up more than the other.
    #
    # IV_call_edge > 0 → calls getting bid up more than puts
    #   → genuine call demand at this strike
    #   → someone paying up specifically for upside
    #
    # IV_put_edge > 0  → puts getting bid up more than calls
    #   → genuine put demand at this strike
    #   → someone paying up specifically for downside
    #
    # This removes the common volatility factor and isolates
    # directional demand. Demand charts will now only show
    # large bars where one side is genuinely outbidding the other.
    iv_call_edge = np.clip(call_iv_z - put_iv_z,  0, None)
    iv_put_edge  = np.clip(put_iv_z  - call_iv_z, 0, None)

    # ── Bid Pressure per side ─────────────────────────────────
    #
    # Estimates whether flow is buyer-aggressive (near ask)
    # or seller-aggressive (near bid).
    # 1.0 = mid at ask → aggressive buying  → opening pressure
    # 0.0 = mid at bid → aggressive selling → closing pressure
    call_mid       = (cbid + cask) / 2.0
    put_mid        = (pbid + pask) / 2.0
    call_spread    = np.abs(cask - cbid) + eps
    put_spread     = np.abs(pask - pbid) + eps
    call_bid_press = np.clip((call_mid - cbid) / call_spread, 0, 1)
    put_bid_press  = np.clip((put_mid  - pbid) / put_spread,  0, 1)

    # ── Proximity Weight ──────────────────────────────────────
    #
    # Used ONLY for GEX signals — not DEX.
    # Exponential decay from spot. Signals far from spot are
    # downweighted — not immediately actionable.
    # ATM band = 0.5% of spot.
    atm_band  = S * 0.005 + eps
    proximity = np.exp(-np.abs(strikes_arr - S) / atm_band)

    # ── Wall Magnitude Normalization ──────────────────────────
    #
    # Anchors demand signals to strikes that have actual OI.
    # Strikes with no OI get near-zero weight even if their
    # Greeks are theoretically large.
    # Uses session-open OI so this is stable through the day.
    total_open_oi  = open_coi + open_poi + eps
    max_open_oi    = total_open_oi.max() + eps
    wall_mag_norm  = total_open_oi / max_open_oi   # 0 to 1

    # ── Hard Wall Classification ──────────────────────────────
    #
    # Fix for reversal/breakout looking identical.
    # Hard threshold creates clear separation:
    #   call_oi > put_oi × 1.5 → pure call wall
    #   put_oi  > call_oi × 1.5 → pure put wall
    #   otherwise → neutral
    #
    # At a pure call wall:
    #   GEX_Reversal  = |put_demand|   (full weight)
    #   GEX_Breakout  = |call_demand|  (full weight)
    #   Clear visual difference between the two charts
    #
    # At neutral strike: both signals dampened (0.5 weight)
    call_wall_flag = np.where(open_coi > open_poi * 1.5, 1.0,
                     np.where(open_poi > open_coi * 1.5, 0.0, 0.5))
    put_wall_flag  = 1.0 - call_wall_flag

    # ═════════════════════════════════════════════════════════
    # GEX_Call_Wall — Call wall magnitude (always positive)
    # ═════════════════════════════════════════════════════════
    #
    # Uses session-open OI for stability — this is the base map.
    # Always positive bars → call walls = ceilings.
    # Tallest bar = strongest resistance for the session.
    #
    # Mechanics:
    #   Dealers short calls (long gamma) → sell into rallies
    #   → price decelerates approaching from below
    #   → compression / ceiling
    #
    # Watch GEX_Breakout at this strike for flip risk.
    # Watch GEX_Reversal at this strike for rejection confirmation.
    gex_call_wall = cg * open_coi * M * S2
    out["GEX_Call_Wall"]   = gex_call_wall
    out["GEX_Call_Wall_$"] = gex_call_wall / 1e9

    # ── Session-open reference snapshot ──────────────────────
    # Preserved as separate columns so chart can overlay
    # original wall map as permanent reference lines.
    out["GEX_Call_Wall_Open"]   = gex_call_wall
    out["GEX_Put_Wall_Open"]    = -(pg * open_poi * M * S2)

    # ═════════════════════════════════════════════════════════
    # GEX_Put_Wall — Put wall magnitude (always negative)
    # ═════════════════════════════════════════════════════════
    #
    # Always negative bars → put walls = floors.
    # Largest negative bar = strongest support for the session.
    #
    # Mechanics:
    #   Dealers short puts (long gamma) → buy into declines
    #   → price approaches fast (falling momentum + put skew)
    #   → dealers absorb the fall → floor / support
    #
    # Watch GEX_Breakout at this strike for breakdown risk.
    # Watch GEX_Reversal for bounce confirmation.
    gex_put_wall = -(pg * open_poi * M * S2)
    out["GEX_Put_Wall"]   = gex_put_wall
    out["GEX_Put_Wall_$"] = gex_put_wall / 1e9

    # ═════════════════════════════════════════════════════════
    # GEX_Call_Demand — Genuine call buying pressure
    # ═════════════════════════════════════════════════════════
    #
    # Three independent signals must agree for a large bar:
    #   1. iv_call_edge   — calls bid up MORE than puts at strike
    #                       (relative IV edge, not absolute)
    #   2. call_vanna     — dealer rehedging sensitivity to IV
    #   3. call_bid_press — flow aggressive at ask (opening)
    #   4. proximity      — near spot emphasis
    #   5. wall_mag_norm  — anchored to OI-significant strikes
    #
    # ALWAYS POSITIVE. Large bar = strong call demand.
    #
    # How to read:
    #   At CALL WALL → rising = BREAKOUT risk (same-side demand)
    #     Gamma flipping: dealers transition from selling to
    #     buying rallies → wall becomes rocket fuel
    #
    #   At PUT WALL → rising = REVERSAL up signal
    #     Buyers positioning for bounce at support
    #     Double dealer buying amplifies the reversal
    #
    # Zero or flat = no call demand = wall holding passively
    gex_call_demand = (iv_call_edge * cv
                       * call_bid_press
                       * proximity
                       * wall_mag_norm
                       * M * S2)
    out["GEX_Call_Demand"]   = gex_call_demand
    out["GEX_Call_Demand_$"] = gex_call_demand / 1e9

    # ═════════════════════════════════════════════════════════
    # GEX_Put_Demand — Genuine put buying pressure
    # ═════════════════════════════════════════════════════════
    #
    # Mirror structure of GEX_Call_Demand but for put side.
    # Uses iv_put_edge (puts bid up MORE than calls) so it
    # will NOT be a mirror image of GEX_Call_Demand.
    #
    # ALWAYS NEGATIVE. Large negative bar = strong put demand.
    # Negative convention keeps directional consistency with
    # GEX_Put_Wall (both negative = both bearish signals).
    #
    # How to read:
    #   At PUT WALL → large negative = BREAKDOWN risk
    #     More puts bought at already-heavy put strike
    #     Gamma flipping: dealers transition to selling with move
    #
    #   At CALL WALL → large negative = REVERSAL down signal
    #     Buyers positioning for rejection at resistance
    #     Double dealer selling amplifies the reversal
    gex_put_demand = -(iv_put_edge * pv
                       * put_bid_press
                       * proximity
                       * wall_mag_norm
                       * M * S2)
    out["GEX_Put_Demand"]   = gex_put_demand
    out["GEX_Put_Demand_$"] = gex_put_demand / 1e9

    # ═════════════════════════════════════════════════════════
    # GEX_Wall_Integrity — Net wall holding power
    # ═════════════════════════════════════════════════════════
    #
    # Measures whether a wall is holding or breaking down.
    # Uses TOTAL demand erosion — any demand at a wall regardless
    # of side reduces integrity because activity = potential
    # transition. A wall with zero demand on both sides is
    # maximally intact.
    #
    # Fix from previous version: opposite-side demand NO LONGER
    # strengthens integrity. It erodes it. The reversal vs
    # breakout distinction is left entirely to GEX_Reversal
    # and GEX_Breakout formulas.
    #
    # Formula:
    #   base = call_wall_flag × |GEX_Call_Wall|
    #        + put_wall_flag  × |GEX_Put_Wall|
    #   erosion = |GEX_Call_Demand| + |GEX_Put_Demand|
    #   integrity = base − erosion
    #
    # How to read:
    #   POSITIVE → wall holding → compression/support active
    #     Fade moves toward this strike
    #
    #   NEGATIVE → wall breaking → transition underway
    #     Follow the move — confirm direction with
    #     GEX_Reversal vs GEX_Breakout
    #
    #   ZERO CROSS (positive → negative) = most actionable
    #     Enter on next bar after zero cross confirmed
    base_integrity = (call_wall_flag * np.abs(gex_call_wall)
                      + put_wall_flag * np.abs(gex_put_wall))
    total_erosion  = np.abs(gex_call_demand) + np.abs(gex_put_demand)

    gex_wall_integrity = base_integrity - total_erosion
    out["GEX_Wall_Integrity"]   = gex_wall_integrity
    out["GEX_Wall_Integrity_$"] = gex_wall_integrity / 1e9

    # ═════════════════════════════════════════════════════════
    # GEX_Reversal — Opposite-side demand at wall
    # ═════════════════════════════════════════════════════════
    #
    # Pure reversal signal: demand building on the OPPOSITE
    # side from the dominant wall at each strike.
    #
    # Hard wall flags create clear visual separation from
    # GEX_Breakout — at a pure call wall:
    #   GEX_Reversal  = 1.0 × |put_demand|  (full weight)
    #   GEX_Breakout  = 1.0 × |call_demand| (full weight)
    #   No ambiguity from overlapping weights
    #
    # ALWAYS POSITIVE. Larger = stronger reversal conviction.
    #
    # How to read:
    #   At CALL WALL with large GEX_Reversal:
    #     Put buyers positioning for rejection at resistance
    #     Price approaching from below → expect stall/reversal down
    #     Confirm with DEX_Flow negative
    #
    #   At PUT WALL with large GEX_Reversal:
    #     Call buyers positioning for bounce at support
    #     Price approaching from above → expect stall/reversal up
    #     Confirm with DEX_Flow positive
    #
    # GEX_Reversal > GEX_Breakout at same strike → FADE the wall
    # GEX_Breakout > GEX_Reversal at same strike → FOLLOW the break
    wall_mag_norm_safe = wall_mag_norm + eps

    gex_reversal = (call_wall_flag * np.abs(gex_put_demand)
                    + put_wall_flag * np.abs(gex_call_demand)) \
                   * proximity * wall_mag_norm_safe
    out["GEX_Reversal"]   = gex_reversal
    out["GEX_Reversal_$"] = gex_reversal / 1e9

    # ═════════════════════════════════════════════════════════
    # GEX_Breakout — Same-side demand at wall
    # ═════════════════════════════════════════════════════════
    #
    # Pure breakout signal: demand building on the SAME side
    # as the dominant wall — wall about to flip from stabilising
    # to destabilising.
    #
    # ALWAYS POSITIVE. Larger = stronger breakout conviction.
    #
    # How to read:
    #   At CALL WALL with large GEX_Breakout:
    #     Call buyers piling in at resistance
    #     Gamma flipping: dealers go from selling to buying
    #     Wall becomes rocket fuel → follow the break UP
    #     Confirm with DEX_Flow positive
    #
    #   At PUT WALL with large GEX_Breakout:
    #     Put buyers piling in at support
    #     Gamma flipping: dealers go from buying to selling
    #     Wall becomes accelerant → follow the break DOWN
    #     Confirm with DEX_Flow negative
    #
    # Decision rule at any wall strike:
    #   GEX_Reversal > GEX_Breakout → fade
    #   GEX_Breakout > GEX_Reversal → follow
    #   Both near zero              → wall passive, wait
    #   Both large                  → conflicted, reduce size
    gex_breakout = (call_wall_flag * np.abs(gex_call_demand)
                    + put_wall_flag * np.abs(gex_put_demand)) \
                   * proximity * wall_mag_norm_safe
    out["GEX_Breakout"]   = gex_breakout
    out["GEX_Breakout_$"] = gex_breakout / 1e9

    # ═════════════════════════════════════════════════════════
    # GEX_Charm_Pin — Afternoon expiration magnet (0DTE)
    # ═════════════════════════════════════════════════════════
    #
    # THE most 0DTE-specific formula. Charm = dDelta/dTime.
    # For 0DTE charm accelerates exponentially after 2pm.
    # Every minute dealers MUST rehedge delta proportional to
    # charm regardless of spot movement — purely mechanical.
    # This creates predictable forced flows into close.
    #
    # Uses bid pressure to confirm flow is real (contracts
    # being opened/held not closed), and wall_mag_norm to
    # anchor to strikes with actual positioning.
    #
    # Formula:
    #   GEX_Charm_Pin =
    #     (call_charm × call_bid_press × call_wall_flag
    #    − put_charm  × put_bid_press  × put_wall_flag)
    #     × wall_mag_norm × proximity × M × S²
    #
    # SIGNED output:
    #   POSITIVE → call charm dominates at this strike
    #     Time-decay forces dealer buying → bullish drift
    #     into close → pin from below
    #
    #   NEGATIVE → put charm dominates
    #     Time-decay forces dealer selling → bearish drift
    #     into close → pin from above
    #
    # How to read:
    #   Mostly flat until ~1:30-2pm then grows rapidly
    #   Largest bar after 2pm = highest-confidence pin strike
    #   for the session close. Price will be magnetically
    #   attracted to this strike purely from time mechanics.
    #   Use as your final 30-min target strike.
    gex_charm_pin = ((cc * call_bid_press * call_wall_flag
                      - pc * put_bid_press * put_wall_flag)
                     * wall_mag_norm * proximity * M * S2)
    out["GEX_Charm_Pin"]   = gex_charm_pin
    out["GEX_Charm_Pin_$"] = gex_charm_pin / 1e9

    # ═════════════════════════════════════════════════════════
    # DEX_Flow — Directional delta flow (proximity-free)
    # ═════════════════════════════════════════════════════════
    #
    # Fix from previous version: proximity REMOVED.
    # Previous mountain shape was caused by proximity weighting
    # overwhelming delta — the chart was showing the exponential
    # decay function not actual delta flow.
    #
    # Shows raw directional hedging pressure at each strike
    # from bid-side flow. No spatial weighting so the chart
    # shows the true distribution across all strikes.
    #
    # Formula:
    #   DEX_Flow = (call_delta × call_bid_press
    #            +  put_delta  × put_bid_press)
    #              × M × S
    #
    # Note: put_delta is already negative from BSM so + sign
    # correctly nets the directional exposure.
    #
    # How to read:
    #   POSITIVE → net call bid pressure at this strike
    #     Buyers aggressively lifting calls
    #     Confirms bullish flow / reversal up / breakout up
    #
    #   NEGATIVE → net put bid pressure at this strike
    #     Buyers aggressively lifting puts
    #     Confirms bearish flow / reversal down / breakdown
    #
    #   Use as final confirmation:
    #     GEX signal bullish + DEX_Flow positive = high confidence
    #     GEX signal bullish + DEX_Flow negative = wait / reduce
    #
    #   Largest absolute bar nearest spot = dominant directional
    #   pressure for the current bar — your bias indicator.
    dex_flow = ((cd * call_bid_press
                 + pd_ * put_bid_press)
                * M * S)
    out["DEX_Flow"]   = dex_flow
    out["DEX_Flow_$"] = dex_flow / 1e9

    # ─────────────────────────────────────────────────────────
    # ADD NEW FORMULAS BELOW THIS LINE
    #
    # Available numpy arrays:
    #   cg, pg            call/put gamma
    #   cv, pv            call/put vanna
    #   cc, pc            call/put charm
    #   cd, pd_           call/put delta
    #   civ, piv          call/put IV
    #   coi, poi          call/put OI (intraday — use open_coi/poi for walls)
    #   open_coi, open_poi session-open OI snapshot (stable)
    #   cbid, cask        call bid/ask
    #   pbid, pask        put bid/ask
    #   cvol, pvol        call/put volume
    #   call_iv_z         call IV Z-score 20-bar rolling
    #   put_iv_z          put IV Z-score 20-bar rolling
    #   iv_call_edge      call IV bid-up vs put (relative, clipped ≥0)
    #   iv_put_edge       put  up vs call (relative, clipped ≥0)
    #   call_bid_press    call bid pressure 0-1
    #   put_bid_press     put bid pressure 0-1
    #   proximity         exponential spot proximity weight
    #   call_wall_flag    1=call wall, 0=put wall, 0.5=neutral
    #   put_wall_flag     1=put wall, 0=call wall, 0.5=neutral
    #   wall_mag_norm     OI magnitude normalised 0-1
    #   S, S2             spot, spot squared
    #   M                 multiplier (100)
    #   eps               small constant for division safety
    #
    # Template:
    #   arr = <numpy expression>
    #   out["GEX_MY_FORMULA"]   = arr
    #   out["GEX_MY_FORMULA_$"] = arr / 1e9
    #   Add "GEX_MY_FORMULA_$" to FORMULA_COLS at top of file.
    # ─────────────────────────────────────────────────────────

    return out

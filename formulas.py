# ─────────────────────────────────────────────────────────────
# formulas.py — GEX / DEX formula engine
# Dynamic intraday formulas — no static OI dependency
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
    "DEX_Proximity_Flow_$",
]


def apply_formulas(df: pd.DataFrame, spot_override: float,
                   intra_date) -> pd.DataFrame:
    """
    Dynamic intraday GEX / DEX formulas for 0DTE SPX.
    Uses IV Z-score (20-bar rolling) instead of raw IV change
    to avoid morning crush contamination and bar-to-bar noise.
    OI used only for wall classification (static map at open).
    All dynamic signals use Greeks, IV Z-score, bid pressure,
    and proximity weighting.

    Requires pipeline.py to have pre-computed per strike:
        call_iv_mean20, call_iv_std20
        put_iv_mean20,  put_iv_std20

    Parameters
    ----------
    df            : wide DataFrame from pipeline.pivot_wide()
                    with rolling IV stats pre-computed
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

    # ── Greeks ────────────────────────────────────────────────
    cg    = col("call_gamma")
    pg    = col("put_gamma")
    cv    = col("call_vanna")
    pv    = col("put_vanna")
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

    # ── Rolling IV stats (pre-computed in pipeline.py) ────────
    civ_mean = col("call_iv_mean20")
    civ_std  = col("call_iv_std20")
    piv_mean = col("put_iv_mean20")
    piv_std  = col("put_iv_std20")

    S   = out["spot_used"].values.astype(float)
    S2  = S ** 2
    M   = SPX_MULT
    eps = 1e-9

    # ═════════════════════════════════════════════════════════
    # BUILDING BLOCKS
    # ═════════════════════════════════════════════════════════

    # ── IV Z-Score (20-bar rolling per strike) ────────────────
    #
    # Measures how abnormal current IV is relative to its own
    # recent history at each strike. Automatically adjusts for
    # morning IV crush since the rolling mean tracks the crush
    # and the Z-score returns to zero once crush stabilises.
    #
    # Z > +1.5 → IV abnormally high → genuine demand building
    # Z < -1.5 → IV abnormally low  → demand fading / closing
    # |Z| < 1  → normal variation   → no strong signal
    #
    # First 20 bars per strike are warmup (min_periods=1 in
    # pipeline means early bars use whatever data is available
    # so signal is weaker but not zero).
    call_iv_z = (civ - civ_mean) / (civ_std + eps)
    put_iv_z  = (piv - piv_mean) / (piv_std  + eps)

    # ── Bid Pressure (0 to 1 per side) ───────────────────────
    #
    # Estimates whether flow is buyer-aggressive (near ask)
    # or seller-aggressive (near bid) without trade-level data.
    #
    # 1.0 = mid at ask  → aggressive buying  → opening pressure
    # 0.0 = mid at bid  → aggressive selling → closing pressure
    # 0.5 = mid at mid  → balanced / neutral
    #
    # We use (mid - bid) / spread rather than raw volume
    # because volume cannot distinguish opening from closing.
    call_mid         = (cbid + cask) / 2.0
    put_mid          = (pbid + pask) / 2.0
    call_spread      = cask - cbid + eps
    put_spread       = pask - pbid + eps
    call_bid_press   = np.clip((call_mid - cbid) / call_spread, 0, 1)
    put_bid_press    = np.clip((put_mid  - pbid) / put_spread,  0, 1)

    # ── Proximity Weight ──────────────────────────────────────
    #
    # Exponential decay from spot. Signals at strikes far from
    # spot are downweighted — they are theoretically interesting
    # but not immediately actionable.
    #
    # ATM band = 0.5% of spot (≈35 pts at SPX 7000).
    # Strike within 5 pts  → weight ≈ 0.87 → strong
    # Strike at 35 pts     → weight ≈ 0.37 → moderate
    # Strike at 70 pts     → weight ≈ 0.14 → weak
    # Strike at 100+ pts   → weight ≈ 0.00 → ignored
    strikes_arr = col("strike") if "strike" in out.columns \
                  else np.zeros(len(out))
    atm_band    = S * 0.005 + eps
    proximity   = np.exp(-np.abs(strikes_arr - S) / atm_band)

    # ── Wall Classification Weights ───────────────────────────
    #
    # Identifies how dominant the call vs put OI is at each
    # strike relative to total OI. Used to weight signals by
    # how strongly a strike qualifies as a wall.
    #
    # call_wall_w near 1.0 → strong call wall
    # put_wall_w  near 1.0 → strong put wall
    # Both near 0.5        → mixed positioning, weaker signal
    total_oi     = coi + poi + eps
    call_wall_w  = coi / total_oi   # 0 to 1
    put_wall_w   = poi / total_oi   # 0 to 1

    # ═════════════════════════════════════════════════════════
    # GEX_Call_Wall — Call wall magnitude (always positive)
    # ═════════════════════════════════════════════════════════
    #
    # Identifies strikes where call OI is dominant.
    # Dealers are assumed short these calls (long gamma).
    # Their hedging OPPOSES price movement → ceiling / resistance.
    # Price decelerates approaching these strikes from below
    # because dealers continuously sell into every uptick.
    #
    # Always POSITIVE so call walls are immediately visible
    # as upward bars on the histogram.
    #
    # How to read:
    #   Tallest positive bar = strongest ceiling for the session.
    #   This is your primary resistance target.
    #   Price approaching from below will slow and may stall here.
    #   Watch GEX_Breakout at this strike for flip risk.
    #   Watch GEX_Reversal at this strike for rejection confirmation.
    #
    # Formula:
    #   call_gamma × call_oi × M × S²
    gex_call_wall = cg * coi * M * S2
    out["GEX_Call_Wall"]   = gex_call_wall
    out["GEX_Call_Wall_$"] = gex_call_wall / 1e9

    # ═════════════════════════════════════════════════════════
    # GEX_Put_Wall — Put wall magnitude (always negative)
    # ═════════════════════════════════════════════════════════
    #
    # Identifies strikes where put OI is dominant.
    # Dealers are assumed short these puts (long gamma).
    # Their hedging OPPOSES downward movement → floor / support.
    # Price approaches put walls FAST (falling markets have
    # higher momentum and put delta accelerates faster due
    # to skew) but dealers are buying into every downtick.
    #
    # Always NEGATIVE so put walls are immediately visible
    # as downward bars — clearly separated from call walls.
    #
    # How to read:
    #   Largest negative bar = strongest floor for the session.
    #   This is your primary support target.
    #   Price approaching from above will be met with dealer buying.
    #   Watch GEX_Breakout at this strike for breakdown risk.
    #   Watch GEX_Reversal at this strike for bounce confirmation.
    #
    # Formula:
    #   −(put_gamma × put_oi × M × S²)
    gex_put_wall = -(pg * poi * M * S2)
    out["GEX_Put_Wall"]   = gex_put_wall
    out["GEX_Put_Wall_$"] = gex_put_wall / 1e9

    # ═════════════════════════════════════════════════════════
    # GEX_Call_Demand — Call buying pressure (dynamic)
    # ═════════════════════════════════════════════════════════
    #
    # Detects genuine call demand building at each strike using
    # three independent live signals multiplied together:
    #
    #   1. call_iv_z     — IV abnormally high vs recent history
    #                      (Z-score, 20-bar rolling per strike)
    #                      Confirms demand is real not just noise
    #
    #   2. call_vanna    — How much delta changes when IV moves
    #                      High vanna = dealers rehedging heavily
    #                      as this call demand builds
    #
    #   3. call_bid_press — Flow closer to ask = buyers aggressive
    #                       Flow closer to bid = sellers / closers
    #                       Distinguishes opening from closing flow
    #
    #   4. proximity     — Downweights far strikes automatically
    #
    # Signal is only strong when ALL THREE agree:
    #   IV abnormally high + vanna active + buyers at ask + near spot
    #
    # How to read:
    #   At a CALL WALL → rising GEX_Call_Demand = BREAKOUT risk
    #     Gamma flipping from stabilising to destabilising
    #     Dealers transitioning from selling to buying with move
    #
    #   At a PUT WALL → rising GEX_Call_Demand = REVERSAL signal
    #     Buyers positioning for bounce at exact support level
    #     Double dealer buying: short put hedge + new short call hedge
    #     Strongest reversal signal when GEX_Put_Wall is large negative
    #
    # Zero or flat → no demand signal → wall holding passively
    # Spikes → imminent transition → watch next 3-5 bars
    gex_call_demand = (call_iv_z * cv * call_bid_press
                       * proximity * M * S2)
    out["GEX_Call_Demand"]   = gex_call_demand
    out["GEX_Call_Demand_$"] = gex_call_demand / 1e9

    # ═════════════════════════════════════════════════════════
    # GEX_Put_Demand — Put buying pressure (dynamic)
    # ═════════════════════════════════════════════════════════
    #
    # Mirror of GEX_Call_Demand for the put side.
    # Detects genuine put demand building using:
    #
    #   1. put_iv_z      — Put IV abnormally high vs recent history
    #   2. put_vanna     — Delta rehedging sensitivity to IV move
    #   3. put_bid_press — Buyers aggressive at ask vs closers at bid
    #   4. proximity     — Near-spot weighting
    #
    # Plotted as NEGATIVE values so put demand reads downward
    # on the chart — directionally consistent with put walls.
    #
    # How to read:
    #   At a PUT WALL → rising |GEX_Put_Demand| = BREAKDOWN risk
    #     More puts being bought at already-heavy put strike
    #     Gamma flipping: dealers transition to selling with move
    #     Price will approach fast AND dealers amplify the fall
    #
    #   At a CALL WALL → rising |GEX_Put_Demand| = REVERSAL signal
    #     Buyers positioning for rejection at exact resistance
    #     Double dealer selling: short call hedge + new short put hedge
    #     Strongest reversal signal when GEX_Call_Wall is large positive
    #
    # Zero → no put demand → wall holding or no wall present
    gex_put_demand = -(put_iv_z * pv * put_bid_press
                       * proximity * M * S2)
    out["GEX_Put_Demand"]   = gex_put_demand
    out["GEX_Put_Demand_$"] = gex_put_demand / 1e9

    # ═════════════════════════════════════════════════════════
    # GEX_Wall_Integrity — Net wall holding power per strike
    # ═════════════════════════════════════════════════════════
    #
    # Combines the static wall strength with the dynamic demand
    # signals to produce a single reading of whether a wall is
    # holding or breaking down at each strike.
    #
    # Formula:
    #   At call-dominant strikes (call_wall_w > 0.5):
    #     Integrity = GEX_Call_Wall − GEX_Call_Demand × call_wall_w
    #                 + GEX_Put_Demand × call_wall_w
    #
    #   At put-dominant strikes (put_wall_w > 0.5):
    #     Integrity = GEX_Put_Wall + GEX_Put_Demand × put_wall_w
    #                 − GEX_Call_Demand × put_wall_w
    #
    # Unified formula (works continuously across all strikes):
    #   Integrity = (call_wall_w × GEX_Call_Wall
    #              + put_wall_w  × GEX_Put_Wall)
    #             − (call_wall_w × GEX_Call_Demand)   ← same-side erodes
    #             + (call_wall_w × GEX_Put_Demand)    ← opposite strengthens
    #             + (put_wall_w  × GEX_Put_Demand)    ← same-side erodes
    #             − (put_wall_w  × GEX_Call_Demand)   ← opposite strengthens
    #
    # How to read:
    #   POSITIVE → wall holding → compression/support active
    #     Fade moves toward this strike
    #     Price likely to stall or reverse here
    #
    #   NEGATIVE → wall breaking → flip/breakdown underway
    #     Do not fade — follow the move through this strike
    #     Confirm with GEX_Breakout or GEX_Reversal
    #
    #   ZERO CROSS (positive → negative) = transition point
    #     Most actionable signal on the chart
    #     Enter on next bar after zero cross confirmed
    base_integrity  = (call_wall_w * gex_call_wall
                       + put_wall_w * gex_put_wall)
    demand_erosion  = (call_wall_w * gex_call_demand
                       + put_wall_w * gex_put_demand)
    opposite_demand = (call_wall_w * gex_put_demand
                       + put_wall_w * gex_call_demand)

    gex_wall_integrity = base_integrity - demand_erosion + opposite_demand
    out["GEX_Wall_Integrity"]   = gex_wall_integrity
    out["GEX_Wall_Integrity_$"] = gex_wall_integrity / 1e9

    # ═════════════════════════════════════════════════════════
    # GEX_Reversal — Opposite-side demand at wall (reversal)
    # ═════════════════════════════════════════════════════════
    #
    # Isolates the pure reversal signal: demand building on the
    # OPPOSITE side from the dominant wall at each strike,
    # weighted by how strong the existing wall is and how close
    # price is to the strike.
    #
    # Logic:
    #   At a CALL WALL: put demand building = reversal down signal
    #     Buyers positioning for rejection at resistance
    #     Combined dealer selling = amplified reversal
    #
    #   At a PUT WALL: call demand building = reversal up signal
    #     Buyers positioning for bounce at support
    #     Combined dealer buying = amplified reversal
    #
    # Formula:
    #   GEX_Reversal = call_wall_w × |GEX_Put_Demand|  (at call walls)
    #                + put_wall_w  × |GEX_Call_Demand| (at put walls)
    #   × proximity × wall_magnitude_weight
    #
    # Always POSITIVE — magnitude shows reversal conviction.
    # Zero = no reversal signal at this strike.
    # Large = strong reversal positioning confirmed.
    #
    # How to use:
    #   Step 1: Find largest GEX_Call_Wall or GEX_Put_Wall near spot
    #   Step 2: Check GEX_Reversal at that same strike
    #   Step 3: If GEX_Reversal rising as price approaches → fade the move
    #   Step 4: Confirm with DEX_Proximity_Flow direction
    wall_magnitude = np.abs(gex_call_wall) + np.abs(gex_put_wall) + eps
    wall_mag_norm  = wall_magnitude / (wall_magnitude.max() + eps)

    gex_reversal = (call_wall_w * np.abs(gex_put_demand)
                    + put_wall_w * np.abs(gex_call_demand)) \
                   * proximity * wall_mag_norm * M
    out["GEX_Reversal"]   = gex_reversal
    out["GEX_Reversal_$"] = gex_reversal / 1e9

    # ═════════════════════════════════════════════════════════
    # GEX_Breakout — Same-side demand at wall (breakout)
    # ═════════════════════════════════════════════════════════
    #
    # Isolates the pure breakout signal: demand building on the
    # SAME side as the dominant wall at each strike, indicating
    # the wall is about to flip from stabilising to destabilising.
    #
    # Logic:
    #   At a CALL WALL: call demand building = breakout up signal
    #     Gamma flipping: dealers go from selling to buying rallies
    #     Wall becomes rocket fuel instead of ceiling
    #
    #   At a PUT WALL: put demand building = breakdown signal
    #     Gamma flipping: dealers go from buying to selling declines
    #     Wall becomes accelerant instead of floor
    #
    # Formula:
    #   GEX_Breakout = call_wall_w × |GEX_Call_Demand| (at call walls)
    #                + put_wall_w  × |GEX_Put_Demand|  (at put walls)
    #   × proximity × wall_magnitude_weight
    #
    # Always POSITIVE — magnitude shows breakout conviction.
    # Zero = no breakout signal.
    # Large = wall flip imminent, do not fade.
    #
    # How to use alongside GEX_Reversal:
    #   GEX_Reversal > GEX_Breakout → fade the wall
    #   GEX_Breakout > GEX_Reversal → follow the break
    #   Both large → conflicted, wait for one to dominate
    #   Both near zero → wall passive, no imminent move
    gex_breakout = (call_wall_w * np.abs(gex_call_demand)
                    + put_wall_w * np.abs(gex_put_demand)) \
                   * proximity * wall_mag_norm * M
    out["GEX_Breakout"]   = gex_breakout
    out["GEX_Breakout_$"] = gex_breakout / 1e9

    # ═════════════════════════════════════════════════════════
    # DEX_Proximity_Flow — Directional delta flow near spot
    # ═════════════════════════════════════════════════════════
    #
    # Measures net directional hedging pressure from bid-side
    # flow at strikes near spot. Confirms the direction implied
    # by the GEX wall signals without relying on volume counts.
    #
    # Formula:
    #   DEX_Proximity_Flow =
    #     (call_delta × call_bid_press
    #    − put_delta  × put_bid_press)   ← put_delta already negative
    #     × proximity × M × S
    #
    # Note: put_delta from BSM is already negative so subtracting
    # a negative bid pressure term means put buying ADDS to the
    # negative directional reading — directionally consistent.
    #
    # How to read:
    #   POSITIVE → net call bid pressure near spot
    #     Buyers lifting calls aggressively close to current price
    #     Confirms bullish bias / reversal up at put wall
    #     Confirms breakout up at call wall
    #
    #   NEGATIVE → net put bid pressure near spot
    #     Buyers lifting puts aggressively close to current price
    #     Confirms bearish bias / reversal down at call wall
    #     Confirms breakdown at put wall
    #
    #   Use as final confirmation:
    #     GEX signal points up + DEX_Proximity_Flow positive = high confidence
    #     GEX signal points up + DEX_Proximity_Flow negative = wait / reduce size
    dex_proximity_flow = ((cd * call_bid_press
                           + pd_ * put_bid_press)
                          * proximity * M * S)
    out["DEX_Proximity_Flow"]   = dex_proximity_flow
    out["DEX_Proximity_Flow_$"] = dex_proximity_flow / 1e9

    # ─────────────────────────────────────────────────────────
    # ADD NEW FORMULAS BELOW THIS LINE
    #
    # Available numpy arrays:
    #   cg, pg          call/put gamma
    #   cv, pv          call/put vanna
    #   cd, pd_         call/put delta
    #   civ, piv        call/put IV
    #   coi, poi        call/put OI (static — use only for classification)
    #   cbid,cask       call bid/ask
    #   pbid,pask       put bid/ask
    #   call_iv_z       call IV Z-score (20-bar rolling)
    #   put_iv_z        put IV Z-score (20-bar rolling)
    #   call_bid_press  call bid pressure 0-1
    #   put_bid_press   put bid pressure 0-1
    #   proximity       exponential spot proximity weight
    #   call_wall_w     call wall dominance weight 0-1
    #   put_wall_w      put wall dominance weight 0-1
    #   S, S2           spot, spot squared
    #   M               multiplier (100)
    #   eps             small constant for division safety
    #
    # Template:
    #   arr = <numpy expression>
    #   out["GEX_MY_FORMULA"]   = arr
    #   out["GEX_MY_FORMULA_$"] = arr / 1e9
    #   Then add "GEX_MY_FORMULA_$" to FORMULA_COLS at the top.
    # ─────────────────────────────────────────────────────────

    return out

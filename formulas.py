# ─────────────────────────────────────────────────────────────
# formulas.py — GEX / DEX formula engine
# ─────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
from config import SPX_MULT

FORMULA_COLS = [
    # ── GEX family ───────────────────────────────────────────
    "GEX_iv_weighted_$",       # Approach 1 — live IV as position proxy
    "GEX_relevance_$",         # Approach 2 — delta-velocity filtered GEX
    "GEX_gir_$",               # Approach 3 — gamma imbalance ratio
    "GEX_vanna_confirmed_$",   # Approach 4 — vanna-flow confirmed GEX
    # ── Supporting signals ───────────────────────────────────
    "strike_relevance",        # delta velocity magnitude — is strike in play?
    "vanna_flow",              # vanna pressure from IV moves — wall reinforced or undermined?
    "vanna_agreement",         # +1 vanna confirms GEX, -1 vanna contradicts GEX
    # ── DPI ──────────────────────────────────────────────────
    "DPI_$",                   # Dealer Pressure Index — combined Greeks score
    # ── DEX ──────────────────────────────────────────────────
    "DEX_$",
]


def apply_formulas(df: pd.DataFrame, spot_override: float,
                   intra_date) -> pd.DataFrame:
    """
    Takes the wide-format merged dataframe (one row per timestamp+strike)
    and computes all GEX / DEX / DPI columns.

    All dollar-normalised outputs are in $Billions (divide raw by 1e9).

    Parameters
    ----------
    df            : wide DataFrame from pipeline.pivot_wide()
    spot_override : fallback spot price when API underlyingPrice is missing
    intra_date    : the session date (available for time-decay formulas)

    Returns
    -------
    df with all formula columns appended.
    """
    out = df.copy()
    out = out.sort_values(["strike", "timestamp"]).reset_index(drop=True)

    # ── Spot price ────────────────────────────────────────────
    spot_col = (out["spot"].fillna(spot_override)
                if "spot" in out.columns
                else pd.Series([spot_override] * len(out), index=out.index))
    out["spot_used"] = spot_col

    # ── Helper: safely get column or zeros ───────────────────
    def col(name):
        return out.get(name, pd.Series(0.0, index=out.index)).fillna(0)

    def col_nan(name):
        """Returns column with NaN preserved — for diff calculations."""
        return out.get(name, pd.Series(np.nan, index=out.index))

    cg   = col("call_gamma")
    pg   = col("put_gamma")
    coi  = col("call_oi")
    poi  = col("put_oi")
    cvol = col("call_volume")
    pvol = col("put_volume")
    cd   = col("call_delta")
    pd_  = col("put_delta")
    cv   = col("call_vanna")
    pv   = col("put_vanna")
    cc   = col("call_charm")
    pc   = col("put_charm")

    civ_raw = col_nan("call_iv")
    piv_raw = col_nan("put_iv")
    civ     = civ_raw.fillna(0)
    piv     = piv_raw.fillna(0)

    S  = spot_col
    S2 = S ** 2
    M  = SPX_MULT

    # ── ATM IV — single reference value per timestamp ─────────
    # Used for normalisation in GIR. Computed as the mean of
    # call and put IV across all strikes at each timestamp,
    # falling back to a session-wide mean if timestamp is absent.
    if "timestamp" in out.columns:
        atm_iv_map = (out.groupby("timestamp")
                        .apply(lambda g: (col_nan("call_iv")
                                          .loc[g.index]
                                          .fillna(col_nan("put_iv")
                                                  .loc[g.index])
                                          .mean()))
                        .rename("_atm_iv"))
        out = out.merge(atm_iv_map, on="timestamp", how="left")
        atm_iv = out["_atm_iv"].fillna(
            civ_raw.fillna(piv_raw).mean()).replace(0, 1e-6)
        out.drop(columns=["_atm_iv"], inplace=True)
    else:
        atm_iv = pd.Series(
            civ_raw.fillna(piv_raw).mean(), index=out.index
        ).replace(0, 1e-6)

    # ─────────────────────────────────────────────────────────
    # APPROACH 1 — IV-Weighted GEX
    # ─────────────────────────────────────────────────────────
    # Replaces OI with live IV as the position weight.
    # Dealers reveal real risk through how they price options —
    # elevated IV at a strike means they're carrying gamma risk
    # there right now, regardless of what prior-day OI says.
    # call_IV weight → call-side gamma pressure (resistance).
    # put_IV weight  → put-side gamma pressure (support).
    # Positive = call IV dominates = resistance / ceiling.
    # Negative = put IV dominates  = support  / floor.
    # No OI dependency — fully live signal from current pricing.
    out["GEX_iv_weighted"]   = (cg * civ - pg * piv) * M * S2
    out["GEX_iv_weighted_$"] = out["GEX_iv_weighted"] / 1e9

    # ─────────────────────────────────────────────────────────
    # APPROACH 2 — Delta-Velocity Strike Relevance + Filtered GEX
    # ─────────────────────────────────────────────────────────
    # Delta velocity: how fast is each option's delta changing
    # bar over bar at this strike. High velocity = spot is moving
    # toward this strike and it is entering the active hedging zone.
    # Low velocity = strike is either far away or already passed.
    #
    # strike_relevance = |Δcall_delta| + |Δput_delta| per bar.
    # High relevance → strike is in play right now.
    # Low relevance  → discount the GEX signal regardless of size.
    #
    # GEX_relevance = GEX_iv_weighted × relevance_score
    # Large GEX bar at low-relevance strike = noise, gets damped.
    # Large GEX bar at high-relevance strike = real wall, amplified.

    cd_prev = (out.groupby("strike")["call_delta"]
                  .shift(1).fillna(out["call_delta"]))
    pd_prev = (out.groupby("strike")["put_delta"]
                  .shift(1).fillna(out["put_delta"]))

    call_delta_velocity = (cd - cd_prev).abs()
    put_delta_velocity  = (pd_ - pd_prev).abs()
    strike_relevance    = call_delta_velocity + put_delta_velocity

    # Normalise relevance to [0, 1] per timestamp so it's a
    # clean multiplier rather than a raw Greek unit
    if "timestamp" in out.columns:
        rel_max = (out.groupby("timestamp")["strike"]
                      .transform(lambda _: strike_relevance
                                           .loc[_.index].max())
                      .replace(0, 1e-6))
    else:
        rel_max = strike_relevance.max() or 1e-6

    relevance_norm = (strike_relevance / rel_max).fillna(0).clip(0, 1)

    out["strike_relevance"] = relevance_norm
    out["GEX_relevance"]    = out["GEX_iv_weighted"] * relevance_norm
    out["GEX_relevance_$"]  = out["GEX_relevance"] / 1e9

    # ─────────────────────────────────────────────────────────
    # APPROACH 3 — Gamma Imbalance Ratio (GIR)
    # ─────────────────────────────────────────────────────────
    # Combines OI and live IV to detect which side (call or put)
    # has dominant dealer risk at each strike, normalised by
    # total OI and ATM IV so it's comparable across strikes and
    # sessions.
    #
    # Numerator: call gamma risk (gamma × OI × IV) vs put gamma risk
    # Denominator: total OI × ATM IV — normalises scale
    #
    # Positive GIR = call-side dominates adjusted for pricing = resistance.
    # Negative GIR = put-side dominates adjusted for pricing  = support.
    # Advantage over raw GEX_signed: a strike with moderate OI
    # but extreme IV skew shows up strongly — closer to the truth
    # than OI alone because dealers priced in their actual risk.

    total_oi_safe = (coi + poi).replace(0, np.nan)
    gir_num       = (cg * coi * civ) - (pg * poi * piv)
    gir_den       = total_oi_safe * atm_iv
    gir_raw       = (gir_num / gir_den).fillna(0)

    # Scale back to dollar terms using S² × M so it plots on
    # a comparable axis to the other GEX formulas
    out["GEX_gir"]   = gir_raw * M * S2
    out["GEX_gir_$"] = out["GEX_gir"] / 1e9

    # ─────────────────────────────────────────────────────────
    # APPROACH 4 — Vanna Flow + Confirmation Filter
    # ─────────────────────────────────────────────────────────
    # Vanna = dDelta/dIV. When IV moves, vanna forces delta
    # hedging even without a spot price move.
    #
    # vanna_flow: net vanna-driven hedging pressure from the IV
    # move at this strike since the previous bar.
    # Positive = IV move is creating buying pressure here.
    # Negative = IV move is creating selling pressure here.
    #
    # vanna_agreement: does vanna reinforce or undermine GEX?
    # +1 = vanna and GEX_iv_weighted agree in sign → trust the wall.
    # -1 = they disagree → wall is being undermined by IV movement,
    #      discount the GEX bar even if it looks large.
    #
    # GEX_vanna_confirmed = GEX_iv_weighted × (1 + agreement × 0.5)
    # Agreement amplifies the signal by 50%. Disagreement damps it
    # to 50% of face value rather than zeroing it — GEX still exists,
    # it just has reduced conviction.

    civ_prev = (out.groupby("strike")["call_iv"]
                   .shift(1).fillna(out["call_iv"])
                   if "call_iv" in out.columns
                   else pd.Series(0.0, index=out.index))
    piv_prev = (out.groupby("strike")["put_iv"]
                   .shift(1).fillna(out["put_iv"])
                   if "put_iv" in out.columns
                   else pd.Series(0.0, index=out.index))

    civ_move = (civ - civ_prev.fillna(0))
    piv_move = (piv - piv_prev.fillna(0))

    vanna_flow = ((cv * coi * civ_move) -
                  (pv * poi * piv_move))
    out["vanna_flow"] = vanna_flow

    gex_sign   = np.sign(out["GEX_iv_weighted"].replace(0, np.nan)
                            .fillna(0))
    vanna_sign = np.sign(vanna_flow.replace(0, np.nan).fillna(0))
    agreement  = np.where(gex_sign == vanna_sign, 1.0,
                          np.where(vanna_sign == 0, 0.0, -1.0))
    out["vanna_agreement"] = agreement

    confirmation_multiplier = 1.0 + agreement * 0.5
    out["GEX_vanna_confirmed"]   = (out["GEX_iv_weighted"]
                                    * confirmation_multiplier)
    out["GEX_vanna_confirmed_$"] = out["GEX_vanna_confirmed"] / 1e9

    # ─────────────────────────────────────────────────────────
    # DPI — Dealer Pressure Index
    # ─────────────────────────────────────────────────────────
    # Combines four sources of dealer hedging pressure into one
    # signed score per strike.
    #
    # Component 1 — Gamma pressure (immediate, spot-driven)
    #   How much do dealers need to hedge RIGHT NOW if spot moves
    #   to this strike. Uses IV-weighted GEX as the best live proxy.
    #
    # Component 2 — Delta pressure (directional, existing position)
    #   What directional hedge do dealers already have at this strike.
    #   High absolute delta = large existing position, low gamma = stable.
    #   Uses DEX logic: call delta positive, put delta negative.
    #
    # Component 3 — Vanna pressure (IV-driven, no spot move needed)
    #   If IV is moving, vanna forces delta adjustments at this strike
    #   independent of spot. Critical for 0DTE where IV can spike fast.
    #   Uses vanna_flow computed above.
    #
    # Component 4 — Charm pressure (time-driven, intraday decay)
    #   As time passes, option deltas decay via charm. Dealers unwind
    #   hedges regardless of price. Accelerates after 2pm ET on 0DTE.
    #   Positive charm pressure = hedge unwind creates buying flow.
    #   Negative = unwind creates selling flow.
    #
    # Weights (tunable — adjust after backtesting):
    #   w1=0.40  gamma is the primary signal for 0DTE walls
    #   w2=0.25  delta pressure is the directional bias
    #   w3=0.20  vanna is the key modifier for IV-volatile sessions
    #   w4=0.15  charm matters most in final 2 hours
    #
    # Positive DPI = net buying pressure from dealer hedging = support.
    # Negative DPI = net selling pressure = resistance.
    # Largest absolute DPI bar = highest conviction strike right now.

    gamma_pressure = out["GEX_iv_weighted"]                    # $raw
    delta_pressure = (cd * coi + pd_ * poi) * M * S            # DEX raw
    vanna_pressure = vanna_flow * S                             # scale to $ terms
    charm_pressure = (cc * coi - pc * poi) * M                 # charm flow raw

    w1, w2, w3, w4 = 0.40, 0.25, 0.20, 0.15

    # Normalise each component to the same scale before weighting
    # so a large vanna_pressure doesn't swamp gamma_pressure.
    # We divide each by its session-wide std (robust to outliers).
    def safe_norm(series):
        std = series.std()
        if std == 0 or np.isnan(std):
            return series.fillna(0)
        return (series / std).fillna(0)

    dpi_raw = (w1 * safe_norm(gamma_pressure) +
               w2 * safe_norm(delta_pressure) +
               w3 * safe_norm(vanna_pressure) +
               w4 * safe_norm(charm_pressure))

    # Re-scale DPI to dollar terms using S² × M so it plots
    # on the same axis as GEX formulas
    out["DPI"]   = dpi_raw * M * S2
    out["DPI_$"] = out["DPI"] / 1e9

    # ── DEX — Delta Exposure ──────────────────────────────────
    # Total directional exposure dealers must hedge per strike.
    # Call delta positive, put delta already negative from BSM.
    # Positive = calls dominate = bullish attractor.
    # Negative = puts dominate  = bearish attractor / floor.
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

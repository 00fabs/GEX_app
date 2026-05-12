import pandas as pd
import numpy as np
from formulas import apply_formulas, FORMULA_COLS

COL_MAP = {
    "optionDelta":    "delta",
    "optionGamma":    "gamma",
    "optionIv":       "iv",
    "optionVanna":    "vanna",
    "optionCharm":    "charm",
    "optionOI":       "oi",
    "optionVolume":   "volume",
    "optionBidPrice": "bid",
    "optionAskPrice": "ask",
}

EWMA_ALPHA = 0.2

EWMA_COLS = [
    "GEX_Call_Demand_$",
    "GEX_Put_Demand_$",
    "GEX_Reversal_$",
    "GEX_Breakout_$",
]

# Stage 1 confirmation thresholds
DEMAND_RATIO_THRESH  = 2.0   # dominant demand must be 2× the other side
REVERSAL_RATIO_THRESH = 1.5  # reversal must be 1.5× breakout for reversal signal
BREAKOUT_RATIO_THRESH = 1.5  # breakout must be 1.5× reversal for breakout signal

# Stage 2 Greek filter thresholds
DELTA_ACC_THRESH     = 0.20  # delta acceleration minimum
GAMMA_DENSITY_THRESH = 0.20  # fraction of wall gamma density required


def pivot_wide(df_all: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-9

    def pivot_side(df, prefix):
        rename = {k: f"{prefix}_{v}"
                  for k, v in COL_MAP.items() if k in df.columns}
        df     = df.rename(columns=rename)
        base   = (["timestamp", "_strike", "underlyingPrice"]
                  if prefix == "call"
                  else ["timestamp", "_strike"])
        keep   = base + list(rename.values())
        return df[[c for c in keep if c in df.columns]].copy()

    calls_p = pivot_side(
        df_all[df_all["_optType"] == "C"].copy(), "call")
    puts_p  = pivot_side(
        df_all[df_all["_optType"] == "P"].copy(), "put")
    merged  = calls_p.merge(puts_p,
                            on=["timestamp", "_strike"],
                            how="outer")
    merged  = merged.rename(columns={"_strike": "strike",
                                     "underlyingPrice": "spot"})
    merged  = merged.sort_values(
        ["timestamp", "strike"]).reset_index(drop=True)

    # ── Rolling IV stats (20-bar per strike) ─────────────────
    for side in ["call", "put"]:
        iv_col = f"{side}_iv"
        if iv_col in merged.columns:
            merged[f"{side}_iv_mean20"] = (
                merged.groupby("strike")[iv_col]
                      .transform(lambda x:
                          x.rolling(20, min_periods=1).mean()))
            merged[f"{side}_iv_std20"] = (
                merged.groupby("strike")[iv_col]
                      .transform(lambda x:
                          x.rolling(20, min_periods=1)
                           .std()
                           .fillna(eps)))
        else:
            merged[f"{side}_iv_mean20"] = 0.0
            merged[f"{side}_iv_std20"]  = eps

    # ── Session-open OI snapshot ──────────────────────────────
    for side in ["call", "put"]:
        oi_col = f"{side}_oi"
        if oi_col in merged.columns:
            merged[f"session_open_{side}_oi"] = (
                merged.groupby("strike")[oi_col]
                      .transform("first"))
        else:
            merged[f"session_open_{side}_oi"] = 0.0

    return merged


def _apply_ewma(result: pd.DataFrame) -> pd.DataFrame:
    df = result.copy()
    if "strike" not in df.columns or "timestamp" not in df.columns:
        return df

    cols_to_smooth = [c for c in EWMA_COLS if c in df.columns]
    if not cols_to_smooth:
        return df

    df = df.sort_values(["strike", "timestamp"]).reset_index(drop=True)
    for c in cols_to_smooth:
        df[c] = (df.groupby("strike")[c]
                   .transform(lambda x:
                       x.ewm(alpha=EWMA_ALPHA,
                             adjust=False).mean()))
    df = df.sort_values(["timestamp", "strike"]).reset_index(drop=True)
    return df


def _apply_cumulative_demand(result: pd.DataFrame) -> pd.DataFrame:
    """
    Builds cumulative demand from session open per strike.
    Monotonically increasing — never drops.
    Weak early bars produce small cumulative values.
    Genuine sustained demand produces large cumulative values.
    This is the evidence accumulator for Stage 1 confirmation.
    """
    df = result.copy()
    if "strike" not in df.columns or "timestamp" not in df.columns:
        return df

    df = df.sort_values(["strike", "timestamp"]).reset_index(drop=True)

    for side in ["call", "put"]:
        demand_col = f"GEX_{side.capitalize()}_Demand_$"
        cumul_col  = f"_cumul_{side}_demand"
        if demand_col in df.columns:
            df[cumul_col] = (
                df.groupby("strike")[demand_col]
                  .transform(lambda x: x.clip(lower=0).cumsum()))
        else:
            df[cumul_col] = 0.0

    df = df.sort_values(["timestamp", "strike"]).reset_index(drop=True)
    return df


def _apply_entry_signal(result: pd.DataFrame) -> pd.DataFrame:
    """
    Computes GEX_Entry_Signal using two-stage logic.

    Stage 1 — Confirmation at wall strike:
      At each bar for each confirmed wall strike find whether
      cumulative put or call demand dominates by DEMAND_RATIO_THRESH,
      and whether reversal or breakout signal dominates accordingly.
      This determines the direction of the expected move.
      A wall strike is confirmed when:
        Reversal signal: cumul_put > cumul_call × ratio AND reversal > breakout × ratio
        Breakout signal: cumul_call > cumul_put × ratio AND breakout > reversal × ratio

    Stage 2 — Entry strike from Greeks:
      Walk strikes between spot and the confirmed wall.
      Find the first strike where both:
        delta_acceleration > DELTA_ACC_THRESH
        gamma_density      > GAMMA_DENSITY_THRESH × wall_gamma_density
      This is the entry strike — where dealer hedging is strong
      enough to influence price before reaching the wall itself.

    Output:
      Positive bar = bullish entry (reversal up or breakout up)
      Negative bar = bearish entry (reversal down or breakdown)
      Magnitude = cumulative_demand × gamma_density_ratio × delta_acc
      Zero at all non-entry strikes.
    """
    df  = result.copy()
    eps = 1e-9

    if "GEX_Entry_Signal_$" not in df.columns:
        df["GEX_Entry_Signal_$"] = 0.0
    df["GEX_Entry_Signal_$"] = 0.0

    if "timestamp" not in df.columns or "strike" not in df.columns:
        return df

    df = df.sort_values(["timestamp", "strike"]).reset_index(drop=True)

    def gcol(grp, name):
        return (pd.to_numeric(grp[name], errors="coerce")
                  .fillna(0).values
                if name in grp.columns
                else np.zeros(len(grp)))

    for ts in df["timestamp"].unique():
        mask = df["timestamp"] == ts
        grp  = df[mask].copy()

        if len(grp) < 3:
            continue

        strikes      = grp["strike"].values.astype(float)
        spot_vals    = (grp["spot_used"].values
                        if "spot_used" in grp.columns
                        else grp["spot"].values
                        if "spot" in grp.columns
                        else np.zeros(len(grp)))
        spot         = float(np.nanmean(spot_vals))

        # ── Inputs ────────────────────────────────────────────
        open_coi      = gcol(grp, "_open_coi")
        open_poi      = gcol(grp, "_open_poi")
        call_wall_flg = gcol(grp, "_call_wall_flag")
        put_wall_flg  = gcol(grp, "_put_wall_flag")
        gamma_press   = gcol(grp, "_gamma_press")
        call_delta_acc= gcol(grp, "_call_delta_acc")
        put_delta_acc = gcol(grp, "_put_delta_acc")

        cumul_call    = gcol(grp, "_cumul_call_demand")
        cumul_put     = gcol(grp, "_cumul_put_demand")
        reversal_vals = gcol(grp, "GEX_Reversal_$")
        breakout_vals = gcol(grp, "GEX_Breakout_$")

        # ── Stage 1: find confirmed wall strikes ──────────────
        # Check all call walls above spot and put walls below spot
        confirmed_signals = []   # list of (wall_idx, direction, cumul_strength)

        # Call walls above spot — check for reversal down or breakout up
        above_mask = strikes > spot
        for i in np.where(above_mask)[0]:
            if call_wall_flg[i] < 0.5:
                continue

            cp  = cumul_put[i]
            cc  = cumul_call[i]
            rev = reversal_vals[i]
            brk = breakout_vals[i]

            # Reversal down: put demand dominates at call wall
            if (cp > cc * DEMAND_RATIO_THRESH and
                    rev > brk * REVERSAL_RATIO_THRESH and
                    cp > 0):
                confirmed_signals.append((i, "reversal_down", cp))

            # Breakout up: call demand dominates at call wall
            elif (cc > cp * DEMAND_RATIO_THRESH and
                      brk > rev * BREAKOUT_RATIO_THRESH and
                      cc > 0):
                confirmed_signals.append((i, "breakout_up", cc))

        # Put walls below spot — check for reversal up or breakdown
        below_mask = strikes < spot
        for i in np.where(below_mask)[0]:
            if put_wall_flg[i] < 0.5:
                continue

            cp  = cumul_put[i]
            cc  = cumul_call[i]
            rev = reversal_vals[i]
            brk = breakout_vals[i]

            # Reversal up: call demand dominates at put wall
            if (cc > cp * DEMAND_RATIO_THRESH and
                    rev > brk * REVERSAL_RATIO_THRESH and
                    cc > 0):
                confirmed_signals.append((i, "reversal_up", cc))

            # Breakdown: put demand dominates at put wall
            elif (cp > cc * DEMAND_RATIO_THRESH and
                      brk > rev * BREAKOUT_RATIO_THRESH and
                      cp > 0):
                confirmed_signals.append((i, "breakdown", cp))

        if not confirmed_signals:
            continue

        # Pick the strongest confirmed signal by cumul_strength
        confirmed_signals.sort(key=lambda x: x[2], reverse=True)
        wall_idx, direction, cumul_strength = confirmed_signals[0]
        wall_strike      = float(strikes[wall_idx])
        wall_gamma_press = float(gamma_press[wall_idx]) + eps

        # ── Stage 2: find entry strike via Greek filters ──────
        if direction in ("reversal_down", "breakout_up"):
            # Wall above spot — walk strikes between spot and wall ascending
            between_mask = (strikes > spot) & (strikes < wall_strike)
        else:
            # Wall below spot — walk strikes between wall and spot descending
            between_mask = (strikes < spot) & (strikes > wall_strike)

        between_indices = np.where(between_mask)[0]

        if len(between_indices) == 0:
            # No strikes between spot and wall
            # Use the strike immediately adjacent to spot toward wall
            if direction in ("reversal_down", "breakout_up"):
                cands = np.where(strikes > spot)[0]
                if len(cands) > 0:
                    between_indices = np.array([cands[0]])
            else:
                cands = np.where(strikes < spot)[0]
                if len(cands) > 0:
                    between_indices = np.array([cands[-1]])

        if len(between_indices) == 0:
            continue

        # Sort toward wall from spot
        if direction in ("reversal_down", "breakout_up"):
            sort_order = np.argsort(strikes[between_mask]
                                    if between_mask.sum() > 0
                                    else strikes[between_indices])
        else:
            sort_order = np.argsort(-strikes[between_mask]
                                     if between_mask.sum() > 0
                                     else -strikes[between_indices])

        sorted_between = between_indices[sort_order] \
                         if len(between_indices) > 1 \
                         else between_indices

        entry_idx   = None
        entry_score = 0.0

        for bi in sorted_between:
            gd_ratio = gamma_press[bi] / wall_gamma_press
            da       = max(float(call_delta_acc[bi]),
                           float(put_delta_acc[bi]))

            if (gd_ratio >= GAMMA_DENSITY_THRESH and
                    da >= DELTA_ACC_THRESH):
                entry_idx   = bi
                entry_score = gd_ratio * da * cumul_strength
                break

        # Fallback: if no strike passes Greek filters use the
        # between strike with highest gamma density
        if entry_idx is None and len(sorted_between) > 0:
            gd_ratios   = gamma_press[sorted_between] / wall_gamma_press
            best_local  = int(np.argmax(gd_ratios))
            entry_idx   = sorted_between[best_local]
            da          = max(float(call_delta_acc[entry_idx]),
                              float(put_delta_acc[entry_idx]))
            entry_score = gd_ratios[best_local] * da * cumul_strength

        if entry_idx is None:
            continue

        # ── Output sign and magnitude ─────────────────────────
        # Positive = bullish entry, Negative = bearish entry
        sign = 1.0 if direction in ("reversal_up", "breakout_up") else -1.0
        entry_global_idx = grp.index[entry_idx]
        df.loc[entry_global_idx,
               "GEX_Entry_Signal_$"] = sign * float(entry_score)

    df = df.sort_values(["timestamp", "strike"]).reset_index(drop=True)
    return df


def build_minute_series(wide_df: pd.DataFrame,
                        spot_override: float,
                        intra_date) -> tuple:

    result = apply_formulas(wide_df, spot_override, intra_date)
    result = _apply_ewma(result)
    result = _apply_cumulative_demand(result)
    result = _apply_entry_signal(result)

    formula_keys = [c.replace("_$", "") for c in FORMULA_COLS]

    series = {}
    for ts, grp in result.groupby("timestamp"):
        ts_key       = pd.Timestamp(ts).strftime("%H:%M")
        strikes_data = {}
        for _, row in grp.iterrows():
            sk    = int(row["strike"])
            entry = {
                "spot": float(
                    row.get("spot_used", spot_override)
                    or spot_override)
            }
            for fkey, fcol in zip(formula_keys, FORMULA_COLS):
                entry[fkey] = float(row.get(fcol, 0) or 0)
            strikes_data[sk] = entry
        series[ts_key] = strikes_data

    sorted_ts   = sorted(series.keys())
    all_strikes = sorted({sk for ts_data in series.values()
                          for sk in ts_data})
    return series, sorted_ts, all_strikes, formula_keys


def build_session_table(wide_df: pd.DataFrame,
                        spot_override: float,
                        intra_date) -> pd.DataFrame:

    result = apply_formulas(wide_df, spot_override, intra_date)
    result = _apply_ewma(result)
    result = _apply_cumulative_demand(result)
    result = _apply_entry_signal(result)
    result = result.reset_index(drop=True)

    greek_cols   = ["call_iv","call_delta","call_gamma",
                    "call_vanna","call_charm","call_oi",
                    "call_volume","put_iv","put_delta",
                    "put_gamma","put_vanna","put_charm",
                    "put_oi","put_volume"]
    formula_cols = [c for c in result.columns if c.endswith("_$")]
    available    = set(result.columns)

    keep = []
    for c in (["timestamp","strike","spot_used"] +
              [c for c in greek_cols   if c in available] +
              [c for c in formula_cols if c in available]):
        if c in available and c not in keep:
            keep.append(c)

    ts_df     = result[keep].copy()
    ts_df     = ts_df.rename(columns={"spot_used": "spot"})
    sort_cols = [c for c in ["timestamp","strike"]
                 if c in ts_df.columns]
    if sort_cols:
        ts_df = ts_df.sort_values(sort_cols).reset_index(drop=True)

    ts_df.rename(
        columns={c: c.replace("_$", " ($)")
                 for c in formula_cols if c in ts_df.columns},
        inplace=True)
    return ts_df

# utils/helpers.py
import pandas as pd
from datetime import datetime, timedelta, time, date
import numpy as np
from scipy.stats import norm

def eat_to_et(d, t):
    eat_dt = datetime.combine(d, t)
    is_edt = 3 <= d.month <= 10
    et_dt  = eat_dt - timedelta(hours=7 if is_edt else 8)
    return et_dt, ("EDT" if is_edt else "EST")

def et_to_eat(dt_et, d):
    is_edt = 3 <= d.month <= 10
    return dt_et + timedelta(hours=7 if is_edt else 8)

def prev_trading_day(d):
    step = 3 if d.weekday() == 0 else 1
    return d - timedelta(days=step)

def fmt_b(val):
    if pd.isna(val) or val is None: return "N/A"
    v = float(val)
    if abs(v) >= 1e9: return f"${v/1e9:.2f}B"
    if abs(v) >= 1e6: return f"${v/1e6:.2f}M"
    if abs(v) >= 1e3: return f"${v/1e3:.1f}K"
    return f"${v:.2f}"

def fmt_df_dollars(df):
    out = df.copy()
    for c in out.columns:
        if "($)" in c:
            out[c] = out[c].apply(fmt_b)
    return out

# BSM formulas
def _d1d2(S, K, T, r, sigma):
    if T <= 1e-8 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan, np.nan
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)

def bsm_vanna(S, K, T, r, sigma):
    d1, d2 = _d1d2(S, K, T, r, sigma)
    if np.isnan(d1): return np.nan
    return -norm.pdf(d1) * d2 / sigma

def bsm_charm(S, K, T, r, sigma):
    d1, d2 = _d1d2(S, K, T, r, sigma)
    if np.isnan(d1) or T <= 1e-8: return np.nan
    return -norm.pdf(d1) * (2*r*T - d2*sigma*np.sqrt(T)) / (2*T*sigma*np.sqrt(T))

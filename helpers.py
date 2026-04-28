# ─────────────────────────────────────────────────────────────
# helpers.py — timezone helpers, formatting, rate-limit state
# ─────────────────────────────────────────────────────────────
import time as time_module
import requests
import pandas as pd
from datetime import datetime, timedelta, time, date
from config import MIN_DELAY

# ── Shared mutable rate-limit clock ──────────────────────────
_last_request_time = [0.0]


def rate_limited_get(url, headers, params):
    elapsed = time_module.time() - _last_request_time[0]
    if elapsed < MIN_DELAY:
        time_module.sleep(MIN_DELAY - elapsed)
    r = requests.get(url, headers=headers, params=params)
    _last_request_time[0] = time_module.time()
    return r


def get_last_request_time():
    return _last_request_time


# ── Timezone ──────────────────────────────────────────────────
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


def session_open_et(d):
    return datetime.combine(d, time(9, 30))


def session_close_et(d):
    return datetime.combine(d, time(16, 0))


# ── Formatting ────────────────────────────────────────────────
def fmt_b(val):
    if pd.isna(val) or val is None:
        return "N/A"
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

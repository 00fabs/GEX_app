# components/charts.py
import streamlit as st
import streamlit.components.v1 as components

def build_histogram_chart(series_data, spot, title):
    if not series_data:
        st.info("No data for this timestamp.")
        return

    series_data = sorted(series_data, key=lambda x: x["strike"])
    strikes     = [d["strike"] for d in series_data]
    values      = [d["value"]  for d in series_data]

    if not values or all(v == 0 for v in values):
        st.info("All values are zero for this timestamp.")
        return

    max_abs = max(abs(v) for v in values) or 1.0

    # Find nearest strike to spot
    spot_idx    = min(range(len(strikes)), key=lambda i: abs(strikes[i] - spot))
    spot_strike = strikes[spot_idx]

    # Build JS arrays for the chart
    strikes_js = str(strikes)
    values_js  = str([round(v, 6) for v in values])
    spot_idx_js = spot_idx
    spot_js    = round(spot, 2)
    spot_strike_js = spot_strike
    max_abs_js = round(max_abs, 6)
    title_escaped = title.replace("'", "\\'")

    html = f'''
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #0e1117;
    color: #d1d4dc;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 12px;
    overflow-x: hidden;
  }}
  .chart-title {{
    padding: 8px 10px 4px;
    font-size: 13px;
    font-weight: 600;
    color: #e0e0e0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .chart-wrapper {{
    width: 100%;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
    cursor: grab;
    background: #0e1117;
    border: 1px solid #1e2130;
    border-radius: 6px;
  }}
  .chart-wrapper:active {{ cursor: grabbing; }}
  .chart-inner {{
    position: relative;
    height: 320px;
    padding: 10px 8px 36px 52px;
  }}
  .y-axis {{
    position: absolute;
    left: 0; top: 10px; bottom: 36px;
    width: 52px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    align-items: flex-end;
    padding-right: 6px;
  }}
  .y-label {{
    font-size: 10px;
    color: #666;
    line-height: 1;
  }}
  .bars-area {{
    position: relative;
    height: 100%;
    display: flex;
    align-items: flex-end;
    gap: 2px;
  }}
  .zero-line {{
    position: absolute;
    left: 0; right: 0;
    height: 1px;
    background: #ffffff44;
    pointer-events: none;
    z-index: 2;
  }}
  .bar-col {{
    display: flex;
    flex-direction: column;
    align-items: center;
    position: relative;
    flex-shrink: 0;
    width: 22px;
  }}
  .bar-col.spot-col .bar-rect {{
    outline: 2px solid #f0c040;
    outline-offset: 1px;
  }}
  .bar-rect {{
    width: 18px;
    border-radius: 2px 2px 0 0;
    transition: opacity 0.1s;
    cursor: pointer;
    position: relative;
    z-index: 3;
  }}
  .bar-rect.negative {{
    border-radius: 0 0 2px 2px;
  }}
  .bar-rect:hover {{ opacity: 0.75; }}
  .x-label {{
    position: absolute;
    bottom: -22px;
    font-size: 9px;
    color: #888;
    white-space: nowrap;
    transform: rotate(-45deg);
    transform-origin: top left;
    left: 2px;
  }}
  .x-label.spot-label {{ color: #f0c040; font-weight: 700; }}
  .tooltip {{
    position: fixed;
    background: #1e2130ee;
    border: 1px solid #444;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 12px;
    pointer-events: none;
    z-index: 9999;
    display: none;
    min-width: 140px;
    box-shadow: 0 4px 16px #0008;
  }}
  .tooltip .t-strike {{ font-weight: 700; color: #26a69a; font-size: 14px; }}
  .tooltip .t-val {{ color: #f0c040; }}
  .tooltip .t-spot {{ color: #aaa; font-size: 11px; margin-top: 3px; }}
  .spot-info {{
    padding: 6px 10px 8px;
    color: #f0c040;
    font-size: 12px;
  }}
  .grid-line {{
    position: absolute;
    left: 0; right: 0;
    height: 1px;
    background: #1e2130;
    pointer-events: none;
    z-index: 1;
  }}
</style>
</head>
<body>
<div class="chart-title">{title_escaped}</div>
<div id="tooltip" class="tooltip">
  <div class="t-strike" id="tt-strike"></div>
  <div class="t-val" id="tt-val"></div>
  <div class="t-spot" id="tt-spot"></div>
</div>
<div class="chart-wrapper">
  <div class="chart-inner" id="chartInner">
    <div class="y-axis" id="yAxis"></div>
    <div class="bars-area" id="barsArea"></div>
  </div>
</div>
<div class="spot-info">▲ Spot: <b>{spot_js}</b> &nbsp;|&nbsp; Nearest strike: <b>{spot_strike_js}</b></div>

<script>
const strikes   = {strikes_js};
const values    = {values_js};
const spotIdx   = {spot_idx_js};
const spot      = {spot_js};
const maxAbs    = {max_abs_js};
const spotStrike = {spot_strike_js};

const barsArea  = document.getElementById('barsArea');
const chartInner = document.getElementById('chartInner');
const yAxis     = document.getElementById('yAxis');
const tooltip   = document.getElementById('tooltip');
const ttStrike  = document.getElementById('tt-strike');
const ttVal     = document.getElementById('tt-val');
const ttSpot    = document.getElementById('tt-spot');

// Chart height available for bars (px), excluding padding
const CHART_H = 274;  // 320 - 10top - 36bottom

// Draw Y-axis labels
const numTicks = 5;
const yLabels = [];
for (let i = 0; i <= numTicks; i++) {{
  const frac = i / numTicks;
  const val  = maxAbs - frac * 2 * maxAbs;
  yLabels.push(val);
}}
yAxis.innerHTML = yLabels.map(v => {{
  const sign = v >= 0 ? '+' : '';
  const disp = Math.abs(v) >= 1 ? v.toFixed(2) : v.toFixed(3);
  return `<span class="y-label">\( {{sign}} \){{disp}}B</span>`;
}}).join('');

// Grid lines
for (let i = 0; i <= numTicks; i++) {{
  const pct = i / numTicks * 100;
  const gl  = document.createElement('div');
  gl.className = 'grid-line';
  gl.style.top = pct + '%';
  barsArea.appendChild(gl);
}}

// Zero line
const zeroLine = document.createElement('div');
zeroLine.className = 'zero-line';
zeroLine.style.top = '50%';
barsArea.appendChild(zeroLine);

// Min width so all bars fit
const barW   = 22;
const gapW   = 2;
const totalW = strikes.length * (barW + gapW);
barsArea.style.minWidth = totalW + 'px';
barsArea.style.height   = CHART_H + 'px';
barsArea.style.position = 'relative';
barsArea.style.alignItems = 'unset';

// Draw bars
strikes.forEach((strike, i) => {{
  const val    = values[i];
  const isPos  = val >= 0;
  const color  = isPos ? '#26a69a' : '#ef5350';
  const pct    = Math.abs(val) / maxAbs;

  const barH   = Math.round(pct * (CHART_H / 2));
  const zeroY  = CHART_H / 2;

  const col    = document.createElement('div');
  col.className = 'bar-col' + (i === spotIdx ? ' spot-col' : '');
  col.style.position = 'absolute';
  col.style.left     = (i * (barW + gapW)) + 'px';
  col.style.bottom   = '0';
  col.style.height   = CHART_H + 'px';
  col.style.width    = barW + 'px';

  const rect   = document.createElement('div');
  rect.className = 'bar-rect' + (isPos ? '' : ' negative');
  rect.style.background = color;
  rect.style.width      = '18px';
  rect.style.height     = barH + 'px';
  rect.style.position   = 'absolute';

  if (isPos) {{
    rect.style.bottom = (CHART_H / 2) + 'px';
    rect.style.top    = 'auto';
  }} else {{
    rect.style.top  = (CHART_H / 2) + 'px';
    rect.style.bottom = 'auto';
  }}

  const showLabel = strikes.length <= 40 || i % 2 === 0;
  if (showLabel) {{
    const lbl = document.createElement('div');
    lbl.className = 'x-label' + (i === spotIdx ? ' spot-label' : '');
    lbl.textContent = strike;
    lbl.style.position = 'absolute';
    lbl.style.bottom   = '-24px';
    lbl.style.left     = '2px';
    col.appendChild(lbl);
  }}

  function showTip(e) {{
    const sign = val >= 0 ? '+' : '';
    ttStrike.textContent = 'Strike: ' + strike;
    ttVal.textContent    = sign + val.toFixed(4) + 'B';
    ttSpot.textContent   = 'Spot: ' + spot + (i === spotIdx ? '  ◀ ATM' : '');
    tooltip.style.display = 'block';
    moveTip(e);
  }}
  function moveTip(e) {{
    const src = e.touches ? e.touches[0] : e;
    tooltip.style.left = (src.clientX + 14) + 'px';
    tooltip.style.top  = (src.clientY - 40) + 'px';
  }}
  function hideTip() {{ tooltip.style.display = 'none'; }}

  rect.addEventListener('mouseenter', showTip);
  rect.addEventListener('mousemove',  moveTip);
  rect.addEventListener('mouseleave', hideTip);
  rect.addEventListener('touchstart', e => {{ showTip(e); e.preventDefault(); }}, {{passive:false}});
  rect.addEventListener('touchmove',  moveTip, {{passive:true}});
  rect.addEventListener('touchend',   hideTip);

  col.appendChild(rect);
  barsArea.appendChild(col);
}});

chartInner.style.minWidth = (totalW + 60) + 'px';
</script>
</body>
</html>
'''

    # Height: 320px chart + title + spot info + x-label overflow
    components.html(html, height=400, scrolling=False)

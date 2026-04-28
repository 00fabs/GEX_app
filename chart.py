# ─────────────────────────────────────────────────────────────
# chart.py — touch-friendly HTML/JS histogram chart
# Features:
#   • Vertical spot price line (exact position, not bar outline)
#   • Crosshair (X+Y lines) with live strike & value readout
#   • Pinch-to-zoom on both axes independently
#   • Pan / drag on both axes
#   • Y-axis drag to stretch vertically
#   • X-axis drag to stretch horizontally (bar spacing)
# ─────────────────────────────────────────────────────────────
import streamlit.components.v1 as components


def build_histogram_chart(series_data: list, spot: float, title: str):
    '''
    Parameters
    ----------
    series_data : list of {"strike": int, "value": float}
    spot        : current spot price (float, exact — not snapped to strike)
    title       : chart title string
    """
    if not series_data:
        import streamlit as st
        st.info("No data for this timestamp.")
        return

    series_data = sorted(series_data, key=lambda x: x["strike"])
    strikes     = [d["strike"] for d in series_data]
    values      = [round(d["value"], 6) for d in series_data]

    if all(v == 0 for v in values):
        import streamlit as st
        st.info("All values are zero for this timestamp.")
        return

    strikes_js = str(strikes)
    values_js  = str(values)
    spot_js    = round(spot, 4)
    title_esc  = title.replace("'", "\\'").replace("`", "\\`")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  background: #0e1117;
  color: #d1d4dc;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  font-size: 12px;
  overflow: hidden;
  touch-action: none;
}}
#title {{
  padding: 7px 10px 3px;
  font-size: 13px;
  font-weight: 600;
  color: #e0e0e0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}
#outer {{
  display: flex;
  width: 100%;
  height: calc(100vh - 58px);
}}
/* ── Y-axis panel ── */
#yAxisPanel {{
  width: 54px;
  flex-shrink: 0;
  position: relative;
  cursor: ns-resize;
  background: #0e1117;
  border-right: 1px solid #1e2130;
  user-select: none;
}}
#yAxisCanvas {{
  display: block;
  width: 54px;
}}
/* ── Chart viewport ── */
#chartViewport {{
  flex: 1;
  position: relative;
  overflow: hidden;
  cursor: crosshair;
}}
canvas#mainCanvas {{
  display: block;
  position: absolute;
  top: 0; left: 0;
}}
/* ── X-axis panel ── */
#xAxisPanel {{
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 36px;
  cursor: ew-resize;
  background: #0e1117;
  border-top: 1px solid #1e2130;
  user-select: none;
}}
#xAxisCanvas {{
  display: block;
  width: 100%;
  height: 36px;
}}
/* ── Crosshair tooltip ── */
#crosshairTip {{
  position: absolute;
  background: #1e2130ee;
  border: 1px solid #555;
  border-radius: 6px;
  padding: 7px 11px;
  font-size: 12px;
  pointer-events: none;
  display: none;
  z-index: 99;
  min-width: 150px;
  box-shadow: 0 4px 16px #0009;
  line-height: 1.7;
}}
#spotInfo {{
  padding: 4px 10px 6px;
  font-size: 12px;
  color: #f0c040;
  height: 22px;
}}
</style>
</head>
<body>
<div id="title">{title_esc}</div>
<div id="outer">
  <div id="yAxisPanel"><canvas id="yAxisCanvas"></canvas></div>
  <div id="chartViewport">
    <canvas id="mainCanvas"></canvas>
    <div id="xAxisPanel"><canvas id="xAxisCanvas"></canvas></div>
    <div id="crosshairTip"></div>
  </div>
</div>
<div id="spotInfo">▲ Spot: <b>{spot_js}</b></div>

<script>
// ── Data ─────────────────────────────────────────────────────
const STRIKES = {strikes_js};
const VALUES  = {values_js};
const SPOT    = {spot_js};
const N       = STRIKES.length;

// ── Canvases ─────────────────────────────────────────────────
const viewport    = document.getElementById('chartViewport');
const mainCanvas  = document.getElementById('mainCanvas');
const yPanel      = document.getElementById('yAxisPanel');
const yCanvas     = document.getElementById('yAxisCanvas');
const xPanel      = document.getElementById('xAxisPanel');
const xCanvas     = document.getElementById('xAxisCanvas');
const tip         = document.getElementById('crosshairTip');

const mCtx = mainCanvas.getContext('2d');
const yCtx = yCanvas.getContext('2d');
const xCtx = xCanvas.getContext('2d');

// ── View state ───────────────────────────────────────────────
// barW    : pixel width per bar (x-zoom)
// yScale  : pixels per unit of value (y-zoom)
// panX    : pixel offset from left edge to first bar centre
// panY    : zero-line position from top of chart area in px (y-pan)
let barW   = 24;
let yScale = 1;          // will be set by fitAll()
let panX   = 40;
let panY   = 0;          // offset from natural centre

const BAR_GAP    = 3;
const X_AXIS_H   = 36;
const Y_AXIS_W   = 54;

// Device pixel ratio
const DPR = window.devicePixelRatio || 1;

// Dimensions (logical px)
let VW = 0, VH = 0;   // viewport width, height (excl x-axis)

function resize() {{
  VW = viewport.clientWidth;
  VH = viewport.clientHeight - X_AXIS_H;

  // Main canvas
  mainCanvas.width  = VW  * DPR;
  mainCanvas.height = VH  * DPR;
  mainCanvas.style.width  = VW  + 'px';
  mainCanvas.style.height = VH  + 'px';
  mCtx.scale(DPR, DPR);

  // Y axis canvas
  yCanvas.width  = Y_AXIS_W * DPR;
  yCanvas.height = VH       * DPR;
  yCanvas.style.height = VH + 'px';
  yCtx.scale(DPR, DPR);

  // X axis canvas
  xCanvas.width  = VW * DPR;
  xCanvas.height = X_AXIS_H * DPR;
  xCtx.scale(DPR, DPR);

  fitAll();
}}

// Fit all bars into view with padding
function fitAll() {{
  const totalW = N * (barW + BAR_GAP);
  panX = (VW - totalW) / 2 + (barW + BAR_GAP) / 2;

  const maxAbs = Math.max(...VALUES.map(Math.abs), 1e-9);
  yScale = (VH * 0.42) / maxAbs;
  panY   = 0;
  draw();
}}

// ── Coordinate helpers ────────────────────────────────────────
// Strike index → x pixel (centre of bar)
function strikeToX(i) {{
  return panX + i * (barW + BAR_GAP);
}}

// Value → y pixel (0 = top of chart area)
function valToY(v) {{
  return VH / 2 - panY - v * yScale;
}}

// x pixel → nearest strike index
function xToStrikeIdx(px) {{
  return Math.round((px - panX) / (barW + BAR_GAP));
}}

// y pixel → value
function yToVal(py) {{
  return (VH / 2 - panY - py) / yScale;
}}

// Spot x position (exact fractional position between strikes)
function spotX() {{
  // linear interpolation between strike positions
  if (N === 0) return panX;
  const s0 = STRIKES[0];
  const step = N > 1 ? (STRIKES[N-1] - s0) / (N-1) : 1;
  const frac = (SPOT - s0) / step;
  return panX + frac * (barW + BAR_GAP);
}}

// ── Draw ──────────────────────────────────────────────────────
function draw() {{
  drawMain();
  drawYAxis();
  drawXAxis();
}}

function drawMain() {{
  mCtx.clearRect(0, 0, VW, VH);

  const zeroY = valToY(0);

  // Grid lines (horizontal)
  const ticks = computeYTicks();
  mCtx.strokeStyle = '#1e2130';
  mCtx.lineWidth   = 1;
  ticks.forEach(t => {{
    const ty = valToY(t);
    if (ty < 0 || ty > VH) return;
    mCtx.beginPath();
    mCtx.moveTo(0, ty);
    mCtx.lineTo(VW, ty);
    mCtx.stroke();
  }});

  // Zero line
  if (zeroY >= 0 && zeroY <= VH) {{
    mCtx.strokeStyle = '#ffffff55';
    mCtx.lineWidth   = 1;
    mCtx.beginPath();
    mCtx.moveTo(0, zeroY);
    mCtx.lineTo(VW, zeroY);
    mCtx.stroke();
  }}

  // Bars
  const halfBar = barW / 2;
  for (let i = 0; i < N; i++) {{
    const x   = strikeToX(i);
    const val = VALUES[i];
    const cy  = valToY(val);
    const pos = val >= 0;

    const barTop  = pos ? cy    : zeroY;
    const barBot  = pos ? zeroY : cy;
    const barH    = Math.abs(barBot - barTop);

    if (x + halfBar < 0 || x - halfBar > VW) continue;  // cull offscreen

    mCtx.fillStyle = pos ? '#26a69a' : '#ef5350';
    mCtx.fillRect(x - halfBar, barTop, barW, Math.max(barH, 1));
  }}

  // ── Spot line (exact continuous position) ─────────────────
  const sx = spotX();
  mCtx.save();
  mCtx.strokeStyle = '#f0c040';
  mCtx.lineWidth   = 2;
  mCtx.setLineDash([6, 4]);
  mCtx.beginPath();
  mCtx.moveTo(sx, 0);
  mCtx.lineTo(sx, VH);
  mCtx.stroke();

  // Spot label at top
  mCtx.fillStyle   = '#f0c040';
  mCtx.font        = 'bold 11px sans-serif';
  mCtx.textAlign   = 'center';
  mCtx.setLineDash([]);
  const labelText  = 'Spot ' + SPOT.toFixed(2);
  const lw         = mCtx.measureText(labelText).width + 10;
  mCtx.fillStyle   = '#1e2130';
  mCtx.fillRect(sx - lw/2, 2, lw, 17);
  mCtx.fillStyle   = '#f0c040';
  mCtx.fillText(labelText, sx, 15);
  mCtx.restore();

  // ── Crosshair ────────────────────────────────────────────
  if (crosshair.visible) {{
    const cx = crosshair.x;
    const cy2 = crosshair.y;

    mCtx.save();
    mCtx.strokeStyle = '#9b59b680';
    mCtx.lineWidth   = 1;
    mCtx.setLineDash([4, 3]);

    // Vertical line
    mCtx.beginPath();
    mCtx.moveTo(cx, 0);
    mCtx.lineTo(cx, VH);
    mCtx.stroke();

    // Horizontal line
    mCtx.beginPath();
    mCtx.moveTo(0, cy2);
    mCtx.lineTo(VW, cy2);
    mCtx.stroke();

    // Dot at intersection
    mCtx.setLineDash([]);
    mCtx.fillStyle = '#9b59b6';
    mCtx.beginPath();
    mCtx.arc(cx, cy2, 4, 0, Math.PI * 2);
    mCtx.fill();

    mCtx.restore();
  }}
}}

function drawYAxis() {{
  yCtx.clearRect(0, 0, Y_AXIS_W, VH);
  const ticks = computeYTicks();
  yCtx.fillStyle  = '#888';
  yCtx.font       = '10px sans-serif';
  yCtx.textAlign  = 'right';

  ticks.forEach(t => {{
    const ty = valToY(t);
    if (ty < 4 || ty > VH - 4) return;
    const label = formatVal(t);
    yCtx.fillStyle = '#888';
    yCtx.fillText(label, Y_AXIS_W - 6, ty + 3);

    // tick mark
    yCtx.strokeStyle = '#2a2e39';
    yCtx.lineWidth   = 1;
    yCtx.beginPath();
    yCtx.moveTo(Y_AXIS_W - 4, ty);
    yCtx.lineTo(Y_AXIS_W,     ty);
    yCtx.stroke();
  }});

  // Crosshair y label
  if (crosshair.visible) {{
    const ty  = crosshair.y;
    const val = yToVal(ty);
    yCtx.fillStyle = '#9b59b6';
    yCtx.font      = 'bold 10px sans-serif';
    yCtx.fillText(formatVal(val), Y_AXIS_W - 6, ty + 3);
  }}
}}

function drawXAxis() {{
  xCtx.clearRect(0, 0, VW, X_AXIS_H);
  xCtx.fillStyle  = '#888';
  xCtx.font       = '10px sans-serif';
  xCtx.textAlign  = 'center';

  // Decide label density based on barW
  const step = barW < 14 ? Math.ceil(20 / (barW + BAR_GAP)) : 1;

  for (let i = 0; i < N; i += step) {{
    const x = strikeToX(i);
    if (x < 20 || x > VW - 10) continue;
    xCtx.fillStyle = '#888';
    xCtx.fillText(STRIKES[i], x, 14);
    xCtx.strokeStyle = '#2a2e39';
    xCtx.lineWidth   = 1;
    xCtx.beginPath();
    xCtx.moveTo(x, 0);
    xCtx.lineTo(x, 5);
    xCtx.stroke();
  }}

  // Crosshair x label
  if (crosshair.visible) {{
    const idx = xToStrikeIdx(crosshair.x);
    if (idx >= 0 && idx < N) {{
      xCtx.fillStyle = '#9b59b6';
      xCtx.font      = 'bold 10px sans-serif';
      xCtx.fillText(STRIKES[idx], crosshair.x, 14);
    }}
  }}
}}

// ── Y tick computation ────────────────────────────────────────
function computeYTicks() {{
  const maxAbs   = Math.max(...VALUES.map(Math.abs), 1e-9);
  const viewAbs  = (VH / 2) / yScale;
  const niceStep = niceNum(viewAbs / 4);
  const ticks    = [];
  const start    = Math.ceil((-viewAbs - panY / yScale) / niceStep) * niceStep;
  for (let t = start; t <= viewAbs - panY / yScale + niceStep; t += niceStep) {{
    ticks.push(parseFloat(t.toPrecision(6)));
  }}
  return ticks;
}}

function niceNum(x) {{
  const e = Math.floor(Math.log10(x));
  const f = x / Math.pow(10, e);
  let nice;
  if      (f < 1.5) nice = 1;
  else if (f < 3)   nice = 2;
  else if (f < 7)   nice = 5;
  else              nice = 10;
  return nice * Math.pow(10, e);
}}

function formatVal(v) {{
  const abs = Math.abs(v);
  if (abs === 0) return '0';
  if (abs >= 1)    return (v > 0 ? '+' : '') + v.toFixed(2) + 'B';
  if (abs >= 0.01) return (v > 0 ? '+' : '') + v.toFixed(3) + 'B';
  return               (v > 0 ? '+' : '') + v.toFixed(4) + 'B';
}}

// ── Crosshair state ───────────────────────────────────────────
const crosshair = {{ visible: false, x: 0, y: 0 }};

function updateCrosshair(px, py) {{
  crosshair.visible = true;
  crosshair.x = px;
  crosshair.y = py;

  // Tooltip content
  const idx = xToStrikeIdx(px);
  const val = yToVal(py);
  let strikeLabel = '';
  let strikeVal   = '';
  if (idx >= 0 && idx < N) {{
    strikeLabel = STRIKES[idx];
    strikeVal   = VALUES[idx];
  }}

  const sign    = strikeVal >= 0 ? '+' : '';
  const curSign = val >= 0 ? '+' : '';

  tip.innerHTML =
    `<span style="color:#26a69a;font-weight:700;font-size:13px">
       Strike: ${{strikeLabel}}
     </span><br>
     Bar value: <span style="color:#f0c040">${{sign}}${{Number(strikeVal).toFixed(4)}}B</span><br>
     Cursor Y:  <span style="color:#aaa">${{curSign}}${{val.toFixed(4)}}B</span><br>
     <span style="color:#777;font-size:10px">Spot: ${{SPOT.toFixed(2)}}</span>`;

  // Position tooltip: keep inside viewport
  const tipW = 170, tipH = 90;
  let tx = px + 14, ty = py - 50;
  if (tx + tipW > VW) tx = px - tipW - 8;
  if (ty < 4)         ty = 4;
  if (ty + tipH > VH) ty = VH - tipH - 4;
  tip.style.left    = tx + 'px';
  tip.style.top     = ty + 'px';
  tip.style.display = 'block';

  draw();
}}

function hideCrosshair() {{
  crosshair.visible = false;
  tip.style.display = 'none';
  draw();
}}

// ── Interaction ───────────────────────────────────────────────
// We track pointer events for:
//   1. Single pointer on chart  → crosshair
//   2. Two pointers on chart    → pinch zoom (x & y independently)
//   3. Single pointer on y-axis → y-stretch
//   4. Single pointer on x-axis → x-stretch

const pointers = new Map();   // pointerId -> {x, y}

let lastPinchDX = null;
let lastPinchDY = null;

// Chart pointer events
viewport.addEventListener('pointermove', onChartMove, {{passive: true}});
viewport.addEventListener('pointerdown', onChartDown, {{passive: true}});
viewport.addEventListener('pointerup',   onChartUp,   {{passive: true}});
viewport.addEventListener('pointerleave',onChartLeave,{{passive: true}});
viewport.addEventListener('pointercancel',onChartLeave,{{passive: true}});

// Prevent default touch scroll inside viewport
viewport.addEventListener('touchmove', e => e.preventDefault(), {{passive: false}});

function onChartDown(e) {{
  pointers.set(e.pointerId, {{ x: e.clientX, y: e.clientY }});
  viewport.setPointerCapture(e.pointerId);
  if (pointers.size === 2) {{
    const pts = [...pointers.values()];
    lastPinchDX = Math.abs(pts[0].x - pts[1].x);
    lastPinchDY = Math.abs(pts[0].y - pts[1].y);
  }}
}}

function onChartMove(e) {{
  const rect = viewport.getBoundingClientRect();
  const cx   = e.clientX - rect.left;
  const cy   = e.clientY - rect.top;

  // Update pointer position
  if (pointers.has(e.pointerId)) {{
    const prev = pointers.get(e.pointerId);

    if (pointers.size === 1) {{
      // Single finger: show crosshair (no pan — pan uses two fingers)
      if (cy < VH) updateCrosshair(cx, cy);

    }} else if (pointers.size === 2) {{
      // Two fingers: pinch zoom
      pointers.set(e.pointerId, {{ x: e.clientX, y: e.clientY }});
      const pts  = [...pointers.values()];
      const newDX = Math.abs(pts[0].x - pts[1].x);
      const newDY = Math.abs(pts[0].y - pts[1].y);

      if (lastPinchDX && newDX > 10) {{
        const scaleX = newDX / lastPinchDX;
        barW = Math.max(4, Math.min(120, barW * scaleX));
      }}
      if (lastPinchDY && newDY > 10) {{
        const scaleY = newDY / lastPinchDY;
        yScale = Math.max(1, yScale * scaleY);
      }}
      lastPinchDX = newDX;
      lastPinchDY = newDY;
      draw();
      return;
    }}
    pointers.set(e.pointerId, {{ x: e.clientX, y: e.clientY }});
  }} else {{
    // Mouse hover (no button)
    if (cy < VH) updateCrosshair(cx, cy);
  }}
}}

function onChartUp(e) {{
  pointers.delete(e.pointerId);
  lastPinchDX = null;
  lastPinchDY = null;
}}
function onChartLeave(e) {{
  pointers.delete(e.pointerId);
  if (pointers.size === 0) hideCrosshair();
}}

// Mouse wheel: zoom x (default) or y (shift+wheel)
viewport.addEventListener('wheel', e => {{
  e.preventDefault();
  const delta = e.deltaY > 0 ? 0.85 : 1.18;
  if (e.shiftKey) {{
    yScale = Math.max(1, yScale * delta);
  }} else {{
    barW = Math.max(4, Math.min(120, barW * delta));
  }}
  draw();
}}, {{ passive: false }});

// ── Y-axis drag: stretch yScale ───────────────────────────────
let yDragStart = null;
let yScaleStart = null;

yPanel.addEventListener('pointerdown', e => {{
  yDragStart  = e.clientY;
  yScaleStart = yScale;
  yPanel.setPointerCapture(e.pointerId);
}}, {{passive: true}});

yPanel.addEventListener('pointermove', e => {{
  if (yDragStart === null) return;
  const dy    = yDragStart - e.clientY;   // drag up = zoom in
  const factor = Math.exp(dy / 120);
  yScale = Math.max(0.5, yScaleStart * factor);
  draw();
}}, {{passive: true}});

yPanel.addEventListener('pointerup',    () => {{ yDragStart = null; }});
yPanel.addEventListener('pointerleave', () => {{ yDragStart = null; }});

// ── X-axis drag: stretch barW ─────────────────────────────────
let xDragStart = null;
let barWStart  = null;

xPanel.addEventListener('pointerdown', e => {{
  xDragStart = e.clientX;
  barWStart  = barW;
  xPanel.setPointerCapture(e.pointerId);
}}, {{passive: true}});

xPanel.addEventListener('pointermove', e => {{
  if (xDragStart === null) return;
  const dx    = e.clientX - xDragStart;   // drag right = wider
  const factor = Math.exp(dx / 200);
  barW = Math.max(4, Math.min(120, barWStart * factor));
  draw();
}}, {{passive: true}});

xPanel.addEventListener('pointerup',    () => {{ xDragStart = null; }});
xPanel.addEventListener('pointerleave', () => {{ xDragStart = null; }});

// ── Two-finger pan on chart (pan X) ──────────────────────────
// Handled in onChartMove when pointers.size === 2 and spread is constant.
// For simplicity we use a dedicated two-finger pan state:
let panStart = null;

// ── Init ──────────────────────────────────────────────────────
window.addEventListener('resize', () => {{
  mCtx.setTransform(1,0,0,1,0,0);
  yCtx.setTransform(1,0,0,1,0,0);
  xCtx.setTransform(1,0,0,1,0,0);
  resize();
}});

resize();
</script>
</body>
</html>'''

    components.html(html, height=460, scrolling=False)

# ─────────────────────────────────────────────────────────────
# chart.py — touch-friendly HTML/JS histogram chart
# Crosshair is permanent: always visible, drag to move,
# stays locked to its strike/value when axes are stretched.
# ─────────────────────────────────────────────────────────────
import streamlit as st
import streamlit.components.v1 as components


def build_histogram_chart(series_data: list, spot: float, title: str):
    if not series_data:
        st.info("No data for this timestamp.")
        return

    series_data = sorted(series_data, key=lambda x: x["strike"])
    strikes     = [d["strike"] for d in series_data]
    values      = [round(d["value"], 6) for d in series_data]

    if all(v == 0 for v in values):
        st.info("All values are zero for this timestamp.")
        return

    strikes_js = str(strikes)
    values_js  = str(values)
    spot_js    = str(round(spot, 4))
    title_safe = title.replace("'", "\\'").replace("`", "\\`")

    html = _CHART_TEMPLATE
    html = html.replace("@@STRIKES@@", strikes_js)
    html = html.replace("@@VALUES@@",  values_js)
    html = html.replace("@@SPOT@@",    spot_js)
    html = html.replace("@@TITLE@@",   title_safe)

    components.html(html, height=460, scrolling=False)


_CHART_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: #0e1117;
  color: #d1d4dc;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  font-size: 12px;
  overflow: hidden;
  touch-action: none;
}
#title {
  padding: 7px 10px 3px;
  font-size: 13px;
  font-weight: 600;
  color: #e0e0e0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
#outer {
  display: flex;
  width: 100%;
  height: calc(100vh - 58px);
}
#yAxisPanel {
  width: 54px;
  flex-shrink: 0;
  position: relative;
  cursor: ns-resize;
  background: #0e1117;
  border-right: 1px solid #1e2130;
  user-select: none;
}
#yAxisCanvas { display: block; width: 54px; }
#chartViewport {
  flex: 1;
  position: relative;
  overflow: hidden;
  /* crosshair drag cursor set dynamically */
}
canvas#mainCanvas {
  display: block;
  position: absolute;
  top: 0; left: 0;
}
#xAxisPanel {
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 36px;
  cursor: ew-resize;
  background: #0e1117;
  border-top: 1px solid #1e2130;
  user-select: none;
}
#xAxisCanvas { display: block; width: 100%; height: 36px; }

/* Permanent crosshair info box — top-right of chart area */
#crosshairTip {
  position: absolute;
  top: 8px;
  right: 8px;
  background: #1e2130cc;
  border: 1px solid #9b59b655;
  border-radius: 6px;
  padding: 7px 11px;
  font-size: 12px;
  pointer-events: none;
  z-index: 99;
  min-width: 155px;
  box-shadow: 0 4px 16px #0009;
  line-height: 1.8;
}
#spotInfo {
  padding: 4px 10px 6px;
  font-size: 12px;
  color: #f0c040;
  height: 22px;
}

/* Drag-handle hint shown on the crosshair dot */
.ch-hint {
  position: absolute;
  background: #9b59b6;
  border-radius: 50%;
  width: 14px;
  height: 14px;
  pointer-events: none;
  transform: translate(-50%, -50%);
  z-index: 50;
  box-shadow: 0 0 0 3px #9b59b630;
}
</style>
</head>
<body>
<div id="title">@@TITLE@@</div>
<div id="outer">
  <div id="yAxisPanel"><canvas id="yAxisCanvas"></canvas></div>
  <div id="chartViewport">
    <canvas id="mainCanvas"></canvas>
    <div id="xAxisPanel"><canvas id="xAxisCanvas"></canvas></div>

    <!-- Permanent crosshair info table — always visible -->
    <div id="crosshairTip">
      <span id="tt-strike" style="color:#26a69a;font-weight:700;font-size:13px">—</span><br>
      Bar: <span id="tt-bar" style="color:#f0c040">—</span><br>
      Y: <span id="tt-y" style="color:#aaa">—</span><br>
      <span style="color:#777;font-size:10px">Spot: @@SPOT@@</span>
    </div>
  </div>
</div>
<div id="spotInfo">&#9650; Spot: <b>@@SPOT@@</b></div>

<script>
const STRIKES = @@STRIKES@@;
const VALUES  = @@VALUES@@;
const SPOT    = @@SPOT@@;
const N       = STRIKES.length;

const viewport   = document.getElementById('chartViewport');
const mainCanvas = document.getElementById('mainCanvas');
const yPanel     = document.getElementById('yAxisPanel');
const yCanvas    = document.getElementById('yAxisCanvas');
const xPanel     = document.getElementById('xAxisPanel');
const xCanvas    = document.getElementById('xAxisCanvas');
const ttStrike   = document.getElementById('tt-strike');
const ttBar      = document.getElementById('tt-bar');
const ttY        = document.getElementById('tt-y');

const mCtx = mainCanvas.getContext('2d');
const yCtx = yCanvas.getContext('2d');
const xCtx = xCanvas.getContext('2d');

// ── View state ────────────────────────────────────────────────
let barW   = 24;
let yScale = 1;
let panX   = 40;
let panY   = 0;

const BAR_GAP  = 3;
const X_AXIS_H = 36;
const Y_AXIS_W = 54;
const DPR      = window.devicePixelRatio || 1;
let VW = 0, VH = 0;

// ── Crosshair state ───────────────────────────────────────────
// Stored as (strikeIndex, value) so it stays locked to the
// same data point when axes are stretched or panned.
const crosshair = {
  strikeIdx: 0,     // index into STRIKES array
  value:     0,     // Y value (in data units, not pixels)
  dragging:  false,
};

// ── Coordinate helpers ────────────────────────────────────────
function strikeToX(i)     { return panX + i * (barW + BAR_GAP); }
function valToY(v)         { return VH / 2 - panY - v * yScale; }
function xToStrikeIdx(px) { return Math.round((px - panX) / (barW + BAR_GAP)); }
function yToVal(py)        { return (VH / 2 - panY - py) / yScale; }

// Crosshair pixel position derived from locked data coords
function crosshairPx() {
  return {
    x: strikeToX(crosshair.strikeIdx),
    y: valToY(crosshair.value),
  };
}

function spotX() {
  if (N === 0) return panX;
  const s0   = STRIKES[0];
  const step = N > 1 ? (STRIKES[N-1] - s0) / (N-1) : 1;
  return panX + ((SPOT - s0) / step) * (barW + BAR_GAP);
}

// ── Resize / fit ──────────────────────────────────────────────
function resize() {
  VW = viewport.clientWidth;
  VH = viewport.clientHeight - X_AXIS_H;

  mainCanvas.width  = VW * DPR;
  mainCanvas.height = VH * DPR;
  mainCanvas.style.width  = VW + 'px';
  mainCanvas.style.height = VH + 'px';
  mCtx.setTransform(DPR, 0, 0, DPR, 0, 0);

  yCanvas.width  = Y_AXIS_W * DPR;
  yCanvas.height = VH * DPR;
  yCanvas.style.height = VH + 'px';
  yCtx.setTransform(DPR, 0, 0, DPR, 0, 0);

  xCanvas.width  = VW * DPR;
  xCanvas.height = X_AXIS_H * DPR;
  xCtx.setTransform(DPR, 0, 0, DPR, 0, 0);

  fitAll();
}

function fitAll() {
  const totalW = N * (barW + BAR_GAP);
  panX = (VW - totalW) / 2 + (barW + BAR_GAP) / 2;
  const maxAbs = Math.max(...VALUES.map(Math.abs), 1e-9);
  yScale = (VH * 0.42) / maxAbs;
  panY   = 0;

  // Initialise crosshair at centre strike, value = that bar's value
  const midIdx = Math.floor(N / 2);
  crosshair.strikeIdx = midIdx;
  crosshair.value     = VALUES[midIdx] || 0;

  updateTip();
  draw();
}

// ── Tip content update ────────────────────────────────────────
function fmtVal(v) {
  const a = Math.abs(v), s = v >= 0 ? '+' : '';
  if (a >= 1)    return s + v.toFixed(2) + 'B';
  if (a >= 0.01) return s + v.toFixed(3) + 'B';
  return              s + v.toFixed(4) + 'B';
}

function updateTip() {
  const idx    = crosshair.strikeIdx;
  const barVal = (idx >= 0 && idx < N) ? VALUES[idx] : null;
  ttStrike.textContent = (idx >= 0 && idx < N) ? 'Strike: ' + STRIKES[idx] : '—';
  ttBar.textContent    = barVal !== null ? fmtVal(barVal) : '—';
  ttY.textContent      = fmtVal(crosshair.value);
}

// ── Draw ──────────────────────────────────────────────────────
function draw() {
  drawMain();
  drawYAxis();
  drawXAxis();
}

function drawMain() {
  mCtx.clearRect(0, 0, VW, VH);
  const zeroY = valToY(0);
  const ticks = computeYTicks();

  // Grid lines
  mCtx.strokeStyle = '#1e2130';
  mCtx.lineWidth = 1;
  ticks.forEach(t => {
    const ty = valToY(t);
    if (ty < 0 || ty > VH) return;
    mCtx.beginPath(); mCtx.moveTo(0, ty); mCtx.lineTo(VW, ty); mCtx.stroke();
  });

  // Zero line
  if (zeroY >= 0 && zeroY <= VH) {
    mCtx.strokeStyle = '#ffffff55';
    mCtx.lineWidth = 1;
    mCtx.beginPath(); mCtx.moveTo(0, zeroY); mCtx.lineTo(VW, zeroY); mCtx.stroke();
  }

  // Bars
  const halfBar = barW / 2;
  for (let i = 0; i < N; i++) {
    const x   = strikeToX(i);
    const val = VALUES[i];
    if (x + halfBar < 0 || x - halfBar > VW) continue;
    const pos    = val >= 0;
    const barTop = pos ? valToY(val) : zeroY;
    const barH   = Math.abs(valToY(val) - zeroY);
    mCtx.fillStyle = pos ? '#26a69a' : '#ef5350';
    mCtx.fillRect(x - halfBar, barTop, barW, Math.max(barH, 1));
  }

  // Spot line
  const sx = spotX();
  mCtx.save();
  mCtx.strokeStyle = '#f0c040';
  mCtx.lineWidth   = 2;
  mCtx.setLineDash([6, 4]);
  mCtx.beginPath(); mCtx.moveTo(sx, 0); mCtx.lineTo(sx, VH); mCtx.stroke();
  mCtx.setLineDash([]);
  mCtx.font = 'bold 11px sans-serif';
  mCtx.textAlign = 'center';
  const labelText = 'Spot ' + SPOT.toFixed(2);
  const lw = mCtx.measureText(labelText).width + 10;
  mCtx.fillStyle = '#1e2130';
  mCtx.fillRect(sx - lw/2, 2, lw, 17);
  mCtx.fillStyle = '#f0c040';
  mCtx.fillText(labelText, sx, 15);
  mCtx.restore();

  // ── Permanent crosshair ───────────────────────────────────
  const cp = crosshairPx();
  mCtx.save();

  // Highlight the bar the crosshair sits on
  const hi = crosshair.strikeIdx;
  if (hi >= 0 && hi < N) {
    const hx  = strikeToX(hi);
    const hv  = VALUES[hi];
    const pos = hv >= 0;
    const barTop = pos ? valToY(hv) : zeroY;
    const barH   = Math.abs(valToY(hv) - zeroY);
    mCtx.fillStyle = pos ? '#26a69a44' : '#ef535044';
    mCtx.fillRect(hx - halfBar - 2, 0, barW + 4, VH);   // full-height column tint
    mCtx.strokeStyle = pos ? '#26a69a' : '#ef5350';
    mCtx.lineWidth   = 2;
    mCtx.strokeRect(hx - halfBar, barTop, barW, Math.max(barH, 1));
  }

  // Crosshair lines — dashed, purple
  mCtx.strokeStyle = '#9b59b6cc';
  mCtx.lineWidth   = 1.5;
  mCtx.setLineDash([5, 4]);

  // Vertical line
  mCtx.beginPath();
  mCtx.moveTo(cp.x, 0);
  mCtx.lineTo(cp.x, VH);
  mCtx.stroke();

  // Horizontal line
  mCtx.beginPath();
  mCtx.moveTo(0, cp.y);
  mCtx.lineTo(VW, cp.y);
  mCtx.stroke();

  // Centre dot (drag handle)
  mCtx.setLineDash([]);
  mCtx.fillStyle = '#9b59b6';
  mCtx.strokeStyle = '#fff';
  mCtx.lineWidth = 1.5;
  mCtx.beginPath();
  mCtx.arc(cp.x, cp.y, 6, 0, Math.PI * 2);
  mCtx.fill();
  mCtx.stroke();

  mCtx.restore();
}

function drawYAxis() {
  yCtx.clearRect(0, 0, Y_AXIS_W, VH);
  const ticks = computeYTicks();
  yCtx.textAlign = 'right';

  ticks.forEach(t => {
    const ty = valToY(t);
    if (ty < 4 || ty > VH - 4) return;
    yCtx.fillStyle   = '#888';
    yCtx.font        = '10px sans-serif';
    yCtx.fillText(formatVal(t), Y_AXIS_W - 6, ty + 3);
    yCtx.strokeStyle = '#2a2e39';
    yCtx.lineWidth   = 1;
    yCtx.beginPath(); yCtx.moveTo(Y_AXIS_W - 4, ty); yCtx.lineTo(Y_AXIS_W, ty); yCtx.stroke();
  });

  // Crosshair Y label on axis — highlighted
  const cp = crosshairPx();
  if (cp.y >= 0 && cp.y <= VH) {
    yCtx.fillStyle    = '#0e1117';
    yCtx.fillRect(0, cp.y - 9, Y_AXIS_W - 2, 18);
    yCtx.fillStyle    = '#9b59b6';
    yCtx.font         = 'bold 10px sans-serif';
    yCtx.fillText(formatVal(crosshair.value), Y_AXIS_W - 6, cp.y + 4);
    // tick mark
    yCtx.strokeStyle  = '#9b59b6';
    yCtx.lineWidth    = 1.5;
    yCtx.beginPath();
    yCtx.moveTo(Y_AXIS_W - 6, cp.y);
    yCtx.lineTo(Y_AXIS_W,     cp.y);
    yCtx.stroke();
  }
}

function drawXAxis() {
  xCtx.clearRect(0, 0, VW, X_AXIS_H);
  xCtx.textAlign = 'center';
  const step = barW < 14 ? Math.ceil(20 / (barW + BAR_GAP)) : 1;

  for (let i = 0; i < N; i += step) {
    const x = strikeToX(i);
    if (x < 20 || x > VW - 10) continue;
    xCtx.fillStyle   = '#888';
    xCtx.font        = '10px sans-serif';
    xCtx.fillText(STRIKES[i], x, 14);
    xCtx.strokeStyle = '#2a2e39';
    xCtx.lineWidth   = 1;
    xCtx.beginPath(); xCtx.moveTo(x, 0); xCtx.lineTo(x, 5); xCtx.stroke();
  }

  // Crosshair X label on axis — highlighted
  const cp  = crosshairPx();
  const idx = crosshair.strikeIdx;
  if (idx >= 0 && idx < N && cp.x >= 0 && cp.x <= VW) {
    xCtx.fillStyle = '#0e1117';
    xCtx.fillRect(cp.x - 22, 0, 44, 22);
    xCtx.fillStyle = '#9b59b6';
    xCtx.font      = 'bold 10px sans-serif';
    xCtx.fillText(STRIKES[idx], cp.x, 14);
    xCtx.strokeStyle = '#9b59b6';
    xCtx.lineWidth   = 1.5;
    xCtx.beginPath(); xCtx.moveTo(cp.x, 0); xCtx.lineTo(cp.x, 6); xCtx.stroke();
  }
}

// ── Y tick computation ────────────────────────────────────────
function computeYTicks() {
  const viewAbs  = (VH / 2) / yScale;
  const niceStep = niceNum(viewAbs / 4);
  const ticks    = [];
  const lo = -(viewAbs + panY / yScale);
  const hi =  (viewAbs - panY / yScale);
  for (let t = Math.ceil(lo / niceStep) * niceStep; t <= hi + niceStep; t += niceStep) {
    ticks.push(parseFloat(t.toPrecision(6)));
  }
  return ticks;
}

function niceNum(x) {
  const e    = Math.floor(Math.log10(x));
  const f    = x / Math.pow(10, e);
  const nice = f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10;
  return nice * Math.pow(10, e);
}

function formatVal(v) {
  const a = Math.abs(v), s = v >= 0 ? '+' : '';
  if (a >= 1)    return s + v.toFixed(2) + 'B';
  if (a >= 0.01) return s + v.toFixed(3) + 'B';
  return              s + v.toFixed(4) + 'B';
}

// ── Crosshair drag ────────────────────────────────────────────
// Dragging the crosshair updates strikeIdx and value,
// which are the "locked" data coordinates.
const DRAG_RADIUS = 28;   // px — how close to dot to start drag

function distToCrosshair(px, py) {
  const cp = crosshairPx();
  return Math.sqrt((px - cp.x) ** 2 + (py - cp.y) ** 2);
}

function moveCrosshairTo(px, py) {
  // X: snap to nearest strike index
  let idx = xToStrikeIdx(px);
  idx = Math.max(0, Math.min(N - 1, idx));
  crosshair.strikeIdx = idx;

  // Y: free-floating value (not snapped)
  crosshair.value = yToVal(py);

  updateTip();
  draw();
}

// ── Pointer events on chart viewport ─────────────────────────
const pointers   = new Map();
let lastPinchDX  = null, lastPinchDY  = null;
let isDraggingCH = false;   // dragging crosshair?
let isPinching   = false;

viewport.addEventListener('pointerdown', e => {
  const rect = viewport.getBoundingClientRect();
  const px   = e.clientX - rect.left;
  const py   = e.clientY - rect.top;

  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  viewport.setPointerCapture(e.pointerId);

  if (pointers.size === 1 && py < VH && distToCrosshair(px, py) < DRAG_RADIUS) {
    isDraggingCH = true;
    viewport.style.cursor = 'grabbing';
  }

  if (pointers.size === 2) {
    isDraggingCH = false;
    isPinching   = true;
    const pts    = [...pointers.values()];
    lastPinchDX  = Math.abs(pts[0].x - pts[1].x);
    lastPinchDY  = Math.abs(pts[0].y - pts[1].y);
  }
}, { passive: true });

viewport.addEventListener('pointermove', e => {
  const rect = viewport.getBoundingClientRect();
  const px   = e.clientX - rect.left;
  const py   = e.clientY - rect.top;

  if (pointers.has(e.pointerId)) {
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  }

  // Two-finger pinch zoom (independent X and Y axes)
  if (pointers.size === 2 && isPinching) {
    const pts   = [...pointers.values()];
    const newDX = Math.abs(pts[0].x - pts[1].x);
    const newDY = Math.abs(pts[0].y - pts[1].y);
    if (lastPinchDX && newDX > 8) {
      // Save crosshair strike position before zoom so it follows
      const prevX = crosshairPx().x;
      barW = Math.max(4, Math.min(120, barW * (newDX / lastPinchDX)));
      // Adjust panX to keep leftmost bar in same visual place
    }
    if (lastPinchDY && newDY > 8) {
      yScale = Math.max(0.5, yScale * (newDY / lastPinchDY));
    }
    lastPinchDX = newDX;
    lastPinchDY = newDY;
    draw();
    return;
  }

  // Single pointer drag: move crosshair
  if (isDraggingCH && py < VH) {
    moveCrosshairTo(px, py);
    return;
  }

  // Hover: show grab cursor when near crosshair dot
  if (!isDraggingCH && py < VH) {
    viewport.style.cursor =
      distToCrosshair(px, py) < DRAG_RADIUS ? 'grab' : 'default';
  }
}, { passive: true });

viewport.addEventListener('pointerup', e => {
  pointers.delete(e.pointerId);
  if (pointers.size < 2) {
    isPinching   = false;
    lastPinchDX  = null;
    lastPinchDY  = null;
  }
  if (pointers.size === 0) {
    isDraggingCH = false;
    viewport.style.cursor = 'default';
  }
}, { passive: true });

viewport.addEventListener('pointerleave', e => {
  pointers.delete(e.pointerId);
  if (pointers.size === 0) {
    isDraggingCH = false;
    viewport.style.cursor = 'default';
  }
}, { passive: true });

viewport.addEventListener('pointercancel', e => {
  pointers.delete(e.pointerId);
  isDraggingCH = false;
}, { passive: true });

viewport.addEventListener('touchmove', e => e.preventDefault(), { passive: false });

// Mouse wheel: zoom X (default) or Y (shift+wheel)
// Crosshair stays on its strike/value — pixel position updates automatically
viewport.addEventListener('wheel', e => {
  e.preventDefault();
  const d = e.deltaY > 0 ? 0.85 : 1.18;
  if (e.shiftKey) { yScale = Math.max(0.5, yScale * d); }
  else            { barW   = Math.max(4, Math.min(120, barW * d)); }
  draw();
}, { passive: false });

// ── Y-axis panel drag: stretch yScale ────────────────────────
let yDragStart = null, yScaleStart = null;
yPanel.addEventListener('pointerdown', e => {
  yDragStart = e.clientY; yScaleStart = yScale;
  yPanel.setPointerCapture(e.pointerId);
}, { passive: true });
yPanel.addEventListener('pointermove', e => {
  if (yDragStart === null) return;
  yScale = Math.max(0.5, yScaleStart * Math.exp((yDragStart - e.clientY) / 120));
  draw();   // crosshair pixel Y auto-updates via crosshairPx()
}, { passive: true });
yPanel.addEventListener('pointerup',    () => { yDragStart = null; });
yPanel.addEventListener('pointerleave', () => { yDragStart = null; });

// ── X-axis panel drag: stretch barW ──────────────────────────
let xDragStart = null, barWStart = null;
xPanel.addEventListener('pointerdown', e => {
  xDragStart = e.clientX; barWStart = barW;
  xPanel.setPointerCapture(e.pointerId);
}, { passive: true });
xPanel.addEventListener('pointermove', e => {
  if (xDragStart === null) return;
  barW = Math.max(4, Math.min(120, barWStart * Math.exp((e.clientX - xDragStart) / 200)));
  draw();   // crosshair pixel X auto-updates via crosshairPx()
}, { passive: true });
xPanel.addEventListener('pointerup',    () => { xDragStart = null; });
xPanel.addEventListener('pointerleave', () => { xDragStart = null; });

// ── Window resize ─────────────────────────────────────────────
window.addEventListener('resize', () => {
  mCtx.setTransform(1,0,0,1,0,0);
  yCtx.setTransform(1,0,0,1,0,0);
  xCtx.setTransform(1,0,0,1,0,0);
  resize();
});

resize();
</script>
</body>
</html>"""

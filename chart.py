# ─────────────────────────────────────────────────────────────
# chart.py — touch-friendly HTML/JS histogram chart
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

    # ── Inject only the dynamic values via simple replacement ──
    # The HTML template uses @@VARNAME@@ placeholders instead of
    # f-string braces so that the JS curly braces are never
    # misinterpreted by Python's f-string parser.
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


# ── Template — no f-string, pure JS braces are safe here ─────
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
  cursor: crosshair;
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
#crosshairTip {
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
}
#spotInfo {
  padding: 4px 10px 6px;
  font-size: 12px;
  color: #f0c040;
  height: 22px;
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
    <div id="crosshairTip"></div>
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
const tip        = document.getElementById('crosshairTip');

const mCtx = mainCanvas.getContext('2d');
const yCtx = yCanvas.getContext('2d');
const xCtx = xCanvas.getContext('2d');

let barW   = 24;
let yScale = 1;
let panX   = 40;
let panY   = 0;

const BAR_GAP  = 3;
const X_AXIS_H = 36;
const Y_AXIS_W = 54;
const DPR      = window.devicePixelRatio || 1;

let VW = 0, VH = 0;

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
  draw();
}

function strikeToX(i)  { return panX + i * (barW + BAR_GAP); }
function valToY(v)     { return VH / 2 - panY - v * yScale; }
function xToStrikeIdx(px) { return Math.round((px - panX) / (barW + BAR_GAP)); }
function yToVal(py)    { return (VH / 2 - panY - py) / yScale; }

function spotX() {
  if (N === 0) return panX;
  const s0   = STRIKES[0];
  const step = N > 1 ? (STRIKES[N-1] - s0) / (N-1) : 1;
  return panX + ((SPOT - s0) / step) * (barW + BAR_GAP);
}

function draw() { drawMain(); drawYAxis(); drawXAxis(); }

function drawMain() {
  mCtx.clearRect(0, 0, VW, VH);
  const zeroY = valToY(0);
  const ticks = computeYTicks();

  // Grid
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

  // Spot line — exact continuous position
  const sx = spotX();
  mCtx.save();
  mCtx.strokeStyle = '#f0c040';
  mCtx.lineWidth   = 2;
  mCtx.setLineDash([6, 4]);
  mCtx.beginPath(); mCtx.moveTo(sx, 0); mCtx.lineTo(sx, VH); mCtx.stroke();

  // Spot label box
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

  // Crosshair
  if (crosshair.visible) {
    mCtx.save();
    mCtx.strokeStyle = '#9b59b680';
    mCtx.lineWidth   = 1;
    mCtx.setLineDash([4, 3]);
    mCtx.beginPath(); mCtx.moveTo(crosshair.x, 0); mCtx.lineTo(crosshair.x, VH); mCtx.stroke();
    mCtx.beginPath(); mCtx.moveTo(0, crosshair.y); mCtx.lineTo(VW, crosshair.y); mCtx.stroke();
    mCtx.setLineDash([]);
    mCtx.fillStyle = '#9b59b6';
    mCtx.beginPath(); mCtx.arc(crosshair.x, crosshair.y, 4, 0, Math.PI * 2); mCtx.fill();
    mCtx.restore();
  }
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
  if (crosshair.visible) {
    yCtx.fillStyle = '#9b59b6';
    yCtx.font      = 'bold 10px sans-serif';
    yCtx.fillText(formatVal(yToVal(crosshair.y)), Y_AXIS_W - 6, crosshair.y + 3);
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
  if (crosshair.visible) {
    const idx = xToStrikeIdx(crosshair.x);
    if (idx >= 0 && idx < N) {
      xCtx.fillStyle = '#9b59b6';
      xCtx.font      = 'bold 10px sans-serif';
      xCtx.fillText(STRIKES[idx], crosshair.x, 14);
    }
  }
}

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
  const e = Math.floor(Math.log10(x));
  const f = x / Math.pow(10, e);
  const nice = f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10;
  return nice * Math.pow(10, e);
}

function formatVal(v) {
  const a = Math.abs(v);
  const s = v > 0 ? '+' : '';
  if (a >= 1)    return s + v.toFixed(2) + 'B';
  if (a >= 0.01) return s + v.toFixed(3) + 'B';
  return              s + v.toFixed(4) + 'B';
}

const crosshair = { visible: false, x: 0, y: 0 };

function updateCrosshair(px, py) {
  crosshair.visible = true;
  crosshair.x = px;
  crosshair.y = py;

  const idx = xToStrikeIdx(px);
  const barVal   = (idx >= 0 && idx < N) ? VALUES[idx] : null;
  const cursorVal = yToVal(py);
  const s  = v => (v >= 0 ? '+' : '') + Number(v).toFixed(4) + 'B';

  tip.innerHTML =
    '<span style="color:#26a69a;font-weight:700;font-size:13px">Strike: ' +
    (idx >= 0 && idx < N ? STRIKES[idx] : '—') + '</span><br>' +
    'Bar value: <span style="color:#f0c040">' + (barVal !== null ? s(barVal) : '—') + '</span><br>' +
    'Cursor Y: <span style="color:#aaa">' + s(cursorVal) + '</span><br>' +
    '<span style="color:#777;font-size:10px">Spot: ' + SPOT.toFixed(2) + '</span>';

  const tipW = 170, tipH = 90;
  let tx = px + 14, ty = py - 50;
  if (tx + tipW > VW) tx = px - tipW - 8;
  if (ty < 4)         ty = 4;
  if (ty + tipH > VH) ty = VH - tipH - 4;
  tip.style.left    = tx + 'px';
  tip.style.top     = ty + 'px';
  tip.style.display = 'block';
  draw();
}

function hideCrosshair() {
  crosshair.visible = false;
  tip.style.display = 'none';
  draw();
}

// ── Pointer handling ──────────────────────────────────────────
const pointers = new Map();
let lastPinchDX = null, lastPinchDY = null;

viewport.addEventListener('pointermove',  onChartMove,  { passive: true });
viewport.addEventListener('pointerdown',  onChartDown,  { passive: true });
viewport.addEventListener('pointerup',    onChartUp,    { passive: true });
viewport.addEventListener('pointerleave', onChartLeave, { passive: true });
viewport.addEventListener('pointercancel',onChartLeave, { passive: true });
viewport.addEventListener('touchmove', e => e.preventDefault(), { passive: false });

function onChartDown(e) {
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  viewport.setPointerCapture(e.pointerId);
  if (pointers.size === 2) {
    const pts = [...pointers.values()];
    lastPinchDX = Math.abs(pts[0].x - pts[1].x);
    lastPinchDY = Math.abs(pts[0].y - pts[1].y);
  }
}

function onChartMove(e) {
  const rect = viewport.getBoundingClientRect();
  const cx   = e.clientX - rect.left;
  const cy   = e.clientY - rect.top;

  if (pointers.has(e.pointerId)) {
    if (pointers.size === 1) {
      if (cy < VH) updateCrosshair(cx, cy);
    } else if (pointers.size === 2) {
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      const pts   = [...pointers.values()];
      const newDX = Math.abs(pts[0].x - pts[1].x);
      const newDY = Math.abs(pts[0].y - pts[1].y);
      if (lastPinchDX && newDX > 10) {
        barW = Math.max(4, Math.min(120, barW * (newDX / lastPinchDX)));
      }
      if (lastPinchDY && newDY > 10) {
        yScale = Math.max(1, yScale * (newDY / lastPinchDY));
      }
      lastPinchDX = newDX;
      lastPinchDY = newDY;
      draw();
      return;
    }
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  } else {
    if (cy < VH) updateCrosshair(cx, cy);
  }
}

function onChartUp(e)    { pointers.delete(e.pointerId); lastPinchDX = null; lastPinchDY = null; }
function onChartLeave(e) { pointers.delete(e.pointerId); if (pointers.size === 0) hideCrosshair(); }

// Mouse wheel: scroll = zoom X, shift+scroll = zoom Y
viewport.addEventListener('wheel', e => {
  e.preventDefault();
  const d = e.deltaY > 0 ? 0.85 : 1.18;
  if (e.shiftKey) { yScale = Math.max(1, yScale * d); }
  else            { barW   = Math.max(4, Math.min(120, barW * d)); }
  draw();
}, { passive: false });

// Y-axis drag → stretch yScale
let yDragStart = null, yScaleStart = null;
yPanel.addEventListener('pointerdown', e => {
  yDragStart = e.clientY; yScaleStart = yScale;
  yPanel.setPointerCapture(e.pointerId);
}, { passive: true });
yPanel.addEventListener('pointermove', e => {
  if (yDragStart === null) return;
  yScale = Math.max(0.5, yScaleStart * Math.exp((yDragStart - e.clientY) / 120));
  draw();
}, { passive: true });
yPanel.addEventListener('pointerup',    () => { yDragStart = null; });
yPanel.addEventListener('pointerleave', () => { yDragStart = null; });

// X-axis drag → stretch barW
let xDragStart = null, barWStart = null;
xPanel.addEventListener('pointerdown', e => {
  xDragStart = e.clientX; barWStart = barW;
  xPanel.setPointerCapture(e.pointerId);
}, { passive: true });
xPanel.addEventListener('pointermove', e => {
  if (xDragStart === null) return;
  barW = Math.max(4, Math.min(120, barWStart * Math.exp((e.clientX - xDragStart) / 200)));
  draw();
}, { passive: true });
xPanel.addEventListener('pointerup',    () => { xDragStart = null; });
xPanel.addEventListener('pointerleave', () => { xDragStart = null; });

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

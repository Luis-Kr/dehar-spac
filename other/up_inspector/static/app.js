/* LEAF up-vector inspector — live client-side re-fold + WebGL fisheye.
 * Re-fold math mirrors dehar.proximal_rs.leaf.recenter_leaf_data:
 *   zen = |s - up|;  az flips by pi where s < up;  h = range*cos(zen) + sensor_height.
 */
"use strict";

const TURBO = [
  [0.0,"#30123b"],[0.07,"#4145ab"],[0.14,"#4675ed"],[0.21,"#39a2fc"],
  [0.29,"#1bcfd4"],[0.36,"#24eca6"],[0.43,"#61fc6c"],[0.50,"#a4fc3b"],
  [0.57,"#d1e834"],[0.64,"#f3c63a"],[0.71,"#fe9b2d"],[0.79,"#f36315"],
  [0.86,"#d93806"],[0.93,"#b11901"],[1.0,"#7a0402"]
];
const DEG = 180 / Math.PI, RAD = Math.PI / 180, TAU = 2 * Math.PI;

const S = {
  days: [], saved: {}, corrections: [], fnIndex: {}, redoItems: [],
  mode: "new", di: 0, scan: null, D: null,
  up: 180, defUp: 180, defSrc: "manual", pLo: 2, pHi: 98, cmin: null, cmax: null,
  showDown: false, view: "points", plotInit: false,
};

const $ = (id) => document.getElementById(id);

/* ── data loading ─────────────────────────────────────────── */
async function boot() {
  const r = await (await fetch("/api/days")).json();
  S.days = r.days; S.saved = r.saved || {}; S.corrections = r.corrections || [];
  S.sh = r.sensor_height || 1.5;
  S.days.forEach((d, di) => d.scans.forEach((s) => { S.fnIndex[s.filename] = { dayIndex: di, ts: s.ts }; }));
  buildRedoItems();
  S.di = firstNewIdx();
  wireEvents();
  await loadCurrent();
}

// the already-corrected scans, resolved to their day + timestamp, sorted in time
function buildRedoItems() {
  S.redoItems = Object.keys(S.saved)
    .map((fn) => {
      const ix = S.fnIndex[fn];
      return ix ? { filename: fn, dayIndex: ix.dayIndex, ts: ix.ts,
                    up_manual: S.saved[fn].up_manual, date: S.saved[fn].date } : null;
    })
    .filter(Boolean)
    .sort((a, b) => a.ts - b.ts);
}

// resume at the first day whose representative scan has not been corrected yet
function firstNewIdx() {
  const i = S.days.findIndex((d) => !(d.rep in S.saved));
  return i < 0 ? 0 : i;
}

// nearest-in-time manual up (the manual curve); null if there are no corrections yet
function nearestUp(ts) {
  let best = null, bd = Infinity;
  for (const c of S.corrections) {
    const dd = Math.abs(c.ts - ts);
    if (dd < bd) { bd = dd; best = c.up; }
  }
  return best;
}

/* ── current-item helpers (mode-aware) ────────────────────── */
function listLen() { return S.mode === "redo" ? S.redoItems.length : S.days.length; }
function curDayIdx() { return S.mode === "redo" ? S.redoItems[S.di].dayIndex : S.di; }
function curFilename() { return S.mode === "redo" ? S.redoItems[S.di].filename : S.days[S.di].rep; }

async function loadCurrent() {
  renderDayChrome();
  if (S.mode === "redo") {
    const it = S.redoItems[S.di];
    await loadScan(it.filename, it.up_manual);   // preload the stored correction to adjust
  } else {
    await loadScan(S.days[S.di].rep, null);
  }
}

let loadSeq = 0;
async function loadScan(filename, presetUp) {
  const seq = ++loadSeq;
  S.scan = filename;
  $("loading").classList.remove("hidden");
  // binary payload: uint32 n, then float32 s[n], az[n], r1[n], r2[n] (NaN = no return)
  const buf = await (await fetch("/api/scan.bin?file=" + encodeURIComponent(filename))).arrayBuffer();
  if (seq !== loadSeq) return;                 // a newer navigation superseded this one
  const n = new Uint32Array(buf, 0, 1)[0];
  S.D = { n, sh: S.sh,
    s: new Float32Array(buf, 4, n), az: new Float32Array(buf, 4 + 4 * n, n),
    r1: new Float32Array(buf, 4 + 8 * n, n), r2: new Float32Array(buf, 4 + 12 * n, n) };
  const day = S.days[curDayIdx()];
  const scan = day.scans.find((s) => s.filename === filename);
  const nu = scan ? nearestUp(scan.ts) : null;
  S.defUp = nu != null ? nu : day.default_up;       // default = nearest manual up (else smoothed)
  S.defSrc = nu != null ? "manual" : "smooth";
  const prior = S.saved[filename];                  // this exact scan already corrected?
  S.up = presetUp != null ? presetUp : (prior ? prior.up_manual : S.defUp);
  $("up-default").textContent = S.defUp.toFixed(1);
  $("def-src").textContent = S.defSrc;
  syncScanSelect(filename);
  S.cmin = S.cmax = null;            // force a fresh colour scale for the new scan
  setUp(S.up, true);
  $("loading").classList.add("hidden");
}

/* ── re-fold + projection ─────────────────────────────────── */
function computeXYH(up) {
  const D = S.D, n = D.n, upr = up * RAD, sh = D.sh;
  const X = new Float32Array(2 * n), Y = new Float32Array(2 * n), H = new Float32Array(2 * n);
  let m = 0;
  for (let i = 0; i < n; i++) {
    const s = D.s[i];
    const zen = Math.abs(s - upr);
    let az = D.az[i]; if (s < upr) az += Math.PI; az %= TAU; if (az < 0) az += TAU;
    const zdeg = zen * DEG, cz = Math.cos(zen);
    const xx = zdeg * Math.sin(az), yy = zdeg * Math.cos(az);
    const a = D.r1[i];
    if (a === a) { X[m] = xx; Y[m] = yy; H[m] = a * cz + sh; m++; }
    const b = D.r2[i];
    if (b === b) { X[m] = xx; Y[m] = yy; H[m] = b * cz + sh; m++; }
  }
  // slice() (not subarray) so each typed array's buffer length == element count;
  // Plotly's WebGL path can misread a subarray view's oversized backing buffer.
  return { X: X.slice(0, m), Y: Y.slice(0, m), H: H.slice(0, m) };
}

function gridMeanHeight(X, Y, H, R, nb) {
  // Bin on-disk points into nb x nb cells; cell = mean height, or null if empty.
  // null cells render transparent in Plotly heatmap, so the raster is a clean disk.
  const size = (2 * R) / nb, R2 = R * R;
  const sum = new Float64Array(nb * nb), cnt = new Int32Array(nb * nb);
  for (let k = 0; k < X.length; k++) {
    const x = X[k], y = Y[k];
    if (x * x + y * y > R2) continue;
    const i = ((x + R) / size) | 0, j = ((y + R) / size) | 0;
    if (i < 0 || i >= nb || j < 0 || j >= nb) continue;
    const idx = j * nb + i; sum[idx] += H[k]; cnt[idx] += 1;
  }
  const c = new Array(nb), z = new Array(nb);
  for (let i = 0; i < nb; i++) c[i] = -R + (i + 0.5) * size;
  for (let j = 0; j < nb; j++) {
    const row = new Array(nb);
    for (let i = 0; i < nb; i++) { const idx = j * nb + i; row[i] = cnt[idx] ? sum[idx] / cnt[idx] : null; }
    z[j] = row;
  }
  return { z, c };
}

function percentile(arr, p) {
  const a = Float32Array.from(arr); a.sort();
  if (!a.length) return 0;
  const k = Math.min(a.length - 1, Math.max(0, Math.round((p / 100) * (a.length - 1))));
  return a[k];
}

/* ── plotting ─────────────────────────────────────────────── */
function fisheyeLayout() {
  const R = S.showDown ? 135 : 90;     // outer radius in zenith degrees
  const rings = S.showDown ? [30, 60, 120] : [30, 60];
  const ring = (r, bold) => ({ type: "circle", xref: "x", yref: "y",
    x0: -r, y0: -r, x1: r, y1: r,
    line: { color: bold ? "#9aa4b2" : "#d7dde5", width: bold ? 1.4 : 1 } });
  const label = (x, y, t, c) => ({ x, y, text: t, showarrow: false,
    font: { size: 12, color: c || "#9aa4b2", family: "Inter" } });
  const shapes = [
    { type: "circle", xref: "x", yref: "y", x0: -R, y0: -R, x1: R, y1: R,
      fillcolor: "#fbfcfd", line: { color: "#cbd2db", width: 1.2 }, layer: "below" },
    ...rings.map((r) => ring(r, false)),
    ring(90, true),                                   // horizon = bold anchor
    { type: "line", x0: -R, y0: 0, x1: R, y1: 0, line: { color: "#eef1f5", width: 1 } },
    { type: "line", x0: 0, y0: -R, x1: 0, y1: R, line: { color: "#eef1f5", width: 1 } },
  ];
  const annos = [
    label(0, R + 4, "N"), label(R + 4, 0, "E"), label(0, -R - 4, "S"), label(-R - 4, 0, "W"),
    label(22, 1.6, "30°"), label(52, 1.6, "60°"), label(86, 1.6, "horizon", "#7c8698"),
  ];
  if (S.showDown) annos.push(label(118, 1.6, "120°"));
  return {
    margin: { l: 8, r: 74, t: 36, b: 8 },
    paper_bgcolor: "#fff", plot_bgcolor: "#fff",
    showlegend: false, hovermode: false, dragmode: "pan",
    xaxis: { visible: false, range: [-R - 8, R + 8], scaleanchor: "y", scaleratio: 1, constrain: "domain" },
    yaxis: { visible: false, range: [-R - 8, R + 8], constrain: "domain" },
    shapes, annotations: annos,
  };
}

function render(recolor) {
  if (!S.D) return;
  const { X, Y, H } = computeXYH(S.up);
  if (recolor || S.cmin == null) {
    S.cmin = percentile(H, S.pLo); S.cmax = percentile(H, S.pHi);
    if (S.cmax <= S.cmin) S.cmax = S.cmin + 1;
    $("crange").textContent = `${S.cmin.toFixed(1)} – ${S.cmax.toFixed(1)} m`;
  }
  const R = S.showDown ? 135 : 90;
  const cbar = { title: { text: "height (m)", side: "right", font: { size: 12 } },
    thickness: 12, len: 0.7, x: 1.0, outlinewidth: 0, tickfont: { size: 11 } };
  let trace;
  if (S.view === "grid") {
    const nb = Math.min(160, Math.round((2 * R) / 1.3));  // ~1.3° cells
    const g = gridMeanHeight(X, Y, H, R, nb);   // mean height per cell; empty -> null (transparent)
    trace = {
      type: "heatmap", z: g.z, x: g.c, y: g.c, hoverinfo: "skip",
      colorscale: TURBO, zmin: S.cmin, zmax: S.cmax, zsmooth: "best", colorbar: cbar,
    };
  } else {
    trace = {
      type: "scattergl", mode: "markers", x: X, y: Y, hoverinfo: "skip",
      marker: { size: 5, color: H, colorscale: TURBO, cmin: S.cmin, cmax: S.cmax,
        colorbar: cbar, opacity: 0.8 },
    };
  }
  const day = S.days[S.di];
  const layout = fisheyeLayout();
  layout.title = { text: `${day.date}  ·  ${S.scan}  ·  up = ${S.up.toFixed(1)}°`,
    font: { size: 12.5, color: "#5b6573", family: "Inter" }, x: 0.5, y: 0.985 };
  if (!S.plotInit) {
    Plotly.newPlot("plot", [trace], layout, { responsive: true, displaylogo: false,
      modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"] });
    S.plotInit = true;
  } else {
    Plotly.react("plot", [trace], layout, { responsive: true, displaylogo: false });
  }
}

/* ── up state + UI sync ───────────────────────────────────── */
let raf = 0;
function setUp(v, recolor) {
  S.up = Math.round(Math.min(215, Math.max(165, v)) * 10) / 10;
  $("up-slider").value = S.up;
  $("up-input").value = S.up.toFixed(1);
  $("up-value").textContent = S.up.toFixed(1);
  const d = S.up - S.defUp;
  $("up-delta").textContent = `Δ ${d >= 0 ? "+" : ""}${d.toFixed(1)}°`;
  if (raf) cancelAnimationFrame(raf);
  raf = requestAnimationFrame(() => render(recolor));
}

function renderDayChrome() {
  const day = S.days[curDayIdx()];
  const curFn = curFilename();
  $("day-date").textContent = day.date;
  $("day-pos").textContent = `${S.di + 1} / ${listLen()}`;
  const tags = [];
  if (curFn in S.saved) tags.push('<span class="tag done">corrected</span>');
  tags.push(day.all_noisy
    ? '<span class="tag noisy">all flagged noisy</span>'
    : `<span class="tag clean">${day.n_clean} clean</span>`);
  $("day-tags").innerHTML = tags.join("");
  const sel = $("scan-select");
  sel.innerHTML = day.scans.map((s) => {
    const mark = s.filename in S.saved ? " · ✓" : (s.filename === day.rep ? " · rep" : "");
    return `<option value="${s.filename}">${String(s.hour).padStart(2, "0")}h · `
      + `${s.quality ? "clean" : "noisy"} · PAI ${s.pai}${mark}</option>`;
  }).join("");
  syncScanSelect(curFn);
  updateProgress();
}

function syncScanSelect(fn) { $("scan-select").value = fn; }

function updateProgress() {
  $("prog-count").textContent = S.mode === "redo" ? "redo done" : `${S.days.length} days`;
  $("prog-saved").textContent = `${Object.keys(S.saved).length} corrected`;
  $("prog-bar").style.width = `${((S.di + 1) / Math.max(1, listLen())) * 100}%`;
}

/* ── actions ──────────────────────────────────────────────── */
async function submit() {
  const day = S.days[curDayIdx()];
  const r = await (await fetch("/api/save", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ date: day.date, filename: S.scan,
      scan_hour: scanHour(S.scan), up_default_smooth: day.default_up, up_manual: S.up,
      color_p_lo: S.pLo, color_p_hi: S.pHi }) })).json();
  S.saved = r.saved; S.corrections = r.corrections;     // accumulate/overwrite into the curve
  buildRedoItems();
  nextItem(true);
}

function scanHour(fn) {
  const s = S.days[curDayIdx()].scans.find((x) => x.filename === fn);
  return s ? s.hour : "";
}

function nextItem(skipDone) {
  let i = S.di + 1;
  if (S.mode === "new" && skipDone)
    while (i < listLen() && (S.days[i].rep in S.saved)) i++;
  if (i >= listLen()) i = Math.min(S.di + 1, listLen() - 1);
  if (i === S.di && S.di === listLen() - 1) { updateProgress(); return; }
  S.di = i; loadCurrent();
}
function prevItem() { if (S.di > 0) { S.di--; loadCurrent(); } }

function setMode(mode) {
  if (mode === S.mode) return;
  if (mode === "redo" && !S.redoItems.length) { alert("No corrections yet to redo."); return; }
  S.mode = mode;
  document.querySelectorAll("#mode-seg .seg").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  S.di = mode === "redo" ? 0 : firstNewIdx();
  loadCurrent();
}

/* ── events ───────────────────────────────────────────────── */
function wireEvents() {
  $("up-slider").addEventListener("input", (e) => setUp(parseFloat(e.target.value), false));
  $("up-input").addEventListener("change", (e) => setUp(parseFloat(e.target.value), false));
  document.querySelectorAll("[data-step]").forEach((b) =>
    b.addEventListener("click", () => setUp(S.up + parseFloat(b.dataset.step), false)));
  $("reset-up").addEventListener("click", () => setUp(S.defUp, true));
  $("rescale").addEventListener("click", () => render(true));
  $("p-lo").addEventListener("change", (e) => { S.pLo = +e.target.value; render(true); });
  $("p-hi").addEventListener("change", (e) => { S.pHi = +e.target.value; render(true); });
  $("show-down").addEventListener("change", (e) => { S.showDown = e.target.checked; render(false); });
  $("view-seg").addEventListener("click", (e) => {
    const b = e.target.closest(".seg"); if (!b) return;
    S.view = b.dataset.view;
    document.querySelectorAll("#view-seg .seg").forEach((x) => x.classList.toggle("active", x === b));
    S.plotInit = false;                       // trace type changed -> full redraw
    render(false);
  });
  $("mode-seg").addEventListener("click", (e) => {
    const b = e.target.closest(".seg"); if (b) setMode(b.dataset.mode);
  });
  $("scan-select").addEventListener("change", (e) => loadScan(e.target.value, null));
  $("submit").addEventListener("click", submit);
  $("skip").addEventListener("click", () => nextItem(false));
  $("prev-day").addEventListener("click", prevItem);
  $("next-day").addEventListener("click", () => nextItem(false));
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    const big = e.shiftKey ? 1 : 0.1;
    if (e.key === "ArrowRight") { setUp(S.up + big, false); e.preventDefault(); }
    else if (e.key === "ArrowLeft") { setUp(S.up - big, false); e.preventDefault(); }
    else if (e.key === "Enter") submit();
    else if (e.key.toLowerCase() === "s") nextItem(false);
    else if (e.key === "]") nextItem(false);
    else if (e.key === "[") prevItem();
  });
}

boot();

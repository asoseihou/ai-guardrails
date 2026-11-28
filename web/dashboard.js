"use strict";
// Cost & quality dashboard: inline SVG charts built with DOM APIs (strict CSP, no
// third-party code). Labels go through textContent, never innerHTML.

const SVG = "http://www.w3.org/2000/svg";
const $ = (id) => document.getElementById(id);
const tooltip = $("tooltip");

const pct = (v) => `${(100 * v).toFixed(1)}%`;
const usd = (v) => `$${v.toFixed(2)}`;
const secs = (v) => `${v.toFixed(2)} s`;
const when = (iso) => iso.replace("T", " ").replace("+00:00", "").slice(0, 16);

function el(name, attrs = {}, parent = null, ns = SVG) {
  const node = ns ? document.createElementNS(ns, name) : document.createElement(name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  if (parent) parent.appendChild(node);
  return node;
}

function text(parent, x, y, value, cls, anchor = "start") {
  const t = el("text", { x, y, class: cls, "text-anchor": anchor }, parent);
  t.textContent = value;
  return t;
}

// ------------------------------------------------------------------ tooltip
function showTooltip(event, title, rows) {
  tooltip.replaceChildren();
  const head = el("div", { class: "tt-title" }, tooltip, null);
  head.textContent = title;
  for (const row of rows) {
    const line = el("div", { class: "tt-row" }, tooltip, null);
    if (row.cls) el("span", { class: `tt-key ${row.cls}` }, line, null);
    const value = el("strong", {}, line, null);
    value.textContent = row.value;
    const label = el("span", { class: "tt-label" }, line, null);
    label.textContent = row.label;
  }
  tooltip.hidden = false;
  const pad = 14;
  const box = tooltip.getBoundingClientRect();
  let x = event.clientX + pad;
  if (x + box.width > window.innerWidth - 8) x = event.clientX - box.width - pad;
  tooltip.style.left = `${x + window.scrollX}px`;
  tooltip.style.top = `${event.clientY + window.scrollY + pad}px`;
}
const hideTooltip = () => { tooltip.hidden = true; };

// ------------------------------------------------------------------ line chart
function niceMax(max, isPct) {
  if (isPct) return Math.min(1, Math.max(0.05, Math.ceil(max * 20) / 20));
  const step = 10 ** Math.floor(Math.log10(max || 1));
  return Math.ceil((max || 1) / step) * step;
}

function lineChart(figure, title, runs, series, format, { isPct = false, fixedMax = null, width = 360 } = {}) {
  figure.replaceChildren();
  const caption = el("figcaption", {}, figure, null);
  caption.textContent = title;
  if (series.length > 1) {
    const legend = el("div", { class: "legend" }, figure, null);
    for (const s of series) {
      const item = el("span", {}, legend, null);
      el("span", { class: `swatch line ${s.cls}` }, item, null);
      item.appendChild(document.createTextNode(s.label));
    }
  }
  const W = width, H = Math.round(width * 0.5), m = { l: 48, r: 54, t: 10, b: 28 };
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": title }, figure);
  const values = series.flatMap((s) => runs.map((r) => s.get(r)));
  const yMax = fixedMax ?? niceMax(Math.max(...values), isPct);
  const yMin = 0;
  const n = runs.length;
  const x = (i) => m.l + (n === 1 ? (W - m.l - m.r) / 2 : (i * (W - m.l - m.r)) / (n - 1));
  const y = (v) => m.t + (1 - (v - yMin) / (yMax - yMin)) * (H - m.t - m.b);
  for (let k = 0; k <= 4; k++) {
    const v = yMin + (k * (yMax - yMin)) / 4;
    el("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), class: k === 0 ? "axis" : "grid" }, svg);
    text(svg, m.l - 6, y(v) + 4, format(v), "tick", "end");
  }
  text(svg, m.l, H - 8, `run 1`, "tick");
  text(svg, W - m.r, H - 8, `run ${n}`, "tick", "end");
  for (const s of series) {
    const points = runs.map((r, i) => `${x(i)},${y(s.get(r))}`).join(" ");
    if (n > 1) el("polyline", { points, class: `series-line ${s.cls}` }, svg);
    runs.forEach((r, i) => el("circle", { cx: x(i), cy: y(s.get(r)), r: 4, class: `series-dot ${s.cls}` }, svg));
    const last = runs[n - 1];
    text(svg, x(n - 1) + 8, y(s.get(last)) + 4, format(s.get(last)), "end-label");
  }
  const hair = el("line", { y1: m.t, y2: H - m.b, class: "crosshair", visibility: "hidden" }, svg);
  const hit = el("rect", { x: m.l - 10, y: 0, width: W - m.l - m.r + 20, height: H, class: "hit", tabindex: 0 }, svg);
  const nearest = (event) => {
    const box = svg.getBoundingClientRect();
    const px = ((event.clientX - box.left) / box.width) * W;
    let best = 0;
    runs.forEach((_, i) => { if (Math.abs(x(i) - px) < Math.abs(x(best) - px)) best = i; });
    return best;
  };
  const move = (event, i) => {
    hair.setAttribute("x1", x(i));
    hair.setAttribute("x2", x(i));
    hair.setAttribute("visibility", "visible");
    const r = runs[i];
    showTooltip(event, `Run ${i + 1} · ${when(r.created_at)} · ${r.command} (noise ×${r.config.noise ?? 1})`,
      series.map((s) => ({ cls: s.cls, value: format(s.get(r)), label: s.label })));
  };
  hit.addEventListener("pointermove", (e) => move(e, nearest(e)));
  hit.addEventListener("pointerleave", () => { hair.setAttribute("visibility", "hidden"); hideTooltip(); });
  hit.addEventListener("focus", () => {
    const box = svg.getBoundingClientRect();
    move({ clientX: box.right - 20, clientY: box.top + 20 }, n - 1);
  });
  hit.addEventListener("blur", () => { hair.setAttribute("visibility", "hidden"); hideTooltip(); });
}

// ------------------------------------------------------------------ per-field bars
function fieldBars(figure, run) {
  figure.replaceChildren();
  const caption = el("figcaption", {}, figure, null);
  caption.textContent = `Error rate per field · run of ${when(run.created_at)}`;
  const series = [
    { key: "raw", label: "Raw model output", cls: "s2", get: (f) => 1 - f.raw_accuracy },
    { key: "final", label: "Published after guardrails", cls: "s1", get: (f) => 1 - f.final_accuracy },
  ];
  const legend = el("div", { class: "legend" }, figure, null);
  for (const s of series) {
    const item = el("span", {}, legend, null);
    el("span", { class: `swatch ${s.cls}` }, item, null);
    item.appendChild(document.createTextNode(s.label));
  }
  const fields = Object.entries(run.per_field).sort((a, b) => (1 - b[1].raw_accuracy) - (1 - a[1].raw_accuracy));
  const row = 30, bar = 10, gap = 2, m = { l: 150, r: 56, t: 6, b: 24 };
  const W = 760, H = m.t + m.b + fields.length * row;
  const max = niceMax(Math.max(0.01, ...fields.map(([, f]) => 1 - f.raw_accuracy)), true);
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Error rate per field" }, figure);
  const x = (v) => m.l + (v / max) * (W - m.l - m.r);
  for (let k = 0; k <= 4; k++) {
    const v = (k * max) / 4;
    el("line", { x1: x(v), x2: x(v), y1: m.t, y2: H - m.b, class: k === 0 ? "axis" : "grid" }, svg);
    text(svg, x(v), H - 8, pct(v), "tick", "middle");
  }
  fields.forEach(([name, f], i) => {
    const top = m.t + i * row + (row - 2 * bar - gap) / 2;
    text(svg, m.l - 8, top + bar + 4, name, "tick", "end");
    series.forEach((s, j) => {
      const v = s.get(f);
      const y0 = top + j * (bar + gap);
      const width = Math.max(v > 0 ? 2 : 0, x(v) - x(0));
      el("rect", { x: x(0), y: y0, width, height: bar, rx: Math.min(2, width / 2), class: `bar ${s.cls}` }, svg);
      if (j === 0 || v > 0) text(svg, x(0) + width + 4, y0 + bar - 1, pct(v), "bar-label");
    });
    const hit = el("rect", { x: 0, y: m.t + i * row, width: W, height: row, class: "hit", tabindex: 0 }, svg);
    const tip = (e) => showTooltip(e, name, series.map((s) => ({ cls: s.cls, value: pct(s.get(f)), label: s.label })));
    hit.addEventListener("pointermove", tip);
    hit.addEventListener("pointerleave", hideTooltip);
    hit.addEventListener("focus", () => { const b = hit.getBoundingClientRect(); tip({ clientX: b.left + 200, clientY: b.top }); });
    hit.addEventListener("blur", hideTooltip);
  });
}

// ------------------------------------------------------------------ KPIs and table
function kpis(runs) {
  const last = runs[runs.length - 1];
  const prev = runs.length > 1 ? runs[runs.length - 2] : null;
  const tiles = [
    ["Field accuracy", last.field_accuracy, pct, true],
    ["Silent-error documents", last.silent_error_rate, pct, false],
    ["Sent to human review", last.review_rate, pct, false],
    ["Cost / 1,000 docs", 1000 * last.cost_per_doc, usd, false],
    ["Latency p95", last.latency_p95, secs, false],
  ];
  const root = $("kpis");
  root.replaceChildren();
  const prevValue = { "Field accuracy": prev?.field_accuracy, "Silent-error documents": prev?.silent_error_rate,
    "Sent to human review": prev?.review_rate, "Cost / 1,000 docs": prev ? 1000 * prev.cost_per_doc : null,
    "Latency p95": prev?.latency_p95 };
  for (const [label, value, format, higherIsBetter] of tiles) {
    const tile = el("div", { class: "kpi" }, root, null);
    const l = el("div", { class: "kpi-label" }, tile, null);
    l.textContent = label;
    const v = el("div", { class: "kpi-value" }, tile, null);
    v.textContent = format(value);
    const before = prevValue[label];
    const d = el("div", { class: "kpi-delta" }, tile, null);
    if (before === null || before === undefined) {
      d.textContent = "first run";
    } else {
      const diff = value - before;
      const better = Math.abs(diff) < 1e-9 ? null : (diff > 0) === higherIsBetter;
      d.textContent = better === null
        ? "no change vs previous run"
        : `${diff > 0 ? "▲" : "▼"} ${format(Math.abs(diff))} vs previous run`;
      if (better !== null) d.classList.add(better ? "better" : "worse");
    }
  }
}

function runsTable(runs) {
  const table = $("runs-table");
  table.replaceChildren();
  const head = el("tr", {}, el("thead", {}, table, null), null);
  const cols = ["#", "Recorded (UTC)", "Command", "Noise", "Seed", "Docs", "Field accuracy",
    "Straight-through", "Review", "Silent errors", "Cost / doc", "p50", "p95", "Retries"];
  for (const c of cols) el("th", {}, head, null).textContent = c;
  const body = el("tbody", {}, table, null);
  runs.forEach((r, i) => {
    const tr = el("tr", {}, body, null);
    const cells = [i + 1, when(r.created_at), r.command, `×${r.config.noise ?? 1}`, r.config.seed ?? "—", r.n_docs,
      pct(r.field_accuracy), pct(r.straight_through), pct(r.review_rate), pct(r.silent_error_rate),
      `$${r.cost_per_doc.toFixed(5)}`, secs(r.latency_p50), secs(r.latency_p95), r.retries];
    for (const c of cells) el("td", {}, tr, null).textContent = String(c);
  });
}

async function main() {
  const res = await fetch("/metrics/runs?limit=200");
  const { runs } = await res.json();
  if (!runs.length) {
    $("empty").hidden = false;
    for (const s of document.querySelectorAll("main > section:not(#empty)")) s.hidden = true;
    return;
  }
  const last = runs[runs.length - 1];
  const provider = last.config.simulated ? "simulated LLM" : `${last.config.provider} · ${last.config.model}`;
  $("run-scope").textContent = `${runs.length} run(s) · latest ${when(last.created_at)} · ${provider}`;
  kpis(runs);
  const one = (label, get) => [{ label, cls: "s1", get }];
  const small = { isPct: true, width: 280 };
  lineChart($("chart-accuracy"), "Field accuracy", runs, one("Field accuracy", (r) => r.field_accuracy), pct, { ...small, fixedMax: 1 });
  lineChart($("chart-review"), "Documents sent to review", runs, one("Review rate", (r) => r.review_rate), pct, small);
  lineChart($("chart-silent"), "Silent-error documents", runs, one("Silent errors", (r) => r.silent_error_rate), pct, small);
  lineChart($("chart-cost"), "Cost per 1,000 documents (USD)", runs, one("Cost / 1,000", (r) => 1000 * r.cost_per_doc), usd, { width: 420 });
  lineChart($("chart-latency"), "Latency per document (s)", runs, [
    { label: "p50", cls: "s1", get: (r) => r.latency_p50 },
    { label: "p95", cls: "s2", get: (r) => r.latency_p95 },
  ], secs, { width: 420 });
  fieldBars($("chart-fields"), last);
  runsTable(runs);
}

main().catch(() => { $("empty").hidden = false; });

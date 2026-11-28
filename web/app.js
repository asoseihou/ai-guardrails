"use strict";
// Single-page UI. No inline script or style (strict CSP); every value coming from
// the API is escaped before it reaches innerHTML. The review token lives in memory
// (and, only if the user asks, in sessionStorage); it is never put in a URL.

const $ = (id) => document.getElementById(id);
const TOKEN_KEY = "docai-review-token";
let apiToken = "";
let lastUpload = null; // the files of the last extraction, to send them to the review queue
let currentItem = null;

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

function show(value) {
  if (value === null || value === undefined) return '<span class="muted">null</span>';
  if (Array.isArray(value)) {
    return value.map((v) => {
      if (v && typeof v === "object") {
        const label = v.name ?? v.description ?? "";
        return `<code>${escapeHtml(label)}: ${escapeHtml(v.value)}</code>`;
      }
      return `<code>${escapeHtml(v)}</code>`;
    }).join(" ");
  }
  return `<code>${escapeHtml(value)}</code>`;
}

function errorText(body) {
  if (!body) return "request failed";
  const detail = body.detail;
  if (detail && Array.isArray(detail.reasons)) return `${detail.message}: ${detail.reasons.join("; ")}`;
  if (Array.isArray(detail)) return detail.map((d) => `${(d.loc || []).join(".")}: ${d.msg}`).join("; ");
  const ref = body.request_id ? ` (request ${body.request_id})` : "";
  return `${typeof detail === "string" ? detail : JSON.stringify(detail)}${ref}`;
}

// ------------------------------------------------------------------ tabs
function selectTab(name) {
  for (const tab of document.querySelectorAll(".tab[data-tab]")) {
    const active = tab.dataset.tab === name;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
    $(`tab-${tab.dataset.tab}`).hidden = !active;
  }
  if (name === "review") loadQueue();
}
for (const tab of document.querySelectorAll(".tab[data-tab]")) {
  tab.addEventListener("click", () => selectTab(tab.dataset.tab));
}

// ------------------------------------------------------------------ health and samples
async function loadHealth() {
  try {
    const h = await (await fetch("/health")).json();
    const provider = h.provider.simulated ? "simulated LLM" : `${h.provider.name} · ${h.provider.model}`;
    $("health").innerHTML =
      `provider <b>${escapeHtml(provider)}</b> · classifier <b>${escapeHtml(h.classifier)}</b>` +
      `<br>guardrails <b>${escapeHtml(h.guardrails)}</b> · v${escapeHtml(h.version)}`;
  } catch {
    $("health").textContent = "API unreachable";
  }
}

async function loadSamples() {
  try {
    const samples = await (await fetch("/samples")).json();
    for (const sample of samples) {
      const opt = document.createElement("option");
      opt.value = sample.name;
      opt.dataset.hasXml = sample.has_xml ? "1" : "";
      opt.textContent = sample.has_xml ? `${sample.name}  (+ XML)` : sample.name;
      $("sample").appendChild(opt);
    }
  } catch { /* samples are optional */ }
}

// ------------------------------------------------------------------ extraction
function renderExtraction(r) {
  const probs = Object.entries(r.probabilities || {})
    .sort((a, b) => b[1] - a[1]).slice(0, 3)
    .map(([k, v]) => `${k} ${(100 * v).toFixed(1)}%`).join(" · ");
  let html = `<div class="summary">
    <span class="pill">type <b>${escapeHtml(r.doc_type)}</b></span>
    ${r.source ? `<span class="pill">source <b>${escapeHtml(r.source)}</b></span>` : ""}
    <span class="muted">${escapeHtml(probs)} (${escapeHtml(r.classifier)})</span>`;
  if (r.extracted) {
    html += r.needs_review
      ? '<span class="pill review">needs human review</span>'
      : '<span class="pill pass">straight-through</span>';
  } else {
    html += '<span class="pill">not an extractable fiscal document</span>';
  }
  html += "</div>";
  if (!r.extracted) return html;
  for (const reason of r.review_reasons || []) html += `<p class="error">${escapeHtml(reason)}</p>`;
  const injected = (r.meta && r.meta.injected_errors) || [];
  if (injected.length) {
    html += `<p class="muted">Simulated LLM injected ${injected.length} error(s): ` +
      injected.map((e) => `<code>${escapeHtml(e.type)}</code> on <code>${escapeHtml(e.field)}</code>`).join(", ") + "</p>";
  }
  if (r.meta && r.meta.retried) {
    html += `<p class="muted">The first response was truncated; recovered with ${escapeHtml(r.meta.attempts)} attempts.</p>`;
  }
  html += "<table><thead><tr><th>Field</th><th>Model output</th><th>Published</th><th>Status</th><th>Why</th></tr></thead><tbody>";
  for (const f of r.fields) {
    html += `<tr><td>${escapeHtml(f.name)}</td><td>${show(f.raw)}</td><td>${show(f.final)}</td>
      <td><span class="status ${escapeHtml(f.status)}">${escapeHtml(f.status)}</span></td>
      <td class="reason">${f.reasons.map(escapeHtml).join("<br>")}</td></tr>`;
  }
  html += "</tbody></table>";
  if (r.usage) {
    const u = r.usage;
    html += `<p class="muted">${u.input_tokens} input / ${u.output_tokens} output tokens · ` +
      `${u.latency_s}s · $${u.cost_usd.toFixed(5)}${u.simulated ? " (simulated)" : ""}</p>`;
  } else if (r.source === "xml") {
    html += '<p class="muted">Read from the fiscal XML: no LLM call.</p>';
  }
  if (r.needs_review) {
    html += '<button type="button" id="send-review" class="secondary">Send to review queue</button> <span id="send-review-state" class="muted"></span>';
  }
  return html;
}

async function getFiles() {
  const chosen = $("file").files[0];
  const xmlChosen = $("xml").files[0] || null;
  if (chosen) return { pdf: chosen, xml: xmlChosen };
  const option = $("sample").selectedOptions[0];
  const sample = $("sample").value;
  if (!sample) return { pdf: null, xml: null };
  const base = `/samples/${encodeURIComponent(sample)}`;
  const pdf = new File([await (await fetch(base)).blob()], sample, { type: "application/pdf" });
  let xml = xmlChosen;
  if (!xml && option.dataset.hasXml && $("use-xml").checked) {
    xml = new File([await (await fetch(`${base}/xml`)).blob()], "document.xml", { type: "application/xml" });
  }
  return { pdf, xml };
}

$("extract-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const out = $("extract-result");
  const { pdf, xml } = await getFiles();
  if (!pdf) {
    out.hidden = false;
    out.innerHTML = '<p class="error">Choose a PDF or a sample first.</p>';
    return;
  }
  const form = new FormData();
  form.append("file", pdf);
  if (xml) form.append("xml", xml);
  form.append("noise", $("noise").value);
  $("extract-btn").disabled = true;
  try {
    const res = await fetch("/extract", { method: "POST", body: form });
    const body = await res.json();
    out.innerHTML = res.ok ? renderExtraction(body) : `<p class="error">${escapeHtml(errorText(body))}</p>`;
    lastUpload = res.ok ? { pdf, xml } : null;
  } catch (err) {
    out.innerHTML = `<p class="error">${escapeHtml(err)}</p>`;
  } finally {
    out.hidden = false;
    $("extract-btn").disabled = false;
  }
});

$("extract-result").addEventListener("click", async (event) => {
  if (event.target.id !== "send-review" || !lastUpload) return;
  const state = $("send-review-state");
  if (!apiToken) {
    state.textContent = "set the API token in the Review queue tab first";
    return;
  }
  const form = new FormData();
  form.append("file", lastUpload.pdf);
  if (lastUpload.xml) form.append("xml", lastUpload.xml);
  const res = await fetch("/review/items", { method: "POST", body: form, headers: authHeaders() });
  const body = await res.json();
  state.textContent = res.ok
    ? (body.item_id ? `queued as ${body.item_id}` : body.detail)
    : errorText(body);
  refreshPendingCount();
});

// ------------------------------------------------------------------ ask
$("ask-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const out = $("ask-result");
  const res = await fetch("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question: $("question").value, k: 3 }),
  });
  const a = await res.json();
  if (!res.ok) {
    out.innerHTML = `<p class="error">${escapeHtml(errorText(a))}</p>`;
  } else {
    out.innerHTML = `<p><b>Answer</b> <span class="muted">(${escapeHtml(a.mode)})</span><br>${escapeHtml(a.answer)}</p>` +
      a.passages.map((p) => `<div class="passage"><div class="cite">[${escapeHtml(p.rank)}] ${escapeHtml(p.title)} — <code>${escapeHtml(p.citation)}</code></div><div class="ptext">${escapeHtml(p.text)}</div></div>`).join("");
  }
  out.hidden = false;
});

// ------------------------------------------------------------------ review queue
function authHeaders(extra = {}) {
  return apiToken ? { ...extra, Authorization: `Bearer ${apiToken}` } : extra;
}

function setToken(value) {
  apiToken = value.trim();
  $("token-state").textContent = apiToken ? "token set (kept in memory)" : "not set";
  try {
    if ($("remember-token").checked && apiToken) sessionStorage.setItem(TOKEN_KEY, apiToken);
    else sessionStorage.removeItem(TOKEN_KEY);
  } catch { /* storage may be unavailable */ }
}

$("token").addEventListener("change", () => setToken($("token").value));
$("remember-token").addEventListener("change", () => setToken($("token").value));

async function refreshPendingCount() {
  try {
    const res = await fetch("/review/items?limit=1", { headers: authHeaders() });
    if (!res.ok) return;
    const body = await res.json();
    $("pending-count").textContent = body.pending ? String(body.pending) : "";
  } catch { /* optional */ }
}

async function loadQueue() {
  const out = $("review-list");
  const res = await fetch(`/review/items?status=${encodeURIComponent($("review-status").value)}`, { headers: authHeaders() });
  const body = await res.json();
  if (!res.ok) {
    out.innerHTML = `<p class="error">${escapeHtml(errorText(body))}</p>`;
    return;
  }
  $("pending-count").textContent = body.pending ? String(body.pending) : "";
  if (!body.items.length) {
    out.innerHTML = '<p class="muted">Nothing here. Extract a document that needs review and send it to the queue, or run <code>python -m docai review enqueue samples/pdf/*.pdf</code>.</p>';
    return;
  }
  out.innerHTML = "<table><thead><tr><th>Created (UTC)</th><th>Type</th><th>File</th><th>Source</th><th>Open fields</th><th>Status</th></tr></thead><tbody>" +
    body.items.map((i) => `<tr class="clickable" data-item="${escapeHtml(i.id)}" tabindex="0">
      <td>${escapeHtml(i.created_at.replace("T", " ").replace("+00:00", ""))}</td><td>${escapeHtml(i.doc_type)}</td>
      <td>${escapeHtml(i.filename || "—")}</td><td>${escapeHtml(i.source)}</td>
      <td>${i.pending_fields.map((f) => `<code>${escapeHtml(f)}</code>`).join(" ") || '<span class="muted">none</span>'}</td>
      <td><span class="pill ${i.status === "approved" ? "pass" : "review"}">${escapeHtml(i.status)}</span></td></tr>`).join("") +
    "</tbody></table>";
}

function openItemFrom(event) {
  const row = event.target.closest("tr[data-item]");
  if (row) loadItem(row.dataset.item);
}
$("review-list").addEventListener("click", openItemFrom);
$("review-list").addEventListener("keydown", (e) => { if (e.key === "Enter") openItemFrom(e); });
$("review-status").addEventListener("change", loadQueue);
$("review-refresh").addEventListener("click", loadQueue);

function reasonsFor(item, field) {
  return item.findings.filter((f) => f.field === field).map((f) => `${f.check}: ${f.reason}`);
}

function renderItem(item) {
  const editable = item.status === "pending";
  let html = `<h3>Item <code>${escapeHtml(item.id)}</code> · ${escapeHtml(item.doc_type)} · ${escapeHtml(item.filename || "")}</h3>`;
  for (const reason of item.review_reasons) html += `<p class="error">${escapeHtml(reason)}</p>`;
  html += "<table><thead><tr><th>Field</th><th>Value</th><th>Status</th><th>Why it is here</th></tr></thead><tbody>";
  for (const [name, value] of Object.entries(item.record)) {
    const status = item.field_status[name] || "ok";
    html += `<tr><td>${escapeHtml(name)}</td><td>${show(value)}</td>
      <td><span class="status ${escapeHtml(status)}">${escapeHtml(status)}</span></td>
      <td class="reason">${reasonsFor(item, name).map(escapeHtml).join("<br>")}</td></tr>`;
  }
  html += "</tbody></table>";
  if (editable) {
    const options = Object.keys(item.record).map((f) =>
      `<option value="${escapeHtml(f)}">${escapeHtml(f)}${item.pending_fields.includes(f) ? " (open)" : ""}</option>`).join("");
    html += `<form id="correct-form" class="review-form">
      <div class="row"><label for="c-field">Field</label><select id="c-field">${options}</select></div>
      <div class="row"><label for="c-value">Value as printed</label><input type="text" id="c-value" required maxlength="5000" placeholder='text, number or JSON list, e.g. ["3526…"]'></div>
      <div class="row"><label for="c-reason">Reason</label><input type="text" id="c-reason" required minlength="3" maxlength="500" placeholder="read from the DESTINATÁRIO block"></div>
      <div class="row"><label for="c-reviewer">Reviewer</label><input type="text" id="c-reviewer" required maxlength="64" placeholder="your name"></div>
      <button type="submit">Save correction</button>
      <button type="button" id="approve-btn" class="secondary">Approve record</button>
      <span id="review-state" class="muted"></span></form>`;
  }
  html += "<h4>Audit trail (append-only)</h4><ol class=\"audit\">" + item.audit.map((a) =>
    `<li><span class="muted">${escapeHtml(a.at)}</span> <b>${escapeHtml(a.actor)}</b> ${escapeHtml(a.action)}` +
    (a.field ? ` <code>${escapeHtml(a.field)}</code>: ${show(a.before)} → ${show(a.after)}` : "") +
    (a.reason ? ` <span class="muted">— ${escapeHtml(a.reason)}</span>` : "") + "</li>").join("") + "</ol>";
  return html;
}

async function loadItem(id) {
  const out = $("review-detail");
  const res = await fetch(`/review/items/${encodeURIComponent(id)}`, { headers: authHeaders() });
  const body = await res.json();
  out.hidden = false;
  if (!res.ok) {
    out.innerHTML = `<p class="error">${escapeHtml(errorText(body))}</p>`;
    return;
  }
  currentItem = body;
  out.innerHTML = renderItem(body);
}

function parseValue(raw) {
  const text = raw.trim();
  if (text.startsWith("[") || text.startsWith("{")) {
    try { return JSON.parse(text); } catch { return text; }
  }
  return text;
}

async function postReview(path, payload) {
  const res = await fetch(path, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(payload),
  });
  return { res, body: await res.json() };
}

$("review-detail").addEventListener("submit", async (event) => {
  if (event.target.id !== "correct-form" || !currentItem) return;
  event.preventDefault();
  const { res, body } = await postReview(`/review/items/${encodeURIComponent(currentItem.id)}/corrections`, {
    field: $("c-field").value,
    value: parseValue($("c-value").value),
    reason: $("c-reason").value,
    reviewer: $("c-reviewer").value,
  });
  if (res.ok) {
    currentItem = body;
    $("review-detail").innerHTML = renderItem(body);
    $("review-state").textContent = "saved";
    loadQueue();
  } else {
    $("review-state").textContent = errorText(body);
  }
});

$("review-detail").addEventListener("click", async (event) => {
  if (event.target.id !== "approve-btn" || !currentItem) return;
  const reviewer = $("c-reviewer").value || "";
  const { res, body } = await postReview(`/review/items/${encodeURIComponent(currentItem.id)}/approve`, {
    reviewer, reason: $("c-reason").value || "approved after review",
  });
  if (res.ok) {
    currentItem = body;
    $("review-detail").innerHTML = renderItem(body);
    loadQueue();
  } else {
    $("review-state").textContent = errorText(body);
  }
});

// ------------------------------------------------------------------ misc
const drop = $("drop");
$("file").addEventListener("change", () => {
  const f = $("file").files[0];
  $("drop-text").textContent = f ? f.name : "Drop a PDF here or click to choose";
});
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  if (e.dataTransfer.files.length) {
    $("file").files = e.dataTransfer.files;
    $("drop-text").textContent = e.dataTransfer.files[0].name;
  }
});
$("noise").addEventListener("input", () => { $("noise-value").textContent = Number($("noise").value).toFixed(1); });

try {
  const saved = sessionStorage.getItem(TOKEN_KEY);
  if (saved) {
    $("token").value = saved;
    $("remember-token").checked = true;
    setToken(saved);
  }
} catch { /* storage may be unavailable */ }
if (location.hash === "#review") selectTab("review");
loadHealth();
loadSamples();
refreshPendingCount();

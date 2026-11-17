/* Run Instructions:
  This script is executed dynamically by the browser when loaded via search.html.
  To apply changes, save this file and hard-refresh your browser (Ctrl+F5).
*/

let PAGE = 1;
const PAGE_SIZE = 20;
let TOTAL_PAGES = 1;

// --------------------------
// Utils
// --------------------------
function esc(s) {
  return String(s || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function debounce(fn, delayMs) {
  let t = null;
  return (...args) => {
    if (t) clearTimeout(t);
    t = setTimeout(() => fn(...args), delayMs);
  };
}

function tokenizeText(t) {
  return String(t || "")
    .toLowerCase()
    .split(/[^a-z0-9]+/g)
    .filter(x => x.length > 2);
}

function highlight(text, query) {
  const qTokens = new Set(tokenizeText(query));
  if (!qTokens.size) return esc(text);
  const words = String(text || "").split(/(\b)/);
  return words
    .map(w => {
      const k = w.toLowerCase();
      if (qTokens.has(k)) return `<mark>${esc(w)}</mark>`;
      return esc(w);
    })
    .join("");
}

function snippet(s, n = 240) {
  const t = String(s || "").replace(/\s+/g, " ").trim();
  if (!t) return "";
  if (t.length <= n) return t;
  return t.slice(0, n - 1) + "…";
}

async function jpost(url, body) {
  url = (window.asaUrl && typeof window.asaUrl === "function") ? window.asaUrl(url) : url;
  const r = await ((window.asaFetch && typeof window.asaFetch === "function") ? window.asaFetch : fetch)(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const j = await r.json().catch(() => null);
  return { ok: r.ok, status: r.status, json: j };
}

async function jget(url) {
  url = (window.asaUrl && typeof window.asaUrl === "function") ? window.asaUrl(url) : url;
  const r = await ((window.asaFetch && typeof window.asaFetch === "function") ? window.asaFetch : fetch)(url);
  const j = await r.json().catch(() => null);
  return { ok: r.ok, status: r.status, json: j };
}

let _LOGGED_IN = null;
async function isLoggedIn() {
  if (_LOGGED_IN !== null) return _LOGGED_IN;
  const r = await jget("/api/bookmarks/me");
  _LOGGED_IN = !!(r.ok && r.json && r.json.logged_in);
  return _LOGGED_IN;
}

function truncateWithEllipsis(s, n) {
  const t = String(s || "").replace(/\s+/g, " ").trim();
  if (!t) return "";
  if (t.length <= n) return t;
  return t.slice(0, Math.max(0, n - 1)) + "…";
}

function normalizeName(x) {
  if (!x) return "";
  if (typeof x === "string") return x.trim();
  if (typeof x === "object") return String(x.name || x.display_name || x.institution || x.author || x.venue || "").trim();
  return "";
}

function listText(x, sep = ", ") {
  if (!x) return "";
  if (Array.isArray(x)) return x.map(normalizeName).filter(Boolean).join(sep);
  return normalizeName(x);
}

function bestUrl(p) {
  const doi = p.doi ? String(p.doi).replace("https://doi.org/", "") : "";
  const doiUrl = doi ? `https://doi.org/${doi}` : "";
  return (
    p.landing_page_url ||
    (p.primary_location && (p.primary_location.landing_page_url || p.primary_location.url)) ||
    (p.open_access && p.open_access.oa_url) ||
    doiUrl ||
    ""
  );
}

function bestPdf(p) {
  return (
    p.pdf_url ||
    (p.best_oa_location && p.best_oa_location.pdf_url) ||
    (p.primary_location && p.primary_location.pdf_url) ||
    ""
  );
}

function venueText(p) {
  const v = p.venue || p.host_venue || (p.primary_location && (p.primary_location.source || p.primary_location.host_venue)) || "";
  if (typeof v === "string") return v;
  if (v && typeof v === "object") return v.display_name || v.name || "";
  return "";
}

function paperTypeBadge(p) {
  const t = (p.paper_type || p.type || p.publication_type || "").toString().trim();
  if (!t) return "";
  return `<span class="badge type">[${esc(t.toUpperCase())}]</span>`;
}

function _normalizedTypeUpper(p) {
  const t = (p.paper_type || p.type || p.publication_type || "").toString().trim();
  return t ? t.toUpperCase() : "";
}

function featureBadges(p) {
  // De-duplicate feature badges against the paper-type badge.
  // Example bug: paperTypeBadge == [DATASET] and p.has_dataset == true -> show [DATASET] twice.
  console.assert(p && typeof p === "object", "featureBadges expects paper object");

  const typeU = _normalizedTypeUpper(p);
  const typeHasDataset = typeU === "DATASET" || typeU.includes("DATASET");
  const typeHasBench = typeU === "BENCH" || typeU === "BENCHMARK" || typeU.includes("BENCH");
  const typeHasCode = typeU === "CODE" || typeU.includes("CODE");

  const out = [];

  if (p.has_code && !typeHasCode) out.push(`<span class="badge" style="background:#10b981;color:#fff;">[CODE]</span>`);
  if (p.has_dataset && !typeHasDataset) out.push(`<span class="badge" style="background:#3b82f6;color:#fff;">[DATASET]</span>`);
  if (p.has_benchmark && !typeHasBench) out.push(`<span class="badge" style="background:#f59e0b;color:#111;">[BENCH]</span>`);

  return out.join("");
}

function tagBadges(p) {
  const out = [];
  const fields = ["species", "modalities", "features", "stages", "domains", "contexts"];
  for (const f of fields) {
    const arr = p[f];
    if (!arr) continue;
    if (Array.isArray(arr)) {
      for (const val of arr) {
        const s = normalizeName(val);
        if (s) out.push(`<span class="tag-badge">${esc(s)}</span>`);
      }
    } else if (typeof arr === "string") {
      const s = arr.trim();
      if (s) out.push(`<span class="tag-badge">${esc(s)}</span>`);
    }
  }
  return out.join(" ");
}

// --------------------------
// Citation.js
// --------------------------
function ensureCitationJsLoaded() {
  // We ship citation-js locally under /static/vendor.
  // Some bundles expose `require('citation-js')` but don't attach `window.Cite`.
  if (window.Cite) return true;
  try {
    if (typeof require === "function") {
      const mod = require("citation-js");
      if (mod) window.Cite = mod;
    }
  } catch (e) {
    // ignore
  }
  if (!window.Cite) {
    console.warn("[Citation Debug] citation-js not available (window.Cite missing); using fallback.");
    return false;
  }
  return true;
}

function buildCslJson(p) {
  const title = p.title || p.display_name || "";
  const year = p.year;
  const venue = venueText(p);

  const authors = [];
  if (Array.isArray(p.authors)) {
    for (const a of p.authors) {
      const name = normalizeName(a);
      if (!name) continue;
      const parts = name.split(/\s+/);
      authors.push({ given: parts.slice(0, -1).join(" "), family: parts.slice(-1).join(" ") });
    }
  }

  const doi = p.doi ? String(p.doi).replace("https://doi.org/", "") : undefined;
  const url = bestUrl(p) || undefined;

  const pt = String(p.paper_type || p.type || "").toLowerCase();
  let cslType = "article-journal";
  if (pt.includes("conference") || pt.includes("proceeding")) cslType = "paper-conference";
  else if (pt.includes("preprint")) cslType = "manuscript";
  else if (pt.includes("book")) cslType = "book";

  return [{
    id: p.doc_id || ("item-" + Math.random().toString(36).slice(2)),
    type: cslType,
    title,
    author: authors.length ? authors : undefined,
    issued: year ? { "date-parts": [[Number(year)]] } : undefined,
    "container-title": venue || undefined,
    DOI: doi,
    URL: url,
  }];
}

async function getCitationText(p, format) {
  const fallback = `${p.title || p.display_name || ""} (${p.year || ""}). ${bestUrl(p)}`.trim();
  const ok = ensureCitationJsLoaded();
  if (!ok) return fallback;

  try {
    const cite = new window.Cite(buildCslJson(p));
    if (format === "bibtex") return cite.format("bibtex").trim();

    return cite.format("bibliography", {
      format: "text",
      template: format,
      lang: "en-US",
    }).trim();
  } catch (err) {
    console.error(`[Citation Debug] citation-js error for paper ${p.doc_id}:`, err, p);
    return fallback;
  }
}

// (pager visibility handled by the setPagerVisible() near buildParams)

// --------------------------
// Facets
// --------------------------
async function loadFacet(field, elementId, limit = null) {
  const el = document.getElementById(elementId);
  if (!el) return;

  try {
    const url = `/api/facets?field=${encodeURIComponent(field)}` + (limit ? `&limit=${encodeURIComponent(String(limit))}` : "");
    const resp = await ((window.asaFetch && typeof window.asaFetch === "function") ? window.asaFetch : fetch)(url);
    const data = await resp.json();

    el.innerHTML = "";
    if (resp.ok && Array.isArray(data.values)) {
      for (const v of data.values) {
        if (!v) continue;
        const opt = document.createElement("option");
        opt.value = v;
        opt.textContent = v;
        el.appendChild(opt);
      }
    }
  } catch (e) {
    console.error("Facet load failed", field, e);
  }
}

function initSelect2Static(selectId, { placeholder = "", allowClear = true } = {}) {
  const el = document.getElementById(selectId);
  if (!el || !window.jQuery || !window.jQuery.fn || !window.jQuery.fn.select2) return;
  const $el = window.jQuery(el);
  $el.select2({
    width: "100%",
    placeholder,
    allowClear,
    closeOnSelect: false,
  });
}

function initSelect2Ajax(selectId, field, {
  placeholder = "Start typing…",
  minimumInputLength = 3,
  delay = 320,
  multiple = true,
  allowClear = true,
  limit = 50,
} = {}) {
  const el = document.getElementById(selectId);
  if (!el || !window.jQuery || !window.jQuery.fn || !window.jQuery.fn.select2) return;

  const $el = window.jQuery(el);

  $el.select2({
    width: "100%",
    placeholder,
    allowClear,
    multiple,
    closeOnSelect: !multiple,
    minimumInputLength,
    ajax: {
      delay,
      transport: function (params, success, failure) {
        const q = (params.data && params.data.term) ? String(params.data.term) : "";
        const url = `/api/facets/search?field=${encodeURIComponent(field)}&q=${encodeURIComponent(q)}&limit=${encodeURIComponent(String(limit))}`;
        ((window.asaFetch && typeof window.asaFetch === "function") ? window.asaFetch : fetch)(url)
          .then(r => r.json().then(j => ({ ok: r.ok, json: j })))
          .then(({ ok, json }) => {
            if (!ok) return success({ results: [] });
            const values = Array.isArray(json.values) ? json.values : [];
            const cleaned = (field === 'species')
              ? values.filter(v => String(v || '').trim().toLowerCase() !== 'others')
              : values;
            success({ results: cleaned.map(v => ({ id: v, text: v })) });
          })
          .catch(err => {
            console.error("select2 ajax failed", field, err);
            failure(err);
          });
      },
      processResults: function (data) {
        // data is already in select2 format from transport.success
        return data;
      },
      cache: true,
    },
  });
}

// --------------------------
// Params
// --------------------------
function buildParams() {
  const q = document.getElementById("q").value.trim();
  const yearMin = document.getElementById("year_min").value.trim();
  const yearMax = document.getElementById("year_max").value.trim();

  // Single selects (select2)
  const paperType = document.getElementById("paper_type").value;
  const tradition = document.getElementById("tradition").value;

  const params = new URLSearchParams();
  params.set("q", q);
  params.set("page", String(PAGE));
  params.set("page_size", String(PAGE_SIZE));

  if (yearMin) params.set("year_min", yearMin);
  if (yearMax) params.set("year_max", yearMax);
  if (paperType) params.set("paper_type", paperType);
  if (tradition) params.set("tradition", tradition);

  const multi = [
    ["species", "species"],
    ["modalities", "modalities"],
    ["features", "features"],
    ["stages", "stages"],
    ["domains", "domains"],
    ["contexts", "contexts"],
    ["venues", "venues"],
    ["institutions", "institutions"],
    ["authors", "authors"],
  ];

  for (const [field, id] of multi) {
    const el = document.getElementById(id);
    if (!el) continue;
    for (const opt of Array.from(el.selectedOptions || [])) {
      if (opt && opt.value) params.append(field, opt.value);
    }
  }

  for (const id of ["has_code", "has_dataset", "has_benchmark"]) {
    const el = document.getElementById(id);
    if (el && el.classList && el.classList.contains("active")) {
      params.set(id, "true");
    }
  }

  return params;
}

function setPagerVisible(show) {
  const pager = document.querySelector(".pager");
  // Use display:none so "Prev/Next" doesn't show before first search.
  if (pager) pager.style.display = show ? "inline-flex" : "none";
  const prev = document.getElementById("prev");
  const next = document.getElementById("next");
  if (prev) prev.disabled = !show;
  if (next) next.disabled = !show;
}

function setPagerState(total) {
  const prev = document.getElementById("prev");
  const next = document.getElementById("next");
  const pageLabel = document.getElementById("pageLabel");

  TOTAL_PAGES = Math.max(1, Math.ceil((Number(total) || 0) / PAGE_SIZE));
  if (PAGE > TOTAL_PAGES) PAGE = TOTAL_PAGES;
  if (PAGE < 1) PAGE = 1;

  if (prev) prev.disabled = PAGE <= 1;
  if (next) next.disabled = PAGE >= TOTAL_PAGES;
  if (pageLabel) pageLabel.textContent = `Page ${PAGE} / ${TOTAL_PAGES}`;
}

async function hydrateSaveButtons(containerEl) {
  if (!containerEl) return;
  const loggedIn = await isLoggedIn();
  if (!loggedIn) return;

  const btns = Array.from(containerEl.querySelectorAll(".save-paper-btn"));
  if (btns.length === 0) return;

  // Batch check to avoid reverse-proxy/WAF rate limits.
  const cap = Math.min(btns.length, 60);
  const docIds = [];
  const byId = {};
  for (const b of btns.slice(0, cap)) {
    const docId = String(b.getAttribute("data-id") || "").trim();
    if (!docId) continue;
    docIds.push(docId);
    byId[docId] = b;
  }
  if (docIds.length === 0) return;

  const rr = await jpost("/api/library/papers/has_batch", { doc_ids: docIds });
  if (!rr.ok || !rr.json || !rr.json.saved) return;
  const savedMap = rr.json.saved;
  for (const [docId, saved] of Object.entries(savedMap)) {
    const b = byId[docId];
    if (!b) continue;
    const s = !!saved;
    b.textContent = s ? "★ Saved" : "☆ Save";
    b.classList.toggle("btn-primary", s);
  }
}

// --------------------------
// Search
// --------------------------
async function doSearch() {
  const qNow = document.getElementById("q").value.trim();
  const itemsEl = document.getElementById("items");
  const statsEl = document.getElementById("stats");
  const pageLabel = document.getElementById("pageLabel");

  if (!qNow) {
    statsEl.textContent = "Enter a query to search.";
    pageLabel.textContent = "Page —";
    itemsEl.innerHTML = "";
    setPagerVisible(false);
    return;
  }

  setPagerVisible(true);

  const params = buildParams();
  const rr = await jget(`/api/search?${params.toString()}`);
  const data = rr.json || {};

  if (!rr.ok) {
    statsEl.textContent = "Search failed.";
    itemsEl.innerHTML = "";
    return;
  }

  const q = qNow;
  const total = data.total || data.total_candidates || data.results?.length || 0;
  statsEl.textContent = `${total} results`;
  setPagerState(total);

  itemsEl.innerHTML = "";
  for (const p of (data.results || [])) {
    const title = p.title || p.display_name || "";
    const year = p.year || "";

    const ai = p.summary || p.ai_summary || (p.ai_metadata && p.ai_metadata.summary) || "";
    const pdfUrl = bestPdf(p);
    const url = bestUrl(p);

    const venueFull = venueText(p);
    const authorsFull = listText(p.authors, ", ");
    const affList = (Array.isArray(p.affiliations) && p.affiliations.length)
      ? p.affiliations
      : (Array.isArray(p.institutions) ? p.institutions : []);
    const instFull = listText(affList, " • ");

    // Truncate on cards; full text in tooltip + modal.
    // Keep affiliations especially short so cards don't get visually "ragged".
    const authorsShort = truncateWithEllipsis(authorsFull, 120);
    const instShort = truncateWithEllipsis(instFull, 50);
    const venueShort = truncateWithEllipsis(venueFull, 90);

    const authorsLine = authorsShort
      ? `<div class="meta-line" title="${esc(authorsFull)}"><span class="meta-pill authors">Authors</span> <span class="meta-value authors">${esc(authorsShort)}</span></div>`
      : "";
    const instLine = instShort
      ? `<div class="meta-line" title="${esc(instFull)}"><span class="meta-pill aff">Affiliations</span> <span class="meta-value aff">${esc(instShort)}</span></div>`
      : "";
    const venueLine = venueShort
      ? `<div class="meta-line" title="${esc(venueFull)}"><span class="meta-pill venue">Venue</span> <span class="meta-value venue meta-venue">${esc(venueShort)}</span></div>`
      : "";

    const card = document.createElement("div");
    card.className = "item card";

    card.innerHTML = `
      <div class="item-head" style="display:flex; justify-content:space-between; align-items:flex-start;">
        <div class="item-title" style="flex:1; padding-right:15px;">
          ${paperTypeBadge(p)}
          ${featureBadges(p)}
          <a class="titlelink" href="${(window.asaUrl && typeof window.asaUrl === "function") ? window.asaUrl(`/paper/${encodeURIComponent(p.doc_id)}`) : `/paper/${encodeURIComponent(p.doc_id)}`}" target="_blank" rel="noopener">
            ${highlight(title, q)}
          </a>
        </div>
        <div class="top-right-actions" style="flex-shrink:0; display:flex; gap:8px;">
          ${pdfUrl ? `<a class="btn primary sm" href="${esc(pdfUrl)}" target="_blank" rel="noopener" style="text-decoration:none;">📄 PDF</a>` : (url ? `<a class="btn sm" href="${esc(url)}" target="_blank" rel="noopener" style="text-decoration:none;">🔗 Link</a>` : "")}
        </div>
      </div>

      <div class="meta muted" style="margin-top:10px; font-size:0.95rem;">
        <div style="font-weight:800; color:var(--text); margin-bottom:6px;">${esc(year ? String(year) : "Year Unknown")}</div>
        ${authorsLine}
        ${instLine}
        ${venueLine}
      </div>

      <div class="tags-container" style="margin-top:12px;">
        ${tagBadges(p)}
      </div>

      ${ai ? `<div class="blk" style="border-left:3px solid var(--accent); margin-top:12px;"><div class="lbl">✨ AI Summary</div><div>${highlight(snippet(ai, 220), q)}</div></div>` : ""}

      <div class="actions" style="margin-top:15px; border-top:1px solid var(--border2); padding-top:12px; display:flex; flex-wrap:wrap; gap:8px;">
        <button class="btn sm" data-action="details" data-id="${esc(p.doc_id)}">🔍 View Details</button>
        <button class="btn sm save-paper-btn" data-id="${esc(p.doc_id)}">☆ Save</button>
        <button class="btn sm copy-card-btn" data-format="bibtex" data-id="${esc(p.doc_id)}">📋 Copy BibTeX</button>
        <button class="btn sm copy-card-btn" data-format="apa" data-id="${esc(p.doc_id)}">📋 Copy APA</button>
        <button class="btn sm copy-link-btn" data-id="${esc(p.doc_id)}">🔗 Copy link</button>
        <a class="btn sm" href="${(window.asaUrl && typeof window.asaUrl === "function") ? window.asaUrl(`/api/paper/${encodeURIComponent(p.doc_id)}/ris`) : `/api/paper/${encodeURIComponent(p.doc_id)}/ris`}" download>RIS</a>
      </div>
    `;

    card.addEventListener("click", async (e) => {
      const t = e.target;
      if (t.tagName === "A" || t.closest("a")) return;

      if (t.classList.contains("copy-card-btn")) {
        e.preventDefault(); e.stopPropagation();
        const fmt = t.getAttribute("data-format") || "bibtex";
        const text = await getCitationText(p, fmt);
        await navigator.clipboard.writeText(text);
        const orig = t.innerText;
        t.innerText = "✓ Copied!";
        setTimeout(() => t.innerText = orig, 1500);
        return;
      }

      if (t.classList.contains("copy-link-btn")) {
        e.preventDefault(); e.stopPropagation();
        const permalink = `${window.location.origin}${(window.asaUrl && typeof window.asaUrl === "function") ? window.asaUrl(`/paper/${encodeURIComponent(p.doc_id)}`) : `/paper/${encodeURIComponent(p.doc_id)}`}`;
        await navigator.clipboard.writeText(permalink);
        const orig = t.innerText;
        t.innerText = "✓ Copied!";
        setTimeout(() => t.innerText = orig, 1500);
        return;
      }

      if (t.classList.contains("save-paper-btn")) {
        e.preventDefault(); e.stopPropagation();
        const docId = t.getAttribute("data-id");
        const rr = await jpost("/api/library/papers/toggle", { doc_id: docId, source: "search" });
        if (!rr.ok && rr.status === 401) {
          window.asaToast && window.asaToast("Sign in to save papers.", "warn");
          window.location.href = (window.asaUrl && typeof window.asaUrl === "function") ? window.asaUrl("/login") : "/login";
          return;
        }
        if (rr.ok && rr.json) {
          const saved = !!rr.json.saved;
          t.textContent = saved ? "★ Saved" : "☆ Save";
          t.classList.toggle("btn-primary", saved);
          window.asaToast && window.asaToast(saved ? "Saved" : "Removed", "ok");
        } else {
          window.asaToast && window.asaToast("Save failed", "error");
        }
        return;
      }

      const action = t.getAttribute("data-action");
      if (action === "details") {
        e.preventDefault();
        await openModal(p.doc_id);
      }
    });

    itemsEl.appendChild(card);
  }

  // Update "Save" button state for logged-in users.
  hydrateSaveButtons(itemsEl).catch(() => null);
}

// --------------------------
// Modal
// --------------------------
function openBackdrop(show) {
  document.getElementById("modalBackdrop").classList.toggle("hidden", !show);
}

function openModalShell(show) {
  document.getElementById("paperModal").classList.toggle("hidden", !show);
  openBackdrop(show);
}

async function openModal(docId) {
  const rr = await jget(`/api/paper/${encodeURIComponent(docId)}`);
  const p = rr.json || {};
  if (!rr.ok) return;

  const title = p.title || p.display_name || "";
  const ai = p.summary || p.ai_summary || (p.ai_metadata && p.ai_metadata.summary) || "";
  const abs = p.abstract_text || p.abstract || "";
  const pdf = bestPdf(p);
  const url = bestUrl(p);

  const venueFull = venueText(p);
  const authorsFull = listText(p.authors, ", ");
  const affList = (Array.isArray(p.affiliations) && p.affiliations.length)
    ? p.affiliations
    : (Array.isArray(p.institutions) ? p.institutions : []);
  const instFull = listText(affList, " • ");

  document.getElementById("mTitle").innerHTML = `${esc(title)} <span class="muted" style="font-weight:500;">(${esc(p.year || "N/A")})</span>`;

  const rightActionsEl = document.getElementById("mTopRightActions");
  rightActionsEl.innerHTML = "";
  if (pdf) rightActionsEl.innerHTML = `<a class="btn primary" href="${esc(pdf)}" target="_blank" rel="noopener">📄 PDF</a>`;
  else if (url) rightActionsEl.innerHTML = `<a class="btn" href="${esc(url)}" target="_blank" rel="noopener">🔗 Link</a>`;

  const authorsLine = authorsFull ? `<div style="margin-bottom:6px; word-break:break-word;"><span class="meta-label">Authors:</span> ${esc(authorsFull)}</div>` : "";
  const instLine = instFull ? `<div style="margin-bottom:6px; word-break:break-word;"><span class="meta-label">Affiliations:</span> ${esc(instFull)}</div>` : "";
  const venueLine = venueFull ? `<div style="margin-bottom:6px; word-break:break-word;"><span class="meta-label">Venue:</span> <span class="meta-venue">${esc(venueFull)}</span></div>` : "";

  document.getElementById("mBody").innerHTML = `
    <div class="modal-grid" style="display:grid; grid-template-columns: 2fr 1fr; gap:20px;">
      <div class="modal-main">
        <div class="meta muted" style="margin-bottom: 16px; font-size: 1.0rem; line-height: 1.5;">
          ${authorsLine}
          ${instLine}
          ${venueLine}
        </div>
        ${ai ? `<div class="blk" style="border-left:4px solid var(--accent); padding-left:12px;"><div class="lbl">✨ AI Summary</div><div style="font-size:1.0rem; line-height:1.6;">${esc(ai)}</div></div>` : ""}
        ${abs ? `<div class="blk"><div class="lbl">Abstract</div><div style="font-size:1.0rem; line-height:1.6;">${esc(abs)}</div></div>` : "<div class='muted'>No abstract available.</div>"}

        <div style="margin-top:20px;">
          <div class="lbl">Semantic Tags</div>
          <div class="tags-container">${tagBadges(p) || "<span class='muted'>None</span>"}</div>
        </div>
      </div>

      <div class="modal-sidebar">
        <div class="blk" style="background:var(--card); padding:18px; border-radius:8px; border:1px solid var(--border2);">
          <div class="lbl" style="border-bottom:1px solid var(--border2); padding-bottom:8px; margin-bottom:12px; font-size:1.05rem;">Details</div>
          <div class="kv" style="display:flex; justify-content:space-between; margin-bottom:10px;"><span>Type</span><span style="font-weight:600;">${esc(p.paper_type || "N/A")}</span></div>
          <div class="kv" style="display:flex; justify-content:space-between; margin-bottom:10px;"><span>Tradition</span><span style="font-weight:600;">${esc(p.research_tradition || "N/A")}</span></div>
          <div class="kv" style="display:flex; justify-content:space-between; margin-bottom:10px;"><span>Code</span><span>${p.has_code ? "✅" : "❌"}</span></div>
          <div class="kv" style="display:flex; justify-content:space-between; margin-bottom:10px;"><span>Dataset</span><span>${p.has_dataset ? "✅" : "❌"}</span></div>
          <div class="kv" style="display:flex; justify-content:space-between; margin-bottom:10px;"><span>Benchmark</span><span>${p.has_benchmark ? "✅" : "❌"}</span></div>
        </div>

        <div class="blk" style="background:var(--card); padding:18px; border-radius:8px; border:1px solid var(--border2); margin-top:15px;">
          <div class="lbl" style="border-bottom:1px solid var(--border2); padding-bottom:8px; margin-bottom:12px; font-size:1.05rem;">Citation Export</div>
          <div class="cite-picker" style="display:flex; flex-direction:column; gap:12px;">
            <select id="mCiteFormatSelect" style="padding:10px; border-radius:10px; border:1px solid var(--border2); font-size:1rem;">
              <option value="bibtex" selected>BibTeX</option>
              <option value="apa">APA</option>
              <option value="chicago-author-date">Chicago</option>
              <option value="vancouver">Vancouver</option>
              <option value="harvard1">Harvard</option>
            </select>
            <button id="mCopyCiteBtn" class="btn primary">📋 Copy Citation</button>
          </div>
        </div>

        <div class="blk" style="background:var(--card); padding:18px; border-radius:8px; border:1px solid var(--border2); margin-top:15px; display:flex; flex-direction:column; gap:10px;">
          <div class="lbl" style="border-bottom:1px solid var(--border2); padding-bottom:8px; margin-bottom:5px; font-size:1.05rem;">Links</div>
          <a class="btn" href="${(window.asaUrl && typeof window.asaUrl === "function") ? window.asaUrl(`/api/paper/${encodeURIComponent(docId)}/ris`) : `/api/paper/${encodeURIComponent(docId)}/ris`}" download>⬇️ Export RIS</a>
          <a class="btn" href="${(window.asaUrl && typeof window.asaUrl === "function") ? window.asaUrl(`/paper/${encodeURIComponent(docId)}`) : `/paper/${encodeURIComponent(docId)}`}" target="_blank" rel="noopener">🔗 Permalink / Paper Page</a>
        </div>
      </div>
    </div>
  `;

  document.getElementById("mCopyCiteBtn").onclick = async () => {
    const fmt = document.getElementById("mCiteFormatSelect").value || "bibtex";
    const text = await getCitationText(p, fmt);
    await navigator.clipboard.writeText(text);
    const btn = document.getElementById("mCopyCiteBtn");
    const orig = btn.innerText;
    btn.innerText = "✓ Copied!";
    setTimeout(() => { btn.innerText = orig; }, 1500);
  };

  openModalShell(true);
}

// --------------------------
// Filters
// --------------------------
function clearFilters() {
  for (const id of ["species", "modalities", "features", "stages", "domains", "contexts", "venues", "institutions", "authors"]) {
    const el = document.getElementById(id);
    if (!el) continue;
    Array.from(el.options).forEach(o => (o.selected = false));
    if (window.jQuery && window.jQuery(el).data('select2')) {
      window.jQuery(el).val(null).trigger('change');
    }
  }

  for (const id of ["year_min", "year_max", "q"]) {
    const el = document.getElementById(id);
    if (el) el.value = "";
  }

  // single selects
  for (const id of ["paper_type", "tradition"]) {
    const el = document.getElementById(id);
    if (!el) continue;
    el.value = "";
    if (window.jQuery && window.jQuery(el).data('select2')) {
      window.jQuery(el).val(null).trigger('change');
    }
  }

  for (const id of ["has_code", "has_dataset", "has_benchmark"]) {
    const el = document.getElementById(id);
    if (el && el.classList) el.classList.remove("active");
  }

  PAGE = 1;
  document.getElementById("items").innerHTML = "";
  document.getElementById("stats").textContent = "—";
  document.getElementById("pageLabel").textContent = "Page —";
  setPagerVisible(false);
}

// --------------------------
// Boot
// --------------------------
document.addEventListener("DOMContentLoaded", async () => {
  setPagerVisible(false);

  // Lightweight initial loads for small/medium facets.
  // Large facets (species/authors/institutions/paper_type/tradition) use Select2 AJAX typeahead.
  Promise.all([
    loadFacet("domains", "domains", 2000),
    loadFacet("modalities", "modalities", 2000),
    loadFacet("features", "features", 2000),
    loadFacet("stages", "stages", 2000),
    loadFacet("contexts", "contexts", 2000),
  ]).catch((e) => {
    console.error("Initial facet bootstrap failed", e);
  });

  // Enhance all selects with Select2.
  initSelect2Static("domains", { placeholder: "Select domains" });
  initSelect2Static("modalities", { placeholder: "Select modalities" });
  initSelect2Static("features", { placeholder: "Select features" });
  initSelect2Static("stages", { placeholder: "Select stages" });
  initSelect2Static("contexts", { placeholder: "Select contexts" });

  initSelect2Ajax("species", "species", { placeholder: "Search species…", minimumInputLength: 3, delay: 320, multiple: true, limit: 120 });
  initSelect2Ajax("venues", "venue", { placeholder: "Search venues…", minimumInputLength: 3, delay: 320, multiple: true, limit: 120 });
  initSelect2Ajax("institutions", "institutions", { placeholder: "Search institutions…", minimumInputLength: 3, delay: 320, multiple: true, limit: 120 });
  initSelect2Ajax("authors", "authors", { placeholder: "Search authors…", minimumInputLength: 3, delay: 320, multiple: true, limit: 120 });

  initSelect2Ajax("paper_type", "paper_type", { placeholder: "Select paper type", minimumInputLength: 0, delay: 250, multiple: false, limit: 50 });
  initSelect2Ajax("tradition", "tradition", { placeholder: "Select research tradition", minimumInputLength: 0, delay: 250, multiple: false, limit: 50 });

  // Controls
  document.getElementById("btnSearch").addEventListener("click", () => { PAGE = 1; doSearch(); });
  const btnSearch2 = document.getElementById("btnSearch2");
  if (btnSearch2) btnSearch2.addEventListener("click", () => { PAGE = 1; doSearch(); closeFilters(); });

  document.getElementById("q").addEventListener("keydown", (e) => {
    if (e.key === "Enter") { PAGE = 1; doSearch(); }
  });

  document.getElementById("prev").addEventListener("click", () => {
    if (PAGE > 1) { PAGE -= 1; doSearch(); }
  });

  document.getElementById("next").addEventListener("click", () => {
    if (PAGE < TOTAL_PAGES) { PAGE += 1; doSearch(); }
  });

  document.getElementById("btnClear").addEventListener("click", () => { clearFilters(); });

  // Resource toggle buttons
  for (const id of ["has_code", "has_dataset", "has_benchmark"]) {
    const el = document.getElementById(id);
    if (!el) continue;
    el.addEventListener("click", () => {
      el.classList.toggle("active");
    });
  }

  // Modal close
  document.getElementById("mClose").addEventListener("click", () => openModalShell(false));
  document.getElementById("modalBackdrop").addEventListener("click", () => openModalShell(false));
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") openModalShell(false); });

  // Sidebar
  const sidebar = document.getElementById("sidebar");
  const btnToggleFilters = document.getElementById("btnToggleFilters");
  const btnCloseFilters = document.getElementById("btnCloseFilters");
  function openFilters() { if (sidebar) { sidebar.classList.add("open"); document.body.style.overflow = "hidden"; } }
  function closeFilters() { if (sidebar) { sidebar.classList.remove("open"); document.body.style.overflow = ""; } }
  window.closeFilters = closeFilters;
  if (btnToggleFilters) btnToggleFilters.addEventListener("click", () => openFilters());
  if (btnCloseFilters) btnCloseFilters.addEventListener("click", () => closeFilters());
});

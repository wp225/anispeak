// AniSpeakArchive static demo.
// Hybrid search (MiniSearch BM25 + bge-small dense vectors, fused with RRF) running in the browser,
// facet filtering, paper pages with citations and similar papers, and pre-generated Ask AI examples.

const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.1";
const PAGE_SIZE = 20;
const RRF_K = 60;
const K_CANDIDATES = 200;
const DENSE_MIN_SIM = 0.60;   // below this, dense matches are treated as noise
const DENSE_REL_WINDOW = 0.12; // keep dense matches within this of the best match

const FACETS = [
  { key: "taxa", label: "Taxa" },
  { key: "modality", label: "Signal modality" },
  { key: "themes", label: "Research theme" },
  { key: "methods", label: "Methods" },
  { key: "topic", label: "OpenAlex topic", single: true, limit: 12, collapsed: true },
];
const SEARCH_EXAMPLES = ["dolphin signature whistles", "how do bats find food in the dark", "deep learning for bird sounds", "honeybee waggle dance"];

const S = {
  papers: [], byId: new Map(), emb: null, dim: 384, meta: null, ask: [], mini: null,
  embedder: null, embedState: "idle", // idle | loading | ready | error
  queryCache: new Map(), mounted: null, searchSeq: 0,
};

const $ = (sel, root = document) => root.querySelector(sel);
const app = $("#app");
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// ---------- data ----------
async function loadData() {
  const [papers, meta, ask, embBuf] = await Promise.all([
    fetch("data/papers.json").then((r) => r.json()),
    fetch("data/meta.json").then((r) => r.json()),
    fetch("data/ask.json").then((r) => r.json()),
    fetch("data/embeddings.bin").then((r) => r.arrayBuffer()),
  ]);
  S.papers = papers;
  S.meta = meta;
  S.ask = ask;
  S.dim = meta.dim;
  S.emb = new Float32Array(embBuf);
  papers.forEach((p, i) => { p._i = i; S.byId.set(p.id, p); });

  S.mini = new MiniSearch({
    fields: ["title", "abstract", "authorsText", "venue", "tagsText"],
    storeFields: [],
    processTerm: (t) => { const w = t.toLowerCase(); return w.length < 2 || STOP.has(w) ? null : w; },
    searchOptions: { boost: { title: 2.5, tagsText: 1.2 }, prefix: (t) => t.length > 3, fuzzy: (t) => (t.length > 5 ? 0.15 : 0), combineWith: "OR" },
  });
  S.mini.addAll(papers.map((p, i) => ({
    id: i, title: p.title, abstract: p.abstract, authorsText: p.authors.join(" "), venue: p.venue || "",
    tagsText: [...p.taxa, ...p.modality, ...p.themes, ...p.methods, p.topic || ""].join(" "),
  })));
}

// ---------- semantic model ----------
async function ensureEmbedder() {
  if (S.embedState !== "idle") return;
  S.embedState = "loading";
  updateModePill();
  try {
    const { pipeline, env } = await import(TRANSFORMERS_URL);
    env.allowLocalModels = false;
    S.embedder = await pipeline("feature-extraction", S.meta.browser_model, { dtype: "q8" });
    S.embedState = "ready";
  } catch (err) {
    console.warn("Semantic model failed to load; using keyword search only.", err);
    S.embedState = "error";
  }
  updateModePill();
  if (S.mounted === "search") runSearch();
}

async function embedQuery(q) {
  if (S.queryCache.has(q)) return S.queryCache.get(q);
  const out = await S.embedder(S.meta.query_prefix + q, { pooling: "cls", normalize: true });
  const v = out.data instanceof Float32Array ? out.data : Float32Array.from(out.data);
  S.queryCache.set(q, v);
  return v;
}

function cosineAll(v) {
  const n = S.papers.length, d = S.dim, out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    const off = i * d;
    for (let j = 0; j < d; j++) s += S.emb[off + j] * v[j];
    out[i] = s;
  }
  return out;
}

function topK(scores, k, minScore = -Infinity) {
  const idx = [];
  for (let i = 0; i < scores.length; i++) if (scores[i] >= minScore) idx.push(i);
  idx.sort((a, b) => scores[b] - scores[a]);
  return idx.slice(0, k);
}

function rrf(rankings) {
  const score = new Map();
  for (const ranking of rankings) ranking.forEach((i, r) => score.set(i, (score.get(i) || 0) + 1 / (RRF_K + r + 1)));
  return [...score.keys()].sort((a, b) => score.get(b) - score.get(a));
}

async function retrieve(q) {
  const lex = S.mini.search(q).slice(0, K_CANDIDATES).map((r) => r.id);
  if (S.embedState !== "ready") return { ids: lex, mode: "keyword" };
  const sims = cosineAll(await embedQuery(q));
  let best = -1;
  for (const s of sims) if (s > best) best = s;
  const dense = topK(sims, K_CANDIDATES, Math.max(DENSE_MIN_SIM, best - DENSE_REL_WINDOW));
  return { ids: rrf([lex, dense]), mode: "hybrid" };
}

// ---------- routing ----------
function parseHash() {
  const h = location.hash.replace(/^#/, "") || "/";
  const [path, qs] = h.split("?");
  return { path, params: new URLSearchParams(qs || "") };
}

function setNav(route) {
  document.querySelectorAll(".nav-link").forEach((a) => a.classList.toggle("active", a.dataset.route === route));
}

function render() {
  const { path, params } = parseHash();
  S.visited = (S.visited || 0) + 1;
  window.scrollTo({ top: 0 });
  if (path.startsWith("/paper/")) { S.mounted = "paper"; setNav(""); return renderPaper(decodeURIComponent(path.slice(7))); }
  if (path === "/ask") { S.mounted = "ask"; setNav("ask"); return renderAsk(params); }
  if (path === "/about") { S.mounted = "about"; setNav("about"); return renderAbout(); }
  setNav("search");
  document.title = "AniSpeakArchive Demo";
  if (S.mounted !== "search") mountSearch();
  syncSearchControls(params);
  runSearch();
}

// ---------- search page ----------
function readState() {
  const { params } = parseHash();
  const st = { q: params.get("q") || "", sort: params.get("sort") || "", page: Math.max(1, parseInt(params.get("page") || "1", 10) || 1), filters: {} };
  for (const f of FACETS) {
    const v = params.get(f.key);
    st.filters[f.key] = v ? v.split("|").filter(Boolean) : [];
  }
  st.yFrom = parseInt(params.get("from") || "", 10) || null;
  st.yTo = parseInt(params.get("to") || "", 10) || null;
  st.oa = params.get("oa") === "1";
  return st;
}

function writeState(patch, { push = false } = {}) {
  const st = { ...readState(), ...patch };
  const p = new URLSearchParams();
  if (st.q) p.set("q", st.q);
  if (st.sort) p.set("sort", st.sort);
  for (const f of FACETS) if (st.filters[f.key]?.length) p.set(f.key, st.filters[f.key].join("|"));
  if (st.yFrom) p.set("from", st.yFrom);
  if (st.yTo) p.set("to", st.yTo);
  if (st.oa) p.set("oa", "1");
  if (st.page > 1) p.set("page", st.page);
  const url = "#/" + (p.toString() ? "?" + p.toString() : "");
  if (push) history.pushState(null, "", url); else history.replaceState(null, "", url);
  runSearch();
}

function mountSearch() {
  S.mounted = "search";
  app.innerHTML = `
    <div class="page-header">
      <h1>Search animal-communication research</h1>
      <p class="muted">Hybrid keyword and semantic search over ${S.papers.length} papers on bioacoustics and animal communication. Everything runs in your browser.</p>
    </div>
    <form class="searchbar" role="search" id="searchForm">
      <input type="search" id="q" placeholder="Try “how do bats find food in the dark”" autocomplete="off" aria-label="Search papers" />
      <button class="btn primary" type="submit">Search</button>
    </form>
    <div class="status-row">
      <span class="mode" id="mode"><span class="dot"></span><span id="modeText"></span></span>
      <span class="muted">Try:</span>
      <span class="examples">${SEARCH_EXAMPLES.map((e) => `<button type="button" class="linkish" data-example="${esc(e)}">${esc(e)}</button>`).join("")}</span>
    </div>
    <div class="layout">
      <aside class="card filters" id="filters" aria-label="Filters"></aside>
      <section>
        <div class="results-head">
          <div id="count" class="muted"></div>
          <label class="muted" style="font-size:14px">Sort
            <select id="sort">
              <option value="">Relevance</option>
              <option value="cited">Most cited</option>
              <option value="new">Newest</option>
              <option value="old">Oldest</option>
            </select>
          </label>
        </div>
        <div class="results" id="results"></div>
        <div class="pager" id="pager"></div>
      </section>
    </div>`;

  const q = $("#q");
  let t;
  q.addEventListener("focus", ensureEmbedder, { once: true });
  q.addEventListener("input", () => {
    ensureEmbedder();
    clearTimeout(t);
    t = setTimeout(() => writeState({ q: q.value.trim(), page: 1 }), 250);
  });
  $("#searchForm").addEventListener("submit", (e) => { e.preventDefault(); clearTimeout(t); writeState({ q: q.value.trim(), page: 1 }, { push: true }); });
  $("#sort").addEventListener("change", (e) => writeState({ sort: e.target.value, page: 1 }));
  app.querySelectorAll("[data-example]").forEach((b) => b.addEventListener("click", () => {
    q.value = b.dataset.example;
    ensureEmbedder();
    writeState({ q: b.dataset.example, page: 1 }, { push: true });
  }));
  updateModePill();
}

function syncSearchControls(params) {
  const q = $("#q");
  if (q && document.activeElement !== q) q.value = params.get("q") || "";
  const sort = $("#sort");
  if (sort) sort.value = params.get("sort") || "";
}

function updateModePill() {
  const pill = $("#mode"), text = $("#modeText");
  if (!pill) return;
  const m = {
    idle: "Keyword search · semantic model loads when you search",
    loading: "Keyword search · loading semantic model…",
    ready: "Hybrid search · keyword + semantic (bge-small)",
    error: "Keyword search · semantic model unavailable",
  }[S.embedState];
  text.textContent = m;
  pill.classList.toggle("ready", S.embedState === "ready");
}

function passes(p, st, skipKey) {
  for (const f of FACETS) {
    if (f.key === skipKey) continue;
    const sel = st.filters[f.key];
    if (!sel.length) continue;
    const vals = f.single ? [p[f.key]] : p[f.key];
    if (!sel.some((s) => vals.includes(s))) return false;
  }
  if (skipKey !== "_year") {
    if (st.yFrom && (p.year || 0) < st.yFrom) return false;
    if (st.yTo && (p.year || 9999) > st.yTo) return false;
  }
  if (skipKey !== "_oa" && st.oa && !p.is_oa) return false;
  return true;
}

async function runSearch() {
  const seq = ++S.searchSeq;
  const st = readState();
  let ids, mode = "browse";
  if (st.q) {
    const r = await retrieve(st.q);
    if (seq !== S.searchSeq) return; // a newer search started
    ids = r.ids;
    mode = r.mode;
  } else {
    ids = S.papers.map((_, i) => i);
  }
  const base = ids.map((i) => S.papers[i]);
  let hits = base.filter((p) => passes(p, st));
  const sort = st.sort || (st.q ? "" : "cited");
  if (sort === "cited") hits = [...hits].sort((a, b) => b.cited_by - a.cited_by);
  else if (sort === "new") hits = [...hits].sort((a, b) => (b.year || 0) - (a.year || 0) || b.cited_by - a.cited_by);
  else if (sort === "old") hits = [...hits].sort((a, b) => (a.year || 9999) - (b.year || 9999));

  renderFilters(base, st);
  renderResults(hits, st, mode);
}

function renderFilters(base, st) {
  const el = $("#filters");
  if (!el) return;
  const openState = {};
  el.querySelectorAll("details[data-key]").forEach((d) => (openState[d.dataset.key] = d.open));
  const anyActive = FACETS.some((f) => st.filters[f.key].length) || st.yFrom || st.yTo || st.oa;

  let html = `<div class="filters-head"><h2>Filters</h2>${anyActive ? `<button class="btn sm" id="clearFilters">Clear</button>` : ""}</div>`;
  for (const f of FACETS) {
    const counts = new Map();
    for (const p of base) {
      if (!passes(p, st, f.key)) continue;
      const vals = f.single ? (p[f.key] ? [p[f.key]] : []) : p[f.key];
      for (const v of vals) counts.set(v, (counts.get(v) || 0) + 1);
    }
    for (const s of st.filters[f.key]) if (!counts.has(s)) counts.set(s, 0);
    let entries = [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
    if (f.limit) entries = entries.filter(([v], i) => i < f.limit || st.filters[f.key].includes(v));
    if (!entries.length) continue;
    const open = openState[f.key] ?? (!f.collapsed || st.filters[f.key].length > 0);
    html += `<details data-key="${f.key}" ${open ? "open" : ""}><summary>${esc(f.label)}</summary><div class="facet-list">` +
      entries.map(([v, c]) => {
        const checked = st.filters[f.key].includes(v);
        return `<label class="facet ${c === 0 ? "zero" : ""}"><input type="checkbox" data-facet="${f.key}" value="${esc(v)}" ${checked ? "checked" : ""}/> <span>${esc(v)}</span><span class="count">${c}</span></label>`;
      }).join("") + `</div></details>`;
  }
  const yOpen = openState._year ?? true;
  html += `<details data-key="_year" ${yOpen ? "open" : ""}><summary>Year</summary>
      <div class="year-row"><input type="number" id="yFrom" placeholder="from" min="1900" max="2100" value="${st.yFrom || ""}" aria-label="From year" />
      <span>–</span><input type="number" id="yTo" placeholder="to" min="1900" max="2100" value="${st.yTo || ""}" aria-label="To year" /></div>
      <label class="facet" style="margin-top:8px"><input type="checkbox" id="oa" ${st.oa ? "checked" : ""}/> <span>Open access only</span></label>
    </details>
    <p class="note">Taxa, modality, theme and method tags are assigned by keyword rules for this demo. The full system uses LLM annotations.</p>`;
  el.innerHTML = html;

  el.querySelectorAll("input[data-facet]").forEach((cb) => cb.addEventListener("change", () => {
    const key = cb.dataset.facet;
    const cur = new Set(readState().filters[key]);
    cb.checked ? cur.add(cb.value) : cur.delete(cb.value);
    writeState({ filters: { ...readState().filters, [key]: [...cur] }, page: 1 });
  }));
  const yr = () => writeState({ yFrom: parseInt($("#yFrom").value, 10) || null, yTo: parseInt($("#yTo").value, 10) || null, page: 1 });
  $("#yFrom").addEventListener("change", yr);
  $("#yTo").addEventListener("change", yr);
  $("#oa").addEventListener("change", (e) => writeState({ oa: e.target.checked, page: 1 }));
  $("#clearFilters")?.addEventListener("click", () => writeState({ filters: Object.fromEntries(FACETS.map((f) => [f.key, []])), yFrom: null, yTo: null, oa: false, page: 1 }));
}

function queryTerms(q) {
  return [...new Set(q.toLowerCase().match(/[a-z0-9]{3,}/g) || [])].filter((t) => !STOP.has(t));
}
const STOP = new Set(("a an the and or of in on at to for with by as is are was were be been being it its this that these those from into over under " +
  "how what which why who whom when where does do did can could should would may might will their they them there use used using about " +
  "than then so such not no also between within we our i you your find").split(" "));

function snippet(text, terms, len = 280) {
  let start = 0;
  if (terms.length) {
    const low = text.toLowerCase();
    const pos = terms.map((t) => low.indexOf(t)).filter((x) => x >= 0).sort((a, b) => a - b)[0];
    if (pos !== undefined && pos > 80) start = text.lastIndexOf(" ", pos - 60) + 1;
  }
  let s = text.slice(start, start + len);
  if (start + len < text.length) s = s.replace(/\s+\S*$/, "") + " …";
  if (start > 0) s = "… " + s;
  let html = esc(s);
  if (terms.length) {
    const rx = new RegExp(`\\b(${terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})\\w*`, "gi");
    html = html.replace(rx, "<mark>$&</mark>");
  }
  return html;
}

function authorLine(p, n = 3) {
  if (!p.authors.length) return "Unknown authors";
  const a = p.authors.slice(0, n).join(", ");
  return p.n_authors > n ? `${a} et al.` : a;
}

function chipsFor(p, max = 6) {
  const tags = [...p.taxa.map((t) => [t, ""]), ...p.themes.map((t) => [t, "alt"]), ...p.modality.filter((m) => m !== "Acoustic").map((t) => [t, "alt"])];
  return tags.slice(0, max).map(([t, c]) => `<span class="chip ${c}">${esc(t)}</span>`).join("");
}

function renderResults(hits, st, mode) {
  const el = $("#results");
  if (!el) return;
  const total = hits.length;
  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const page = Math.min(st.page, pages);
  const shown = hits.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);
  const terms = queryTerms(st.q);

  $("#count").textContent = st.q
    ? `${total} result${total === 1 ? "" : "s"} for “${st.q}”${mode === "keyword" && S.embedState !== "ready" ? " (keyword only)" : ""}`
    : `${total} papers · browse or search`;

  if (!total) {
    el.innerHTML = `<div class="card empty"><p><strong>No matching papers.</strong></p>
      <p class="muted">${S.embedState === "loading" ? "The semantic model is still loading. Results will update when it's ready." : "Try different words or clear some filters. This demo covers only a sample of papers."}</p></div>`;
  } else {
    el.innerHTML = shown.map((p) => `
      <a class="card result" href="#/paper/${encodeURIComponent(p.id)}">
        <h3>${esc(p.title)}</h3>
        <div class="meta"><span>${esc(authorLine(p))}</span>${p.year ? `<span>${p.year}</span>` : ""}${p.venue ? `<span>${esc(p.venue)}</span>` : ""}<span>Cited by ${p.cited_by.toLocaleString()}</span>${p.is_oa ? `<span>Open access</span>` : ""}</div>
        <p class="snippet">${snippet(p.abstract, terms)}</p>
        <div class="chips">${chipsFor(p)}</div>
      </a>`).join("");
  }

  const pager = $("#pager");
  pager.innerHTML = pages > 1
    ? `<button class="btn sm" ${page <= 1 ? "disabled" : ""} data-page="${page - 1}">← Prev</button><span class="muted">Page ${page} of ${pages}</span><button class="btn sm" ${page >= pages ? "disabled" : ""} data-page="${page + 1}">Next →</button>`
    : "";
  pager.querySelectorAll("[data-page]").forEach((b) => b.addEventListener("click", () => { writeState({ page: +b.dataset.page }, { push: true }); window.scrollTo({ top: 0 }); }));
}

// ---------- paper page ----------
function similarTo(p, k = 5) {
  const v = S.emb.subarray(p._i * S.dim, (p._i + 1) * S.dim);
  const sims = cosineAll(v);
  sims[p._i] = -Infinity;
  return topK(sims, k).map((i) => S.papers[i]);
}

function bibtex(p) {
  const first = (p.authors[0] || "anon").split(" ").pop().toLowerCase().replace(/[^a-z]/g, "");
  const word = (p.title.toLowerCase().match(/[a-z]{4,}/) || ["paper"])[0];
  const lines = [
    `@article{${first}${p.year || ""}${word},`,
    `  title   = {${p.title}},`,
    `  author  = {${p.authors.join(" and ")}},`,
    p.venue ? `  journal = {${p.venue}},` : null,
    p.year ? `  year    = {${p.year}},` : null,
    p.doi ? `  doi     = {${p.doi}},` : null,
    `}`,
  ];
  return lines.filter(Boolean).join("\n");
}

function apa(p) {
  const fmt = (name) => {
    const parts = name.trim().split(/\s+/);
    const last = parts.pop();
    return `${last}, ${parts.map((x) => x[0] + ".").join(" ")}`.trim().replace(/,\s*$/, "");
  };
  let a = p.authors.slice(0, 20).map(fmt);
  a = a.length > 1 ? a.slice(0, -1).join(", ") + ", & " + a[a.length - 1] : a[0] || "";
  return `${a} (${p.year || "n.d."}). ${p.title}.${p.venue ? " " + p.venue + "." : ""}${p.doi ? " https://doi.org/" + p.doi : ""}`;
}

function renderPaper(id) {
  const p = S.byId.get(id);
  if (!p) {
    app.innerHTML = `<div class="card empty"><p><strong>Paper not found.</strong></p><p><a href="#/">Back to search</a></p></div>`;
    return;
  }
  document.title = `${p.title} · AniSpeakArchive Demo`;
  const sim = similarTo(p);
  const tagRows = [["Taxa", p.taxa], ["Modality", p.modality], ["Themes", p.themes], ["Methods", p.methods]].filter(([, v]) => v.length);
  app.innerHTML = `
    <article class="paper">
      <a class="back" href="#/" id="backLink">← Back</a>
      <div class="card">
        <div class="meta">${p.year ? `<span>${p.year}</span>` : ""}${p.venue ? `<span>${esc(p.venue)}</span>` : ""}<span>Cited by ${p.cited_by.toLocaleString()}</span>${p.is_oa ? "<span>Open access</span>" : ""}</div>
        <h1>${esc(p.title)}</h1>
        <p class="authors muted">${esc(p.authors.join(", "))}${p.n_authors > p.authors.length ? ` and ${p.n_authors - p.authors.length} more` : ""}</p>
        ${p.institutions.length ? `<p class="muted" style="font-size:13px;margin:0">${esc(p.institutions.join(" · "))}</p>` : ""}
        <div class="links">
          ${p.doi ? `<a class="btn sm" href="https://doi.org/${esc(p.doi)}" target="_blank" rel="noopener">DOI ↗</a>` : ""}
          ${p.oa_url ? `<a class="btn sm" href="${esc(p.oa_url)}" target="_blank" rel="noopener">Full text ↗</a>` : ""}
          <a class="btn sm" href="https://openalex.org/${esc(p.id)}" target="_blank" rel="noopener">OpenAlex ↗</a>
        </div>
        <section><h2>Abstract</h2><p class="abstract">${esc(p.abstract)}</p></section>
        ${tagRows.length ? `<section><h2>Tags</h2>${tagRows.map(([k, v]) => `<div class="chips" style="margin-bottom:6px"><span class="muted" style="font-size:13px;min-width:76px">${k}</span>${v.map((t) => `<a class="chip" style="text-decoration:none" href="#/?${k === "Taxa" ? "taxa" : k === "Modality" ? "modality" : k === "Themes" ? "themes" : "methods"}=${encodeURIComponent(t)}">${esc(t)}</a>`).join("")}</div>`).join("")}${p.topic ? `<div class="chips"><span class="muted" style="font-size:13px;min-width:76px">Topic</span><span class="chip alt">${esc(p.topic)}</span></div>` : ""}</section>` : ""}
        <section class="cite-box"><h2>Cite</h2>
          <div class="cite-tabs"><button class="btn sm active" data-fmt="apa">APA</button><button class="btn sm" data-fmt="bib">BibTeX</button><button class="btn sm" id="copyCite" style="margin-left:auto">Copy</button></div>
          <pre class="cite" id="cite"></pre>
        </section>
      </div>
      <section class="card" style="margin-top:14px"><h2>Similar papers</h2>
        <div class="similar">${sim.map((s) => `<a href="#/paper/${encodeURIComponent(s.id)}"><div>${esc(s.title)}</div><div class="meta">${esc(authorLine(s, 2))}${s.year ? ` · ${s.year}` : ""}</div></a>`).join("")}</div>
      </section>
    </article>`;
  if (S.visited > 1) $("#backLink").addEventListener("click", (e) => { e.preventDefault(); history.back(); });
  const cite = $("#cite");
  let fmt = "apa";
  const show = () => (cite.textContent = fmt === "apa" ? apa(p) : bibtex(p));
  show();
  app.querySelectorAll("[data-fmt]").forEach((b) => b.addEventListener("click", () => {
    fmt = b.dataset.fmt;
    app.querySelectorAll("[data-fmt]").forEach((x) => x.classList.toggle("active", x === b));
    show();
  }));
  $("#copyCite").addEventListener("click", async (e) => {
    try { await navigator.clipboard.writeText(cite.textContent); e.target.textContent = "Copied"; }
    catch { e.target.textContent = "Select & copy"; }
    setTimeout(() => (e.target.textContent = "Copy"), 1500);
  });
}

// ---------- Ask AI ----------
function renderAsk(params) {
  document.title = "Ask AI · AniSpeakArchive Demo";
  const active = params.has("q") ? parseInt(params.get("q"), 10) : -1;
  app.innerHTML = `
    <div class="ask">
      <div class="page-header">
        <h1>Ask AI</h1>
        <p class="muted">In the full system, Ask AI plans search queries, retrieves papers from the archive, and writes an answer grounded in them with numbered citations.</p>
      </div>
      <div class="ask-banner"><strong>Demo:</strong> live generation needs a GPU-hosted LLM, so this page replays pre-generated answers. Each answer cites only papers retrieved from this demo's sample. Pick a question:</div>
      <div class="starters">${S.ask.map((a, i) => `<button class="starter ${i === active ? "active" : ""}" data-i="${i}">${esc(a.question)}</button>`).join("")}</div>
      <div class="thread" id="thread"></div>
      <form class="ask-input" id="askForm">
        <input id="askQ" placeholder="Or type your own question…" maxlength="300" aria-label="Your question" />
        <button class="btn primary" type="submit">Ask</button>
      </form>
    </div>`;
  app.querySelectorAll(".starter").forEach((b) => b.addEventListener("click", () => history.pushState(null, "", `#/ask?q=${b.dataset.i}`) || renderAsk(new URLSearchParams(`q=${b.dataset.i}`))));
  $("#askForm").addEventListener("submit", (e) => { e.preventDefault(); askFree($("#askQ").value.trim()); });
  if (active >= 0 && S.ask[active]) playAnswer(S.ask[active]);
}

function sleep(ms) { return new Promise((r) => setTimeout(r, matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : ms)); }

async function playAnswer(a) {
  const thread = $("#thread");
  const token = Symbol();
  thread._token = token;
  thread.innerHTML = `<div class="bubble-q">${esc(a.question)}</div><div class="steps" id="steps"></div>`;
  const steps = $("#steps");
  const step = async (txt, ms) => {
    const d = document.createElement("div");
    d.className = "doing"; d.textContent = txt; steps.appendChild(d);
    await sleep(ms);
    d.className = "done";
  };
  await step("Planning search queries", 450);
  if (thread._token !== token) return;
  await step(`Searching ${S.papers.length} papers (hybrid BM25 + dense)`, 550);
  if (thread._token !== token) return;
  await step(`Grounding answer in top ${a.n_retrieved} records`, 500);
  if (thread._token !== token) return;

  const idFor = new Map(a.references.map((r) => [r.citation_id, r.id]));
  let html = marked.parse(a.answer_markdown);
  html = html.replace(/\[(\d+)\]/g, (m, n) => idFor.has(+n)
    ? `<a class="cite-ref" href="#/paper/${encodeURIComponent(idFor.get(+n))}" title="${esc(S.byId.get(idFor.get(+n))?.title || "")}">${n}</a>`
    : m);
  const card = document.createElement("div");
  card.className = "card answer flash";
  card.innerHTML = `${html}
    <div class="answer-foot"><span class="conf">Confidence: ${esc(a.confidence)}</span>${a.limitations ? `<span class="muted">${esc(a.limitations)}</span>` : ""}</div>
    <div class="refs"><h3 style="margin:0 0 8px;font-size:14px">References</h3><ol>${a.references.map((r) => `<li><span class="num">[${r.citation_id}]</span><span><a href="#/paper/${encodeURIComponent(r.id)}">${esc(r.title)}</a> <span class="muted">· ${esc(r.authors[0] || "")}${r.n_authors > 1 ? " et al." : ""}${r.year ? `, ${r.year}` : ""}</span></span></li>`).join("")}</ol></div>`;
  thread.appendChild(card);
}

async function askFree(q) {
  if (!q) return;
  const norm = (s) => s.toLowerCase().replace(/[^a-z0-9 ]/g, "").trim();
  const i = S.ask.findIndex((a) => norm(a.question) === norm(q));
  if (i >= 0) { history.pushState(null, "", `#/ask?q=${i}`); return renderAsk(new URLSearchParams(`q=${i}`)); }

  app.querySelectorAll(".starter").forEach((b) => b.classList.remove("active"));
  const thread = $("#thread");
  thread._token = Symbol();
  thread.innerHTML = `<div class="bubble-q">${esc(q)}</div><div class="card"><p style="margin-top:0"><strong>Live answers aren't available in this static demo.</strong> The full system sends retrieved papers to a GPU-hosted LLM. Here are the papers it would retrieve for your question:</p><div class="similar" id="freeHits"><span class="muted">Searching…</span></div></div>`;
  ensureEmbedder();
  const { ids } = await retrieve(q);
  const hits = ids.slice(0, 6).map((j) => S.papers[j]);
  $("#freeHits").innerHTML = hits.length
    ? hits.map((s) => `<a href="#/paper/${encodeURIComponent(s.id)}"><div>${esc(s.title)}</div><div class="meta">${esc(authorLine(s, 2))}${s.year ? ` · ${s.year}` : ""}</div></a>`).join("") +
      `<p class="muted" style="font-size:13px;margin:6px 0 0">${S.embedState === "ready" ? "Hybrid search." : "Keyword search (semantic model still loading)."} <a href="#/?q=${encodeURIComponent(q)}">See all results →</a></p>`
    : `<span class="muted">No matching papers in this sample.</span>`;
}

// ---------- about ----------
function renderAbout() {
  document.title = "About · AniSpeakArchive Demo";
  const years = S.papers.map((p) => p.year).filter(Boolean);
  app.innerHTML = `
    <div class="prose">
      <div class="page-header"><h1>About this demo</h1></div>
      <p>AniSpeakArchive is a retrieval system for research on animal communication and bioacoustics. This page is a small, static version of it that runs entirely in your browser.</p>
      <div class="stats">
        <div class="card stat"><div class="n">${S.papers.length}</div><div class="l">papers</div></div>
        <div class="card stat"><div class="n">${Math.min(...years)}–${Math.max(...years)}</div><div class="l">publication years</div></div>
        <div class="card stat"><div class="n">${S.dim}</div><div class="l">embedding dimensions</div></div>
        <div class="card stat"><div class="n">${S.ask.length}</div><div class="l">Ask AI examples</div></div>
      </div>
      <h2>How search works here</h2>
      <ul>
        <li><strong>Keyword (BM25)</strong> with MiniSearch over title, abstract, authors, venue and tags.</li>
        <li><strong>Semantic</strong>: paper vectors were precomputed with <code>BAAI/bge-small-en-v1.5</code>. Your query is embedded in the browser by the same model (via transformers.js, ~34 MB, cached after the first load).</li>
        <li>The two rankings are merged with <strong>reciprocal rank fusion</strong> (k = 60), as in the full system.</li>
      </ul>
      <h2>Demo vs. full system</h2>
      <ul>
        <li><strong>Corpus:</strong> the full pipeline filters OpenAlex with a fine-tuned SciBERT classifier and an LLM. This demo uses ${S.papers.length} papers from targeted OpenAlex searches.</li>
        <li><strong>Facets:</strong> the full system annotates species, modality, stages and more with an LLM. Here, tags come from keyword rules and will sometimes be wrong.</li>
        <li><strong>Models:</strong> the full system uses <code>bge-m3</code> embeddings and a cross-encoder reranker on GPU. The demo uses <code>bge-small</code> with no reranker.</li>
        <li><strong>Ask AI:</strong> the full system generates grounded answers live with a GPU-hosted LLM. The demo shows ${S.ask.length} pre-generated answers, each citing only papers retrieved from this sample.</li>
        <li><strong>Accounts, bookmarks and analytics</strong> are not included.</li>
      </ul>
      <h2>Data</h2>
      <p>Paper metadata and abstracts come from <a href="https://openalex.org" target="_blank" rel="noopener">OpenAlex</a> (CC0). Data built ${esc(S.meta.built)}.</p>
    </div>`;
}

// ---------- boot ----------
window.addEventListener("hashchange", render);
loadData().then(render).catch((err) => {
  console.error(err);
  app.innerHTML = `<div class="card empty"><p><strong>Couldn't load the demo data.</strong></p><p class="muted">If you opened index.html directly from disk, serve the folder instead, for example <code>python -m http.server</code> in demo/site.</p></div>`;
});

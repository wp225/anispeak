/* Run Instructions:
  This script is executed dynamically by the browser when loaded via paper.html.
  To apply changes, simply save this file and hard-refresh your browser (Ctrl+F5).
*/

function esc(s) {
  return String(s || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function bestUrl(p) {
  const raw =
    p.landing_page_url
    || (p.primary_location && p.primary_location.landing_page_url)
    || (p.primary_location && p.primary_location.url)
    || (p.open_access && p.open_access.oa_url)
    || p.doi
    || "";

  if (!raw) return "";
  if (raw.startsWith("http://") || raw.startsWith("https://")) return raw;
  if (raw.startsWith("10.")) return `https://doi.org/${raw}`;
  return raw;
}

function bestPdf(p) {
  return (p.best_oa_location && p.best_oa_location.pdf_url)
    || (p.primary_location && p.primary_location.pdf_url)
    || (p.open_access && p.open_access.oa_url)
    || "";
}

function tagBadges(p) {
    const out = [];
    const fields = ["species", "modalities", "features", "stages", "domains", "contexts"];
    for (const f of fields) {
        if (Array.isArray(p[f])) {
            for (const val of p[f]) {
                out.push(`<span class="tag-badge">${esc(val)}</span>`);
            }
        }
    }
    return out.join(" ");
}

function buildCslJson(p) {
  console.assert(typeof p === 'object', "Paper object must be valid to build CSL JSON");
  const title = p.title || p.display_name || "";
  const year = p.year;
  const authors = [];
  if (Array.isArray(p.authors)) {
    for (const a of p.authors) {
      if (typeof a === "string" && a.trim()) {
        const parts = a.trim().split(/\s+/);
        authors.push({ given: parts.slice(0, -1).join(" "), family: parts.slice(-1).join(" ") });
      } else if (a && typeof a === "object" && typeof a.name === "string") {
        const parts = a.name.trim().split(/\s+/);
        authors.push({ given: parts.slice(0, -1).join(" "), family: parts.slice(-1).join(" ") });
      }
    }
  }

  return [{
    id: p.doc_id || ("item-" + Math.random().toString(36).substr(2, 9)),
    type: "article-journal",
    title: title,
    author: authors.length ? authors : undefined,
    issued: year ? { "date-parts": [[Number(year)]] } : undefined,
    DOI: p.doi ? p.doi.replace("https://doi.org/", "") : undefined,
    URL: bestUrl(p) || undefined,
  }];
}

async function getCitationText(p, format) {
    console.assert(typeof p === 'object', "Paper object must be valid to generate citation text");
    const csl = buildCslJson(p);

    try {
        // Fix for "Cite is not defined" error in the browser environment
        const CiteModule = (typeof require !== 'undefined') ? require('citation-js') : (window.cite || window.Cite);
        const CiteCtor = (typeof CiteModule === "function") ? CiteModule : CiteModule.Cite;
        console.assert(typeof CiteCtor === "function", "Citation-js constructor not found");
        const cite = new CiteCtor(csl);
        let citeText = "";

        if (format === "bibtex") {
            citeText = cite.format("bibtex").trim();
        } else {
            citeText = cite.format("bibliography", { format: "text", template: format, lang: "en-US" }).trim();
        }

        return citeText;
    } catch (err) {
        console.error(`[Citation Debug] Citation.js error for paper ${p.doc_id}:`, err);
        return `${p.title} (${p.year}). ${bestUrl(p)}`;
    }
}

async function loadPaper() {
  const fetchFn =
    (window.asaFetch && typeof window.asaFetch === "function")
      ? window.asaFetch
      : fetch;

  const url =
    (window.asaUrl && typeof window.asaUrl === "function")
      ? window.asaUrl(`/api/paper/${encodeURIComponent(DOC_ID)}`)
      : `/api/paper/${encodeURIComponent(DOC_ID)}`;

  const resp = await fetchFn(url);
  const el = document.getElementById("paper");
  console.assert(el, "Expected #paper element to exist");

  if (!resp.ok) {
    el.innerHTML = "<p>Not found.</p>";
    return;
  }

  const p = await resp.json();
  console.assert(p && typeof p === "object", "Expected paper JSON object");

  const title = p.title || p.display_name || "";
  const url2 = bestUrl(p);
  const pdf = bestPdf(p);
  const ai = p.summary || p.ai_summary || (p.ai_metadata && p.ai_metadata.summary) || "";
  const abs = p.abstract_text || p.abstract || "";

  const venueFull = p.venue || p.host_venue || "";
  const authorsStr = Array.isArray(p.authors) ? p.authors.join(", ") : "";
  const instStr = Array.isArray(p.institutions) ? p.institutions.join(" • ") : "";


  const venueText = venueFull ? `<div style="margin-bottom: 6px;"><span style="font-weight:600;">Venue:</span> <span style="font-style:italic;">${esc(venueFull)}</span></div>` : "";
  const authorsText = authorsStr ? `<div style="margin-bottom: 6px;"><span style="font-weight:600;">Authors:</span> ${esc(authorsStr)}</div>` : "";
  const instText = instStr ? `<div style="margin-bottom: 6px;"><span style="font-weight:600;">Affiliations:</span> ${esc(instStr)}</div>` : "";

  // Render the exact synced grid from search.js modal
  el.innerHTML = `
    <div class="card" style="margin-bottom: 30px; padding: 30px;">
      <div class="paper-head" style="display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 1px solid var(--border2); padding-bottom: 15px; margin-bottom: 25px;">
        <div style="flex: 1; padding-right: 20px;">
          <h1 style="margin: 0; font-size: 1.8rem; line-height: 1.3;">${esc(title)} <span class="muted" style="font-weight:500; font-size: 1.4rem;">(${esc(p.year || "N/A")})</span></h1>
        </div>
        <div class="paper-actions" style="display:flex; gap: 10px; flex-shrink: 0;">
          ${pdf ? `<a class="btn primary" href="${esc(pdf)}" target="_blank" rel="noopener">📄 PDF</a>` : ""}
          ${url2 && !pdf ? `<a class="btn primary" href="${esc(url2)}" target="_blank" rel="noopener">📄 PDF (Landing)</a>` : ""}
        </div>
      </div>

      <div class="modal-grid" style="display: grid; grid-template-columns: 2fr 1fr; gap: 35px;">
          <div class="modal-main">
            <div class="meta muted" style="margin-bottom: 25px; font-size: 1.05rem; line-height: 1.5;">
                ${authorsText}
                ${instText}
                ${venueText}
            </div>
            ${ai ? `<div class="blk" style="border-left: 4px solid var(--accent); padding-left: 15px; margin-bottom: 25px;"><div class="lbl">✨ AI Summary</div><div style="font-size: 1.05rem; line-height: 1.6;">${esc(ai)}</div></div>` : ""}
            ${abs ? `<div class="blk"><div class="lbl">Abstract</div><div style="font-size: 1.05rem; line-height: 1.6;">${esc(abs)}</div></div>` : "<div class='muted'>No abstract available.</div>"}

            <div style="margin-top:30px;">
                <div class="lbl">Semantic Tags</div>
                <div class="tags-container">${tagBadges(p) || "<span class='muted'>None</span>"}</div>
            </div>
          </div>

          <div class="modal-sidebar">
            <div class="blk" style="background:var(--card); padding: 20px; border-radius: 8px; border: 1px solid var(--border2);">
              <div class="lbl" style="border-bottom: 1px solid var(--border2); padding-bottom: 8px; margin-bottom: 12px; font-size: 1.1rem;">Details</div>
              <div class="kv" style="display:flex; justify-content:space-between; margin-bottom:12px;"><span>Type</span><span style="font-weight:600;">${esc(p.paper_type || "N/A")}</span></div>
              <div class="kv" style="display:flex; justify-content:space-between; margin-bottom:12px;"><span>Tradition</span><span style="font-weight:600;">${esc(p.research_tradition || "N/A")}</span></div>
              <div class="kv" style="display:flex; justify-content:space-between; margin-bottom:12px;"><span>Code</span><span>${p.has_code ? "✅" : "❌"}</span></div>
              <div class="kv" style="display:flex; justify-content:space-between; margin-bottom:12px;"><span>Dataset</span><span>${p.has_dataset ? "✅" : "❌"}</span></div>
            </div>

            <div class="blk" style="background:var(--card); padding: 20px; border-radius: 8px; border: 1px solid var(--border2); margin-top: 25px;">
               <div class="lbl" style="border-bottom: 1px solid var(--border2); padding-bottom: 8px; margin-bottom: 12px; font-size: 1.1rem;">Citation Export</div>
               <div class="cite-picker" style="display:flex; flex-direction:column; gap: 12px;">
                  <select id="pCiteFormatSelect" style="padding: 10px; border-radius: 4px; border: 1px solid var(--border2); font-size: 1rem;">
                      <option value="bibtex" selected>BibTeX</option>
                      <option value="apa">APA</option>
                      <option value="chicago-author-date">Chicago</option>
                      <option value="vancouver">Vancouver</option>
                      <option value="harvard1">Harvard</option>
                  </select>
                  <button id="pCopyCiteBtn" class="btn primary">📋 Copy Citation</button>
               </div>
            </div>

            <div class="blk" style="background:var(--card); padding: 20px; border-radius: 8px; border: 1px solid var(--border2); margin-top: 25px;">
               <div class="lbl" style="border-bottom: 1px solid var(--border2); padding-bottom: 8px; margin-bottom: 12px; font-size: 1.1rem;">Downloads</div>
               <div style="display:flex; flex-direction:column; gap: 12px;">
                  <a class="btn" href="${(window.asaUrl && typeof window.asaUrl === "function") ? window.asaUrl(`/api/paper/${encodeURIComponent(DOC_ID)}/ris`) : `/api/paper/${encodeURIComponent(DOC_ID)}/ris`}" download style="text-align: center;">⬇️ Export RIS</a>
               </div>
            </div>
          </div>
      </div>
    </div>
  `;

  document.getElementById("pCopyCiteBtn").onclick = async () => {
      const format = document.getElementById("pCiteFormatSelect").value || "bibtex";
      const text = await getCitationText(p, format);
      await navigator.clipboard.writeText(text);

      const btn = document.getElementById("pCopyCiteBtn");
      const orig = btn.innerText;
      btn.innerText = "✓ Copied!";
      setTimeout(() => { btn.innerText = orig; }, 2000);
  };

  loadSimilarPapers();
}

async function loadSimilarPapers() {
  const section = document.getElementById("similarPapers");
  if (!section) return;

  const fetchFn = (window.asaFetch && typeof window.asaFetch === "function") ? window.asaFetch : fetch;
  const apiUrl = (window.asaUrl && typeof window.asaUrl === "function")
    ? window.asaUrl(`/api/paper/${encodeURIComponent(DOC_ID)}/similar`)
    : `/api/paper/${encodeURIComponent(DOC_ID)}/similar`;

  try {
    const resp = await fetchFn(apiUrl);
    if (!resp.ok) return;
    const data = await resp.json();
    const results = data.results || [];
    if (results.length === 0) { section.style.display = "none"; return; }

    section.innerHTML = `
      <h2 style="font-size:1.2rem; font-weight:800; margin:0 0 16px; letter-spacing:-.01em;">Similar Papers</h2>
      <div id="similarList" style="display:flex; flex-direction:column; gap:12px;"></div>
    `;

    const list = section.querySelector("#similarList");
    for (const p of results) {
      const title = p.title || "(untitled)";
      const year  = p.year  || "";
      const venue = p.venue || "";
      const href  = (window.asaUrl && typeof window.asaUrl === "function")
        ? window.asaUrl(`/paper/${encodeURIComponent(p.doc_id)}`)
        : `/paper/${encodeURIComponent(p.doc_id)}`;

      const authorsArr = Array.isArray(p.authors) ? p.authors : [];
      const authorsText = authorsArr.slice(0, 3).map(a => esc(typeof a === "string" ? a : (a.name || ""))).join(", ")
        + (authorsArr.length > 3 ? " <span style='color:var(--muted)'>+" + (authorsArr.length - 3) + " more</span>" : "");

      const speciesArr = Array.isArray(p.species) ? p.species.slice(0, 3) : [];
      const speciesBadges = speciesArr.map(s => `<span class="tag-badge">${esc(s)}</span>`).join(" ");

      const card = document.createElement("a");
      card.href = href;
      card.className = "similar-paper-card";
      card.innerHTML = `
        <div style="font-weight:700; font-size:14px; line-height:1.4; margin-bottom:4px;">${esc(title)}</div>
        ${authorsText ? `<div style="font-size:12px; color:var(--muted); margin-bottom:4px;">${authorsText}</div>` : ""}
        <div style="font-size:12px; color:var(--muted);">${year ? esc(String(year)) : ""}${year && venue ? " · " : ""}${venue ? esc(venue) : ""}</div>
        ${speciesBadges ? `<div style="margin-top:6px;">${speciesBadges}</div>` : ""}
      `;
      list.appendChild(card);
    }
  } catch (e) {
    section.style.display = "none";
  }
}

document.addEventListener("DOMContentLoaded", loadPaper);

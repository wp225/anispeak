(function () {
  "use strict";

  function esc(s) {
    return String(s || "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  async function jget(url) {
    url = (window.asaUrl && typeof window.asaUrl === "function") ? window.asaUrl(url) : url;
    const r = await ((window.asaFetch && typeof window.asaFetch === "function") ? window.asaFetch : fetch)(url);
    const j = await r.json().catch(() => null);
    return { ok: r.ok, status: r.status, json: j };
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

  async function jdel(url) {
    url = (window.asaUrl && typeof window.asaUrl === "function") ? window.asaUrl(url) : url;
    const r = await ((window.asaFetch && typeof window.asaFetch === "function") ? window.asaFetch : fetch)(url, { method: "DELETE" });
    const j = await r.json().catch(() => null);
    return { ok: r.ok, status: r.status, json: j };
  }

  function toast(msg, kind) {
    const host = document.getElementById("toastHost");
    if (!host) return;
    const d = document.createElement("div");
    d.className = "toast " + (kind || "info");
    d.textContent = msg;
    host.appendChild(d);
    setTimeout(() => d.classList.add("show"), 10);
    setTimeout(() => {
      d.classList.remove("show");
      setTimeout(() => d.remove(), 250);
    }, 3000);
  }

  async function fetchPaperMeta(docId) {
    const r = await jget("/api/paper/" + encodeURIComponent(docId));
    if (!r.ok || !r.json) return { doc_id: docId, title: docId };
    return r.json;
  }

  function tabInit() {
    const btns = document.querySelectorAll(".tab-btn");
    btns.forEach((b) => {
      b.addEventListener("click", () => {
        btns.forEach((x) => x.classList.remove("active"));
        b.classList.add("active");
        const tab = b.getAttribute("data-tab");
        ["papers", "collections"].forEach((t) => {
          const pane = document.getElementById("tab-" + t);
          if (!pane) return;
          pane.style.display = t === tab ? "block" : "none";
        });
      });
    });
  }

  async function renderCollections() {
    const pane = document.getElementById("tab-collections");
    if (!pane) return;
    const [r, papersR] = await Promise.all([jget("/api/library/collections"), jget("/api/library/papers")]);
    if (!r.ok) {
      pane.innerHTML = "<div class='muted'>Failed to load buckets.</div>";
      return;
    }
    const items = (r.json && r.json.items) || [];
    const papers = (papersR.ok && papersR.json && papersR.json.items) ? papersR.json.items : [];
    const byBucket = new Map();
    for (const p of papers) {
      const key = (p.collection_id === null || p.collection_id === undefined) ? "" : String(p.collection_id);
      if (!byBucket.has(key)) byBucket.set(key, []);
      byBucket.get(key).push(p);
    }
    let html = "";
    html += "<div style='display:flex; gap:10px; flex-wrap:wrap; align-items:center;'>";
    html += "<input id='newBucketName' class='input' style='max-width:320px;' placeholder='New bucket name' />";
    html += "<button class='btn sm' id='createBucketBtn'>Create</button>";
    html += "</div>";
    html += "<div style='margin-top:14px;'>";
    if (items.length === 0) {
      html += "<div class='muted'>No buckets yet. Create one to organize saved papers.</div>";
    } else {
      // Accordion view: show buckets, and lazily load paper metadata when expanded.
      html += "<div class='bucket-accordion-wrap'>";

      // Unassigned first
      const unassignedCount = (byBucket.get("") || []).length;
      html += "<details class='bucket-accordion' data-bucket-id=''>";
      html += "<summary><span>Unassigned</span><span class='bucket-count'>" + esc(String(unassignedCount)) + "</span></summary>";
      html += "<div class='bucket-body'><div class='muted'>Expand to load papers…</div></div>";
      html += "</details>";

      for (const c of items) {
        const k = String(c.id);
        const count = (byBucket.get(k) || []).length;
        html += "<details class='bucket-accordion' data-bucket-id='" + esc(k) + "'>";
        html += "<summary><span>" + esc(c.name) + "</span><span class='bucket-count'>" + esc(String(count)) + "</span></summary>";
        html += "<div class='bucket-actions'>";
        html += "<button class='btn sm danger' data-del-bucket='" + esc(c.id) + "'>Delete</button>";
        html += "</div>";
        html += "<div class='bucket-body'><div class='muted'>Expand to load papers…</div></div>";
        html += "</details>";
      }
      html += "</div>";
    }
    html += "</div>";
    pane.innerHTML = html;

    // Lazy load bucket contents on first open.
    pane.querySelectorAll("details.bucket-accordion").forEach((det) => {
      det.addEventListener("toggle", async () => {
        if (!det.open) return;
        if (det.getAttribute("data-loaded") === "1") return;
        det.setAttribute("data-loaded", "1");
        const bucketId = det.getAttribute("data-bucket-id") || "";
        const list = byBucket.get(bucketId) || [];
        const body = det.querySelector(".bucket-body");
        if (!body) return;
        if (list.length === 0) {
          body.innerHTML = "<div class='muted'>No papers in this bucket.</div>";
          return;
        }

        body.innerHTML = "<div class='muted'>Loading papers…</div>";
        const cap = 100;
        const metas = [];
        for (const it of list.slice(0, cap)) {
          metas.push(await fetchPaperMeta(it.doc_id));
        }

        let out = "";
        out += "<div class='bucket-paper-list'>";
        for (let i = 0; i < Math.min(list.length, cap); i++) {
          const it = list[i];
          const p = metas[i] || { title: it.doc_id };
          const title = p.title || p.display_name || it.doc_id;
          const year = p.year ? String(p.year) : "";
          const href = (window.asaUrl && typeof window.asaUrl === "function")
            ? window.asaUrl("/paper/" + encodeURIComponent(it.doc_id))
            : ("/paper/" + encodeURIComponent(it.doc_id));
          out += "<div class='bucket-paper'>";
          out += "<a class='bucket-paper-title' href='" + esc(href) + "' target='_blank' rel='noopener'>" + esc(title) + "</a>";
          out += year ? "<div class='bucket-paper-meta'>" + esc(year) + "</div>" : "";
          out += "</div>";
        }
        out += "</div>";
        if (list.length > cap) out += "<div class='muted' style='margin-top:8px;'>Showing first " + esc(String(cap)) + " papers.</div>";
        body.innerHTML = out;
      });
    });

    const createBtn = document.getElementById("createBucketBtn");
    if (createBtn) {
      createBtn.addEventListener("click", async () => {
        const name = String(document.getElementById("newBucketName").value || "").trim();
        const rr = await jpost("/api/library/collections", { name });
        if (rr.ok) {
          toast("Bucket created", "ok");
          await renderAll();
        } else {
          toast("Could not create bucket", "error");
        }
      });
    }
    pane.querySelectorAll("[data-del-bucket]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const id = btn.getAttribute("data-del-bucket");
        if (!confirm("Delete this bucket? Saved papers will become unassigned.")) return;
        const rr = await jdel("/api/library/collections/" + encodeURIComponent(id));
        if (rr.ok) {
          toast("Bucket deleted", "ok");
          await renderAll();
        }
      });
    });
  }

  async function renderPapers() {
    const pane = document.getElementById("tab-papers");
    if (!pane) return;
    const [papersR, colR] = await Promise.all([jget("/api/library/papers"), jget("/api/library/collections")]);
    if (!papersR.ok) {
      pane.innerHTML = "<div class='muted'>Failed to load saved papers.</div>";
      return;
    }
    const items = (papersR.json && papersR.json.items) || [];
    const cols = (colR.ok && colR.json && colR.json.items) ? colR.json.items : [];
    const colOpts = ["<option value=''>Unassigned</option>"]
      .concat(cols.map((c) => "<option value='" + esc(c.id) + "'>" + esc(c.name) + "</option>"))
      .join("");

    if (items.length === 0) {
      pane.innerHTML = "<div class='muted'>No saved papers yet. Use the Save button on a paper or search result.</div>";
      return;
    }

    // Fetch metadata lazily to avoid a thundering herd.
    pane.innerHTML = "<div class='muted'>Loading…</div>";

    const metas = [];
    for (const it of items.slice(0, 50)) {
      // cap initial; users can paginate later
      metas.push(await fetchPaperMeta(it.doc_id));
    }

    let html = "";
    html += "<div class='profile-grid'>";
    for (let i = 0; i < Math.min(items.length, 50); i++) {
      const it = items[i];
      const p = metas[i] || { title: it.doc_id };
      const title = p.title || p.display_name || it.doc_id;
      const year = p.year || "";
      html += "<div class='bookmark-item saved-paper-card'>";
      html += "<div class='bookmark-head'>";
      const paperHref = (window.asaUrl && typeof window.asaUrl === "function")
        ? window.asaUrl("/paper/" + encodeURIComponent(it.doc_id))
        : ("/paper/" + encodeURIComponent(it.doc_id));
      html += "<a class='bookmark-title' href='" + esc(paperHref) + "'>" + esc(title) + "</a>";
      html += "<div class='bookmark-actions'>";
      html += "<a class='btn sm btn-primary' target='_blank' rel='noopener' href='" + esc(paperHref) + "'>Open details</a>";
      html += "<button class='btn sm danger' data-unsave='" + esc(it.doc_id) + "'>Remove</button>";
      html += "</div>";
      html += "</div>";
      html += year ? "<div class='bookmark-meta'>" + esc(String(year)) + "</div>" : "";
      html += "<div style='display:flex; gap:10px; flex-wrap:wrap; margin-top:10px; align-items:center;'>";
      html += "<label class='muted' style='font-size:12px;'>Bucket</label>";
      html += "<select class='input' style='min-width:220px;' data-bucket='" + esc(it.doc_id) + "'>" + colOpts + "</select>";
      html += "<a class='btn sm' href='" + ((window.asaUrl && typeof window.asaUrl === "function") ? window.asaUrl("/api/citation/" + encodeURIComponent(it.doc_id) + "?format=bibtex") : ("/api/citation/" + encodeURIComponent(it.doc_id) + "?format=bibtex")) + "'>BibTeX</a>";
      html += "<a class='btn sm' href='" + ((window.asaUrl && typeof window.asaUrl === "function") ? window.asaUrl("/api/citation/" + encodeURIComponent(it.doc_id) + "?format=apa") : ("/api/citation/" + encodeURIComponent(it.doc_id) + "?format=apa")) + "'>APA</a>";
      html += "</div>";
      html += "<div style='margin-top:10px;'>";
      html += "<textarea class='input' rows='3' placeholder='Add a note…' data-note='" + esc(it.doc_id) + "'>" + esc(it.note || "") + "</textarea>";
      html += "<div style='margin-top:8px; display:flex; gap:10px;'>";
      html += "<button class='btn sm' data-save-note='" + esc(it.doc_id) + "'>Save note</button>";
      html += "</div>";
      html += "</div>";
      html += "</div>";
    }
    html += "</div>";
    if (items.length > 50) {
      html += "<div class='muted' style='margin-top:12px;'>Showing first 50 saved papers.</div>";
    }
    pane.innerHTML = html;

    // set selected bucket values
    pane.querySelectorAll("[data-bucket]").forEach((sel) => {
      const docId = sel.getAttribute("data-bucket");
      const it = items.find((x) => x.doc_id === docId);
      sel.value = it && it.collection_id ? String(it.collection_id) : "";
      sel.addEventListener("change", async () => {
        const v = sel.value;
        const rr = await jpost("/api/library/papers/update", { doc_id: docId, collection_id: v ? Number(v) : null });
        if (rr.ok) toast("Updated", "ok");
        else toast("Update failed", "error");
      });
    });

    pane.querySelectorAll("[data-save-note]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const docId = btn.getAttribute("data-save-note");
        const ta = pane.querySelector("[data-note='" + CSS.escape(docId) + "']");
        const note = ta ? ta.value : "";
        const rr = await jpost("/api/library/papers/update", { doc_id: docId, note });
        if (rr.ok) toast("Note saved", "ok");
        else toast("Could not save note", "error");
      });
    });

    pane.querySelectorAll("[data-unsave]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const docId = btn.getAttribute("data-unsave");
        const rr = await jpost("/api/library/papers/toggle", { doc_id: docId, source: "paper" });
        if (rr.ok) {
          toast("Removed", "ok");
          await renderAll();
        }
      });
    });
  }

  async function renderAll() {
    await Promise.all([renderPapers(), renderCollections()]);
  }

  document.addEventListener("DOMContentLoaded", function () {
    tabInit();
    renderAll().catch((e) => console.warn("profile render failed", e));
  });
})();
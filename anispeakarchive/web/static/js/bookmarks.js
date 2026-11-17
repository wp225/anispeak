(function () {
  "use strict";

  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  }

  async function getJSON(url, opts) {
    url = (window.asaUrl && typeof window.asaUrl === "function") ? window.asaUrl(url) : url;
    const r = await ((window.asaFetch && typeof window.asaFetch === "function") ? window.asaFetch : fetch)(url, opts || {});
    let j = null;
    try {
      j = await r.json();
    } catch (_) {
      j = null;
    }
    return { ok: r.ok, status: r.status, json: j };
  }

  async function initPaperBookmark() {
    if (typeof DOC_ID === "undefined") return;
    const host = document.getElementById("paperActions");
    if (!host) return;

    // If auth isn't enabled server-side, the endpoint won't exist.
    const me = await getJSON("/api/bookmarks/me");
    if (!me.ok && me.status === 404) return;

    const btn = el("button", "btn", "Save");
    btn.type = "button";
    host.appendChild(btn);

    const bib = el("a", "btn sm", "BibTeX");
    bib.href = (window.asaUrl && typeof window.asaUrl === "function") ? window.asaUrl("/api/citation/" + encodeURIComponent(DOC_ID) + "?format=bibtex") : ("/api/citation/" + encodeURIComponent(DOC_ID) + "?format=bibtex");
    bib.style.textDecoration = "none";
    const apa = el("a", "btn sm", "APA");
    apa.href = (window.asaUrl && typeof window.asaUrl === "function") ? window.asaUrl("/api/citation/" + encodeURIComponent(DOC_ID) + "?format=apa") : ("/api/citation/" + encodeURIComponent(DOC_ID) + "?format=apa");
    apa.style.textDecoration = "none";
    host.appendChild(bib);
    host.appendChild(apa);

    async function refresh() {
      const h = await getJSON("/api/library/papers/has?doc_id=" + encodeURIComponent(DOC_ID));
      if (!h.ok) {
        btn.classList.remove("btn-primary");
        btn.textContent = "Save";
        return;
      }
      const saved = !!(h.json && h.json.saved);
      if (saved) {
        btn.classList.add("btn-primary");
        btn.textContent = "Saved";
      } else {
        btn.classList.remove("btn-primary");
        btn.textContent = "Save";
      }
    }

    btn.addEventListener("click", async function () {
      const r = await getJSON("/api/library/papers/toggle", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ doc_id: DOC_ID, source: "paper" }),
      });
      if (!r.ok && r.status === 401) {
        window.asaToast && window.asaToast("Sign in to save papers.", "warn");
        window.location.href = (window.asaUrl && typeof window.asaUrl === "function") ? window.asaUrl("/login") : "/login";
        return;
      }
      if (r.ok && r.json) {
        window.asaToast && window.asaToast(r.json.saved ? "Saved" : "Removed", "ok");
      }
      await refresh();
    });

    // If not logged in, keep as a sign-in CTA.
    if (!(me.json && me.json.logged_in)) {
      btn.textContent = "Sign in to save";
      btn.addEventListener("click", function () {
        window.location.href = (window.asaUrl && typeof window.asaUrl === "function") ? window.asaUrl("/login") : "/login";
      }, { once: true });
      return;
    }

    await refresh();
  }

  document.addEventListener("DOMContentLoaded", function () {
    initPaperBookmark().catch(function (e) {
      console.warn("[bookmarks] init failed", e);
    });
  });
})();

// Global UI behavior (no backend/API changes)

function asaToast(msg, kind) {
  const host = document.getElementById("toastHost");
  if (!host) return;
  const d = document.createElement("div");
  d.className = "toast " + (kind || "info");
  d.textContent = String(msg || "");
  host.appendChild(d);
  setTimeout(() => d.classList.add("show"), 10);
  setTimeout(() => {
    d.classList.remove("show");
    setTimeout(() => d.remove(), 250);
  }, 3200);
}

// Expose globally for other scripts.
window.asaToast = asaToast;

function asaScriptRoot() {
  const r = (typeof window.__ASA_SCRIPT_ROOT === "string") ? window.__ASA_SCRIPT_ROOT : "";
  // script_root is "" at site root; otherwise like "/anispeakarchive"
  return (r && r !== "/") ? r.replace(/\/$/, "") : "";
}

function asaUrl(path) {
  const p = String(path || "");
  // Only prefix absolute app paths.
  if (!p.startsWith("/")) return p;

  const root = asaScriptRoot();
  if (!root) return p;

  // Idempotent: if already rooted, don't double-prefix.
  if (p === root) return p;
  if (p.startsWith(root + "/")) return p;

  return root + p;
}

// Expose globally for other scripts.
window.asaUrl = asaUrl;

// Wrapper to ensure fetch targets respect subpath deployments.
window.asaFetch = function(input, init) {
  if (typeof input === "string") return fetch(asaUrl(input), init);
  return fetch(input, init);
};


function toggleClass(el, cls, on) {
  if (!el) return;
  el.classList.toggle(cls, !!on);
}

document.addEventListener("DOMContentLoaded", () => {
  // Render server flash messages as toasts.
  try {
    const flashes = Array.isArray(window.__ASA_FLASHES) ? window.__ASA_FLASHES : [];
    for (const it of flashes) {
      if (!Array.isArray(it) || it.length < 2) continue;
      const category = String(it[0] || "info").toLowerCase();
      const msg = String(it[1] || "");
      const kind = (category === "ok") ? "ok" : (category === "error") ? "error" : (category === "warn") ? "warn" : "info";
      asaToast(msg, kind);
    }
    // Prevent double-render on hot reload.
    window.__ASA_FLASHES = [];
  } catch (e) {
    console.warn("[toast] failed to render flashes", e);
  }

  // Mobile navbar
  const navToggle = document.getElementById("navToggle");
  const navLinks = document.getElementById("navLinks");
  if (navToggle && navLinks) {
    navToggle.addEventListener("click", () => {
      const isOpen = navLinks.classList.toggle("open");
      navToggle.setAttribute("aria-expanded", isOpen ? "true" : "false");
    });

    document.addEventListener("click", (e) => {
      if (!(e.target instanceof Element)) return;
      if (!navLinks.classList.contains("open")) return;
      if (navLinks.contains(e.target) || navToggle.contains(e.target)) return;
      navLinks.classList.remove("open");
      navToggle.setAttribute("aria-expanded", "false");
    });
  }
});

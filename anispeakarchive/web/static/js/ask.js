const SCRIPT_ROOT = window.__ASA_SCRIPT_ROOT || '';
const MAX_ASK_CHARS = 2000;

let conversations = [];
let activeId = null;
let activeMessages = [];
let isLoading = false;

const $messages = document.getElementById('askMessages');
const $empty = document.getElementById('askEmpty');
const $input = document.getElementById('askInput');
const $sendBtn = document.getElementById('askSend');
const $newBtn = document.getElementById('btnNewChat');
const $historyList = document.getElementById('askHistoryList');
const $sidebar = document.getElementById('askSidebar');
const $toggleBtn = document.getElementById('btnToggleSidebar');
const $clearBtn = document.getElementById('btnClearHistory');
const $charCount = document.getElementById('askCharCount');

async function apiFetch(url, opts = {}) {
  const res = await fetch(SCRIPT_ROOT + url, {
    headers: { 'Content-Type': 'application/json', ...(opts.headers || {}) },
    ...opts,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Server error ${res.status}`);
  return data;
}

async function streamAsk(body, onEvent) {
  const res = await fetch(SCRIPT_ROOT + '/api/ask/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `Server error ${res.status}`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf('\n\n')) >= 0) {
      const chunk = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      let event = 'message';
      let data = '';
      for (const line of chunk.split('\n')) {
        if (line.startsWith('event: ')) event = line.slice(7).trim();
        if (line.startsWith('data: ')) data += line.slice(6);
      }
      if (data) onEvent(event, JSON.parse(data));
    }
  }
}

async function loadHistory() {
  try {
    const data = await apiFetch('/api/ask/conversations');
    conversations = data.conversations || [];
  } catch {
    conversations = [];
  }
}

async function deleteConversation(id) {
  try { await apiFetch(`/api/ask/conversations/${id}`, { method: 'DELETE' }); } catch {}
  conversations = conversations.filter(c => c.id !== id);
  if (activeId === id) {
    activeId = null;
    activeMessages = [];
    renderMessages([]);
  }
  renderHistory();
}

async function clearAllConversations() {
  if (!confirm('Delete all conversation history?')) return;
  try { await apiFetch('/api/ask/conversations', { method: 'DELETE' }); } catch {}
  conversations = [];
  activeId = null;
  activeMessages = [];
  renderMessages([]);
  renderHistory();
}

async function loadConversation(id) {
  activeId = id;
  renderHistory();
  try {
    const data = await apiFetch(`/api/ask/conversations/${id}/messages`);
    activeMessages = (data.messages || []).map(m => {
      if (m.role === 'assistant' && m.response_json) {
        try {
          const payload = JSON.parse(m.response_json);
          return {
            role: 'assistant',
            content: payload.answer || m.content,
            references: payload.references || [],
            meta: payload.model_output || null,
            answerHtml: payload.answer_html || payload.answerHtml || null,
          };
        } catch {}
      }
      return { role: m.role, content: m.content, references: [], meta: null };
    });
    renderMessages(activeMessages);
  } catch {
    renderMessages([]);
  }
}

function newConversation() {
  activeId = null;
  activeMessages = [];
  renderMessages([]);
  renderHistory();
  $input.focus();
}

function groupByDate(convs) {
  const groups = {};
  const now = new Date();
  const today = now.toDateString();
  const yest = new Date(now - 864e5).toDateString();
  for (const c of convs) {
    const d = new Date(c.updated_at * 1000).toDateString();
    const label = d === today ? 'Today' : d === yest ? 'Yesterday' : new Date(c.updated_at * 1000).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
    (groups[label] = groups[label] || []).push(c);
  }
  return groups;
}

function renderHistory() {
  $historyList.innerHTML = '';
  if (!conversations.length) {
    $historyList.innerHTML = '<p class="ask-history-empty">No conversations yet.</p>';
    return;
  }
  const groups = groupByDate(conversations);
  for (const [label, convs] of Object.entries(groups)) {
    const groupEl = document.createElement('div');
    groupEl.innerHTML = `<div class="ask-history-group-label">${label}</div>`;
    $historyList.appendChild(groupEl);
    for (const conv of convs) {
      const btn = document.createElement('button');
      btn.className = 'ask-history-item' + (conv.id === activeId ? ' active' : '');
      btn.innerHTML = `<span class="ask-history-item-title">${escHtml(conv.title)}</span><button class="ask-history-delete" data-del="${conv.id}" title="Delete">✕</button>`;
      btn.addEventListener('click', e => { if (!e.target.closest('[data-del]')) loadConversation(conv.id); });
      btn.querySelector('[data-del]').addEventListener('click', e => { e.stopPropagation(); deleteConversation(conv.id); });
      $historyList.appendChild(btn);
    }
  }
}

function renderMessages(messages) {
  document.querySelectorAll('.ask-message-row, .ask-typing-row').forEach(el => el.remove());
  $empty.style.display = messages.length ? 'none' : '';
  for (const msg of messages) appendBubble(msg.role, msg.content, msg.references || [], msg.meta || null, msg.answerHtml || msg.answer_html || null);
  scrollToBottom();
}

function bubbleInnerHtml(role, content, references = [], meta = null, answerHtml = null) {
  let html = `<div class="ask-bubble">${role === 'assistant' ? (answerHtml || renderPlainText(content)) : escHtml(content)}</div>`;
  if (role === 'assistant' && meta) {
    const lim = meta.limitations ? `<div class="ask-meta-line">${escHtml(meta.limitations)}</div>` : '';
    html += `<div class="ask-meta-line">Confidence: ${escHtml(meta.confidence || 'unknown')}</div>${lim}`;
  }
  if (role === 'assistant' && references.length) {
    html += renderReferences(references);
  }
  return html;
}

function appendBubble(role, content, references = [], meta = null, answerHtml = null) {
  $empty.style.display = 'none';
  const row = document.createElement('div');
  row.className = `ask-message-row ${role}`;
  row.innerHTML = bubbleInnerHtml(role, content, references, meta, answerHtml);
  $messages.appendChild(row);
  scrollToBottom();
  return row;
}

function renderReferences(refs) {
  const cards = refs.map(ref => {
    const authors = Array.isArray(ref.authors) ? ref.authors.join(', ') : '';
    const paperUrl = ref.doc_id ? `/paper/${encodeURIComponent(ref.doc_id)}` : '';
    const doiUrl = ref.doi ? `https://doi.org/${String(ref.doi).replace(/^https?:\/\/doi\.org\//i, '')}` : '';
    return `
      <div class="ask-ref-card${paperUrl ? ' ask-ref-card--clickable' : ''}"
           ${paperUrl ? `onclick="window.open('${escAttr(paperUrl)}','_blank')"` : ''}>
        <div class="ask-ref-num">[${escHtml(String(ref.citation_id))}]</div>
        <div class="ask-ref-title">${escHtml(ref.title || '')}</div>
        <div class="ask-ref-authors">${escHtml(authors)}</div>
        ${doiUrl ? `<a class="ask-ref-doi" href="${escHtml(doiUrl)}" target="_blank" rel="noopener" onclick="event.stopPropagation()">DOI</a>` : ''}
      </div>
    `;
  }).join('');
  return `<div class="ask-refs-section"><div class="ask-refs-label">References</div><div class="ask-refs-grid">${cards}</div></div>`;
}

function createStreamingBubble() {
  const row = document.createElement('div');
  row.className = 'ask-message-row assistant';
  row.innerHTML = bubbleInnerHtml('assistant', '');
  $messages.appendChild(row);
  scrollToBottom();
  return row;
}

function updateStreamingBubble(row, content) {
  if (!row) return;
  row.innerHTML = bubbleInnerHtml('assistant', content || '');
  scrollToBottom();
}

function scrollToBottom() {
  $messages.scrollTop = $messages.scrollHeight;
}

function renderPlainText(text) {
  return `<div class="ask-md-p" style="white-space: pre-wrap;">${escHtml(String(text || ''))}</div>`;
}

function escHtml(s) {
  return String(s || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function escAttr(s) {
  return escHtml(s).replace(/'/g, '&#39;');
}

async function sendMessage() {
  if (isLoading) return;
  const message = $input.value.trim();
  if (!message || message.length > MAX_ASK_CHARS) return;

  $input.value = '';
  autoResize();
  updateCharCount();
  setLoading(true);

  activeMessages.push({ role: 'user', content: message });
  appendBubble('user', message);
  const streamingRow = createStreamingBubble();
  let streamedText = '';

  try {
    let finalPayload = null;
    await streamAsk({ message, conversation_id: activeId }, (event, payload) => {
      if (event === 'conversation') {
        activeId = payload.conversation_id;
        const existing = conversations.find(c => c.id === payload.conversation_id);
        if (!existing) {
          conversations.unshift({
            id: payload.conversation_id,
            title: payload.title,
            created_at: payload.created_at,
            updated_at: payload.created_at,
          });
        } else {
          existing.title = payload.title;
          existing.updated_at = payload.created_at;
        }
        renderHistory();
      } else if (event === 'token') {
        streamedText += String(payload.text || '');
        updateStreamingBubble(streamingRow, streamedText);
      } else if (event === 'final') {
        finalPayload = payload;
      } else if (event === 'error') {
        throw new Error(payload.error || 'Ask AI failed');
      }
    });

    streamingRow.remove();
    if (!finalPayload) throw new Error('Ask AI returned no final payload');
    const answer = finalPayload.answer || '(no response)';
    const refs = finalPayload.references || [];
    const meta = finalPayload.model_output || null;
    const answerHtml = finalPayload.answer_html || finalPayload.answerHtml || null;
    activeMessages.push({ role: 'assistant', content: answer, references: refs, meta, answerHtml });
    appendBubble('assistant', answer, refs, meta, answerHtml);
    const conv = conversations.find(c => c.id === activeId);
    if (conv) conv.updated_at = Math.floor(Date.now() / 1000);
    renderHistory();
  } catch (err) {
    streamingRow.remove();
    appendBubble('assistant', `Request failed: ${err.message}`);
    activeMessages.push({ role: 'assistant', content: `Request failed: ${err.message}` });
  } finally {
    setLoading(false);
    $input.focus();
  }
}

function setLoading(val) {
  isLoading = val;
  $sendBtn.disabled = val || !$input.value.trim() || $input.value.trim().length > MAX_ASK_CHARS;
  $input.disabled = val;
}

function autoResize() {
  $input.style.height = 'auto';
  $input.style.height = Math.min($input.scrollHeight, 160) + 'px';
}

function updateCharCount() {
  const n = $input.value.length;
  $charCount.textContent = `${n}/${MAX_ASK_CHARS}`;
  $charCount.style.color = n > MAX_ASK_CHARS ? '#b91c1c' : '';
}

$sendBtn.addEventListener('click', sendMessage);
$input.addEventListener('input', () => {
  autoResize();
  updateCharCount();
  $sendBtn.disabled = isLoading || !$input.value.trim() || $input.value.trim().length > MAX_ASK_CHARS;
});
$input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});


$newBtn.addEventListener('click', newConversation);
$clearBtn.addEventListener('click', clearAllConversations);
$toggleBtn.addEventListener('click', () => $sidebar.classList.toggle('open'));
document.querySelectorAll('.ask-starter-btn').forEach(btn => btn.addEventListener('click', () => {
  $input.value = btn.textContent.trim();
  updateCharCount();
  autoResize();
  $sendBtn.disabled = false;
  sendMessage();
}));

(async function boot() {
  updateCharCount();
  await loadHistory();
  renderHistory();
})();

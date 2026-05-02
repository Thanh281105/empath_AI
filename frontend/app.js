/**
 * ArXiv RAG Frontend — App Logic (v2: Ethereal AI Edition)
 * WebSocket Streaming + SQLite Chat History
 */

let ws = null;
let wsReconnectAttempt = 0;
const MAX_RECONNECT = 5;

let isProcessing = false;
let chatHistory = [];
let currentSessionId = null;

// Streaming state
let currentStreamingEl = null;
let currentStreamingText = '';

// ═══════════════════════════════════════════════════════
// Init
// ═══════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
    connectWebSocket();
    loadSessions();

    // Enter key submits
    const input = document.getElementById('queryInput');
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit(e);
        }
    });
});

// ═══════════════════════════════════════════════════════
// WebSocket
// ═══════════════════════════════════════════════════════
function connectWebSocket() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${location.host}/ws/chat`;
    try { ws = new WebSocket(wsUrl); } catch { return; }

    ws.onopen = () => {
        wsReconnectAttempt = 0;
        updateStatus('connected');
    };

    ws.onmessage = (e) => {
        try { handleWsMessage(JSON.parse(e.data)); } catch {}
    };

    ws.onclose = () => {
        updateStatus('disconnected');
        if (wsReconnectAttempt < MAX_RECONNECT) {
            const delay = Math.min(1000 * 2 ** wsReconnectAttempt, 30000);
            wsReconnectAttempt++;
            setTimeout(connectWebSocket, delay);
        }
    };

    ws.onerror = () => {};
}

function updateStatus(s) {
    const el = document.getElementById('topNavStatus');
    if (!el) return;
    const dot = el.querySelector('span:first-child');
    const txt = el.querySelector('span:last-child');
    if (s === 'connected') {
        dot.className = 'w-2 h-2 rounded-full bg-tertiary animate-pulse';
        txt.textContent = 'System Online';
    } else {
        dot.className = 'w-2 h-2 rounded-full bg-amber-400 animate-pulse';
        txt.textContent = 'Reconnecting...';
    }
}

// ═══════════════════════════════════════════════════════
// WebSocket Message Handler
// ═══════════════════════════════════════════════════════
function handleWsMessage(data) {
    if (data.type === 'status') {
        updateTypingText(data.message || 'Đang xử lý...');
    } else if (data.type === 'stream') {
        removeTyping();
        appendStreamToken(data.token || '');
    } else if (data.type === 'answer') {
        removeTyping();
        // Update currentSessionId from server
        if (data.session_id) currentSessionId = data.session_id;
        finalizeStream(data.answer, data.sources || [], data.agent_trace, data.processing_time_ms || 0);
        isProcessing = false;
        setSubmitEnabled(true);
        chatHistory.push({ role: 'assistant', content: data.answer });
        // Reload sessions list (new session appeared)
        loadSessions();
    } else if (data.type === 'error') {
        removeTyping();
        clearStream();
        isProcessing = false;
        setSubmitEnabled(true);
        addAIMessage(`❌ ${data.message || 'Lỗi không xác định'}`, [], null, 0);
    }
}

// ═══════════════════════════════════════════════════════
// Streaming
// ═══════════════════════════════════════════════════════
function appendStreamToken(token) {
    if (!currentStreamingEl) {
        const area = document.getElementById('messagesArea');
        const div = document.createElement('div');
        div.id = 'streaming-msg';
        div.className = 'flex flex-col items-start group';
        div.innerHTML = `
            <div class="flex items-center gap-3 mb-2">
                <span class="text-xs font-semibold text-primary">EmpathAI</span>
            </div>
            <div class="bg-surface-container-low p-6 rounded-2xl rounded-tl-none border border-primary/5 text-on-surface max-w-[90%] shadow-2xl relative overflow-hidden">
                <div class="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-tertiary to-transparent opacity-40"></div>
                <div class="streaming-text prose text-base leading-relaxed"></div>
                <span class="inline-block w-2 h-4 bg-primary animate-pulse ml-1"></span>
            </div>`;
        area.appendChild(div);
        currentStreamingEl = div.querySelector('.streaming-text');
        currentStreamingText = '';
        scrollToBottom();
    }
    currentStreamingText += token;
    currentStreamingEl.innerHTML = formatMarkdown(currentStreamingText);
    scrollToBottom();
}

function finalizeStream(answer, sources, trace, timeMs) {
    const el = document.getElementById('streaming-msg');
    if (el) el.remove();
    clearStream();
    addAIMessage(answer, sources, trace, timeMs);
}

function clearStream() {
    currentStreamingEl = null;
    currentStreamingText = '';
}

// ═══════════════════════════════════════════════════════
// Submit
// ═══════════════════════════════════════════════════════
async function handleSubmit(e) {
    if (e && e.preventDefault) e.preventDefault();
    if (isProcessing) return;

    const input = document.getElementById('queryInput');
    const modelSelect = document.getElementById('modelSelect');
    const model = modelSelect.value;
    const q = input.value.trim();
    if (!q) return;

    // Switch to chat view
    showChatView();
    addUserMessage(q);
    chatHistory.push({ role: 'user', content: q });
    input.value = '';

    if (model === 'compare') {
        await runCompare(q);
        return;
    }

    if (model === 'empath') {
        submitEmpath(q);
    } else {
        submitHttpModel(model, q);
    }
}

function submitEmpath(q) {
    showTyping();
    isProcessing = true;
    setSubmitEnabled(false);
    clearStream();

    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            question: q,
            session_id: currentSessionId,
            top_k: 5,
            history: chatHistory.slice(-10)
        }));
    } else {
        removeTyping();
        isProcessing = false;
        setSubmitEnabled(true);
        addAIMessage('⚠️ Không thể kết nối server. Kiểm tra Docker + Rust + Python workers.', [], null, 0);
    }
}

async function submitHttpModel(model, q) {
    const ports = { req1: 8001, req2: 8002, req3: 8003 };
    const labels = { req1: 'LLM Only', req2: 'LLM Fine-tune', req3: 'LLM + RAG' };
    const port = ports[model];
    const label = labels[model];

    showTyping();
    isProcessing = true;
    setSubmitEnabled(false);

    try {
        const res = await fetch(`http://localhost:${port}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: q, history: chatHistory.slice(-10) })
        });
        removeTyping();
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        addAIMessage(data.answer, [], null, data.processing_time_ms || 0, label);
        chatHistory.push({ role: 'assistant', content: data.answer });
    } catch (err) {
        removeTyping();
        addAIMessage(`❌ Lỗi kết nối ${label}: ${err.message}`, [], null, 0, label);
    } finally {
        isProcessing = false;
        setSubmitEnabled(true);
    }
}

async function runCompare(q) {
    isProcessing = true;
    setSubmitEnabled(false);
    const area = document.getElementById('messagesArea');
    area.innerHTML = '';
    area.classList.remove('hidden');
    // Re-add user message so it's visible above the compare grid
    addUserMessage(q);

    const models = [
        { id: 'empath', label: 'EmpathAI (Full LangGraph)', type: 'ws' },
        { id: 'req1', label: 'LLM Only (Groq)', type: 'http', port: 8001 },
        { id: 'req2', label: 'LLM Fine-tune', type: 'http', port: 8002 },
        { id: 'req3', label: 'LLM + RAG', type: 'http', port: 8003 },
    ];

    // Create 4-panel grid
    const grid = document.createElement('div');
    grid.className = 'grid grid-cols-1 md:grid-cols-2 gap-4';
    grid.id = 'compare-grid';

    models.forEach(m => {
        const panel = document.createElement('div');
        panel.className = 'bg-surface-container-low rounded-2xl border border-primary/5 p-4 flex flex-col min-h-[200px]';
        panel.id = `compare-panel-${m.id}`;
        panel.innerHTML = `
            <div class="flex items-center gap-2 mb-3 pb-3 border-b border-outline-variant/10">
                <span class="material-symbols-outlined text-primary text-lg">model_training</span>
                <span class="text-sm font-bold text-primary">${m.label}</span>
                <span id="compare-status-${m.id}" class="ml-auto text-[10px] text-tertiary animate-pulse">Đang xử lý...</span>
            </div>
            <div id="compare-content-${m.id}" class="text-base text-on-surface/90 leading-relaxed flex-1">
                <div class="flex items-center gap-2 text-on-surface-variant"><span class="w-2 h-2 rounded-full bg-primary/50 animate-bounce"></span><span class="text-xs">Đang phản hồi...</span></div>
            </div>
        `;
        grid.appendChild(panel);
    });

    area.appendChild(grid);
    scrollToBottom();

    // Run all in parallel
    const promises = models.map(m => runCompareModel(m, q));
    await Promise.all(promises);

    isProcessing = false;
    setSubmitEnabled(true);
    loadSessions(); // refresh history sidebar — EmpathAI session was saved during compare
}

async function runCompareModel(m, q) {
    const contentEl = document.getElementById(`compare-content-${m.id}`);
    const statusEl = document.getElementById(`compare-status-${m.id}`);

    try {
        let answer = '';
        let timeMs = 0;
        if (m.type === 'ws') {
            answer = await compareEmpathWS(q);
        } else {
            const res = await fetch(`http://localhost:${m.port}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: q, history: [] })
            });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            answer = data.answer;
            timeMs = data.processing_time_ms || 0;
        }
        contentEl.innerHTML = formatMarkdown(answer);
        statusEl.textContent = timeMs > 0 ? `${timeMs}ms` : 'Hoàn thành';
        statusEl.className = 'ml-auto text-[10px] text-tertiary';
    } catch (err) {
        contentEl.innerHTML = `<span class="text-error text-xs">❌ ${err.message}</span>`;
        statusEl.textContent = 'Lỗi';
        statusEl.className = 'ml-auto text-[10px] text-error';
    }
}

function compareEmpathWS(q) {
    return new Promise((resolve, reject) => {
        if (!ws || ws.readyState !== WebSocket.OPEN) {
            reject(new Error('WebSocket disconnected'));
            return;
        }
        const tempId = 'compare-' + Date.now();
        const originalHandler = ws.onmessage;

        let answer = '';
        const timeout = setTimeout(() => {
            ws.onmessage = originalHandler;
            reject(new Error('Timeout'));
        }, 180000); // 3 minutes — matches slowest model (LLM+RAG ~150s)

        ws.onmessage = (e) => {
            try {
                const data = JSON.parse(e.data);
                if (data.type === 'answer') {
                    answer = data.answer;
                } else if (data.type === 'error') {
                    clearTimeout(timeout);
                    ws.onmessage = originalHandler;
                    reject(new Error(data.message));
                    return;
                }
                if (answer && data.type === 'answer') {
                    clearTimeout(timeout);
                    ws.onmessage = originalHandler;
                    resolve(answer);
                }
            } catch {}
        };

        ws.send(JSON.stringify({
            question: q,
            session_id: null,
            top_k: 5,
            history: []
        }));
    });
}

function fillExample(text) {
    document.getElementById('queryInput').value = text;
    handleSubmit();
}

// ═══════════════════════════════════════════════════════
// View Switching (Welcome ↔ Chat)
// ═══════════════════════════════════════════════════════
function showChatView() {
    document.getElementById('welcomeView').classList.add('hidden');
    const area = document.getElementById('messagesArea');
    area.classList.remove('hidden');
}

function showWelcomeView() {
    document.getElementById('welcomeView').classList.remove('hidden');
    document.getElementById('messagesArea').classList.add('hidden');
    document.getElementById('messagesArea').innerHTML = '';
    chatHistory = [];
    currentSessionId = null;
}

function newResearch() {
    showWelcomeView();
}

// ═══════════════════════════════════════════════════════
// Chat History (SQLite API)
// ═══════════════════════════════════════════════════════
async function loadSessions() {
    try {
        const res = await fetch('/api/sessions');
        const data = await res.json();
        renderSessionList(data.sessions || []);
    } catch {
        // API not available yet
    }
}

function renderSessionList(sessions) {
    const container = document.getElementById('sessionList');
    container.innerHTML = '<p class="px-4 text-[11px] font-bold text-on-surface-variant/50 tracking-tighter uppercase mb-2">Recent Sessions</p>';

    sessions.forEach(s => {
        const div = document.createElement('div');
        div.className = 'session-item flex items-center justify-between px-4 py-2 rounded-lg hover:bg-surface-variant cursor-pointer transition-all';
        div.innerHTML = `
            <span class="text-xs text-on-surface-variant hover:text-primary transition-colors truncate flex-1" title="${esc(s.title)}">${esc(s.title || 'Untitled')}</span>
            <button onclick="event.stopPropagation(); deleteSession('${s.id}')" class="session-delete p-1 text-on-surface-variant hover:text-error transition-colors rounded">
                <span class="material-symbols-outlined text-[14px]">delete</span>
            </button>`;
        div.addEventListener('click', () => loadSession(s.id));
        container.appendChild(div);
    });
}

async function loadSession(sessionId) {
    try {
        const res = await fetch(`/api/sessions/${sessionId}/messages`);
        const data = await res.json();

        currentSessionId = sessionId;
        chatHistory = [];
        showChatView();
        document.getElementById('messagesArea').innerHTML = '';

        (data.messages || []).forEach(msg => {
            chatHistory.push({ role: msg.role, content: msg.content });
            if (msg.role === 'user') {
                addUserMessage(msg.content);
            } else {
                addAIMessage(msg.content, [], null, 0);
            }
        });
    } catch (e) {
        console.error('Load session failed:', e);
    }
}

async function deleteSession(sessionId) {
    try {
        await fetch(`/api/sessions/${sessionId}`, { method: 'DELETE' });
        loadSessions();
        if (currentSessionId === sessionId) {
            showWelcomeView();
        }
    } catch {}
}

function toggleHistoryPanel() {
    loadSessions();
}

// ═══════════════════════════════════════════════════════
// Message Rendering
// ═══════════════════════════════════════════════════════
function addUserMessage(text) {
    const area = document.getElementById('messagesArea');
    const div = document.createElement('div');
    div.className = 'flex flex-col items-end group';
    div.innerHTML = `
        <div class="flex items-center gap-3 mb-2">
            <span class="text-xs font-semibold text-secondary">Customer</span>
        </div>
        <div class="glass p-5 rounded-2xl rounded-tr-none border border-outline-variant/10 text-on-surface max-w-[80%] shadow-xl">
            <p class="text-base leading-relaxed">${esc(text)}</p>
        </div>`;
    area.appendChild(div);
    scrollToBottom();
}

function addAIMessage(answer, sources, trace, timeMs, label) {
    const area = document.getElementById('messagesArea');
    const div = document.createElement('div');
    div.className = 'flex flex-col items-start group';

    const modelLabel = label || 'EmpathAI + LangGraph';
    const senderName = label ? label.split(' ')[0] : 'EmpathAI';

    // Sources section
    let sourcesHtml = '';
    if (sources && sources.length > 0) {
        sourcesHtml = `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3 mt-4">
                ${sources.slice(0, 3).map(s => `
                    <div class="glass p-4 rounded-xl border border-outline-variant/10 hover:border-primary/20 transition-all cursor-pointer">
                        <div class="text-[10px] font-bold text-primary uppercase mb-1">Chinh Sach: ${esc(s.category || 'CSKH')}</div>
                        <div class="text-xs font-semibold text-on-surface leading-tight">${esc(s.doc_title || 'Policy')}</div>
                        ${s.policy_id ? `<div class="text-[10px] text-on-surface-variant mt-1 font-mono">ID: ${s.policy_id}</div>` : ''}
                    </div>
                `).join('')}
            </div>`;
    }

    // Badges
    const traceId = 'trace-' + Date.now();
    let badgesHtml = `
        <div class="flex flex-wrap items-center gap-3 pt-3">
            <div class="px-3 py-1.5 rounded-full bg-secondary-container/30 border border-secondary/20 flex items-center gap-2">
                <span class="material-symbols-outlined text-[14px] text-secondary">hub</span>
                <span class="text-[10px] font-bold text-secondary uppercase tracking-tight">${modelLabel}</span>
            </div>
            ${timeMs > 0 ? `
                <div class="px-3 py-1.5 rounded-full bg-surface-variant flex items-center gap-2">
                    <span class="material-symbols-outlined text-[14px] text-tertiary">timer</span>
                    <span class="text-[10px] font-medium text-on-surface-variant">${timeMs}ms</span>
                </div>` : ''}
            ${trace ? `
                <button onclick="showTrace('${traceId}')" class="px-3 py-1.5 rounded-full bg-primary/10 hover:bg-primary/20 border border-primary/20 text-primary transition-all flex items-center gap-2 group/btn">
                    <span class="material-symbols-outlined text-[14px]">account_tree</span>
                    <span class="text-[10px] font-bold uppercase tracking-tight">Agent Trace</span>
                </button>` : ''}
        </div>`;

    div.innerHTML = `
        <div class="flex items-center gap-3 mb-2">
            <span class="text-xs font-semibold text-primary">${senderName}</span>
        </div>
        <div class="bg-surface-container-low p-6 rounded-2xl rounded-tl-none border border-primary/5 text-on-surface max-w-[90%] shadow-2xl relative overflow-hidden">
            <div class="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-tertiary to-transparent opacity-40"></div>
            <div class="text-base leading-relaxed text-on-surface/90">${formatMarkdown(answer)}</div>
            ${badgesHtml}
        </div>
        ${sourcesHtml}
        ${trace ? `<div id="${traceId}" class="hidden">${JSON.stringify(trace)}</div>` : ''}`;

    area.appendChild(div);
    scrollToBottom();
}

// ═══════════════════════════════════════════════════════
// Typing Indicator
// ═══════════════════════════════════════════════════════
function showTyping() {
    removeTyping();
    const area = document.getElementById('messagesArea');
    const div = document.createElement('div');
    div.id = 'typing-indicator';
    div.className = 'flex flex-col items-start';
    div.innerHTML = `
        <div class="flex items-center gap-3 mb-2">
            <span class="text-xs font-semibold text-primary">EmpathAI</span>
        </div>
        <div class="bg-surface-container-low p-6 rounded-2xl rounded-tl-none border border-primary/5 max-w-[60%] shadow-xl">
            <div class="flex items-center gap-3">
                <div class="flex gap-1">
                    <span class="w-2.5 h-2.5 rounded-full bg-primary/50 animate-bounce" style="animation-delay:0ms"></span>
                    <span class="w-2.5 h-2.5 rounded-full bg-primary/50 animate-bounce" style="animation-delay:150ms"></span>
                    <span class="w-2.5 h-2.5 rounded-full bg-primary/50 animate-bounce" style="animation-delay:300ms"></span>
                </div>
                <span id="typing-text" class="text-xs text-on-surface-variant">Đang phân tích câu hỏi...</span>
            </div>
        </div>`;
    area.appendChild(div);
    scrollToBottom();
}

function updateTypingText(msg) {
    const el = document.getElementById('typing-text');
    if (el) el.textContent = msg;
}

function removeTyping() {
    const el = document.getElementById('typing-indicator');
    if (el) el.remove();
}

// ═══════════════════════════════════════════════════════
// Agent Trace Modal
// ═══════════════════════════════════════════════════════
function showTrace(dataId) {
    const dataEl = document.getElementById(dataId);
    if (!dataEl) return;
    const trace = JSON.parse(dataEl.textContent);
    const modal = document.getElementById('traceModal');
    const content = document.getElementById('traceContent');

    const steps = [
        { icon: 'alt_route', title: 'Router', body: `Intent: <strong>${trace.router_decision || 'N/A'}</strong>` },
        { icon: 'psychology', title: 'Sentiment Analysis', body: `Sentiment: <strong>${trace.sentiment_detected || 'N/A'}</strong> (Score: ${trace.sentiment_score || 0})` },
        { icon: 'search', title: 'Hybrid Search + Grade', body: `Retrieved: <strong>${trace.retrieved_count || 0}</strong> docs<br>Rewrites: ${trace.grade_rewrite_count || 0}` },
        { icon: 'edit_note', title: 'Empathy Writer', body: `<span class="text-xs">${esc((trace.writer_answer || trace.inquiry_answer || 'N/A').substring(0, 200))}...</span>` },
        { icon: 'fact_check', title: `Quality Checker ${trace.reviewer_triggered ? '(TRIGGERED)' : '(SKIPPED)'}`, body: trace.reviewer_triggered ? (trace.reviewer_result?.is_approved ? '✅ Approved empathy' : ('❌ Issues: ' + (trace.reviewer_result?.issues || []).join(', '))) : '⚡ Skipped (not a complaint)' },
    ];

    content.innerHTML = steps.map((s, i) => `
        <div class="p-4 rounded-xl bg-surface-container border border-outline-variant/10">
            <div class="flex items-center gap-2 mb-2">
                <span class="material-symbols-outlined text-primary text-lg">${s.icon}</span>
                <span class="text-sm font-bold text-on-surface">${i + 1}. ${s.title}</span>
            </div>
            <div class="text-sm text-on-surface-variant">${s.body}</div>
        </div>
    `).join('');

    modal.classList.remove('hidden');
    modal.classList.add('flex');
}

function closeTrace(e) {
    const modal = document.getElementById('traceModal');
    if (e && e.target !== modal) return;
    modal.classList.add('hidden');
    modal.classList.remove('flex');
}

// ═══════════════════════════════════════════════════════
// Utilities
// ═══════════════════════════════════════════════════════
function esc(text) {
    const d = document.createElement('div');
    d.textContent = text || '';
    return d.innerHTML;
}

function formatMarkdown(text) {
    if (!text) return '';
    return text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`([^`]+)`/g, '<code class="px-1 py-0.5 bg-surface-variant rounded text-xs">$1</code>')
        .replace(/\n\n/g, '</p><p class="mt-2">')
        .replace(/\n/g, '<br>')
        .replace(/^(.*)$/, '<p>$1</p>');
}

function scrollToBottom() {
    const c = document.getElementById('chatCanvas');
    setTimeout(() => c.scrollTop = c.scrollHeight, 50);
}

function setSubmitEnabled(enabled) {
    document.getElementById('submitBtn').disabled = !enabled;
}

// Keyboard shortcut
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeTrace();
});

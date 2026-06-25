/**
 * Mini RAG E-Learning Platform — Frontend Application
 * Handles upload, chat, AI tools, and document management.
 */

const API_BASE = '/api';
let currentAction = 'chat';
let isProcessing = false;

// ── DOM Elements ──────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const els = {
  teacherId: $('#teacherId'),
  uploadZone: $('#uploadZone'),
  fileInput: $('#fileInput'),
  uploadProgress: $('#uploadProgress'),
  uploadFilename: $('#uploadFilename'),
  uploadPercent: $('#uploadPercent'),
  uploadFill: $('#uploadFill'),
  uploadStatus: $('#uploadStatus'),
  documentsList: $('#documentsList'),
  documentsEmpty: $('#documentsEmpty'),
  indexStats: $('#indexStats'),
  statVectors: $('#statVectors'),
  statDocs: $('#statDocs'),
  statPages: $('#statPages'),
  chatMessages: $('#chatMessages'),
  chatInput: $('#chatInput'),
  sendBtn: $('#sendBtn'),
  fromPage: $('#fromPage'),
  toPage: $('#toPage'),
  clearPageFilter: $('#clearPageFilter'),
  activeToolName: $('#activeToolName'),
  refreshDocs: $('#refreshDocs'),
  toastContainer: $('#toastContainer'),
};

// ── Initialization ────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  setupUpload();
  setupChat();
  setupTools();
  setupPageFilter();
  loadDocuments();
});

// ══════════════════════════════════════════════════════════
// UPLOAD HANDLING
// ══════════════════════════════════════════════════════════

function setupUpload() {
  const zone = els.uploadZone;
  const input = els.fileInput;

  zone.addEventListener('click', () => input.click());
  input.addEventListener('change', (e) => { if (e.target.files[0]) handleUpload(e.target.files[0]); });

  // Drag & drop
  zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('upload-zone--dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('upload-zone--dragover'));
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('upload-zone--dragover');
    if (e.dataTransfer.files[0]) handleUpload(e.dataTransfer.files[0]);
  });

  els.refreshDocs.addEventListener('click', loadDocuments);
}

async function handleUpload(file) {
  // Client-side validation
  if (!file.name.toLowerCase().endsWith('.pdf')) {
    showToast('Unsupported file type. Please upload a PDF document only.', 'error');
    return;
  }
  if (file.size > 20 * 1024 * 1024) {
    showToast('File size exceeds the allowed limit (20MB).', 'error');
    return;
  }

  const teacherId = els.teacherId.value.trim();
  if (!teacherId) { showToast('Please enter a Teacher ID first.', 'error'); return; }

  // Show progress
  els.uploadProgress.hidden = false;
  els.uploadFilename.textContent = file.name;
  setUploadProgress(0, 'Uploading...');

  const formData = new FormData();
  formData.append('file', file);
  formData.append('teacher_id', teacherId);

  try {
    // Simulate progress during upload
    const progressInterval = setInterval(() => {
      const current = parseInt(els.uploadPercent.textContent);
      if (current < 90) setUploadProgress(current + Math.random() * 15, 'Processing...');
    }, 500);

    const res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: formData });
    clearInterval(progressInterval);

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Upload failed');
    }

    const data = await res.json();
    setUploadProgress(100, 'Complete!');
    showToast(`✅ "${data.filename}" uploaded — ${data.total_pages} pages, ${data.total_chunks} chunks`, 'success');

    // Reset after delay
    setTimeout(() => {
      els.uploadProgress.hidden = true;
      els.fileInput.value = '';
    }, 2000);

    loadDocuments();
  } catch (err) {
    setUploadProgress(0, 'Failed');
    showToast(err.message, 'error');
    setTimeout(() => { els.uploadProgress.hidden = true; }, 3000);
  }
}

function setUploadProgress(percent, status) {
  percent = Math.min(100, Math.round(percent));
  els.uploadPercent.textContent = `${percent}%`;
  els.uploadFill.style.width = `${percent}%`;
  if (status) els.uploadStatus.textContent = status;
}

// ══════════════════════════════════════════════════════════
// DOCUMENTS MANAGEMENT
// ══════════════════════════════════════════════════════════

async function loadDocuments() {
  const teacherId = els.teacherId.value.trim();
  if (!teacherId) return;

  try {
    const res = await fetch(`${API_BASE}/documents?teacher_id=${encodeURIComponent(teacherId)}`);
    if (!res.ok) throw new Error('Failed to load documents');
    const data = await res.json();

    renderDocuments(data.documents);
    renderStats(data.index_stats);
  } catch (err) {
    console.error('Load documents error:', err);
  }
}

function renderDocuments(docs) {
  if (!docs || docs.length === 0) {
    els.documentsList.innerHTML = `
      <div class="documents-empty">
        <span class="documents-empty__icon">📁</span>
        <p>No documents uploaded yet</p>
        <p class="documents-empty__hint">Upload a PDF to get started</p>
      </div>`;
    return;
  }

  els.documentsList.innerHTML = docs.map(doc => `
    <div class="doc-item" data-id="${doc.id}">
      <span class="doc-item__icon">📄</span>
      <div class="doc-item__info">
        <div class="doc-item__name" title="${doc.original_filename}">${doc.original_filename}</div>
        <div class="doc-item__meta">${doc.total_pages} pages • ${formatSize(doc.file_size_bytes)}</div>
      </div>
      <button class="doc-item__delete" onclick="deleteDocument(${doc.id})" title="Delete">🗑️</button>
    </div>
  `).join('');
}

function renderStats(stats) {
  if (!stats || stats.total_vectors === 0) {
    els.indexStats.hidden = true;
    return;
  }
  els.indexStats.hidden = false;
  els.statVectors.textContent = stats.total_vectors.toLocaleString();
  els.statDocs.textContent = stats.total_documents;
  els.statPages.textContent = stats.total_pages;
}

async function deleteDocument(docId) {
  const teacherId = els.teacherId.value.trim();
  if (!confirm('Delete this document and all its data?')) return;

  try {
    const res = await fetch(`${API_BASE}/documents/${docId}?teacher_id=${encodeURIComponent(teacherId)}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Delete failed');
    showToast('Document deleted', 'info');
    loadDocuments();
  } catch (err) {
    showToast(err.message, 'error');
  }
}

// ══════════════════════════════════════════════════════════
// CHAT INTERFACE
// ══════════════════════════════════════════════════════════

function setupChat() {
  els.sendBtn.addEventListener('click', sendMessage);
  els.chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });

  // Auto-resize textarea
  els.chatInput.addEventListener('input', () => {
    els.chatInput.style.height = 'auto';
    els.chatInput.style.height = Math.min(els.chatInput.scrollHeight, 120) + 'px';
  });
}

async function sendMessage() {
  const message = els.chatInput.value.trim();
  const teacherId = els.teacherId.value.trim();

  if (!message || isProcessing) return;
  if (!teacherId) { showToast('Please enter a Teacher ID.', 'error'); return; }

  // Add user message to chat
  addMessage(message, 'user');
  els.chatInput.value = '';
  els.chatInput.style.height = 'auto';

  // Show typing indicator
  const typingEl = showTyping();
  isProcessing = true;
  els.sendBtn.disabled = true;

  // Build request
  const body = { teacher_id: teacherId, message };
  const fromPage = parseInt(els.fromPage.value);
  const toPage = parseInt(els.toPage.value);
  if (!isNaN(fromPage) && fromPage > 0) body.from_page = fromPage;
  if (!isNaN(toPage) && toPage > 0) body.to_page = toPage;

  try {
    const res = await fetch(`${API_BASE}/${currentAction}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    removeTyping(typingEl);

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Request failed');
    }

    const data = await res.json();
    addMessage(data.response, 'ai', data.sources, data.action);
  } catch (err) {
    removeTyping(typingEl);
    addMessage(`❌ Error: ${err.message}`, 'ai');
    showToast(err.message, 'error');
  } finally {
    isProcessing = false;
    els.sendBtn.disabled = false;
    els.chatInput.focus();
  }
}

function addMessage(text, role, sources = [], action = '') {
  const msgDiv = document.createElement('div');
  msgDiv.className = `message message--${role}`;

  const avatar = role === 'user' ? '👤' : '🧠';
  const actionBadge = action && role === 'ai' ? `<span style="font-size:0.7rem;opacity:0.6;display:block;margin-bottom:0.3rem;">📌 ${action.toUpperCase()}</span>` : '';

  // Format AI text with basic markdown-like rendering
  const formattedText = role === 'ai' ? formatResponse(text) : escapeHtml(text);

  let sourcesHtml = '';
  if (sources && sources.length > 0) {
    const teacherId = els.teacherId.value.trim();
    const srcLinks = sources.map(s => {
      const pdfUrl = `${API_BASE}/documents/${s.doc_id}/pdf?teacher_id=${encodeURIComponent(teacherId)}#page=${s.page}`;
      const tooltip = `${s.filename || 'PDF'} — Page ${s.page}`;
      return `<a href="${pdfUrl}" target="_blank" rel="noopener noreferrer" class="source-link" title="${tooltip}">🔗 Page ${s.page}</a>`;
    }).join('');
    sourcesHtml = `<div class="message__sources">${srcLinks}</div>`;
  }

  msgDiv.innerHTML = `
    <div class="message__avatar">${avatar}</div>
    <div class="message__content">
      <div class="message__bubble">${actionBadge}${formattedText}</div>
      ${sourcesHtml}
    </div>`;

  els.chatMessages.appendChild(msgDiv);
  els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
}

function showTyping() {
  const div = document.createElement('div');
  div.className = 'message message--ai';
  div.id = 'typingMsg';
  div.innerHTML = `
    <div class="message__avatar">🧠</div>
    <div class="message__content">
      <div class="message__bubble">
        <div class="typing-indicator"><span></span><span></span><span></span></div>
      </div>
    </div>`;
  els.chatMessages.appendChild(div);
  els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
  return div;
}

function removeTyping(el) { if (el && el.parentNode) el.parentNode.removeChild(el); }

// ══════════════════════════════════════════════════════════
// AI TOOLS
// ══════════════════════════════════════════════════════════

function setupTools() {
  const toolLabels = {
    chat: '💬 Ask Question', summarize: '📝 Summarize', quiz: '❓ Generate Quiz',
    explain: '💡 Explain Simply', flashcards: '📇 Flashcards', translate: '🌍 Translate',
  };
  const placeholders = {
    chat: 'Ask a question about your course materials...',
    summarize: 'What would you like summarized? (e.g., "Summarize chapter 3")',
    quiz: 'What topic should the quiz cover?',
    explain: 'What concept would you like explained simply?',
    flashcards: 'What topic should the flashcards cover?',
    translate: 'What content would you like translated?',
  };

  $$('.tool-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const action = btn.dataset.action;
      currentAction = action;

      $$('.tool-btn').forEach(b => b.classList.remove('tool-btn--active'));
      btn.classList.add('tool-btn--active');

      els.activeToolName.textContent = toolLabels[action] || action;
      els.chatInput.placeholder = placeholders[action] || 'Type your message...';
      els.chatInput.focus();
    });
  });

  // Set initial active
  $('[data-action="chat"]').classList.add('tool-btn--active');
}

function setupPageFilter() {
  els.clearPageFilter.addEventListener('click', () => {
    els.fromPage.value = '';
    els.toPage.value = '';
  });
}

// ══════════════════════════════════════════════════════════
// UTILITIES
// ══════════════════════════════════════════════════════════

function formatResponse(text) {
  let html = escapeHtml(text);
  // Bold: **text**
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  // Headers: lines starting with Q1:, Q2:, etc
  html = html.replace(/^(Q\d+:.*)/gm, '<strong>$1</strong>');
  // Flashcard formatting
  html = html.replace(/^(📇.*)/gm, '<strong>$1</strong>');
  html = html.replace(/^(▶.*)/gm, '<em>$1</em>');
  html = html.replace(/^(◀.*)/gm, '<em>$1</em>');
  // Correct answer highlight
  html = html.replace(/^(✅.*)/gm, '<strong style="color:#22c55e;">$1</strong>');
  // Line breaks
  html = html.replace(/\n/g, '<br>');
  return html;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function formatSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.className = `toast toast--${type}`;
  const icons = { success: '✅', error: '❌', info: 'ℹ️' };
  toast.innerHTML = `<span>${icons[type] || 'ℹ️'}</span><span>${escapeHtml(message)}</span>`;
  els.toastContainer.appendChild(toast);

  setTimeout(() => {
    toast.style.animation = 'toastOut 0.3s ease forwards';
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

// Reload docs when teacher ID changes
let teacherDebounce;
els.teacherId.addEventListener('input', () => {
  clearTimeout(teacherDebounce);
  teacherDebounce = setTimeout(loadDocuments, 600);
});

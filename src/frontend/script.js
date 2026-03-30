const uploadZone  = document.getElementById('upload-zone');
const fileInput   = document.getElementById('file-input');
const previewWrap = document.getElementById('preview-wrap');
const previewImg  = document.getElementById('preview-img');
const analyzeBtn  = document.getElementById('analyze-btn');
const errorBox    = document.getElementById('error-box');
const errorMsg    = document.getElementById('error-msg');
const loading     = document.getElementById('loading');
const results     = document.getElementById('results');
const resetBtn    = document.getElementById('reset-btn');

// ── State ──
let selectedFile = null;

// ── Drag & Drop ──
uploadZone.addEventListener('dragover', e => {
  e.preventDefault();
  uploadZone.classList.add('drag-over');
});

uploadZone.addEventListener('dragleave', () => {
  uploadZone.classList.remove('drag-over');
});

uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');

  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) {
    handleFile(file);
  } else {
    showError('Please drop a valid image file.');
  }
});

// ── File Input ──
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

// ── Handle File ──
function handleFile(file) {
  selectedFile = file;
  hideError();
  hideResults();

  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    previewWrap.style.display = 'block';
    analyzeBtn.style.display = 'block';
  };
  reader.readAsDataURL(file);
}

// ── Analyze Button ──
analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  setLoading(true);
  hideError();
  hideResults();

  const formData = new FormData();
  formData.append('file', selectedFile);

  const API_BASE_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
    ? 'http://127.0.0.1:8000' 
    : ''; // Use relative path if deployed

  try {
    const res = await fetch(`${API_BASE_URL}/process`, {
      method: 'POST',
      body: formData
    });

    if (!res.ok) {
      let detail = `Server error ${res.status}`;
      try {
        const json = await res.json();
        detail = json.detail || json.message || detail;
      } catch (_) {}
      throw new Error(detail);
    }

    const data = await res.json();
    console.log("Backend response:", data);

    showResults(data);

  } catch (err) {
    if (err.message.includes('fetch')) {
      showError('Backend not reachable. Make sure FastAPI is running.');
    } else {
      showError(err.message);
    }
  } finally {
    setLoading(false);
  }
});

// ── SHOW RESULTS (FIXED) ──
function showResults(data) {

  // Raw OCR text
  document.querySelector('#raw-text pre').textContent =
    data.raw_text || '(no data)';

  // Cleaned text (IMPORTANT FIX)
  document.querySelector('#clean-text p').textContent =
    data.cleaned_text || '(no data)';

  // Insights / Summary (IMPORTANT FIX)
  document.querySelector('#summary-text p').textContent =
    formatInsights(data.insights);

  results.style.display = 'block';
  resetBtn.style.display = 'block';
}

// ── Format insights nicely ──
function formatInsights(insights) {
  if (!insights) return '(no data)';

  if (typeof insights === 'string') return insights;

  return Object.entries(insights)
    .map(([key, value]) => `${key}: ${value}`)
    .join('\n');
}

// ── Hide Results ──
function hideResults() {
  results.style.display = 'none';
  resetBtn.style.display = 'none';
}

// ── Loading ──
function setLoading(on) {
  loading.style.display = on ? 'block' : 'none';
  analyzeBtn.disabled = on;
}

// ── Error ──
function showError(msg) {
  errorMsg.textContent = msg;
  errorBox.style.display = 'flex';
}

function hideError() {
  errorBox.style.display = 'none';
}

// ── Reset ──
resetBtn.addEventListener('click', () => {
  selectedFile = null;
  fileInput.value = '';
  previewWrap.style.display = 'none';
  previewImg.src = '';
  analyzeBtn.style.display = 'none';
  hideError();
  hideResults();
});

// ── Copy Buttons ──
document.querySelectorAll('.copy-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const target = document.getElementById(btn.dataset.target);
    const text = target.querySelector('pre, p')?.textContent || '';

    navigator.clipboard.writeText(text);

    const old = btn.textContent;
    btn.textContent = 'Copied!';
    setTimeout(() => btn.textContent = old, 1500);
  });
});
/**
 * SoyCare AI - Frontend Interactive Controller
 * Handles image selection, live webcam stream, asynchronous API calls,
 * Grad-CAM view toggling, dashboard metrics, and paginated scan history.
 */

// DOM Elements
const fileInput = document.querySelector('#imageFileInput');
const dropzone = document.querySelector('#dropzone');
const dropzonePrompt = document.querySelector('#dropzonePrompt');
const previewWrapper = document.querySelector('#previewWrapper');
const imagePreview = document.querySelector('#imagePreview');
const btnClearImage = document.querySelector('#btnClearImage');
const btnDiagnose = document.querySelector('#btnDiagnose');

// Mode Tabs
const tabUpload = document.querySelector('#tabUpload');
const tabCamera = document.querySelector('#tabCamera');
const uploadContainer = document.querySelector('#uploadContainer');
const cameraContainer = document.querySelector('#cameraContainer');
const webcamVideo = document.querySelector('#webcamVideo');
const cameraCanvas = document.querySelector('#cameraCanvas');
const btnSnapPhoto = document.querySelector('#btnSnapPhoto');
const btnToggleCamera = document.querySelector('#btnToggleCamera');

// Results Elements
const resultsSection = document.querySelector('#resultsSection');
const riskBadge = document.querySelector('#riskBadge');
const resultDiseaseName = document.querySelector('#resultDiseaseName');
const resultPathogen = document.querySelector('#resultPathogen');
const resultConfidence = document.querySelector('#resultConfidence');
const resImgOriginal = document.querySelector('#resImgOriginal');
const resImgHeatmap = document.querySelector('#resImgHeatmap');
const topProbabilitiesList = document.querySelector('#topProbabilitiesList');
const diagnosticNoteText = document.querySelector('#diagnosticNoteText');
const visualContainer = document.querySelector('#visualContainer');

// Recommendation Lists
const listImmediateActions = document.querySelector('#listImmediateActions');
const listChemicalControl = document.querySelector('#listChemicalControl');
const listCulturalManagement = document.querySelector('#listCulturalManagement');

// History Table & Filters
const historyTableBody = document.querySelector('#historyTableBody');
const filterSearch = document.querySelector('#filterSearch');
const filterDisease = document.querySelector('#filterDisease');
const filterRisk = document.querySelector('#filterRisk');
const btnPrevPage = document.querySelector('#btnPrevPage');
const btnNextPage = document.querySelector('#btnNextPage');
const paginationInfo = document.querySelector('#paginationInfo');
const btnPrintReport = document.querySelector('#btnPrintReport');

// State Variables
let currentFile = null;
let currentBase64 = null;
let mediaStream = null;
let facingMode = 'environment';
let currentPage = 1;
const pageLimit = 6;
let searchDebounceTimer = null;

// ==========================================================================
// File Upload & Drag-and-Drop
// ==========================================================================
function setFile(file) {
  if (!file || !file.type.startsWith('image/')) {
    alert('Please select a valid image file (JPG, PNG, WEBP).');
    return;
  }
  currentFile = file;
  currentBase64 = null;

  const reader = new FileReader();
  reader.onload = (e) => {
    imagePreview.src = e.target.result;
    dropzonePrompt.classList.add('hidden');
    previewWrapper.classList.remove('hidden');
    btnDiagnose.disabled = false;
  };
  reader.readAsDataURL(file);
}

function clearSelection() {
  currentFile = null;
  currentBase64 = null;
  if (fileInput) fileInput.value = '';
  if (imagePreview) imagePreview.src = '';
  if (previewWrapper) previewWrapper.classList.add('hidden');
  if (dropzonePrompt) dropzonePrompt.classList.remove('hidden');
  if (btnDiagnose) btnDiagnose.disabled = true;
}

if (fileInput) {
  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  });
}

if (btnClearImage) {
  btnClearImage.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    clearSelection();
  });
}

// Drag & Drop
if (dropzone) {
  ['dragenter', 'dragover'].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove('dragover');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    if (dt && dt.files && dt.files[0]) {
      setFile(dt.files[0]);
    }
  });
}

// ==========================================================================
// Live Webcam Stream
// ==========================================================================
if (tabUpload && tabCamera) {
  tabUpload.addEventListener('click', () => {
    tabUpload.classList.add('active');
    tabCamera.classList.remove('active');
    if (uploadContainer) uploadContainer.classList.remove('hidden');
    if (cameraContainer) cameraContainer.classList.add('hidden');
    stopCamera();
  });

  tabCamera.addEventListener('click', () => {
    tabCamera.classList.add('active');
    tabUpload.classList.remove('active');
    if (cameraContainer) cameraContainer.classList.remove('hidden');
    if (uploadContainer) uploadContainer.classList.add('hidden');
    startCamera();
  });
}

async function startCamera() {
  stopCamera();
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('Camera access is not supported by your browser in this environment.');
    }
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: facingMode, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false
    });
    if (webcamVideo) {
      webcamVideo.srcObject = mediaStream;
      await webcamVideo.play();
    }
  } catch (err) {
    alert('Unable to access device camera: ' + err.message);
  }
}

function stopCamera() {
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
}

if (btnToggleCamera) {
  btnToggleCamera.addEventListener('click', () => {
    facingMode = facingMode === 'environment' ? 'user' : 'environment';
    startCamera();
  });
}

if (btnSnapPhoto) {
  btnSnapPhoto.addEventListener('click', () => {
    if (!webcamVideo || !webcamVideo.videoWidth || !webcamVideo.videoHeight) {
      alert('Camera stream is not ready yet. Please wait for the video feed.');
      return;
    }

    cameraCanvas.width = webcamVideo.videoWidth;
    cameraCanvas.height = webcamVideo.videoHeight;
    const ctx = cameraCanvas.getContext('2d');
    ctx.drawImage(webcamVideo, 0, 0);

    try {
      currentBase64 = cameraCanvas.toDataURL('image/jpeg', 0.92);
    } catch (err) {
      alert('Failed to capture frame: ' + err.message);
      return;
    }
    currentFile = null;

    if (imagePreview) imagePreview.src = currentBase64;
    if (tabUpload) tabUpload.click();
    if (dropzonePrompt) dropzonePrompt.classList.add('hidden');
    if (previewWrapper) previewWrapper.classList.remove('hidden');
    if (btnDiagnose) btnDiagnose.disabled = false;
  });
}

// ==========================================================================
// Diagnosis Execution
// ==========================================================================
if (btnDiagnose) {
  btnDiagnose.addEventListener('click', async () => {
    if (!currentFile && !currentBase64) {
      alert('Please upload an image or capture a leaf photo first.');
      return;
    }

    btnDiagnose.disabled = true;
    const btnText = btnDiagnose.querySelector('.btn-text');
    if (btnText) btnText.textContent = 'Analyzing Specimen…';

    try {
      let response;
      if (currentFile) {
        const formData = new FormData();
        formData.append('image', currentFile);
        response = await fetch('/api/predict', { method: 'POST', body: formData });
      } else {
        response = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image_base64: currentBase64 })
        });
      }

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || 'Server error occurred during diagnosis.');
      }

      renderDiagnosisResults(data);
      if (resultsSection) {
        resultsSection.classList.remove('hidden');
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }

      // Refresh telemetry & table
      loadDashboardMetrics();
      loadHistory(1);
    } catch (err) {
      console.error('Diagnosis Error:', err);
      alert('Diagnosis Error: ' + err.message);
    } finally {
      btnDiagnose.disabled = false;
      if (btnText) btnText.textContent = 'Run Disease Diagnosis';
    }
  });
}

function renderDiagnosisResults(data) {
  if (!data) return;

  // Title & Badges
  if (resultDiseaseName) resultDiseaseName.textContent = data.disease || 'Unknown';
  if (resultPathogen) resultPathogen.textContent = (data.recommendation && data.recommendation.pathogen) || 'Pathogen Profile Available';
  if (resultConfidence) resultConfidence.textContent = `${data.confidence}%`;

  if (riskBadge) {
    const riskStr = (data.risk_level || 'ASSESSMENT').toString();
    riskBadge.textContent = riskStr.toUpperCase();
    riskBadge.className = 'risk-badge';
    if (riskStr === 'High Risk') riskBadge.classList.add('high-risk');
    else if (riskStr === 'Moderate') riskBadge.classList.add('moderate');
    else riskBadge.classList.add('healthy');
  }

  // Images & Heatmap
  if (resImgOriginal) resImgOriginal.src = data.image_url || '';
  if (resImgHeatmap) resImgHeatmap.src = data.heatmap_url || data.image_url || '';

  // Class Probabilities
  if (topProbabilitiesList) {
    topProbabilitiesList.innerHTML = (data.top_predictions || []).map((pred) => `
      <div class="prob-row">
        <div class="prob-meta">
          <strong>${pred.disease}</strong>
          <span>${pred.confidence}%</span>
        </div>
        <div class="prob-track">
          <div class="prob-fill" style="width: ${pred.confidence}%"></div>
        </div>
      </div>
    `).join('');
  }

  // Diagnostic Note
  if (diagnosticNoteText) {
    diagnosticNoteText.textContent = data.note || 'AI-assisted result. Confirm chemical-control decisions with a certified agronomist.';
  }

  // Recommendations Lists
  const rec = data.recommendation || {};
  if (listImmediateActions) {
    listImmediateActions.innerHTML = (rec.immediate_action || []).map((act) => `<li>${act}</li>`).join('');
  }
  if (listChemicalControl) {
    listChemicalControl.innerHTML = (rec.chemical_control || []).map((act) => `<li>${act}</li>`).join('');
  }
  if (listCulturalManagement) {
    listCulturalManagement.innerHTML = (rec.cultural_management || []).map((act) => `<li>${act}</li>`).join('');
  }
}

// ==========================================================================
// Visual Inspection View Toggles (Grad-CAM)
// ==========================================================================
document.querySelectorAll('.btn-toggle').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.btn-toggle').forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');

    const view = btn.getAttribute('data-view');
    const originalFrame = document.querySelector('.original-frame');
    const heatmapFrame = document.querySelector('.heatmap-frame');

    if (visualContainer && originalFrame && heatmapFrame) {
      if (view === 'split') {
        visualContainer.style.gridTemplateColumns = '1fr 1fr';
        originalFrame.classList.remove('hidden');
        heatmapFrame.classList.remove('hidden');
      } else if (view === 'heatmap') {
        visualContainer.style.gridTemplateColumns = '1fr';
        originalFrame.classList.add('hidden');
        heatmapFrame.classList.remove('hidden');
      } else if (view === 'original') {
        visualContainer.style.gridTemplateColumns = '1fr';
        originalFrame.classList.remove('hidden');
        heatmapFrame.classList.add('hidden');
      }
    }
  });
});

// Recommendation Tabs
document.querySelectorAll('.rec-tab-btn').forEach((tab) => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.rec-tab-btn').forEach((t) => t.classList.remove('active'));
    document.querySelectorAll('.rec-panel').forEach((p) => p.classList.remove('active'));

    tab.classList.add('active');
    const targetSelector = tab.getAttribute('data-target');
    if (targetSelector) {
      const target = document.querySelector(targetSelector);
      if (target) target.classList.add('active');
    }
  });
});

// Print Report
if (btnPrintReport) {
  btnPrintReport.addEventListener('click', () => {
    window.print();
  });
}

// ==========================================================================
// Dashboard Metrics Telemetry
// ==========================================================================
async function loadDashboardMetrics() {
  try {
    const res = await fetch('/api/dashboard');
    const data = await res.json();

    const statTotal = document.querySelector('#statTotalScans');
    const statAvg = document.querySelector('#statAvgConfidence');
    const statHigh = document.querySelector('#statHighRisk');
    const statTop = document.querySelector('#statTopDisease');

    if (statTotal) statTotal.textContent = data.total_detections || 0;
    if (statAvg) statAvg.textContent = data.total_detections ? `${data.average_confidence}%` : '—';
    if (statHigh) statHigh.textContent = data.high_risk_detections || 0;
    if (statTop) statTop.textContent = (data.disease_counts && data.disease_counts[0] && data.disease_counts[0].disease) || '—';
  } catch (err) {
    console.error('Failed to load dashboard metrics:', err);
  }
}

// ==========================================================================
// Scan History Table & Pagination
// ==========================================================================
async function loadHistory(page = 1) {
  currentPage = page;
  const search = filterSearch ? encodeURIComponent(filterSearch.value.trim()) : '';
  const disease = filterDisease ? encodeURIComponent(filterDisease.value) : '';
  const risk = filterRisk ? encodeURIComponent(filterRisk.value) : '';

  const url = `/api/history?page=${page}&limit=${pageLimit}&search=${search}&disease=${disease}&risk=${risk}`;

  try {
    const res = await fetch(url);
    const data = await res.json();

    if (!historyTableBody) return;

    if (!data.records || data.records.length === 0) {
      historyTableBody.innerHTML = `<tr><td colspan="8" class="text-center py-4">No matching scan records found.</td></tr>`;
      if (paginationInfo) paginationInfo.textContent = 'Showing 0 of 0 records';
      if (btnPrevPage) btnPrevPage.disabled = true;
      if (btnNextPage) btnNextPage.disabled = true;
      return;
    }

    historyTableBody.innerHTML = data.records.map((r) => {
      let dateStr = r.detected_at || '';
      try {
        dateStr = new Date(r.detected_at).toLocaleString();
      } catch (e) {}

      let badgeClass = 'moderate';
      if (r.risk_level === 'High Risk') badgeClass = 'high';
      else if (r.risk_level === 'Healthy') badgeClass = 'healthy';

      return `
        <tr>
          <td><strong>#${r.id}</strong></td>
          <td>${dateStr}</td>
          <td>
            <img src="/uploads/${r.filename}" alt="Leaf" class="table-thumb" onerror="this.src='/uploads/heatmaps/heatmap_${r.filename}'">
          </td>
          <td><strong>${r.disease}</strong></td>
          <td>${r.confidence}%</td>
          <td><span class="badge-risk-table ${badgeClass}">${r.risk_level}</span></td>
          <td><small>${r.source || 'upload'}</small></td>
          <td>
            <button class="btn btn-secondary btn-sm" onclick="deleteRecord(${r.id})">Delete</button>
          </td>
        </tr>
      `;
    }).join('');

    if (paginationInfo) {
      paginationInfo.textContent = `Showing page ${data.page} of ${data.total_pages} (${data.total_records} total records)`;
    }
    if (btnPrevPage) btnPrevPage.disabled = data.page <= 1;
    if (btnNextPage) btnNextPage.disabled = data.page >= data.total_pages;
  } catch (err) {
    console.error('Failed to fetch history:', err);
  }
}

async function deleteRecord(recordId) {
  if (!confirm(`Delete scan record #${recordId}?`)) return;
  try {
    const res = await fetch(`/api/history/${recordId}`, { method: 'DELETE' });
    if (res.ok) {
      loadHistory(currentPage);
      loadDashboardMetrics();
    }
  } catch (err) {
    alert('Failed to delete record: ' + err.message);
  }
}

// Attach filter listeners
if (filterSearch) {
  filterSearch.addEventListener('input', () => {
    clearTimeout(searchDebounceTimer);
    searchDebounceTimer = setTimeout(() => loadHistory(1), 300);
  });
}

if (filterDisease) filterDisease.addEventListener('change', () => loadHistory(1));
if (filterRisk) filterRisk.addEventListener('change', () => loadHistory(1));

if (btnPrevPage) {
  btnPrevPage.addEventListener('click', () => {
    if (currentPage > 1) loadHistory(currentPage - 1);
  });
}

if (btnNextPage) {
  btnNextPage.addEventListener('click', () => {
    loadHistory(currentPage + 1);
  });
}

// ==========================================================================
// Disease Library Knowledge Base Loader
// ==========================================================================
async function loadDiseaseLibrary() {
  const container = document.querySelector('#diseaseKbGrid');
  if (!container) return;
  try {
    const res = await fetch('/api/diseases');
    const data = await res.json();

    container.innerHTML = Object.entries(data).map(([name, info]) => `
      <div class="kb-card">
        <h3>${name}</h3>
        <span class="kb-pathogen">${info.pathogen || 'Diagnostic Profile'}</span>
        <p class="kb-symptoms"><strong>Symptoms:</strong> ${info.symptoms || 'Foliar lesions'}</p>
        <div class="kb-meta-item">
          <strong>Favorable Climate:</strong> ${info.conditions || 'Field conditions vary.'}
        </div>
      </div>
    `).join('');
  } catch (err) {
    console.error('Failed to load disease library:', err);
  }
}

// Initial Boot
loadDashboardMetrics();
loadHistory(1);
loadDiseaseLibrary();

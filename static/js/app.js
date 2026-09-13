const input = document.querySelector('#imageInput');
const preview = document.querySelector('#preview');
const scanButton = document.querySelector('#scanButton');
const result = document.querySelector('#result');

input.addEventListener('change', () => {
  const file = input.files[0];
  if (!file) return;
  preview.src = URL.createObjectURL(file); preview.style.display = 'block';
  scanButton.disabled = false; scanButton.classList.remove('disabled'); result.classList.add('hidden');
});

scanButton.addEventListener('click', async () => {
  scanButton.textContent = 'Analysing…'; scanButton.disabled = true;
  const formData = new FormData(); formData.append('image', input.files[0]);
  try {
    const response = await fetch('/api/predict', { method: 'POST', body: formData });
    const data = await response.json(); if (!response.ok) throw new Error(data.error);
    const actions = data.recommendation.immediate_action.map(action => `<li>${action}</li>`).join('');
    result.innerHTML = `<p class="eyebrow">ASSESSMENT · ${data.risk_level.toUpperCase()} RISK</p><h3>${data.disease}</h3><strong>${data.confidence}% confidence</strong><p>${data.recommendation.symptoms}</p><p><b>Recommended next actions</b></p><ul>${actions}</ul><p><small>${data.note}</small></p>`;
    result.classList.remove('hidden'); result.scrollIntoView({behavior:'smooth', block:'nearest'}); loadDashboard();
  } catch (error) { result.innerHTML = `<p>${error.message}</p>`; result.classList.remove('hidden'); }
  finally { scanButton.innerHTML = 'Analyse image <span>→</span>'; scanButton.disabled = false; }
});

async function loadDashboard() { const response = await fetch('/api/dashboard'); const data = await response.json(); document.querySelector('#totalDetections').textContent=data.total_detections; document.querySelector('#averageConfidence').textContent=data.total_detections ? `${data.average_confidence}%` : '—'; document.querySelector('#topDisease').textContent=data.disease_counts[0]?.disease || '—'; }
loadDashboard();

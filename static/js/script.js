const previewImg = document.getElementById('previewImg');
const fileInput = document.getElementById('fileInput');
const predictBtn = document.getElementById('predictBtn');
const resultBox = document.getElementById('result');
const previewImg = document.getElementById('previewImg');
const matchedImg = document.getElementById('matchedImg');
const matchedLabel = document.getElementById('matchedLabel');
const matchedSim = document.getElementById('matchedSim');
const fileInput = document.getElementById('fileInput');
const predictBtn = document.getElementById('predictBtn');
const resultBox = document.getElementById('result');

fileInput.addEventListener('change', e => {
  const file = e.target.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  resultBox.textContent = '';
  if (matchedImg) matchedImg.src = '/static/img/placeholder.svg';
  if (matchedLabel) matchedLabel.textContent = 'No match yet';
  if (matchedSim) matchedSim.textContent = '';
});

predictBtn.addEventListener('click', async () => {
  const file = fileInput.files[0];
  if (!file) { resultBox.textContent = 'Please choose an image.'; return }
  predictBtn.disabled = true; predictBtn.textContent = 'Predicting...';
  const form = new FormData(); form.append('file', file);
  try {
    const res = await fetch('/predict', { method: 'POST', body: form });
    const data = await res.json();
    if (data.error) {
      resultBox.textContent = data.error
    } else {
      resultBox.innerHTML = `<strong>Prediction:</strong> ${data.class} <br><span class="small">Confidence: ${(data.confidence * 100).toFixed(2)}%</span>`
      if (data.matched_path && matchedImg) {
        matchedImg.src = `/matched?path=${encodeURIComponent(data.matched_path)}`;
        matchedLabel.textContent = data.matched_label || data.matched_path;
        if (data.similarity !== undefined && matchedSim) matchedSim.textContent = `Similarity: ${(data.similarity * 100).toFixed(2)}%`;
      }
    }
  } catch (err) { resultBox.textContent = 'Prediction failed.' }
  predictBtn.disabled = false; predictBtn.textContent = 'Predict'
});

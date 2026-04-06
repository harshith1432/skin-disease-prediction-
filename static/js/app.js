// Elements
const previewImg = document.getElementById('previewImg');
const matchedImg = document.getElementById('matchedImg');
const matchedLabel = document.getElementById('matchedLabel');
const matchedSim = document.getElementById('matchedSim');
const fileInput = document.getElementById('fileInput');
const predictBtn = document.getElementById('predictBtn');
const resultBox = document.getElementById('result');
const historyList = document.getElementById('historyList');
const chatWindow = document.getElementById('chatWindow');
const chatInput = document.getElementById('chatInput');
const chatSend = document.getElementById('chatSend');
const confidenceValue = document.getElementById('confidenceValue');
const gaugeFill = document.getElementById('gaugeFill');
const predictionLabel = document.getElementById('predictionLabel');
const topPredictions = document.getElementById('topPredictions');
const suggestionsEl = document.getElementById('suggestions');

// UI State Helpers
function showLoading(btn, isLoading) {
  if (isLoading) {
    btn.disabled = true;
    btn.dataset.originalText = btn.innerHTML;
    btn.innerHTML = `<i class="bi bi-gear-fill animate-spin me-2"></i> Analyzing...`;
  } else {
    btn.disabled = false;
    btn.innerHTML = btn.dataset.originalText || btn.innerHTML;
  }
}

function animateGauge(percent) {
  const arcLength = 283; // 2 * PI * 45
  const offset = arcLength - (arcLength * percent) / 100;
  if (gaugeFill) gaugeFill.style.strokeDashoffset = offset;
  if (confidenceValue) {
    let current = 0;
    const interval = setInterval(() => {
      if (current >= percent) {
        confidenceValue.textContent = `${percent.toFixed(0)}%`;
        clearInterval(interval);
      } else {
        current += 1;
        confidenceValue.textContent = `${current}%`;
      }
    }, 20);
  }
}

// History Management
async function fetchHistory() {
  try {
    const res = await fetch('/history');
    if (!res.ok) throw new Error('Failed to fetch');
    const data = await res.json();
    renderHistory(data);
  } catch (err) {
    console.error('Failed to fetch history:', err);
  }
}

function renderHistory(scans) {
  if (!historyList) return;
  if (!scans || scans.length === 0) {
    historyList.innerHTML = `<div class="text-center py-5 text-muted x-small">No scans yet</div>`;
    return;
  }

  historyList.innerHTML = scans.map(s => `
    <div class="history-item" onclick='displayHistoricalScan(${JSON.stringify(s)})'>
      <div class="bg-light rounded p-2" style="width: 40px; height: 40px; overflow: hidden; display: flex; align-items: center; justify-content: center;">
        <img src="${s.image_path ? '/static/uploads/' + s.image_path : '/static/img/placeholder.svg'}" style="width:100%; height:100%; object-fit: cover; border-radius: 4px;">
      </div>
      <div class="flex-grow-1 overflow-hidden">
        <div class="small fw-bold truncate">${s.label}</div>
        <div class="x-small text-muted">${new Date(s.timestamp).toLocaleDateString()}</div>
      </div>
      <div class="x-small fw-bold text-primary">${(s.confidence * 100).toFixed(0)}%</div>
    </div>
  `).join('');
}

window.displayHistoricalScan = (scan) => {
  // Update Center Column
  if (predictionLabel) predictionLabel.textContent = scan.label;
  animateGauge(scan.confidence * 100);
  
  // Parse Top K if exists
  let topK = [];
  try { topK = typeof scan.top_k === 'string' ? JSON.parse(scan.top_k) : (scan.top_k || []); } catch(e) {}
  
  if (topPredictions) {
    topPredictions.innerHTML = `<h6 class="text-muted small fw-bold mb-3 uppercase">Differential Diagnosis (Top 3)</h6>` + 
      topK.map(p => {
        const pct = (p.confidence * 100).toFixed(1);
        return `
          <div class="mb-3">
            <div class="d-flex justify-content-between mb-1">
              <span class="small fw-semibold text-dark">${p.class}</span>
              <span class="x-small text-muted">${pct}%</span>
            </div>
            <div class="confidence-meter">
              <div class="confidence-fill" style="width: ${pct}%"></div>
            </div>
          </div>`;
      }).join('');
  }

  // Update Dataset Match (if available in scan data)
  if (matchedImg) {
    if (scan.matched_path) {
      matchedImg.src = `/matched?path=${encodeURIComponent(scan.matched_path)}`;
      matchedImg.classList.remove('d-none');
      if (typeof matchedLabel !== 'undefined' && matchedLabel) {
        matchedLabel.textContent = scan.matched_label || 'Reference Case';
      }
      const matchedPlaceholder = document.getElementById('matchedPlaceholder');
      if (matchedPlaceholder) matchedPlaceholder.classList.add('d-none');
    } else {
      matchedImg.classList.add('d-none');
      const matchedPlaceholder = document.getElementById('matchedPlaceholder');
      if (matchedPlaceholder) matchedPlaceholder.classList.remove('d-none');
    }
  }

  // Set preview if image path exists (fallback to placeholder)
  if (previewImg) {
    const rawPath = scan.image_path;
    // Handle both relative filenames and absolute-styled paths
    const finalPath = rawPath ? `/static/uploads/${rawPath.split('/').pop()}` : null;
    
    if (finalPath) {
      previewImg.src = finalPath;
      previewImg.classList.remove('d-none');
      const container = document.getElementById('previewContainer');
      const prompt = document.getElementById('uploadPrompt');
      if (container) container.classList.remove('d-none');
      if (prompt) prompt.classList.add('d-none');
    }
  }
  
  if (resultBox) resultBox.classList.remove('d-none');
  const placeholder = document.getElementById('resultPlaceholder');
  if (placeholder) placeholder.classList.add('d-none');

  // Trigger Assistant recommendation
  window._lastPrediction = scan.label;
  if (typeof addAssistantMessage === 'function') {
      addAssistantMessage(`I've loaded your historical scan for **${scan.label}**. Analysis confirmed with ${(scan.confidence * 100).toFixed(1)}% confidence.`);
  }
};

// Prediction Handler
if (predictBtn) predictBtn.addEventListener('click', async () => {
  const file = fileInput.files[0];
  if (!file) { 
    alert('Please select an image first.');
    return;
  }

  showLoading(predictBtn, true);
  const form = new FormData(); 
  form.append('file', file);
  
  try {
    const res = await fetch('/predict', { method: 'POST', body: form });
    const data = await res.json();
    
    if (data.error) {
      alert(data.error);
    } else {
      // Update UI with new result
      if (predictionLabel) predictionLabel.textContent = data.class;
      animateGauge(data.confidence * 100);
      
      if (topPredictions) {
        topPredictions.innerHTML = `<h6 class="text-muted small fw-bold mb-3 uppercase">Differential Diagnosis (Top 3)</h6>` + 
          (data.top || []).map(p => {
            const pct = (p.confidence * 100).toFixed(1);
            return `
              <div class="mb-3">
                <div class="d-flex justify-content-between mb-1">
                  <span class="small fw-semibold text-dark">${p.class}</span>
                  <span class="x-small text-muted">${pct}%</span>
                </div>
                <div class="confidence-meter">
                  <div class="confidence-fill" style="width: ${pct}%"></div>
                </div>
              </div>`;
          }).join('');
      }

      // Update Guidance asynchronously via HF LLM
      if (suggestionsEl) {
        suggestionsEl.innerHTML = `
          <div class="text-center py-3">
             <div class="spinner-border text-primary spinner-border-sm mb-2" role="status"></div>
             <p class="x-small text-muted mb-0">Dr. Derma AI is generating personalized prescriptions and home remedies...</p>
          </div>
        `;
        
        fetch('/generate_ai_insight', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ label: data.class })
        })
        .then(res => res.json())
        .then(ai_data => {
          if (ai_data.insight) {
            // Render markdown insight
            suggestionsEl.innerHTML = `
              <div class="p-3 mb-2 bg-white rounded border-start border-4 border-primary small" style="max-height: 400px; overflow-y: auto; font-size: 0.85rem;">
                ${marked.parse(ai_data.insight)}
              </div>
            `;
            addAssistantMessage("I've finished formulating a clinical insight regarding **" + data.class + "**. Please check the Smart Guidance panel for details on home remedies and OTC prescriptions.");
          } else {
            suggestionsEl.innerHTML = `<p class="x-small text-danger mb-0">Could not generate AI insights at this time. ${ai_data.error || ''}</p>`;
          }
        })
        .catch(e => {
          suggestionsEl.innerHTML = `<p class="x-small text-danger mb-0">Error connecting to AI Assistant.</p>`;
        });
      }

      // Update Dataset Match
      if (data.matched_path && matchedImg) {
        matchedImg.src = `/matched?path=${encodeURIComponent(data.matched_path)}`;
        matchedImg.classList.remove('d-none');
        if (matchedLabel) matchedLabel.textContent = data.matched_label || 'Reference Case';
        if (matchedSim) matchedSim.textContent = `Similarity: ${(data.similarity * 100).toFixed(1)}%`;
        const matchedPlaceholder = document.getElementById('matchedPlaceholder');
        if (matchedPlaceholder) matchedPlaceholder.classList.add('d-none');
      } else {
        if (matchedImg) matchedImg.classList.add('d-none');
        const matchedPlaceholder = document.getElementById('matchedPlaceholder');
        if (matchedPlaceholder) matchedPlaceholder.classList.remove('d-none');
      }

      if (resultBox) resultBox.classList.remove('d-none');
      const placeholder = document.getElementById('resultPlaceholder');
      if (placeholder) placeholder.classList.add('d-none');
      
      // Refresh History Component
      fetchHistory();
      
      // Context for chat
      window._lastPrediction = data.class;
      addAssistantMessage(`Analysis complete. I have identified this as **${data.class}**. How can I help with care instructions?`);
    }
  } catch (err) {
    console.error(err);
  } finally {
    showLoading(predictBtn, false);
  }
});

// Assistant Helpers
function addAssistantMessage(text, isMarkdown = false) {
  if (!chatWindow) return;
  const msg = document.createElement('div');
  msg.className = 'chat-bubble chat-bubble-ai mb-3 shadow-none border fade-in';
  
  const content = isMarkdown && window.marked ? marked.parse(text) : `<div class="small">${text}</div>`;
  
  msg.innerHTML = `
    <div class="d-flex align-items-center gap-2 mb-2">
      <div class="bg-primary-container rounded-circle p-1 d-flex align-items-center justify-content-center" style="width:24px; height:24px"><i class="bi bi-robot text-white x-small"></i></div>
      <span class="small fw-bold text-dark">Dr. Derma AI</span>
    </div>
    <div class="ai-msg-content" style="font-size: 0.85rem;">${content}</div>
  `;
  chatWindow.appendChild(msg);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

// Chat logic
if (chatSend) {
  chatSend.onclick = async () => {
    const q = chatInput.value.trim();
    if (!q) return;

    // User Bubble
    const uMsg = document.createElement('div');
    uMsg.className = 'chat-bubble chat-bubble-user mb-3 animate-slide-up align-self-end';
    uMsg.innerHTML = `<div class="p-2 bg-primary text-white rounded-3 small">${q}</div>`;
    chatWindow.appendChild(uMsg);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    chatInput.value = '';

    // Show Loading
    const loadingMsg = document.createElement('div');
    loadingMsg.className = 'chat-bubble chat-bubble-ai mb-3 shadow-none border fade-in align-self-start';
    loadingMsg.id = 'chatLoadingBubble';
    loadingMsg.innerHTML = `
      <div class="d-flex align-items-center gap-2 mb-2">
        <div class="bg-primary-container rounded-circle p-1 d-flex align-items-center justify-content-center" style="width:24px; height:24px"><i class="bi bi-robot text-white x-small"></i></div>
        <span class="small fw-bold text-dark">Dr. Derma AI</span>
      </div>
      <div class="small text-muted"><i class="bi bi-three-dots animate-pulse"></i> Generating insight...</div>
    `;
    chatWindow.appendChild(loadingMsg);
    chatWindow.scrollTop = chatWindow.scrollHeight;

    try {
      const res = await fetch('/chat', { 
        method: 'POST', 
        headers: {'Content-Type': 'application/json'}, 
        body: JSON.stringify({ question: q, label: window._lastPrediction || '' }) 
      });
      const data = await res.json();
      
      // Remove loading
      const loader = document.getElementById('chatLoadingBubble');
      if (loader) loader.remove();

      if (data.error) {
        addAssistantMessage("I'm sorry, I'm having trouble connecting to my specialized AI brain right now. " + data.error);
      } else {
        const ans = data.answer.answer || data.answer;
        addAssistantMessage(ans, true); // pass true to parse as markdown
      }
    } catch (e) {
      const loader = document.getElementById('chatLoadingBubble');
      if (loader) loader.remove();
      addAssistantMessage("I'm sorry, I'm having trouble connecting right now.");
    }
  };
}

// Init
document.addEventListener('DOMContentLoaded', () => {
  fetchHistory();
});

// File Preview
if (fileInput) {
  fileInput.addEventListener('change', e => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (re => {
        previewImg.src = re.target.result;
        previewImg.classList.remove('d-none');
        
        const container = document.getElementById('previewContainer');
        const prompt = document.getElementById('uploadPrompt');
        if (container) container.classList.remove('d-none');
        if (prompt) prompt.classList.add('d-none');
      });
      reader.readAsDataURL(file);

      if (resultBox) resultBox.classList.add('d-none');
      const placeholder = document.getElementById('resultPlaceholder');
      if (placeholder) placeholder.classList.remove('d-none');
    }
  });
}

// Styles Helper
const style = document.createElement('style');
style.innerHTML = `
  @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
  .animate-spin { display: inline-block; animation: spin 1s linear infinite; }
  .confidence-meter { height: 6px; background: #E2E8F0; border-radius: 10px; overflow: hidden; }
  .confidence-fill { height: 100%; background: var(--color-primary); border-radius: 10px; transition: width 1s ease; }
  .bg-primary-light { background: rgba(13, 148, 136, 0.05); }
  .bg-primary-container { background: var(--color-primary); }
  .bg-secondary-light { background: rgba(16, 185, 129, 0.1); }
  .font-heading { font-family: 'Figtree', sans-serif; }
`;
document.head.appendChild(style);

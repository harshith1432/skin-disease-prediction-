const previewImg = document.getElementById('previewImg');
const matchedImg = document.getElementById('matchedImg');
const matchedLabel = document.getElementById('matchedLabel');
const matchedSim = document.getElementById('matchedSim');
const fileInput = document.getElementById('fileInput');
const predictBtn = document.getElementById('predictBtn');
const resultBox = document.getElementById('result');
const chatWindow = document.getElementById('chatWindow');
const chatInput = document.getElementById('chatInput');
const chatSend = document.getElementById('chatSend');

if (fileInput) {
  fileInput.addEventListener('change', e => {
    const file = e.target.files[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    if (previewImg) previewImg.src = url;
    if (resultBox) resultBox.textContent = '';
    if (matchedImg) matchedImg.src = '/static/img/placeholder.svg';
    if (matchedLabel) matchedLabel.textContent = 'No match yet';
    if (matchedSim) matchedSim.textContent = '';
  });
}

if (predictBtn) predictBtn.addEventListener('click', async () => {
  const file = fileInput.files[0];
  if (!file) { resultBox.textContent = 'Please choose an image.'; return }

  // show loading state on button
  predictBtn.disabled = true;
  predictBtn.classList.add('loading');
  const spinner = document.createElement('span');
  spinner.className = 'spinner-border spinner-border-sm spinner-in-btn';
  predictBtn.appendChild(spinner);

  const form = new FormData(); form.append('file', file);
  try {
    const res = await fetch('/predict', { method: 'POST', body: form });
    if (res.status === 401 || res.status === 302) {
      // not logged in — redirect to login
      window.location = '/login';
      return;
    }
    const data = await res.json();
    if (data.error) {
      resultBox.textContent = data.error
    } else {
      // Prediction summary (show top-3)
      if (data.top && Array.isArray(data.top) && data.top.length){
        let html = `<strong>Top predictions</strong><div class="mt-2">`;
        data.top.forEach(p => {
          const pct = (p.confidence * 100).toFixed(2);
          html += `<div class="small d-flex align-items-center mb-1"><div style="flex:1">${p.class}</div><div style="width:90px;text-align:right;margin-left:8px">${pct}%</div></div>`;
        });
        html += `</div>`;
        resultBox.innerHTML = html;
      } else {
        resultBox.innerHTML = `<strong>Prediction:</strong> ${data.class} <br><span class="small">Confidence: ${(data.confidence * 100).toFixed(2)}%</span>`
      }

      // Render suggestions into #suggestions if present (dashboard)
      const suggestionsEl = document.getElementById('suggestions')
      if (suggestionsEl) {
        if (data.suggestions) {
          const s = data.suggestions
          const klass = data.class ? data.class.toString().toLowerCase() : ''
          const iconPath = `/static/img/remedy/${encodeURIComponent(klass)}.svg`
          let card = ''
          card += `<div class="suggestion-card p-3 fade-in">
                    <div class="d-flex align-items-start">
                      <div class="me-2" style="width:44px;height:44px;flex-shrink:0">
                        <img src="${iconPath}" alt="icon" style="width:44px;height:44px;border-radius:8px;border:1px solid #eef5ff;background:white" onerror="this.style.display='none'"/>
                      </div>
                      <div>
                        ${s.title ? `<div class="fw-bold">${s.title}</div>` : ''}
                        ${s.short ? `<div class="small text-muted mb-2">${s.short}</div>` : ''}
                      </div>
                    </div>`
          if (s.do && s.do.length) card += `<div class="mt-2"><strong>Do</strong><ul class="small">${s.do.map(i=>`<li>${i}</li>`).join('')}</ul></div>`
          if (s.dont && s.dont.length) card += `<div class="mt-1"><strong>Don't</strong><ul class="small text-danger">${s.dont.map(i=>`<li>${i}</li>`).join('')}</ul></div>`
          if (s.medicines && s.medicines.length) card += `<div class="mt-1"><strong>Medicines / treatments</strong><ul class="small">${s.medicines.map(i=>`<li>${i}</li>`).join('')}</ul></div>`
          if (s.care_types && s.care_types.length) card += `<div class="mt-1"><strong>Care types:</strong> <div class="small">${s.care_types.join(', ')}</div></div>`
          if (s.disclaimer) card += `<div class="mt-2 small text-muted">${s.disclaimer}</div>`
          if (s.home_remedies && s.home_remedies.length) card += `<div class="mt-2"><strong>Home remedies</strong><ul class="small">${s.home_remedies.map(i=>`<li>${i}</li>`).join('')}</ul></div>`
          card += `</div>`
          suggestionsEl.innerHTML = card
        } else {
          suggestionsEl.innerHTML = `<div class="text-muted small">No suggestion available.</div>`
        }
      }
      if (data.matched_path && matchedImg) {
        // ensure class for animation
        matchedImg.classList.add('matched-img');
        matchedImg.classList.add('enter');
        matchedImg.classList.remove('show');
        matchedImg.style.opacity = 0;
        matchedImg.onload = () => {
          matchedImg.classList.add('show');
          matchedImg.style.opacity = 1;
        }
        matchedImg.src = `/matched?path=${encodeURIComponent(data.matched_path)}`;
        matchedLabel.textContent = data.matched_label || data.matched_path;
        if (data.similarity !== undefined && matchedSim) matchedSim.textContent = `Similarity: ${(data.similarity * 100).toFixed(2)}%`;
      }
      // attach last predicted label for chat context (use top-1)
      window._lastPrediction = (data.top && data.top[0] && data.top[0].class) ? data.top[0].class : (data.class || null)
    }
  } catch (err) { resultBox.textContent = 'Prediction failed.' }

  // restore button
  predictBtn.disabled = false;
  predictBtn.classList.remove('loading');
  const existing = predictBtn.querySelector('.spinner-in-btn');
  if (existing) existing.remove();
  predictBtn.textContent = 'Predict'
});

// Chat send handler
if (chatSend && chatInput && chatWindow) {
  async function sendChat(){
    const q = chatInput.value && chatInput.value.trim();
    if (!q) return;
    const userMsg = document.createElement('div'); userMsg.className='mb-2'; userMsg.innerHTML = `<div class="small text-muted">You</div><div>${q}</div>`;
    chatWindow.appendChild(userMsg); chatWindow.scrollTop = chatWindow.scrollHeight; chatInput.value='';
    // send to server
    try{
      const payload = { question: q, label: window._lastPrediction };
      const res = await fetch('/chat', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload) });
      const data = await res.json();
      if (data.error){
        const err = document.createElement('div'); err.className='text-danger small'; err.textContent = data.error; chatWindow.appendChild(err);
      } else {
        const ans = data.answer;
        const bot = document.createElement('div'); bot.className='mb-3';
        bot.innerHTML = `<div class="small text-muted">Assistant</div>`;
        if (ans.type === 'home_remedies' || ans.type === 'medicines'){
          bot.innerHTML += `<ul class="small">${ans.items.map(x=>`<li>${x}</li>`).join('')}</ul>`
        } else if (ans.type === 'summary' || ans.type === 'escalation' || ans.type === 'generic'){
          bot.innerHTML += `<div class="small">${ans.answer}</div>`
        } else {
          bot.innerHTML += `<div class="small">${JSON.stringify(ans)}</div>`
        }
        chatWindow.appendChild(bot);
        // render retrieved dataset thumbnails (if any)
        if (data.retrieved && Array.isArray(data.retrieved) && data.retrieved.length){
          const grid = document.createElement('div');
          grid.className = 'retrieved-grid mt-2';
          data.retrieved.forEach(r => {
            try{
              const item = document.createElement('div'); item.className='retrieved-item';
              const img = document.createElement('img');
              img.src = `/matched?path=${encodeURIComponent(r.path)}`;
              img.alt = r.label || '';
              img.title = `${r.label || ''} • ${(r.score*100).toFixed(1)}%`;
              img.className = 'retrieved-thumb';
              // clicking a retrieved example sets preview & matched panels
              img.addEventListener('click', ()=>{
                try{
                  if (previewImg) previewImg.src = img.src;
                  if (matchedImg) matchedImg.src = img.src;
                  if (matchedLabel) matchedLabel.textContent = r.label || r.path;
                  if (matchedSim) matchedSim.textContent = `Similarity: ${(r.score*100).toFixed(2)}%`;
                }catch(e){/* ignore */}
              });
              const cap = document.createElement('div'); cap.className='retrieved-caption';
              // clickable link that opens the matched image in a new tab
              const link = document.createElement('a');
              link.href = `/matched?path=${encodeURIComponent(r.path)}`;
              link.target = '_blank';
              link.rel = 'noopener noreferrer';
              link.className = 'retrieved-link small';
              link.textContent = r.label || '(example)';
              const scoreSpan = document.createElement('div');
              scoreSpan.className = 'retrieved-score small text-muted';
              scoreSpan.textContent = `${(r.score*100).toFixed(1)}%`;
              cap.appendChild(link);
              cap.appendChild(scoreSpan);
              item.appendChild(img); item.appendChild(cap);
              grid.appendChild(item);
            }catch(e){/* ignore item */}
          });
          chatWindow.appendChild(grid);
        }
        chatWindow.scrollTop = chatWindow.scrollHeight;
      }
    } catch(e){
      const err = document.createElement('div'); err.className='text-danger small'; err.textContent = 'Chat failed.'; chatWindow.appendChild(err);
    }
  }
  chatSend.addEventListener('click', sendChat);
  chatInput.addEventListener('keydown', e => { if (e.key === 'Enter') sendChat(); });
}

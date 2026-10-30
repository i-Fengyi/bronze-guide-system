const $ = (id) => document.getElementById(id);
const setText = (id, value) => { const node = $(id); if (node) node.textContent = value; };
const state = { exhibit: null, vesselType: null, decision: null, history: [], stream: null, busy: false };
const fileInput = $('file-input');
const dropzone = $('dropzone');

async function api(url, options = {}) {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `请求失败 (${response.status})`);
  return body;
}

async function loadStatus() {
  try {
    const data = await api('/api/status');
    const qwen = data.api_key_configured ? '问答已配置' : '待配置千问 API';
    const vision = data.vision_model_loaded ? '视觉模型已加载' : '视觉权重待安装';
    $('system-status').textContent = `${qwen} · ${vision}`;
    document.querySelector('.status-dot').classList.toggle('ready', data.api_key_configured);
    const exhibits = await api('/api/exhibits');
    $('catalog-count').textContent = `藏品 ${exhibits.length}`;
  } catch (error) { $('system-status').textContent = error.message; }
}

function showPreview(file) {
  if (!file || !file.type.startsWith('image/')) return;
  $('preview-image').src = URL.createObjectURL(file);
  $('preview').classList.remove('hidden');
  dropzone.classList.add('hidden');
  identify(file);
}

async function identify(file) {
  setBusy(true);
  $('result-panel').classList.add('hidden');
  const form = new FormData(); form.append('file', file);
  try {
    const data = await api('/api/identify', { method: 'POST', body: form });
    state.exhibit = data.exhibit;
    renderExhibit(data);
  } catch (error) { addMessage('assistant', error.message); }
  finally { setBusy(false); }
}

function renderExhibit(data) {
  const e = data.exhibit;
  const type = data.vessel_type;
  state.vesselType = type;
  state.decision = data.decision;
  $('result-panel').classList.remove('hidden');
  const uncertain = data.decision === 'retry_with_top3';
  $('result-name').textContent = e?.name || (uncertain ? '暂未识别' : `${type.name}（器型识别）`);
  $('result-period').textContent = e?.period || decisionText(data.decision);
  $('type-score').textContent = `器型置信度 ${(type.confidence * 100).toFixed(1)}%`;
  $('result-score').textContent = data.exhibit_similarity == null ? '无可靠展品匹配' : `展品相似度 ${data.exhibit_similarity.toFixed(3)}`;
  $('result-description').textContent = e?.description || decisionDescription(data.decision, type.name);
  setText('result-icon', uncertain ? '?' : '✓');
  setText('type-candidate-heading', uncertain ? '可能的器型（点击确认）' : '器型候选');
  setText('exhibit-candidate-heading', data.allow_exhibit_confirmation ? '可能的展品（点击确认）' : '同器型展品候选');
  $('type-candidate-list').replaceChildren(...data.type_candidates.map((c, index) => {
    const chip = document.createElement(data.allow_type_confirmation ? 'button' : 'span'); chip.className = 'candidate';
    chip.textContent = `${index + 1}. ${c.name} · ${(c.confidence * 100).toFixed(1)}%`;
    if (data.allow_type_confirmation) { chip.type = 'button'; chip.addEventListener('click', () => confirmType(c)); }
    return chip;
  }));
  $('candidate-list').replaceChildren(...data.candidates.map((c, index) => {
    const chip = document.createElement(data.allow_exhibit_confirmation ? 'button' : 'span'); chip.className = 'candidate';
    chip.textContent = `${index + 1}. ${c.name} · ${c.score.toFixed(3)}`;
    if (data.allow_exhibit_confirmation) { chip.type = 'button'; chip.addEventListener('click', () => confirmExhibit(c.exhibit_id)); }
    return chip;
  }));
  const context = e ? `当前展品：${e.name}` : (uncertain ? '等待人工确认器型' : `当前器型：${type.name}`);
  $('chat-context').innerHTML = `<span class="context-dot active"></span><span>${escapeHtml(context)}</span>`;
  state.exhibit = uncertain ? null : e;
  $('guide-button').disabled = uncertain;
  if (data.needs_confirmation) addMessage('assistant', '器型识别置信度较低。请让器物完整入镜、保持光线均匀并换一个正对角度重新拍摄；上方已保留 Top-3 候选供参考。');
}

function decisionText(decision) { return decision === 'specific_exhibit' ? '已匹配具体展品' : decision === 'vessel_type_only' ? '仅确认器型' : '建议重新拍摄'; }
function decisionDescription(decision, name) { return decision === 'vessel_type_only' ? `系统对“${name}”器型判断较有把握，但未找到相似度足够高的具体馆藏，因此只按器型提供介绍。若你认为候选展品正确，可以点击下方候选进行确认。` : '当前画面与系统已收录的青铜器器型匹配度不足，暂不直接下结论。请调整角度、距离和光线后重新拍摄，或从候选中人工确认。'; }

function escapeHtml(value) { return value.replace(/[&<>"']/g, (c) => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;' }[c])); }

function addMessage(role, content, sources = []) {
  const row = document.createElement('div'); row.className = `message-row ${role}`;
  const avatar = document.createElement('div'); avatar.className = 'message-avatar'; avatar.textContent = role === 'user' ? '我' : '鼎';
  const item = document.createElement('div'); item.className = `message ${role}`; item.textContent = content;
  if (sources.length) {
    const links = document.createElement('div'); links.className = 'message-sources';
    for (const source of sources) { const a = document.createElement('a'); a.className = 'source-link'; a.href = source.url; a.target = '_blank'; a.rel = 'noopener noreferrer'; a.textContent = `来源：${source.name}`; links.append(a); }
    item.append(links);
  }
  row.append(avatar, item); $('messages').append(row); $('messages').scrollTop = $('messages').scrollHeight;
}

function confirmType(candidate) {
  state.vesselType = candidate; state.exhibit = null; state.decision = 'vessel_type_only';
  $('result-name').textContent = `${candidate.name}（已确认器型）`;
  $('result-period').textContent = '用户已确认器型';
  $('result-description').textContent = `已按“${candidate.name}”作为当前器型继续。你可以点击“生成展品讲解”获取器型介绍，或在右侧继续提问。`;
  $('chat-context').innerHTML = `<span class="context-dot active"></span><span>当前器型：${escapeHtml(candidate.name)}</span>`;
  $('guide-button').disabled = false;
  addMessage('assistant', `已按你的选择确认器型为“${candidate.name}”。`);
}

async function confirmExhibit(exhibitId) {
  try {
    const exhibit = await api(`/api/exhibits/${encodeURIComponent(exhibitId)}`);
    state.exhibit = exhibit; state.vesselType = { label: exhibit.vessel_type, name: exhibit.name };
    state.decision = 'specific_exhibit';
    $('result-name').textContent = exhibit.name; $('result-period').textContent = exhibit.period;
    $('result-description').textContent = exhibit.description;
    $('chat-context').innerHTML = `<span class="context-dot active"></span><span>当前展品：${escapeHtml(exhibit.name)}</span>`;
    $('guide-button').disabled = false;
    addMessage('assistant', `已按你的选择切换到“${exhibit.name}”，后续问答将围绕这件展品展开。`);
  } catch (error) { addMessage('assistant', error.message); }
}

async function sendQuestion(question) {
  if (!question.trim() || state.busy) return;
  addMessage('user', question);
  const previous = [...state.history];
  state.history.push({ role: 'user', content: question });
  const typing = document.createElement('div'); typing.className = 'typing'; typing.textContent = '正在检索资料并整理回答…'; $('messages').append(typing);
  setBusy(true);
  try {
    const data = await api('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question, exhibit_id: state.exhibit?.id || null, vessel_type: state.vesselType?.label || null, history: previous }) });
    typing.remove(); addMessage('assistant', data.answer, data.sources); state.history.push({ role: 'assistant', content: data.answer });
  } catch (error) { typing.remove(); addMessage('assistant', error.message); state.history.pop(); }
  finally { setBusy(false); }
}

function setBusy(value) { state.busy = value; $('send-button').disabled = value; $('capture-button').disabled = value || !state.stream; $('guide-button').disabled = value || state.decision === 'retry_with_top3'; }

async function startCamera() {
  try {
    stopCamera(); state.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
    $('camera-video').srcObject = state.stream; $('camera-view').classList.remove('hidden'); dropzone.classList.add('hidden'); $('preview').classList.add('hidden'); $('capture-button').disabled = false;
  } catch (error) { addMessage('assistant', `无法开启摄像头：${error.message}。请检查浏览器权限，或改用图片上传。`); }
}

function stopCamera() { if (state.stream) state.stream.getTracks().forEach(track => track.stop()); state.stream = null; }

document.querySelectorAll('.mode-tab').forEach(tab => tab.addEventListener('click', async () => {
  const camera = tab.dataset.mode === 'camera';
  document.querySelectorAll('.mode-tab').forEach(button => { button.classList.toggle('active', button === tab); button.setAttribute('aria-selected', String(button === tab)); });
  $('result-panel').classList.add('hidden'); $('preview').classList.add('hidden');
  if (camera) await startCamera(); else { stopCamera(); $('camera-view').classList.add('hidden'); dropzone.classList.remove('hidden'); }
}));

$('choose-image').addEventListener('click', (event) => { event.stopPropagation(); fileInput.click(); });
dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') fileInput.click(); });
fileInput.addEventListener('change', () => showPreview(fileInput.files[0]));
dropzone.addEventListener('dragover', (event) => { event.preventDefault(); dropzone.classList.add('dragover'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', (event) => { event.preventDefault(); dropzone.classList.remove('dragover'); showPreview(event.dataTransfer.files[0]); });
$('clear-preview').addEventListener('click', () => { $('preview').classList.add('hidden'); dropzone.classList.remove('hidden'); $('result-panel').classList.add('hidden'); state.exhibit = null; state.vesselType = null; state.decision = null; fileInput.value = ''; });
$('capture-button').addEventListener('click', () => {
  const video = $('camera-video'), canvas = $('camera-canvas'); if (!video.videoWidth) return;
  canvas.width = video.videoWidth; canvas.height = video.videoHeight; canvas.getContext('2d').drawImage(video, 0, 0);
  canvas.toBlob(blob => { $('preview-image').src = URL.createObjectURL(blob); $('preview').classList.remove('hidden'); const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' }); identify(file); }, 'image/jpeg', .9);
});
$('chat-form').addEventListener('submit', event => { event.preventDefault(); const input = $('question-input'); const question = input.value; input.value = ''; sendQuestion(question); });
$('question-input').addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); $('chat-form').requestSubmit(); } });
$('suggestions').addEventListener('click', event => { if (event.target.tagName === 'BUTTON') sendQuestion(event.target.textContent); });
$('guide-button').addEventListener('click', async () => {
  if (!state.exhibit && !state.vesselType) return; setBusy(true); const typing = document.createElement('div'); typing.className = 'typing'; typing.textContent = '正在整理讲解…'; $('messages').append(typing);
  const form = new FormData(); form.append('exhibit_id', state.exhibit?.id || ''); form.append('vessel_type', state.vesselType?.label || ''); form.append('history', JSON.stringify(state.history));
  try { const data = await api('/api/guide', { method: 'POST', body: form }); typing.remove(); addMessage('assistant', data.answer, data.sources); state.history.push({ role: 'assistant', content: data.answer }); }
  catch (error) { typing.remove(); addMessage('assistant', error.message); }
  finally { setBusy(false); }
});
$('clear-chat').addEventListener('click', () => { state.history = []; $('messages').innerHTML = '<div class="welcome-message"><div class="assistant-mark">鼎</div><div><strong>你好，我是青铜器导览助手。</strong><p>识别展品后，我可以介绍它的年代、器型和纹饰。回答会附上资料来源。</p></div></div>'; });
loadStatus();

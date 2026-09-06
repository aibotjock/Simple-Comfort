/* Simple-Comfort front-end — vanilla JS, no build step. */
"use strict";

const state = {
  mode: "t2v",
  model: "ltx25",
  continuity: "new",
  aspect_ratio: "16:9",
  quality: "balanced",
  models: {},          // key -> status object
  customWorkflows: {}, // slug -> record (with schema)
  activeWorkflow: null,
  currentJob: null,
  ws: null,
  refFile: null,
  continueFrom: null,  // clip_id to continue
  lastClip: null,
  projectId: null,
  currentResultClip: null,
};

const MODEL_LABELS = { ltx25: "LTX 2.5", wan22: "Wan 2.2", framepack: "FramePack" };

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

/* --------------------------- init --------------------------- */
document.addEventListener("DOMContentLoaded", () => {
  wirePillGroups();
  wireDropzone();
  wireAdvanced();
  wireProQuality();
  wirePromptCounter();
  wireLipsyncState();
  wireAnnealToggle();
  wireButtons();
  wirePanels();
  wireLibraryForms();
  wireExportCards();
  wireWorkflowPanel();
  refreshStatus();
  refreshModels();
  refreshGallery();
  loadAvatars();
  loadVoices();
  loadBrollCategories();
  loadCustomWorkflows();
  setInterval(refreshStatus, 5000);

  // Ctrl+Enter to generate.
  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      generate();
    }
  });
});

/* --------------------------- pill groups --------------------------- */
function wirePillGroups() {
  $$(".pill-group").forEach((group) => {
    const name = group.dataset.name;
    group.addEventListener("click", (e) => {
      const btn = e.target.closest(".pill");
      if (!btn || btn.disabled) return;
      group.querySelectorAll(".pill").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      onPillChange(name, btn.dataset.value);
    });
  });
}

function onPillChange(name, value) {
  if (name === "mode") {
    state.mode = value;
    $("#ref-field").classList.toggle("hidden", value !== "i2v");
  } else if (name === "model") {
    state.model = value;
    const badge = $("#model-badge");
    if (badge) badge.textContent = MODEL_LABELS[value] || value;
    applyModelConstraints();
  } else if (name === "continuity") {
    state.continuity = value;
    handleContinuity(value);
  } else if (name === "aspect_ratio") {
    state.aspect_ratio = value;
  } else if (name === "quality") {
    state.quality = value;
    applyQualityDefaults(value);
  }
}

function applyQualityDefaults(q) {
  const map = { draft: [15, 3.0], balanced: [25, 3.5], quality: [40, 5.0] };
  const [steps, guidance] = map[q] || map.balanced;
  $("#steps").value = steps; $("#steps-val").textContent = steps;
  $("#guidance").value = guidance; $("#guidance-val").textContent = guidance;
}

/* --------------------------- prompt counter / autogrow --------------------------- */
function wirePromptCounter() {
  const ta = $("#prompt");
  const counter = $("#prompt-counter");
  if (!ta || !counter) return;
  const update = () => {
    const len = ta.value.length;
    counter.textContent = `${len}`;
    // auto-grow up to ~8 rows
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 260) + "px";
  };
  ta.addEventListener("input", update);
  update();
}

/* --------------------------- lip-sync state label --------------------------- */
function wireLipsyncState() {
  const cb = $("#lipsync");
  const label = $("#lipsync-state");
  if (!cb || !label) return;
  const update = () => { label.textContent = cb.checked ? "ON" : "OFF"; };
  cb.addEventListener("change", update);
  update();
}

/* --------------------------- anneal settings toggle --------------------------- */
function wireAnnealToggle() {
  const btn = $("#anneal-toggle");
  const panel = $("#anneal-settings");
  if (!btn || !panel) return;
  btn.addEventListener("click", () => {
    panel.classList.toggle("open");
    const caret = btn.querySelector(".summary-caret");
    if (caret) caret.style.transform = panel.classList.contains("open") ? "rotate(180deg)" : "";
  });
}

/* --------------------------- models --------------------------- */
async function refreshModels() {
  try {
    const res = await fetch("/api/models");
    const data = await res.json();
    state.models = {};
    (data.models || []).forEach((m) => { state.models[m.model] = m; });
    updateModelDots();
    applyModelConstraints();

    // GPU worker selector.
    const workerField = $("#worker-field");
    const workerSel = $("#worker");
    if (data.multi_gpu && data.workers) {
      workerSel.innerHTML = "";
      data.workers.forEach((w) => {
        const opt = document.createElement("option");
        opt.value = w.index;
        opt.textContent = `${w.label} ${w.alive ? "●" : "○"}`;
        workerSel.appendChild(opt);
      });
      workerField.classList.remove("hidden");
    } else {
      workerField.classList.add("hidden");
    }
  } catch (err) {
    console.error("models fetch failed", err);
  }
}

function updateModelDots() {
  $$("#model-group .pill").forEach((pill) => {
    const key = pill.dataset.value;
    const dot = pill.querySelector(".model-dot");
    const m = state.models[key];
    if (dot) dot.className = "dot model-dot " + dotClass(m);
    if (m) {
      pill.title = m.available
        ? `${m.display_name}: ready (T2V:${m.t2v_capable ? "yes" : "no"}, I2V:${m.i2v_capable ? "yes" : "no"})`
        : `${m.display_name}: ${m.detail || m.status}`;
    }
  });
}

function dotClass(m) {
  if (!m) return "dot-yellow";
  if (m.status === "unreachable") return "dot-red";
  return m.available ? "dot-green" : "dot-red";
}

function applyModelConstraints() {
  const m = state.models[state.model];
  // Disable modes not supported by the current model.
  $$("#mode-group .pill").forEach((pill) => {
    const mode = pill.dataset.value;
    let capable = true;
    if (m && m.available) {
      capable = mode === "t2v" ? m.t2v_capable : m.i2v_capable;
    } else if (m && !m.available) {
      capable = false;
    }
    pill.disabled = !capable;
    if (!capable && pill.classList.contains("active")) {
      // Switch to the other mode if current is disabled.
      const other = mode === "t2v" ? "i2v" : "t2v";
      const otherPill = $(`#mode-group .pill[data-value="${other}"]`);
      if (otherPill && !otherPill.disabled) {
        pill.classList.remove("active");
        otherPill.classList.add("active");
        state.mode = other;
        $("#ref-field").classList.toggle("hidden", other !== "i2v");
      }
    }
  });

  // Segment length only meaningful for multi-section models.
  const segField = $("#segment") ? $("#segment").closest(".adv-field") : null;
  if (segField) segField.style.display = state.model === "framepack" ? "none" : "flex";

  updateFormHint();
}

function updateFormHint() {
  const hint = $("#form-hint");
  const m = state.models[state.model];
  if (m && !m.available) {
    hint.textContent = `${m.display_name} unavailable — ${m.detail || m.status}`;
  } else {
    hint.textContent = "";
  }
}

/* --------------------------- status --------------------------- */
async function refreshStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    const dot = $("#conn-dot");
    const label = $("#conn-label");
    if (data.connected) {
      dot.className = "dot dot-green";
      label.textContent = data.multi_gpu
        ? `Connected (${data.workers.filter(w => w.alive).length} workers)`
        : "Connected";
    } else {
      dot.className = "dot dot-red";
      label.textContent = "Disconnected";
    }
    renderGpuChips(data);
    updateStatusBar(data);
  } catch (err) {
    $("#conn-dot").className = "dot dot-red";
    $("#conn-label").textContent = "Server error";
    const chips = $("#gpu-chips");
    if (chips) chips.hidden = true;
    const sbDot = $("#sb-dot");
    if (sbDot) {
      sbDot.className = "dot dot-red";
      $("#sb-conn").textContent = "Server error";
    }
  }
}

function updateStatusBar(data) {
  const sbDot = $("#sb-dot");
  if (!sbDot) return;
  if (data.connected) {
    sbDot.className = "dot dot-green";
    $("#sb-conn").textContent = "ComfyUI connected";
  } else {
    sbDot.className = "dot dot-red";
    $("#sb-conn").textContent = "ComfyUI disconnected";
  }
  $("#sb-ffmpeg").textContent = "FFmpeg: " + (data.ffmpeg ? "ok" : "⚠ missing");
  if (data.multi_gpu && Array.isArray(data.workers)) {
    const alive = data.workers.filter((w) => w.alive).length;
    $("#sb-gpu").textContent = `GPU: ${alive}/${data.workers.length} workers`;
  } else {
    $("#sb-gpu").textContent = "GPU: " + (data.connected ? "1 worker" : "—");
  }
  $("#sb-project").textContent = state.projectId
    ? "Project: " + String(state.projectId).slice(0, 8)
    : "No project";
}

/* Render one status chip per GPU in the header when multi-GPU is active.
   Dot color: green = alive & idle, yellow = alive & busy, red = offline.
   A small badge shows jobs_completed per GPU (e.g. "GPU 0 ✓12"). */
function renderGpuChips(data) {
  const wrap = $("#gpu-chips");
  if (!wrap) return;
  const workers = Array.isArray(data.workers) ? data.workers : [];
  // Only show per-GPU chips for multi-GPU setups; keep single connection
  // indicator for single-endpoint configs.
  if (!data.multi_gpu || workers.length <= 1) {
    wrap.hidden = true;
    wrap.innerHTML = "";
    return;
  }
  wrap.hidden = false;
  wrap.innerHTML = "";
  workers.forEach((w) => {
    const chip = document.createElement("span");
    chip.className = "gpu-chip";
    let dotClass = "dot-red";
    let title = "offline";
    if (w.alive && w.busy) {
      dotClass = "dot-yellow";
      title = "busy";
    } else if (w.alive) {
      dotClass = "dot-green";
      title = "idle";
    }
    chip.title = `${w.label}: ${title} — ${w.jobs_completed || 0} jobs completed`;

    const dot = document.createElement("span");
    dot.className = "dot " + dotClass;
    chip.appendChild(dot);

    const name = document.createElement("span");
    name.className = "gpu-chip-name";
    name.textContent = w.label;
    chip.appendChild(name);

    const badge = document.createElement("span");
    badge.className = "gpu-chip-badge";
    badge.textContent = "✓" + (w.jobs_completed || 0);
    chip.appendChild(badge);

    wrap.appendChild(chip);
  });
}

/* --------------------------- dropzone --------------------------- */
function wireDropzone() {
  const dz = $("#dropzone");
  const input = $("#ref-input");
  dz.addEventListener("click", () => input.click());
  input.addEventListener("change", () => {
    if (input.files && input.files[0]) setRefImage(input.files[0]);
  });
  ["dragover", "dragenter"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("dragover"); }));
  ["dragleave", "drop"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("dragover"); }));
  dz.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files[0];
    if (f) setRefImage(f);
  });
}

function setRefImage(file) {
  state.refFile = file;
  const img = $("#ref-preview");
  img.src = URL.createObjectURL(file);
  img.classList.remove("hidden");
  $("#dropzone-inner").classList.add("hidden");
  $("#dropzone").classList.add("has-image");
}

/* --------------------------- advanced --------------------------- */
function wireAdvanced() {
  const bind = (id, valId) => {
    const el = $(id), val = $(valId);
    if (el && val) el.addEventListener("input", () => { val.textContent = el.value; });
  };
  bind("#steps", "#steps-val");
  bind("#guidance", "#guidance-val");
  bind("#segment", "#segment-val");
  bind("#motion", "#motion-val");
}

/* --------------------------- continuity --------------------------- */
async function handleContinuity(value) {
  const panel = $("#continuity-panel");
  if (value === "continue") {
    try {
      const res = await fetch("/api/last-clip");
      const data = await res.json();
      if (!data.clip) {
        alert("No previous clip found. Generate a clip first.");
        selectPill("continuity-group", "new");
        state.continuity = "new";
        return;
      }
      applyContinueFrom(data.clip);
      panel.classList.remove("hidden");
    } catch (err) {
      console.error(err);
    }
  } else {
    panel.classList.add("hidden");
    state.continueFrom = null;
  }
}

function applyContinueFrom(clip) {
  state.continueFrom = clip.clip_id;
  state.projectId = clip.project_id;
  state.lastClip = clip;
  $("#cont-thumb").src = clip.thumbnail_url || clip.last_frame_url || "";
  $("#cont-prev-prompt").textContent = (clip.metadata && clip.metadata.prompt) || "—";

  // Preserve model / aspect / lip-sync / quality from the parent.
  const md = clip.metadata || {};
  if (md.model_key) { selectPill("model-group", md.model_key); state.model = md.model_key; const b = $("#model-badge"); if (b) b.textContent = MODEL_LABELS[md.model_key] || md.model_key; }
  if (md.aspect_ratio) { selectPill("aspect-group", md.aspect_ratio); state.aspect_ratio = md.aspect_ratio; }
  if (md.quality) { selectPill("quality-group", md.quality); state.quality = md.quality; }
  if (typeof md.lip_sync === "boolean") { $("#lipsync").checked = md.lip_sync; const ls = $("#lipsync-state"); if (ls) ls.textContent = md.lip_sync ? "ON" : "OFF"; }
  if (md.fps) $("#fps").value = String(md.fps);
  applyModelConstraints();
}

function selectPill(groupId, value) {
  const group = document.getElementById(groupId);
  if (!group) return;
  group.querySelectorAll(".pill").forEach((p) => {
    p.classList.toggle("active", p.dataset.value === value);
  });
}

/* --------------------------- buttons --------------------------- */
function wireButtons() {
  $("#generate-btn").addEventListener("click", generate);
  $("#cancel-btn").addEventListener("click", cancelJob);
  $("#continue-btn").addEventListener("click", () => {
    if (state.currentResultClip) {
      selectPill("continuity-group", "continue");
      state.continuity = "continue";
      applyContinueFrom(state.currentResultClip);
      $("#continuity-panel").classList.remove("hidden");
      $("#form-card").scrollIntoView({ behavior: "smooth" });
    }
  });
  $("#play-btn").addEventListener("click", () => { const v = $("#output-video"); v.play(); });
  $("#open-btn").addEventListener("click", () => {
    if (state.currentResultClip && state.currentResultClip.video_url) {
      window.open(state.currentResultClip.video_url, "_blank");
    }
  });
  const brollJump = $("#broll-jump-btn");
  if (brollJump) brollJump.addEventListener("click", () => openPanel("broll"));
}

/* --------------------------- generate --------------------------- */
async function generate() {
  const btn = $("#generate-btn");
  const hint = $("#form-hint");
  hint.textContent = "";

  const m = state.models[state.model];
  if (m && !m.available) {
    hint.textContent = `${m.display_name} is not ready.`;
    return;
  }
  if (state.mode === "i2v" && !state.refFile && !state.continueFrom) {
    hint.textContent = "Please add a reference image for Image-to-Video.";
    return;
  }

  let prompt = $("#prompt").value.trim();
  if (state.continuity === "continue") {
    const change = $("#cont-change").value.trim();
    if (change) prompt = change;
  }
  if (!prompt && state.mode === "t2v") {
    hint.textContent = "Please enter a prompt.";
    return;
  }

  const fd = new FormData();
  fd.append("prompt", prompt);
  fd.append("model", state.model);
  fd.append("mode", state.mode);
  fd.append("aspect_ratio", state.aspect_ratio);
  fd.append("quality", state.quality);
  fd.append("lip_sync", $("#lipsync").checked ? "true" : "false");
  const seed = $("#seed").value;
  if (seed) fd.append("seed", seed);
  fd.append("fps", $("#fps").value);
  fd.append("steps", $("#steps").value);
  fd.append("guidance", $("#guidance").value);
  const resolution = $("#resolution").value;
  if (resolution) {
    const [w, h] = resolution.split("x");
    fd.append("width", w); fd.append("height", h);
  }
  fd.append("segment_length", $("#segment").value);
  fd.append("motion_strength", $("#motion").value);

  // Professional quality controls.
  fd.append("denoise", $("#pro-denoise").checked ? "true" : "false");
  const interp = $("#pro-interp").value;
  if (interp) fd.append("interpolate_fps", interp);
  fd.append("camera_motion", $("#pro-camera").value);
  fd.append("face_stability", $("#pro-face").value);
  fd.append("upscale", $("#pro-upscale").value);
  fd.append("anti_drift", $("#pro-antidrift").value);

  // Optional avatar / voice linkage.
  const linkAvatar = $("#link-avatar").value;
  if (linkAvatar) fd.append("avatar_id", linkAvatar);
  const linkVoice = $("#link-voice").value;
  if (linkVoice) fd.append("voice_id", linkVoice);

  const workerField = $("#worker-field");
  if (!workerField.classList.contains("hidden")) {
    fd.append("worker_index", $("#worker").value);
  }
  if (state.continuity === "continue" && state.continueFrom) {
    fd.append("continue_from", state.continueFrom);
  }
  if (state.projectId) fd.append("project_id", state.projectId);
  if (state.refFile) fd.append("image", state.refFile);

  btn.disabled = true;
  btn.classList.add("generating");
  btn.textContent = "STARTING…";

  try {
    const res = await fetch("/api/generate", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) {
      hint.textContent = data.detail || "Generation failed to start.";
      resetGenerateBtn();
      return;
    }
    state.currentJob = data.job_id;
    state.projectId = data.project_id;
    showProgress();
    openWebSocket(data.job_id);
  } catch (err) {
    hint.textContent = "Network error: " + err.message;
    resetGenerateBtn();
  }
}

function resetGenerateBtn() {
  const btn = $("#generate-btn");
  btn.disabled = false;
  btn.classList.remove("generating");
  btn.textContent = "GENERATE 30s CLIP";
  const wfBtn = $("#wf-generate");
  if (wfBtn) {
    wfBtn.disabled = false;
    wfBtn.classList.remove("generating");
    wfBtn.textContent = "GENERATE WITH WORKFLOW";
  }
}

/* --------------------------- progress / websocket --------------------------- */
function showProgress() {
  $("#progress-card").classList.remove("hidden");
  $("#output-card").classList.add("hidden");
  $("#progress-fill").style.width = "0%";
  $("#progress-section").textContent = "Starting…";
  $("#progress-pct").textContent = "0%";
  $("#progress-eta").textContent = "";
  $("#progress-msg").textContent = "";
  $("#preview-frame").classList.add("hidden");
  $("#progress-card").scrollIntoView({ behavior: "smooth" });
}

function openWebSocket(jobId) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/progress/${jobId}`);
  state.ws = ws;
  ws.onmessage = (evt) => {
    const data = JSON.parse(evt.data);
    if (data.heartbeat) return;
    if (data.error) { $("#progress-msg").textContent = data.error; return; }
    updateProgress(data);
    if (data.status === "completed") { onComplete(data); ws.close(); }
    else if (data.status === "failed") { onFailed(data); ws.close(); }
    else if (data.status === "cancelled") { onCancelled(data); ws.close(); }
  };
  ws.onerror = () => { $("#progress-msg").textContent = "Connection lost — polling…"; pollJob(jobId); };
}

function updateProgress(data) {
  const pct = Math.round((data.progress || 0) * 100);
  $("#progress-fill").style.width = pct + "%";
  $("#progress-pct").textContent = pct + "%";
  $("#progress-section").textContent =
    `Section ${data.current_section || 1}/${data.total_sections || 1}`;
  if (data.eta != null) {
    $("#progress-eta").textContent = "ETA " + formatTime(data.eta);
  }
  if (data.message) $("#progress-msg").textContent = data.message;
  if (data.preview) {
    const img = $("#preview-frame");
    img.src = data.preview;
    img.classList.remove("hidden");
  }
}

function formatTime(sec) {
  sec = Math.round(sec);
  if (sec < 60) return sec + "s";
  return Math.floor(sec / 60) + "m " + (sec % 60) + "s";
}

async function pollJob(jobId) {
  // Fallback polling if WS drops.
  const timer = setInterval(async () => {
    try {
      const res = await fetch("/api/job/" + jobId);
      if (!res.ok) return;
      const data = await res.json();
      updateProgress(data);
      if (["completed", "failed", "cancelled"].includes(data.status)) {
        clearInterval(timer);
        if (data.status === "completed") onComplete(data);
        else if (data.status === "failed") onFailed(data);
        else onCancelled(data);
      }
    } catch (e) { /* keep polling */ }
  }, 2000);
}

function renderMeta(clip) {
  const strip = $("#meta-strip");
  if (!strip) return;
  const md = clip.metadata || {};
  const modelName = MODEL_LABELS[md.model_key] || md.model || md.model_key || "—";
  const pills = [
    ["Model", modelName],
    ["Seed", md.seed != null ? md.seed : "random"],
    ["Res", md.resolution || "—"],
    ["FPS", md.fps != null ? md.fps : "—"],
    ["Quality", md.quality || "—"],
    ["Sections", md.num_sections != null ? md.num_sections : "—"],
    ["Lip-Sync", md.lip_sync ? "on" : "off"],
  ];
  strip.innerHTML = pills
    .map(([k, v]) => `<span class="meta-pill">${k} <b>${escapeHtml(String(v))}</b></span>`)
    .join("");
}

function onComplete(data) {
  resetGenerateBtn();
  $("#progress-card").classList.add("hidden");
  const clip = data.result;
  state.currentResultClip = clip;
  const card = $("#output-card");
  card.classList.remove("hidden");
  const video = $("#output-video");
  video.src = clip.video_url + "?t=" + Date.now();
  video.load();
  renderMeta(clip);
  card.scrollIntoView({ behavior: "smooth" });
  refreshGallery();
}

function onFailed(data) {
  resetGenerateBtn();
  $("#progress-msg").textContent = "❌ " + (data.error || "Generation failed");
  $("#form-hint").textContent = data.error || "Generation failed";
}

function onCancelled(data) {
  resetGenerateBtn();
  $("#progress-card").classList.add("hidden");
  $("#form-hint").textContent = "Generation cancelled.";
}

async function cancelJob() {
  if (!state.currentJob) return;
  await fetch("/api/cancel/" + state.currentJob, { method: "POST" });
  $("#progress-msg").textContent = "Cancelling…";
}

/* --------------------------- gallery --------------------------- */
async function refreshGallery() {
  try {
    const res = await fetch("/api/clips");
    const data = await res.json();
    const gallery = $("#gallery");
    gallery.innerHTML = "";
    const clips = data.clips || [];
    if (!clips.length) {
      gallery.innerHTML = `<div class="gallery-empty">No clips yet — generate your first 30-second clip above.</div>`;
      return;
    }
    clips.forEach((clip) => {
      const item = document.createElement("div");
      item.className = "gallery-item";
      const md = clip.metadata || {};
      const modelName = MODEL_LABELS[md.model_key] || md.model || "";
      item.innerHTML = `
        <img src="${clip.thumbnail_url || clip.first_frame_url || ""}" alt="clip" />
        <div class="g-meta">${escapeHtml(modelName)} · ${escapeHtml(md.quality || "")}</div>
        <button class="g-continue">Continue →</button>`;
      item.querySelector("img").addEventListener("click", () => playClip(clip));
      item.querySelector(".g-continue").addEventListener("click", (e) => {
        e.stopPropagation();
        selectPill("continuity-group", "continue");
        state.continuity = "continue";
        applyContinueFrom(clip);
        $("#continuity-panel").classList.remove("hidden");
        $("#form-card").scrollIntoView({ behavior: "smooth" });
      });
      gallery.appendChild(item);
    });
  } catch (err) {
    console.error("gallery fetch failed", err);
  }
}

function playClip(clip) {
  state.currentResultClip = clip;
  const card = $("#output-card");
  card.classList.remove("hidden");
  const video = $("#output-video");
  video.src = clip.video_url + "?t=" + Date.now();
  video.load(); video.play();
  renderMeta(clip);
  card.scrollIntoView({ behavior: "smooth" });
}

/* =================================================================== */
/* ==========================  PANELS & LIBRARIES  =================== */
/* =================================================================== */

/* --------------------------- pro-quality sliders --------------------------- */
function wireProQuality() {
  const bind = (id, valId) => {
    const el = $(id), val = $(valId);
    if (el && val) el.addEventListener("input", () => { val.textContent = el.value; });
  };
  bind("#pro-camera", "#pro-camera-val");
  bind("#pro-face", "#pro-face-val");
  bind("#pro-upscale", "#pro-upscale-val");
  bind("#pro-antidrift", "#pro-antidrift-val");
}

/* --------------------------- sliding panels / tabs --------------------------- */
function wirePanels() {
  const scrim = $("#panel-scrim");
  $$(".tab-btn[data-panel]").forEach((btn) => {
    btn.addEventListener("click", () => togglePanel(btn.dataset.panel));
  });
  $$(".panel-close[data-panel]").forEach((btn) => {
    btn.addEventListener("click", () => closePanels());
  });
  if (scrim) scrim.addEventListener("click", () => closePanels());
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closePanels();
  });
}

function openPanel(name) {
  closePanels();
  const panel = $("#panel-" + name);
  const scrim = $("#panel-scrim");
  if (!panel) return;
  panel.classList.add("open");
  panel.setAttribute("aria-hidden", "false");
  if (scrim) scrim.classList.remove("hidden");
  const tab = $(`.tab-btn[data-panel="${name}"]`);
  if (tab) tab.classList.add("active");
  if (name === "avatars") loadAvatars();
  else if (name === "voices") loadVoices();
  else if (name === "sequence") loadSequence();
  else if (name === "broll") loadBrollCategories();
}

function togglePanel(name) {
  const panel = $("#panel-" + name);
  if (panel && panel.classList.contains("open")) {
    closePanels();
  } else {
    openPanel(name);
  }
}

function closePanels() {
  $$(".side-panel").forEach((p) => {
    p.classList.remove("open");
    p.setAttribute("aria-hidden", "true");
  });
  $$(".tab-btn").forEach((t) => t.classList.remove("active"));
  const scrim = $("#panel-scrim");
  if (scrim) scrim.classList.add("hidden");
}

/* --------------------------- envelope helper --------------------------- */
async function apiJson(url, opts) {
  const res = await fetch(url, opts);
  let body = null;
  try { body = await res.json(); } catch (e) { body = null; }
  if (!res.ok || (body && body.success === false)) {
    const msg = (body && (body.error || body.detail)) || `Request failed (${res.status})`;
    throw new Error(msg);
  }
  if (body && typeof body === "object" && "success" in body && "data" in body) {
    return body.data;
  }
  return body;
}

/* --------------------------- avatars --------------------------- */
let _avatars = [];

async function loadAvatars() {
  try {
    const data = await apiJson("/api/avatars");
    _avatars = data.avatars || [];
    renderAvatars(_avatars);
    populateAvatarSelects(_avatars);
    populateVoiceAvatarSelects(_avatars);
  } catch (err) {
    console.error("avatars load failed", err);
  }
}

function renderAvatars(avatars) {
  const grid = $("#avatar-grid");
  if (!grid) return;
  const selected = $("#link-avatar") ? $("#link-avatar").value : "";
  grid.innerHTML = "";
  avatars.forEach((av) => {
    const cell = document.createElement("div");
    cell.className = "avatar-cell" + (av.placeholder ? " placeholder" : "");
    if (!av.placeholder && av.id && av.id === selected) cell.classList.add("selected");
    if (av.placeholder) {
      cell.innerHTML = `<div class="avatar-empty">＋<span>Create Avatar</span></div>`;
      cell.addEventListener("click", () => $("#avatar-name").focus());
    } else {
      cell.innerHTML = `
        ${av.voice_id ? '<div class="avatar-badge">🎙 voice</div>' : ""}
        <img src="${av.thumbnail_url || av.image_url || ""}" alt="${escapeHtml(av.name)}" />
        <div class="avatar-name">${escapeHtml(av.name || "Unnamed")}</div>
        <div class="avatar-actions">
          <button class="av-sel${av.id === selected ? " active" : ""}">Select</button>
          <button class="av-edit" title="Rename">✎</button>
          <button class="av-del" title="Delete">✕</button>
        </div>`;
      cell.querySelector(".av-sel").addEventListener("click", () => selectAvatar(av.id));
      cell.querySelector(".av-edit").addEventListener("click", () => renameAvatar(av));
      cell.querySelector(".av-del").addEventListener("click", () => deleteAvatar(av.id));
    }
    grid.appendChild(cell);
  });
}

function selectAvatar(avatarId) {
  const sel = $("#link-avatar");
  if (sel) sel.value = (sel.value === avatarId) ? "" : avatarId;
  renderAvatars(_avatars);
}

async function renameAvatar(av) {
  const name = prompt("Rename avatar:", av.name || "");
  if (name == null) return;
  try {
    await apiJson("/api/avatars/" + av.id, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    loadAvatars();
  } catch (err) {
    alert("Rename failed: " + err.message);
  }
}

function populateAvatarSelects(avatars) {
  const real = avatars.filter((a) => !a.placeholder && a.id);
  const linkSel = $("#link-avatar");
  if (!linkSel) return;
  const prev = linkSel.value;
  linkSel.innerHTML = `<option value="">No avatar</option>`;
  real.forEach((a) => {
    const opt = document.createElement("option");
    opt.value = a.id;
    opt.textContent = a.name || a.id;
    linkSel.appendChild(opt);
  });
  if (prev) linkSel.value = prev;
}

function populateVoiceAvatarSelects(avatars) {
  const real = avatars.filter((a) => !a.placeholder && a.id);
  // Add-voice form select + any per-row assign selects.
  const targets = [$("#voice-avatar"), ...$$(".voice-assign select")];
  targets.forEach((sel) => {
    if (!sel) return;
    const prev = sel.value;
    const firstLabel = sel.classList.contains("va-row") ? "Unassigned" : "Link to avatar (optional)";
    sel.innerHTML = `<option value="">${firstLabel}</option>`;
    real.forEach((a) => {
      const opt = document.createElement("option");
      opt.value = a.id;
      opt.textContent = a.name || a.id;
      sel.appendChild(opt);
    });
    if (prev) sel.value = prev;
  });
}

async function deleteAvatar(avatarId) {
  if (!avatarId) return;
  if (!confirm("Delete this avatar?")) return;
  try {
    await apiJson("/api/avatars/" + avatarId, { method: "DELETE" });
    loadAvatars();
  } catch (err) {
    alert("Delete failed: " + err.message);
  }
}

/* --------------------------- voices --------------------------- */
async function loadVoices() {
  try {
    const data = await apiJson("/api/voices");
    renderVoices(data.voices || []);
    populateVoiceSelect(data.voices || []);
  } catch (err) {
    console.error("voices load failed", err);
  }
}

function renderVoices(voices) {
  const list = $("#voice-list");
  if (!list) return;
  list.innerHTML = "";
  if (!voices.length) {
    list.innerHTML = `<div class="voice-empty">No voices yet.</div>`;
    return;
  }
  const realAvatars = _avatars.filter((a) => !a.placeholder && a.id);
  voices.forEach((v) => {
    const row = document.createElement("div");
    row.className = "voice-row";
    const dur = v.duration_seconds ? `${Math.round(v.duration_seconds)}s` : "";
    const options = ['<option value="">Unassigned</option>']
      .concat(realAvatars.map((a) =>
        `<option value="${a.id}" ${a.id === v.avatar_id ? "selected" : ""}>${escapeHtml(a.name || a.id)}</option>`))
      .join("");
    row.innerHTML = `
      <div class="voice-top">
        <span class="voice-ico">🎙</span>
        <div class="voice-info">
          <div class="voice-name">${escapeHtml(v.name || "Unnamed")}</div>
          <div class="voice-meta">${escapeHtml(v.sample_filename || "")} ${dur}</div>
        </div>
        <button class="voice-del" title="Delete">✕</button>
      </div>
      ${v.sample_url ? `<audio controls preload="none" src="${v.sample_url}"></audio>` : ""}
      <div class="voice-assign">
        <label>Avatar</label>
        <select class="input va-row">${options}</select>
      </div>`;
    row.querySelector(".voice-del").addEventListener("click", () => deleteVoice(v.id));
    row.querySelector(".va-row").addEventListener("change", (e) => assignVoiceAvatar(v.id, e.target.value));
    list.appendChild(row);
  });
}

function populateVoiceSelect(voices) {
  const sel = $("#link-voice");
  if (!sel) return;
  const prev = sel.value;
  sel.innerHTML = `<option value="">No voice</option>`;
  voices.forEach((v) => {
    const opt = document.createElement("option");
    opt.value = v.id;
    opt.textContent = v.name || v.id;
    sel.appendChild(opt);
  });
  if (prev) sel.value = prev;
}

async function assignVoiceAvatar(voiceId, avatarId) {
  try {
    await apiJson("/api/voices/" + voiceId + "/assign-avatar", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ avatar_id: avatarId || null }),
    });
    loadAvatars();
  } catch (err) {
    alert("Assign failed: " + err.message);
  }
}

async function deleteVoice(voiceId) {
  if (!voiceId) return;
  if (!confirm("Delete this voice?")) return;
  try {
    await apiJson("/api/voices/" + voiceId, { method: "DELETE" });
    loadVoices();
  } catch (err) {
    alert("Delete failed: " + err.message);
  }
}

/* --------------------------- library + panel forms --------------------------- */
function wireLibraryForms() {
  const avatarForm = $("#avatar-form");
  if (avatarForm) {
    avatarForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const hint = $("#avatar-hint");
      const file = $("#avatar-image").files[0];
      if (!file) { hint.textContent = "Please choose an image."; return; }
      const fd = new FormData();
      fd.append("name", $("#avatar-name").value.trim());
      fd.append("notes", $("#avatar-notes").value.trim());
      fd.append("image", file);
      hint.textContent = "Saving…";
      try {
        await apiJson("/api/avatars", { method: "POST", body: fd });
        avatarForm.reset();
        hint.textContent = "Saved.";
        loadAvatars();
      } catch (err) {
        hint.textContent = err.message;
      }
    });
  }

  const voiceForm = $("#voice-form");
  if (voiceForm) {
    voiceForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const hint = $("#voice-hint");
      const file = $("#voice-sample").files[0];
      if (!file) { hint.textContent = "Please choose an audio file."; return; }
      const fd = new FormData();
      fd.append("name", $("#voice-name").value.trim());
      fd.append("notes", $("#voice-notes").value.trim());
      const avatarId = $("#voice-avatar").value;
      if (avatarId) fd.append("avatar_id", avatarId);
      fd.append("sample", file);
      hint.textContent = "Saving…";
      try {
        await apiJson("/api/voices", { method: "POST", body: fd });
        voiceForm.reset();
        hint.textContent = "Saved.";
        loadVoices();
        loadAvatars();
      } catch (err) {
        hint.textContent = err.message;
      }
    });
  }

  const brollForm = $("#broll-form");
  if (brollForm) {
    brollForm.addEventListener("submit", (e) => { e.preventDefault(); submitBroll(); });
    const subject = $("#broll-subject");
    if (subject) subject.addEventListener("input", updateBrollPreview);
  }

  const exportForm = $("#export-form");
  if (exportForm) {
    exportForm.addEventListener("submit", (e) => { e.preventDefault(); downloadExport(); });
  }

  const seqAdd = $("#seq-add-btn");
  if (seqAdd) seqAdd.addEventListener("click", addCurrentClipToSequence);
  const anneal = $("#anneal-btn");
  if (anneal) anneal.addEventListener("click", annealCurrentClip);
}

/* --------------------------- export cards --------------------------- */
function wireExportCards() {
  $$(".export-card[data-value]").forEach((card) => {
    card.addEventListener("click", () => {
      $$(".export-card").forEach((c) => c.classList.remove("selected"));
      card.classList.add("selected");
      const hidden = $("#export-type");
      if (hidden) hidden.value = card.dataset.value;
    });
  });
}

/* --------------------------- b-roll --------------------------- */
let _brollCategories = [];
let _brollSelected = null;

async function loadBrollCategories() {
  const box = $("#broll-category");
  if (!box || box.dataset.loaded === "1") return;
  try {
    const data = await apiJson("/api/broll/categories");
    _brollCategories = data.categories || [];
    box.innerHTML = "";
    _brollCategories.forEach((c, i) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "broll-cat" + (i === 0 ? " active" : "");
      b.dataset.value = c.value;
      b.textContent = c.label;
      b.addEventListener("click", () => {
        box.querySelectorAll(".broll-cat").forEach((x) => x.classList.remove("active"));
        b.classList.add("active");
        _brollSelected = c.value;
        updateBrollPreview();
      });
      box.appendChild(b);
    });
    _brollSelected = _brollCategories.length ? _brollCategories[0].value : null;
    box.dataset.loaded = "1";
    updateBrollPreview();
  } catch (err) {
    console.error("broll categories load failed", err);
  }
}

function updateBrollPreview() {
  const preview = $("#broll-preview");
  if (!preview) return;
  const subject = ($("#broll-subject").value || "").trim();
  const cat = _brollCategories.find((c) => c.value === _brollSelected);
  if (!subject) { preview.textContent = ""; return; }
  const prefix = cat ? cat.prompt_prefix : "";
  preview.textContent = prefix ? `${prefix} ${subject}` : subject;
}

function selectedBrollVariants() {
  const active = $("#broll-variants .pill.active");
  return active ? parseInt(active.dataset.value, 10) || 1 : 1;
}

async function submitBroll() {
  const hint = $("#broll-hint");
  const subject = ($("#broll-subject").value || "").trim();
  if (!subject) { hint.textContent = "Enter a subject / description."; return; }
  if (!_brollSelected) { hint.textContent = "Choose a category."; return; }
  const m = state.models[state.model];
  if (m && !m.available) { hint.textContent = `${m.display_name} is not ready.`; return; }
  const payload = {
    model: state.model,
    category: _brollSelected,
    subject,
    aspect_ratio: state.aspect_ratio,
    quality: state.quality,
    variants: selectedBrollVariants(),
    project_id: state.projectId || undefined,
  };
  hint.textContent = "Submitting…";
  try {
    const data = await apiJson("/api/broll/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.projectId = data.project_id;
    const n = (data.jobs || []).length;
    hint.textContent = `Started ${n} b-roll job${n === 1 ? "" : "s"}. Results appear in the gallery.`;
    refreshStatus();
    let ticks = 0;
    const t = setInterval(() => {
      refreshGallery();
      if (++ticks > 30) clearInterval(t);
    }, 4000);
  } catch (err) {
    hint.textContent = err.message;
  }
}

/* --------------------------- sequence --------------------------- */
async function loadSequence() {
  const list = $("#seq-list");
  const empty = $("#seq-empty");
  const proj = $("#seq-project");
  if (!list) return;
  if (proj) proj.innerHTML = "Project: <b>" + (state.projectId ? escapeHtml(String(state.projectId).slice(0, 8)) : "—") + "</b>";
  if (!state.projectId) {
    list.innerHTML = "";
    if (empty) { empty.textContent = "No project yet. Generate a clip first."; empty.classList.remove("hidden"); }
    return;
  }
  try {
    const data = await apiJson("/api/sequence/" + state.projectId);
    renderSequence(data.sequence || []);
  } catch (err) {
    console.error("sequence load failed", err);
  }
}

function renderSequence(items) {
  const list = $("#seq-list");
  const empty = $("#seq-empty");
  if (!list) return;
  list.innerHTML = "";
  if (!items.length) {
    if (empty) { empty.textContent = "Generate a clip and click “＋ Sequence” to build your edit."; empty.classList.remove("hidden"); }
    return;
  }
  if (empty) empty.classList.add("hidden");
  items.forEach((it, idx) => {
    const row = document.createElement("div");
    row.className = "seq-row";
    row.innerHTML = `
      <span class="seq-idx">${idx + 1}</span>
      <img class="seq-thumb" src="${it.thumbnail_url || ""}" alt="" />
      <div class="seq-info">
        <div class="seq-label">${escapeHtml(it.label || it.clip_id || "clip")}</div>
        <div class="seq-type">${escapeHtml(it.type || "clip")}</div>
      </div>
      <div class="seq-actions">
        <button class="seq-up" title="Move up" ${idx === 0 ? "disabled" : ""}>▲</button>
        <button class="seq-down" title="Move down" ${idx === items.length - 1 ? "disabled" : ""}>▼</button>
        <button class="seq-del" title="Remove">✕</button>
      </div>`;
    row.querySelector(".seq-up").addEventListener("click", () => moveSequenceItem(it.id, "up"));
    row.querySelector(".seq-down").addEventListener("click", () => moveSequenceItem(it.id, "down"));
    row.querySelector(".seq-del").addEventListener("click", () => removeSequenceItem(it.id));
    list.appendChild(row);
  });
}

async function addCurrentClipToSequence() {
  const clip = state.currentResultClip;
  if (!clip) { alert("No clip selected."); return; }
  const projectId = clip.project_id || state.projectId;
  if (!projectId) { alert("No project for this clip."); return; }
  try {
    const md = clip.metadata || {};
    const label = (md.prompt || "").slice(0, 48) || clip.clip_id;
    const data = await apiJson("/api/sequence/" + projectId, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ clip_id: clip.clip_id, label }),
    });
    renderSequence(data.sequence || []);
    openPanel("sequence");
  } catch (err) {
    alert("Add to sequence failed: " + err.message);
  }
}

async function moveSequenceItem(itemId, direction) {
  if (!state.projectId) return;
  try {
    const data = await apiJson("/api/sequence/" + state.projectId + "/move", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ item_id: itemId, direction }),
    });
    renderSequence(data.sequence || []);
  } catch (err) {
    alert("Move failed: " + err.message);
  }
}

async function removeSequenceItem(itemId) {
  if (!state.projectId) return;
  try {
    const data = await apiJson(
      "/api/sequence/" + state.projectId + "/" + itemId,
      { method: "DELETE" }
    );
    renderSequence(data.sequence || []);
  } catch (err) {
    alert("Remove failed: " + err.message);
  }
}

/* --------------------------- annealing / bridge --------------------------- */
async function annealCurrentClip() {
  const clip = state.currentResultClip;
  if (!clip) { alert("No clip selected."); return; }
  const btn = $("#anneal-btn");
  if (btn) { btn.disabled = true; btn.textContent = "Bridging…"; }
  try {
    const data = await apiJson("/api/clips/" + clip.clip_id + "/anneal-to-next", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: state.model }),
    });
    state.projectId = data.project_id;
    state.currentJob = data.job_id;
    showProgress();
    openWebSocket(data.job_id);
  } catch (err) {
    alert("Bridge failed: " + err.message);
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = "⤳ Bridge"; }
  }
}

/* --------------------------- export --------------------------- */
async function downloadExport() {
  const hint = $("#export-hint");
  const projectId = state.projectId;
  if (!projectId) { hint.textContent = "No project to export yet."; return; }
  const exportType = $("#export-type").value;
  hint.textContent = "Building export…";
  try {
    const res = await fetch("/api/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project_id: projectId, export_type: exportType }),
    });
    if (!res.ok) {
      let msg = `Export failed (${res.status})`;
      try { const j = await res.json(); msg = j.error || j.detail || msg; } catch (e) {}
      hint.textContent = msg;
      return;
    }
    const blob = await res.blob();
    const disp = res.headers.get("Content-Disposition") || "";
    const match = disp.match(/filename="?([^"]+)"?/);
    const filename = match ? match[1] : "simple-comfort-export.zip";
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 4000);
    hint.textContent = "Downloaded " + filename;
  } catch (err) {
    hint.textContent = "Export failed: " + err.message;
  }
}

/* --------------------------- custom workflows --------------------------- */
const WF_GROUP_ORDER = ["prompts", "generation", "images", "models", "advanced"];
const WF_GROUP_LABELS = {
  prompts: "Prompts",
  generation: "Generation",
  images: "Images",
  models: "Models & Loaders",
  advanced: "Advanced",
};

function wireWorkflowPanel() {
  const drop = $("#wf-drop");
  const input = $("#wf-input");
  if (!drop || !input) return;

  drop.addEventListener("click", (e) => {
    if (e.target !== input) input.click();
  });
  ["dragover", "dragenter"].forEach((ev) =>
    drop.addEventListener(ev, (e) => {
      e.preventDefault();
      drop.classList.add("drag");
    })
  );
  ["dragleave", "drop"].forEach((ev) =>
    drop.addEventListener(ev, (e) => {
      e.preventDefault();
      drop.classList.remove("drag");
    })
  );
  drop.addEventListener("drop", (e) => uploadWorkflows(e.dataTransfer.files));
  input.addEventListener("change", () => { uploadWorkflows(input.files); input.value = ""; });

  $("#wf-delete").addEventListener("click", deleteWorkflow);
  $("#wf-generate").addEventListener("click", submitCustomWorkflow);
}

async function loadCustomWorkflows() {
  try {
    const res = await fetch("/api/custom-workflows");
    const data = await res.json();
    (data.workflows || []).forEach((wf) => { state.customWorkflows[wf.slug] = wf; });
    renderWorkflowList();
  } catch (err) {
    console.warn("custom workflows load failed", err);
  }
}

async function uploadWorkflows(files) {
  const hint = $("#wf-hint");
  let lastSlug = null;
  for (const file of Array.from(files || [])) {
    if (!/\.json$/i.test(file.name)) {
      hint.textContent = "Only .json workflow files are supported.";
      continue;
    }
    hint.textContent = "Importing " + file.name + "…";
    const fd = new FormData();
    fd.append("file", file);
    try {
      const res = await fetch("/api/custom-workflows", { method: "POST", body: fd });
      const data = await res.json();
      if (!data.success) {
        hint.textContent = data.error || "Import failed.";
        continue;
      }
      state.customWorkflows[data.data.slug] = data.data;
      hint.textContent = "Imported " + data.data.name;
      lastSlug = data.data.slug;
    } catch (err) {
      hint.textContent = "Import failed: " + err.message;
    }
  }
  renderWorkflowList();
  if (lastSlug) selectWorkflow(lastSlug);
}

function renderWorkflowList() {
  const list = $("#wf-list");
  if (!list) return;
  const records = Object.values(state.customWorkflows);
  if (!records.length) {
    list.innerHTML = '<div class="wf-empty">No workflows imported yet.</div>';
    return;
  }
  list.innerHTML = records
    .map((wf) => {
      const active = wf.slug === state.activeWorkflow ? " active" : "";
      const outs = (wf.schema.output_nodes || []).join(", ") || "no save node";
      return `<button type="button" class="wf-card${active}" data-slug="${escapeHtml(wf.slug)}">
        <span class="wf-card-name">${escapeHtml(wf.name)}</span>
        <span class="wf-card-meta">${wf.schema.node_count || "?"} nodes · ${escapeHtml(wf.format)} · ${escapeHtml(outs)}</span>
      </button>`;
    })
    .join("");
  $$("#wf-list .wf-card").forEach((card) =>
    card.addEventListener("click", () => selectWorkflow(card.dataset.slug))
  );
}

function selectWorkflow(slug) {
  const rec = state.customWorkflows[slug];
  if (!rec) return;
  state.activeWorkflow = slug;
  renderWorkflowList();
  $("#wf-detail").classList.remove("hidden");
  $("#wf-name").textContent = rec.name;
  $("#wf-meta").textContent =
    `${rec.schema.node_count} nodes · ${rec.format} · ` +
    ((rec.schema.output_nodes || []).join(", ") || "no save node");
  renderWorkflowForm(rec);
  $("#wf-detail").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderWorkflowForm(rec) {
  const form = $("#wf-form");
  const fields = (rec.schema && rec.schema.fields) || [];
  if (!fields.length) {
    form.innerHTML = '<div class="wf-empty">No editable inputs found — the workflow will run as-is.</div>';
    return;
  }
  const byGroup = {};
  fields.forEach((f) => {
    (byGroup[f.group] = byGroup[f.group] || []).push(f);
  });
  let html = "";
  WF_GROUP_ORDER.forEach((group) => {
    const groupFields = byGroup[group];
    if (!groupFields) return;
    const collapsed = group === "advanced" || group === "models";
    html += `<details class="wf-group"${collapsed ? "" : " open"}>` +
      `<summary>${WF_GROUP_LABELS[group] || group} <span class="summary-caret">▾</span></summary>`;
    groupFields.forEach((f) => { html += workflowFieldHtml(f); });
    html += "</details>";
  });
  // Any custom group not in WF_GROUP_ORDER.
  Object.keys(byGroup).forEach((group) => {
    if (WF_GROUP_ORDER.includes(group)) return;
    html += `<details class="wf-group"><summary>${escapeHtml(group)} <span class="summary-caret">▾</span></summary>`;
    byGroup[group].forEach((f) => { html += workflowFieldHtml(f); });
    html += "</details>";
  });
  form.innerHTML = html;

  // Seed dice buttons.
  $$("#wf-form .wf-dice").forEach((btn) =>
    btn.addEventListener("click", () => {
      const input = document.getElementById("wf-f-" + btn.dataset.field);
      if (input) input.value = Math.floor(Math.random() * 4294967295);
    })
  );
}

function workflowFieldHtml(f) {
  const id = "wf-f-" + f.id;
  const label = `<label class="wf-label" for="${id}">${escapeHtml(f.label)}</label>`;
  let control = "";
  if (f.kind === "text") {
    control = f.multiline
      ? `<textarea id="${id}" class="textarea wf-input" rows="3" data-field="${escapeHtml(f.id)}">${escapeHtml(f.default)}</textarea>`
      : `<input type="text" id="${id}" class="input wf-input" data-field="${escapeHtml(f.id)}" value="${escapeHtml(f.default)}" />`;
  } else if (f.kind === "int" || f.kind === "float" || f.kind === "seed") {
    control = `<div class="wf-seed-row">` +
      `<input type="number" id="${id}" class="input wf-input" data-field="${escapeHtml(f.id)}" value="${escapeHtml(f.default)}"` +
      (f.min != null ? ` min="${f.min}"` : "") +
      (f.max != null ? ` max="${f.max}"` : "") +
      (f.step != null ? ` step="${f.step}"` : f.kind === "float" ? ' step="0.1"' : "") +
      ` />` +
      (f.kind === "seed" ? `<button type="button" class="btn-secondary wf-dice" data-field="${escapeHtml(f.id)}" title="Random seed">🎲</button>` : "") +
      `</div>`;
  } else if (f.kind === "bool") {
    control = `<label class="switch"><input type="checkbox" id="${id}" data-field="${escapeHtml(f.id)}" ${f.default ? "checked" : ""} /><span class="slider"></span></label>`;
  } else if (f.kind === "combo") {
    const opts = f.options || [];
    const known = opts.some((o) => String(o) === String(f.default));
    control = `<select id="${id}" class="input wf-input" data-field="${escapeHtml(f.id)}">` +
      (known ? "" : `<option value="${escapeHtml(f.default)}" selected>${escapeHtml(f.default)}</option>`) +
      opts.map((o) =>
        `<option value="${escapeHtml(o)}"${String(o) === String(f.default) ? " selected" : ""}>${escapeHtml(o)}</option>`
      ).join("") +
      `</select>`;
  } else if (f.kind === "image") {
    control = `<input type="file" id="${id}" class="input wf-input wf-file" accept="image/*" data-field="${escapeHtml(f.id)}" />` +
      `<div class="wf-file-name" data-name="${escapeHtml(f.id)}">${escapeHtml(f.default || "")}</div>`;
  }
  return `<div class="wf-field">${label}${control}</div>`;
}

async function submitCustomWorkflow() {
  const slug = state.activeWorkflow;
  const rec = slug && state.customWorkflows[slug];
  const btn = $("#wf-generate");
  if (!rec) return;

  const values = {};
  const files = {};
  $$("#wf-form [data-field]").forEach((el) => {
    const key = el.dataset.field;
    if (el.type === "checkbox") values[key] = el.checked;
    else if (el.type === "file") {
      if (el.files && el.files[0]) files[key] = el.files[0];
    } else values[key] = el.value;
  });

  // Read chosen images as data URLs for server-side staging.
  for (const [key, file] of Object.entries(files)) {
    values[key] = await new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => resolve("");
      reader.readAsDataURL(file);
    });
  }

  btn.disabled = true;
  btn.classList.add("generating");
  btn.textContent = "STARTING…";

  try {
    const res = await fetch(`/api/custom-workflows/${encodeURIComponent(slug)}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        values,
        worker_index: null,
        project_id: state.projectId || null,
      }),
    });
    const data = await res.json();
    if (!res.ok || data.error) {
      $("#wf-hint").textContent = data.error || "Generation failed to start.";
      resetGenerateBtn();
      return;
    }
    state.currentJob = data.job_id;
    state.projectId = data.project_id;
    closePanels();
    showProgress();
    openWebSocket(data.job_id);
  } catch (err) {
    $("#wf-hint").textContent = "Network error: " + err.message;
    resetGenerateBtn();
  }
}

async function deleteWorkflow() {
  const slug = state.activeWorkflow;
  if (!slug || !state.customWorkflows[slug]) return;
  if (!confirm(`Remove workflow "${state.customWorkflows[slug].name}"?`)) return;
  try {
    await fetch(`/api/custom-workflows/${encodeURIComponent(slug)}`, { method: "DELETE" });
  } catch (err) { /* ignore */ }
  delete state.customWorkflows[slug];
  state.activeWorkflow = null;
  $("#wf-detail").classList.add("hidden");
  renderWorkflowList();
}

/* --------------------------- utils --------------------------- */
function escapeHtml(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

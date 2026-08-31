/* Simple-Comfort front-end — vanilla JS, no build step. */
"use strict";

const state = {
  mode: "t2v",
  model: "ltx23",
  continuity: "new",
  aspect_ratio: "16:9",
  quality: "balanced",
  models: {},          // key -> status object
  currentJob: null,
  ws: null,
  refFile: null,
  continueFrom: null,  // clip_id to continue
  lastClip: null,
  projectId: null,
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

/* --------------------------- init --------------------------- */
document.addEventListener("DOMContentLoaded", () => {
  wirePillGroups();
  wireDropzone();
  wireAdvanced();
  wireButtons();
  refreshStatus();
  refreshModels();
  refreshGallery();
  setInterval(refreshStatus, 15000);

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
    dot.className = "dot model-dot " + dotClass(m);
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

  // Show motion field only for models that support it (LTX/Wan).
  $("#motion-field").classList.toggle("hidden", state.model === "framepack" ? false : false);
  // Segment length only meaningful for multi-section models.
  const segField = $("#segment").closest(".adv-field");
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
        ? `ComfyUI connected (${data.workers.filter(w => w.alive).length} workers)`
        : "ComfyUI connected";
    } else {
      dot.className = "dot dot-red";
      label.textContent = "ComfyUI disconnected";
    }
    if (!data.ffmpeg) {
      label.textContent += " · FFmpeg missing";
    }
  } catch (err) {
    $("#conn-dot").className = "dot dot-red";
    $("#conn-label").textContent = "Server error";
  }
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
}

/* --------------------------- advanced --------------------------- */
function wireAdvanced() {
  const bind = (id, valId) => {
    const el = $(id), val = $(valId);
    el.addEventListener("input", () => { val.textContent = el.value; });
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
  if (md.model_key) { selectPill("model-group", md.model_key); state.model = md.model_key; }
  if (md.aspect_ratio) { selectPill("aspect-group", md.aspect_ratio); state.aspect_ratio = md.aspect_ratio; }
  if (md.quality) { selectPill("quality-group", md.quality); state.quality = md.quality; }
  if (typeof md.lip_sync === "boolean") $("#lipsync").checked = md.lip_sync;
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
  btn.textContent = "GENERATE 30s CLIP";
}

/* --------------------------- progress / websocket --------------------------- */
function showProgress() {
  $("#progress-card").classList.remove("hidden");
  $("#output-card").classList.add("hidden");
  $("#progress-fill").style.width = "0%";
  $("#progress-section").textContent = "Starting…";
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
  $("#progress-section").textContent =
    `Section ${data.current_section}/${data.total_sections} — ${pct}%`;
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
  const md = clip.metadata || {};
  $("#output-meta").textContent =
    `Model: ${md.model}\nMode: ${md.mode}\nSeed: ${md.seed ?? "random"}\n` +
    `Resolution: ${md.resolution} @ ${md.fps}fps\nQuality: ${md.quality}\n` +
    `Sections: ${md.num_sections}\nLip-Sync: ${md.lip_sync ? "on" : "off"}`;
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
    (data.clips || []).forEach((clip) => {
      const item = document.createElement("div");
      item.className = "gallery-item";
      const md = clip.metadata || {};
      item.innerHTML = `
        <img src="${clip.thumbnail_url || clip.first_frame_url || ""}" alt="clip" />
        <div class="g-meta">${md.model || ""} · ${md.quality || ""}</div>
        <button class="g-continue">Continue</button>`;
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
  const md = clip.metadata || {};
  $("#output-meta").textContent =
    `Model: ${md.model}\nSeed: ${md.seed ?? "random"}\nResolution: ${md.resolution} @ ${md.fps}fps`;
  card.scrollIntoView({ behavior: "smooth" });
}

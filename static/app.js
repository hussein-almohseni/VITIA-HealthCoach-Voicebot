"use strict";

// ---------------------------------------------------------------------------
// Strings for the patient-facing side
// ---------------------------------------------------------------------------
const UI = {
  en: {
    placeholder: "Type a message",
    tap: "Tap to talk",
    listening: "Listening… tap to finish",
    yourTurn: "Your turn, I'm listening",
    thinking: "Working…",
    speaking: "Speaking… tap to interrupt",
    paused: "Didn't hear anything. Tap to talk",
    handsFree: "Hands-free",
    online: "Online",
    offline: "Offline: saving on this device",
    empty: ["Try saying", "“My blood pressure was 120 over 80 this morning”"],
    suggestions: [
      "My blood pressure was 120 over 80 this morning",
      "Book an appointment next Tuesday afternoon",
      "What does my plan cover for physiotherapy?",
      "Show me a graph of my blood pressure this month",
      "Do I have high blood pressure?",
      "When should I take my lisinopril?",
    ],
  },
  es: {
    placeholder: "Escribe un mensaje",
    tap: "Toca para hablar",
    listening: "Escuchando… toca para terminar",
    yourTurn: "Te toca, te escucho",
    thinking: "Procesando…",
    speaking: "Hablando… toca para interrumpir",
    paused: "No escuché nada. Toca para hablar",
    handsFree: "Manos libres",
    online: "En línea",
    offline: "Sin conexión: guardando en este dispositivo",
    empty: ["Prueba a decir", "“Mi presión fue 120 sobre 80 esta mañana”"],
    suggestions: [
      "Mi presión fue 120 sobre 80 esta mañana",
      "Quiero una cita el martes por la tarde",
      "¿Qué cubre mi plan de fisioterapia?",
      "Muéstrame una gráfica de mi peso este mes",
      "¿Tengo la presión alta?",
      "¿Cuándo debo tomar el lisinopril?",
    ],
  },
};

const STAGES = [
  { key: "pause", label: "Pause detection", color: "var(--s-pause)" },
  { key: "asr", label: "Final transcript", color: "var(--s-asr)" },
  { key: "understand", label: "Understand", color: "var(--s-understand)" },
  { key: "generate", label: "Generate answer", color: "var(--s-generate)" },
  { key: "voice", label: "Synthesize voice", color: "var(--s-voice)" },
];
const TARGET_MS = 2000;
// Hands-free: how long to keep the mic open with no speech before giving up.
const NO_SPEECH_TIMEOUT_MS = 8000;
// Pause between the end of the reply audio and reopening the mic.
const REOPEN_DELAY_MS = 250;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const state = {
  lang: "en",
  ws: null,
  recording: false,
  busy: false,
  audioCtx: null,
  micNode: null,
  micStream: null,
  playHead: 0,
  sources: [],
  turn: null,        // { botEl, userEl, inputEndAt, firstAudioPlayed, turnId, clientMs }
  turns: [],
  settings: {},
  handsFree: loadPref("handsFree", true),
  conversationActive: false, // set by a voice turn; cleared by typing, timeout, or tapping to stop
  awaitingReply: false,       // between sending a turn and turn_done
  pendingDecodes: 0,
  heardSpeech: false,
  listenTimer: null,
  reopenTimer: null,
  lastHint: null,
};

function loadPref(key, fallback) {
  try {
    const v = localStorage.getItem(key);
    return v === null ? fallback : v === "true";
  } catch (_) {
    return fallback;
  }
}

function savePref(key, value) {
  try { localStorage.setItem(key, String(value)); } catch (_) { /* storage unavailable */ }
}

const $ = (sel) => document.querySelector(sel);
const chat = $("#chat");

// ---------------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------------
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => ws.send(JSON.stringify({ type: "hello", lang: state.lang }));
  ws.onmessage = (ev) => onEvent(JSON.parse(ev.data));
  ws.onclose = () => {
    setRecording(false);
    toast("Connection to the local server closed. Reconnecting…", true);
    setTimeout(connect, 1500);
  };
  state.ws = ws;
}

function send(obj) {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) state.ws.send(JSON.stringify(obj));
}

function onEvent(e) {
  switch (e.type) {
    case "partial":
      ensureTurn();
      setUserText(e.text, true);
      state.heardSpeech = true;
      clearTimeout(state.listenTimer);
      break;
    case "transcript":
      ensureTurn();
      setUserText(e.text || "(nothing heard)", false);
      setBusy(true);
      break;
    case "mic_off":
      if (state.turn && !state.turn.inputEndAt) state.turn.inputEndAt = performance.now();
      setRecording(false);
      if (e.reason !== "error") {
        state.awaitingReply = true;
        setBusy(true);
      }
      break;
    case "reply_text":
      appendBot(e.text);
      break;
    case "audio":
      playWav(e.wav);
      break;
    case "chart":
      addChart(e);
      break;
    case "options":
      ensureTurn().options = e.options;  // shown under the reply when the turn ends
      break;
    case "turn_done":
      onTurnDone(e);
      break;
    case "records_changed":
      refreshRecords();
      refreshQueue();
      break;
    case "sync":
      if (e.message) toast(e.message, e.status === "conflict");
      refreshRecords();
      refreshQueue();
      break;
    case "status":
      refreshState();
      break;
    case "error":
      toast(e.message, true);
      state.awaitingReply = false;
      endConversation();
      setBusy(false);
      break;
  }
}

// ---------------------------------------------------------------------------
// Chat rendering
// ---------------------------------------------------------------------------
function clearEmpty() {
  const empty = chat.querySelector(".empty-chat");
  if (empty) empty.remove();
}

function renderEmpty() {
  const [a, b] = UI[state.lang].empty;
  chat.innerHTML = `<div class="empty-chat"><strong>${a}</strong>${b}</div>`;
}

function ensureTurn() {
  if (!state.turn) state.turn = { botEl: null, userEl: null, inputEndAt: null, firstAudioPlayed: null, turnId: null, clientMs: null };
  return state.turn;
}

function setUserText(text, partial) {
  clearEmpty();
  const t = ensureTurn();
  if (!t.userEl) {
    t.userEl = document.createElement("div");
    chat.appendChild(t.userEl);
  }
  t.userEl.className = "msg user" + (partial ? " partial" : "");
  t.userEl.textContent = text;
  scrollChat();
}

function appendBot(text) {
  clearEmpty();
  const t = ensureTurn();
  if (!t.botEl) {
    t.botEl = document.createElement("div");
    t.botEl.className = "msg bot";
    chat.appendChild(t.botEl);
  }
  t.botEl.textContent = t.botEl.textContent ? `${t.botEl.textContent} ${text}` : text;
  scrollChat();
}

function addMeta(text, cls) {
  const el = document.createElement("div");
  el.className = "msg-meta " + (cls || "");
  el.textContent = text;
  chat.appendChild(el);
  scrollChat();
}

function scrollChat() {
  chat.scrollTop = chat.scrollHeight;
}

function addOptions(options) {
  const wrap = document.createElement("div");
  wrap.className = "option-chips";
  options.forEach((o, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "option-chip";
    const d = new Date(o.start);
    b.textContent = d.toLocaleString(state.lang === "es" ? "es-MX" : "en-US",
      { weekday: "short", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
    b.addEventListener("click", () => sendText(state.lang === "es" ? `opción ${i + 1}` : `option ${i + 1}`));
    wrap.appendChild(b);
  });
  chat.appendChild(wrap);
  scrollChat();
}

// ---------------------------------------------------------------------------
// Charts (SVG, drawn from the patient's stored data only)
// ---------------------------------------------------------------------------
function addChart(e) {
  clearEmpty();
  const card = document.createElement("div");
  card.className = "chart-card";
  const colors = ["var(--accent)", "var(--llm)"];
  const W = 340, H = 170, P = { l: 34, r: 10, t: 10, b: 22 };
  const all = e.series.flatMap((s) => s.points);
  if (!all.length) {
    card.innerHTML = `<div class="t">${esc(e.title)}</div><div class="muted small">No readings in this period.</div>`;
    chat.appendChild(card);
    return;
  }
  const xs = all.map((p) => new Date(p[0]).getTime());
  const ys = all.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs) || x0 + 1;
  let y0 = Math.min(...ys), y1 = Math.max(...ys);
  const pad = Math.max(2, (y1 - y0) * 0.15);
  y0 = Math.floor(y0 - pad); y1 = Math.ceil(y1 + pad);
  const sx = (x) => P.l + ((x - x0) / Math.max(1, x1 - x0)) * (W - P.l - P.r);
  const sy = (y) => H - P.b - ((y - y0) / Math.max(1, y1 - y0)) * (H - P.t - P.b);
  const fmtD = (t) => new Date(t).toLocaleDateString(state.lang === "es" ? "es-MX" : "en-US", { month: "short", day: "numeric" });
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(e.title)}">`;
  for (let i = 0; i <= 3; i++) {
    const yv = y0 + ((y1 - y0) * i) / 3;
    const y = sy(yv);
    svg += `<line x1="${P.l}" x2="${W - P.r}" y1="${y}" y2="${y}" stroke="var(--line)" stroke-width="1"/>`;
    svg += `<text x="${P.l - 6}" y="${y + 3.5}" text-anchor="end" font-size="10" fill="var(--muted)">${Math.round(yv)}</text>`;
  }
  svg += `<text x="${P.l}" y="${H - 6}" font-size="10" fill="var(--muted)">${fmtD(x0)}</text>`;
  svg += `<text x="${W - P.r}" y="${H - 6}" font-size="10" fill="var(--muted)" text-anchor="end">${fmtD(x1)}</text>`;
  e.series.forEach((s, i) => {
    if (!s.points.length) return;
    const pts = s.points.map((p) => `${sx(new Date(p[0]).getTime()).toFixed(1)},${sy(p[1]).toFixed(1)}`);
    svg += `<polyline fill="none" stroke="${colors[i % 2]}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" points="${pts.join(" ")}"/>`;
    s.points.forEach((p) => {
      svg += `<circle cx="${sx(new Date(p[0]).getTime()).toFixed(1)}" cy="${sy(p[1]).toFixed(1)}" r="2.6" fill="${colors[i % 2]}"><title>${fmtD(p[0])}: ${p[1]} ${esc(e.unit)}</title></circle>`;
    });
  });
  svg += `</svg>`;
  const legend = e.series.map((s, i) => `<span><i style="background:${colors[i % 2]}"></i>${esc(s.name)} (${esc(e.unit)})</span>`).join("");
  card.innerHTML = `<div class="t">${esc(e.title)}</div>${svg}<div class="chart-legend">${legend}</div>`;
  chat.appendChild(card);
  scrollChat();
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ---------------------------------------------------------------------------
// Audio out: play sentences back to back as they arrive
// ---------------------------------------------------------------------------
function audioContext() {
  if (!state.audioCtx) state.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  if (state.audioCtx.state === "suspended") state.audioCtx.resume();
  return state.audioCtx;
}

async function playWav(b64) {
  const t = state.turn;  // the turn this audio belongs to, even if it ends while decoding
  const ctx = audioContext();
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  let buffer;
  state.pendingDecodes += 1;
  state.replyStarted = true;
  try {
    buffer = await ctx.decodeAudioData(bytes.buffer);
  } catch (err) {
    console.warn("Audio decode failed", err);
    return;
  } finally {
    state.pendingDecodes -= 1;
  }
  if (state.recording) return; // the patient interrupted while this was decoding
  const src = ctx.createBufferSource();
  src.buffer = buffer;
  src.connect(ctx.destination);
  const startAt = Math.max(ctx.currentTime + 0.01, state.playHead);
  src.start(startAt);
  state.playHead = startAt + buffer.duration;
  state.sources.push(src);
  updateMicHint();
  src.onended = () => {
    state.sources = state.sources.filter((s) => s !== src);
    updateMicHint();
    maybeAutoListen();
  };
  if (t && t.firstAudioPlayed == null) {
    const delayMs = (startAt - ctx.currentTime) * 1000;
    t.firstAudioPlayed = performance.now() + delayMs;
    if (t.inputEndAt) {
      t.clientMs = Math.round(t.firstAudioPlayed - t.inputEndAt);
      if (t.turnId) send({ type: "client_timing", turn_id: t.turnId, ms: t.clientMs });
    }
  }
}

function stopPlayback() {
  const playing = state.sources;
  state.sources = [];
  playing.forEach((s) => {
    s.onended = null;
    try { s.stop(); } catch (_) { /* already stopped */ }
  });
  state.playHead = 0;
}

// A short two-note tone that tells the patient the mic is open.
function playListenTone() {
  const ctx = audioContext();
  const t0 = ctx.currentTime + 0.01;
  [[660, 0], [880, 0.09]].forEach(([freq, offset]) => {
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.frequency.value = freq;
    osc.type = "sine";
    gain.gain.setValueAtTime(0.0001, t0 + offset);
    gain.gain.exponentialRampToValueAtTime(0.12, t0 + offset + 0.015);
    gain.gain.exponentialRampToValueAtTime(0.0001, t0 + offset + 0.08);
    osc.connect(gain).connect(ctx.destination);
    osc.start(t0 + offset);
    osc.stop(t0 + offset + 0.09);
  });
  return 200; // ms until the tone has finished
}

// ---------------------------------------------------------------------------
// Hands-free loop: reopen the mic once the reply has finished playing
// ---------------------------------------------------------------------------
function replyFinished() {
  return !state.awaitingReply && state.sources.length === 0 && state.pendingDecodes === 0;
}

function maybeAutoListen() {
  clearTimeout(state.reopenTimer);
  if (!state.handsFree || !state.conversationActive || state.recording || !replyFinished()) return;
  state.reopenTimer = setTimeout(() => {
    if (!state.handsFree || !state.conversationActive || state.recording || !replyFinished()) return;
    startListening({ auto: true });
  }, REOPEN_DELAY_MS);
}

function endConversation(reason) {
  state.conversationActive = false;
  clearTimeout(state.reopenTimer);
  clearTimeout(state.listenTimer);
  state.lastHint = reason || null;
  updateMicHint();
}

// ---------------------------------------------------------------------------
// Microphone in: 16 kHz float32 chunks over the socket
// ---------------------------------------------------------------------------
async function ensureMic() {
  if (state.micNode) return;
  const ctx = audioContext();
  state.micStream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });
  await ctx.audioWorklet.addModule("/static/pcm-worklet.js");
  const source = ctx.createMediaStreamSource(state.micStream);
  const node = new AudioWorkletNode(ctx, "pcm-capture");
  node.port.onmessage = (ev) => {
    const chunk = ev.data;
    if (state.recording && state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(chunk.buffer);
      let sum = 0;
      for (let i = 0; i < chunk.length; i++) sum += chunk[i] * chunk[i];
      const level = Math.min(1, Math.sqrt(sum / chunk.length) * 6);
      $("#meter-fill").style.width = `${Math.round(level * 100)}%`;
    }
  };
  source.connect(node);
  state.micNode = node;
}

async function toggleMic() {
  if (state.recording) {
    clearTimeout(state.listenTimer);
    if (!state.heardSpeech) {
      // Tapped while listening and nothing was said: close the mic, end hands-free.
      send({ type: "cancel" });
      setRecording(false);
      endConversation();
      if (state.turn && !state.turn.userEl) state.turn = null;
      return;
    }
    if (state.turn) state.turn.inputEndAt = performance.now();
    send({ type: "stop" });
    setRecording(false);
    state.awaitingReply = true;
    setBusy(true);
    return;
  }
  // Tapping while the assistant is speaking interrupts it.
  state.conversationActive = true;
  await startListening({ auto: false });
}

async function startListening({ auto }) {
  try {
    await ensureMic();
  } catch (err) {
    toast("Microphone not available: " + err.message + ". You can type instead.", true);
    endConversation();
    return;
  }
  clearTimeout(state.reopenTimer);
  stopPlayback();
  state.turn = null;
  ensureTurn();
  state.heardSpeech = false;
  state.replyStarted = false;
  state.lastHint = null;
  if (auto) await new Promise((r) => setTimeout(r, playListenTone()));
  if (state.recording) return;
  send({ type: "start" });
  setRecording(true, auto);
  clearTimeout(state.listenTimer);
  state.listenTimer = setTimeout(onNoSpeech, NO_SPEECH_TIMEOUT_MS);
}

function onNoSpeech() {
  if (!state.recording || state.heardSpeech) return;
  send({ type: "cancel" });
  setRecording(false);
  if (state.turn && !state.turn.userEl) state.turn = null;
  endConversation("paused");
}

function setRecording(on, auto = false) {
  state.recording = on;
  state.autoListening = on && auto;
  const mic = $("#mic");
  mic.classList.toggle("recording", on);
  if (!on) $("#meter-fill").style.width = "0%";
  updateMicHint();
}

function setBusy(on) {
  state.busy = on;
  $("#mic").classList.toggle("busy", on && !state.recording);
  updateMicHint();
}

function updateMicHint() {
  const s = UI[state.lang];
  const speaking = state.sources.length > 0 || state.pendingDecodes > 0 || (state.busy && state.replyStarted);
  const aboutToListen = state.handsFree && state.conversationActive && !state.busy;
  let hint = s.tap;
  if (state.recording) hint = state.autoListening ? s.yourTurn : s.listening;
  else if (speaking) hint = s.speaking;
  else if (state.busy) hint = s.thinking;
  else if (aboutToListen) hint = s.yourTurn;
  else if (state.lastHint === "paused") hint = s.paused;
  const el = $("#mic-hint");
  if (el.textContent !== hint) el.textContent = hint;
  $("#mic").classList.toggle("speaking", !state.recording && speaking);
}

// ---------------------------------------------------------------------------
// Typed turns
// ---------------------------------------------------------------------------
function sendText(text) {
  text = text.trim();
  if (!text) return;
  if (state.recording) {
    send({ type: "cancel" });
    setRecording(false);
  }
  endConversation();  // typing pauses hands-free; tap the mic to resume
  stopPlayback();
  state.turn = null;
  const t = ensureTurn();
  setUserText(text, false);
  t.inputEndAt = performance.now();
  state.awaitingReply = true;
  state.replyStarted = false;
  setBusy(true);
  send({ type: "text", text });
}

// ---------------------------------------------------------------------------
// Latency panel
// ---------------------------------------------------------------------------
function stagesFor(tm) {
  const pause = tm.vad_wait_ms || 0;
  const asr = tm.asr_final_ms || 0;
  const understand = tm.understand_ms || 0;
  const voice = tm.tts_first_ms || 0;
  const generate = Math.max(0, (tm.first_audio_ms || 0) - understand - voice);
  return { pause, asr, understand, generate, voice };
}

function totalFor(tm) {
  if (tm.speech_end_to_first_audio_ms != null) return tm.speech_end_to_first_audio_ms;
  const s = stagesFor(tm);
  return s.pause + s.asr + (tm.first_audio_ms || 0);
}

function routeKind(route) {
  if (/safety|guard:blocked|refuse/.test(route)) return "safety";
  if (/llm:/.test(route)) return "llm";
  return "rules";
}

function routeLabel(route) {
  const kind = routeKind(route);
  if (/llm:/.test(route)) {
    if (/retrieval:/.test(route) && /rules:plan_question/.test(route)) return "Rules + Gemma answer";
    return "Gemma";
  }
  if (/retrieval:/.test(route)) return "Rules + Gemma answer";
  if (kind === "safety") return "Safety template";
  return "Rules only";
}

function onTurnDone(e) {
  state.awaitingReply = false;
  setBusy(false);
  const t = ensureTurn();
  t.turnId = e.turn_id;
  if (t.clientMs != null) send({ type: "client_timing", turn_id: t.turnId, ms: t.clientMs });
  const total = totalFor(e.timings);
  const entry = {
    id: e.turn_id,
    said: t.userEl ? t.userEl.textContent : "",
    route: e.route,
    usedLLM: (e.timings.llm_prompt_tokens != null) || /llm:/.test(e.route),
    total,
    timings: e.timings,
    input: e.input_method,
  };
  state.turns.push(entry);
  if (t.options) addOptions(t.options);
  addMeta(`${(total / 1000).toFixed(2)} s · ${routeLabel(e.route)}`, "");
  renderLatency(entry);
  state.turn = null;
  // Two empty transcripts in a row usually means background noise: stop the loop.
  state.emptyTurns = e.route === "empty" ? (state.emptyTurns || 0) + 1 : 0;
  if (state.emptyTurns >= 2) endConversation("paused");
  maybeAutoListen();
}

function renderLatency(entry) {
  const tm = entry.timings;
  const total = entry.total;
  const hero = $("#hero-ms");
  hero.textContent = (total / 1000).toFixed(2);
  const cls = total <= TARGET_MS ? "good" : total <= TARGET_MS * 1.75 ? "warn" : "bad";
  hero.parentElement.className = "hero-value " + cls;
  const kind = entry.usedLLM ? "llm" : routeKind(entry.route);
  const r = $("#hero-route");
  r.className = "badge " + kind;
  r.textContent = routeLabel(entry.route);
  $("#hero-target").textContent = entry.input === "text"
    ? "Typed turn: measured from send. Target: under 2.0 s"
    : "Target: under 2.0 s";

  // Waterfall
  const st = stagesFor(tm);
  const sum = Object.values(st).reduce((a, b) => a + b, 0) || 1;
  const scale = Math.max(sum, TARGET_MS * 1.15);
  let html = `<div class="wf-wrap"><div class="wf-track">`;
  STAGES.forEach((s) => {
    const v = st[s.key];
    if (!v) return;
    const pct = (v / scale) * 100;
    html += `<div class="wf-seg" style="width:${pct}%;background:${s.color}" title="${s.label}: ${v} ms">${pct > 9 ? `<span>${v} ms</span>` : ""}</div>`;
  });
  html += `</div><div class="wf-target" style="left:${(TARGET_MS / scale) * 100}%"></div></div>`;
  html += `<div class="wf-axis"><span>0 s</span><span>${(scale / 1000).toFixed(1)} s</span></div>`;
  const rows = [
    ["Pause detection", tm.vad_wait_ms],
    ["Final transcript", tm.asr_final_ms],
    ["Understand (route)", tm.understand_ms],
    ["Gemma routing call", tm.llm_route_ms],
    ["Plan retrieval", tm.retrieval_ms],
    ["Gemma first sentence", tm.llm_first_sentence_ms],
    ["First sentence voice", tm.tts_first_ms],
    ["Gemma prompt tokens", tm.llm_prompt_tokens, ""],
    ["Gemma prompt time", tm.llm_prompt_ms],
    ["Gemma output tokens", tm.llm_output_tokens, ""],
    ["Gemma output time", tm.llm_output_ms],
    ["Whole reply spoken (synthesis)", tm.tts_total_ms],
    ["Audio length heard", tm.audio_s != null ? Math.round(tm.audio_s * 1000) : null],
  ].filter((x) => x[1] != null);
  html += `<div class="wf-detail">${rows.map(([k, v, u]) => `<div>${k}<b>${v}${u === "" ? "" : " ms"}</b></div>`).join("")}</div>`;
  html += `<div class="muted small" style="margin-top:8px">Path: <code>${esc(entry.route)}</code></div>`;
  $("#waterfall").innerHTML = html;

  // Stats
  const voice = state.turns;
  const med = (arr) => {
    if (!arr.length) return null;
    const s = [...arr].sort((a, b) => a - b);
    const m = Math.floor(s.length / 2);
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  };
  const fmt = (v) => (v == null ? "–" : `${(v / 1000).toFixed(2)} s`);
  $("#stat-rules").textContent = fmt(med(voice.filter((x) => !x.usedLLM).map((x) => x.total)));
  $("#stat-llm").textContent = fmt(med(voice.filter((x) => x.usedLLM).map((x) => x.total)));
  const under = voice.filter((x) => x.total <= TARGET_MS).length;
  $("#stat-under").textContent = `${under} of ${voice.length}`;

  // Table
  const tbody = $("#turn-table tbody");
  tbody.innerHTML = [...state.turns].reverse().map((x) => {
    const c = x.total <= TARGET_MS ? "ok" : x.total <= TARGET_MS * 1.75 ? "slow" : "too-slow";
    const k = x.usedLLM ? "llm" : routeKind(x.route);
    return `<tr><td>${x.id}</td><td class="said">${esc(x.said)}</td><td><span class="badge ${k}">${esc(routeLabel(x.route))}</span></td><td class="num ${c}">${(x.total / 1000).toFixed(2)} s</td></tr>`;
  }).join("");
}

function renderLegend() {
  $("#legend").innerHTML = STAGES.map((s) => `<span><i style="background:${s.color}"></i>${s.label}</span>`).join("");
}

// ---------------------------------------------------------------------------
// Records, queue, settings
// ---------------------------------------------------------------------------
const METRIC_LABEL = { blood_pressure: "Blood pressure", weight: "Weight", glucose: "Blood glucose", heart_rate: "Heart rate" };

function fmtDateTime(iso) {
  if (!iso) return "";
  return new Date(iso).toLocaleString("en-US", { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
}

function fmtReading(m) {
  if (m.metric === "blood_pressure") return `${m.systolic}/${m.diastolic} mmHg`;
  let extra = "";
  if (m.metric === "weight" && m.unit === "kg") extra = ` (${(m.value * 2.20462).toFixed(1)} lb)`;
  if (m.metric === "weight" && m.unit === "lb") extra = ` (${m.canonical_value} kg)`;
  if (m.metric === "glucose" && m.unit === "mmol/L") extra = ` (${m.canonical_value} mg/dL)`;
  return `${m.value} ${m.unit}${extra}`;
}

async function refreshRecords() {
  try {
    const [rec, sum] = await Promise.all([
      fetch("/api/records").then((r) => r.json()),
      fetch("/api/provider_summary").then((r) => r.json()),
    ]);
    $("#provider-summary").innerHTML = sum.metrics.map((m) => `
      <div class="summary-card">
        <div class="m">${esc(m.metric)} · ${m.count} readings</div>
        <div class="latest">${esc(m.latest)}</div>
        <div class="range">Latest ${fmtDateTime(m.latest_at)} · ${esc(m.latest_method.replace("_", " "))}</div>
        <div class="range">Range ${esc(m.min)} to ${esc(m.max)} · Avg ${esc(m.avg)}</div>
      </div>`).join("");
    const cap = rec.captured;
    $("#captured-table tbody").innerHTML = cap.length ? cap.map((m) => `
      <tr><td>${fmtDateTime(m.measured_at)}<div class="muted small">saved ${fmtDateTime(m.recorded_at)}</div></td>
      <td><b>${esc(METRIC_LABEL[m.metric])}</b><br>${esc(fmtReading(m))}</td>
      <td>${esc(m.input_method)} · ${esc(m.lang)}</td>
      <td class="said">${esc(m.source_text || "")}</td>
      <td><span class="status ${m.sync_status}">${m.sync_status === "pending_sync" ? "queued" : "synced"}</span></td></tr>`).join("")
      : `<tr><td colspan="5" class="empty">Nothing captured yet. Log a reading by voice.</td></tr>`;
    const ap = rec.appointments;
    const provider = { patel: "Dr. Patel", ruiz: "Ana Ruiz (coach)" };
    $("#appt-table tbody").innerHTML = ap.length ? ap.map((a) => `
      <tr><td>${fmtDateTime(a.start_at)}</td><td>${esc(provider[a.provider] || a.provider)}</td>
      <td>${fmtDateTime(a.created_at)} · ${esc(a.input_method)}</td>
      <td><span class="status ${a.status}">${a.status === "pending_sync" ? "saved locally, syncing" : a.status}</span></td></tr>`).join("")
      : `<tr><td colspan="4" class="empty">No appointments booked in this demo yet.</td></tr>`;
  } catch (err) {
    console.warn(err);
  }
}

let queueItems = [];
async function refreshQueue() {
  try {
    const q = await fetch("/api/outbox").then((r) => r.json());
    queueItems = q.items;
    renderQueue();
  } catch (err) {
    console.warn(err);
  }
}

function renderQueue() {
  const el = $("#queue");
  const items = queueItems.slice(0, 12);
  if (!items.length) {
    el.innerHTML = `<div class="muted small">Empty. Saved readings and bookings appear here on their way to the clinic.</div>`;
    return;
  }
  const now = Date.now();
  el.innerHTML = items.map((i) => {
    let eta = "";
    if (i.status === "queued") {
      const left = Math.max(0, Math.round((new Date(i.deliver_after).getTime() - now) / 1000));
      eta = !state.settings.online ? "waiting for network" : left > 0 ? `delivers in ${Math.floor(left / 60)}:${String(left % 60).padStart(2, "0")}` : "delivering…";
    } else {
      eta = `${i.status} ${fmtDateTime(i.delivered_at)}`;
    }
    return `<div class="queue-item"><span>${i.kind === "appointment" ? "Appointment" : "Reading"} #${i.ref_id} <span class="status ${i.status}">${i.status}</span></span><span class="eta">${eta}</span></div>`;
  }).join("");
}

async function refreshState() {
  try {
    const s = await fetch("/api/state").then((r) => r.json());
    state.settings = s.settings;
    const st = s.status;
    const chip = (label, value) => {
      const bad = /unavailable/.test(value);
      const wait = /loading/.test(value);
      return `<span class="chip ${bad ? "bad" : wait ? "wait" : ""}" title="${esc(value)}">${label}</span>`;
    };
    $("#status-chips").innerHTML = chip("Speech in", st.asr) + chip("Gemma", st.llm) + chip("Voice", st.tts);
    const banner = $("#banner");
    if (st.errors && st.errors.length) {
      banner.innerHTML = st.errors.map((e) => `<div>${esc(e)}</div>`).join("");
      banner.hidden = false;
    } else {
      banner.hidden = true;
    }
    $("#online").checked = s.settings.online;
    $("#conflict").checked = s.settings.simulate_conflict;
    $("#auto-stop").checked = s.settings.auto_stop;
    if (document.activeElement !== $("#bus-delay")) $("#bus-delay").value = s.settings.bus_delay_s;
    const net = $("#net-label");
    net.textContent = s.settings.online ? UI[state.lang].online : UI[state.lang].offline;
    net.classList.toggle("off", !s.settings.online);
    const p = s.plans[state.lang] || {};
    $("#plan-meta").textContent = p.version
      ? `Version ${p.version} · last synced ${fmtDateTime(p.synced_at)} · answers come only from this copy` : "–";
    renderQueue();
  } catch (err) {
    console.warn(err);
  }
}

async function postSettings(body) {
  await fetch("/api/settings", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  refreshState();
}

async function loadPlanText() {
  const p = await fetch(`/api/plan/${state.lang}`).then((r) => r.json());
  $("#plan-text").textContent = p.content;
}

// ---------------------------------------------------------------------------
// Language and wiring
// ---------------------------------------------------------------------------
function setLang(lang) {
  state.lang = lang;
  document.documentElement.lang = lang;
  document.querySelectorAll(".seg-btn").forEach((b) => b.classList.toggle("active", b.dataset.lang === lang));
  $("#text-input").placeholder = UI[lang].placeholder;
  $("#hands-free-label").textContent = UI[lang].handsFree;
  $("#suggestions").innerHTML = UI[lang].suggestions.map((s) => `<button type="button" class="suggestion">${esc(s)}</button>`).join("");
  document.querySelectorAll(".suggestion").forEach((b) => b.addEventListener("click", () => sendText(b.textContent)));
  updateMicHint();
  send({ type: "set_lang", lang });
  if (!chat.querySelector(".msg")) renderEmpty();
  refreshState();
  loadPlanText();
}

function toast(text, bad) {
  const el = document.createElement("div");
  el.className = "toast" + (bad ? " bad" : "");
  el.textContent = text;
  $("#toasts").appendChild(el);
  setTimeout(() => el.remove(), 6000);
}

function init() {
  document.querySelectorAll(".seg-btn").forEach((b) => b.addEventListener("click", () => setLang(b.dataset.lang)));
  document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === t));
    ["latency", "records", "sync"].forEach((k) => { $(`#tab-${k}`).hidden = k !== t.dataset.tab; });
    if (t.dataset.tab === "records") refreshRecords();
    if (t.dataset.tab === "sync") { refreshQueue(); loadPlanText(); }
  }));
  $("#mic").addEventListener("click", toggleMic);
  document.addEventListener("keydown", (e) => {
    if (e.code === "Space" && document.activeElement === document.body) {
      e.preventDefault();
      toggleMic();
    }
  });
  $("#text-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const input = $("#text-input");
    sendText(input.value);
    input.value = "";
  });
  $("#hands-free").checked = state.handsFree;
  $("#hands-free").addEventListener("change", (e) => {
    state.handsFree = e.target.checked;
    savePref("handsFree", state.handsFree);
    if (!state.handsFree) clearTimeout(state.reopenTimer);
    else maybeAutoListen();
  });
  $("#reset-chat").addEventListener("click", () => {
    if (state.recording) {
      send({ type: "cancel" });
      setRecording(false);
    }
    endConversation();
    state.awaitingReply = false;
    setBusy(false);
    stopPlayback();
    send({ type: "reset" });
    state.turn = null;
    chat.innerHTML = "";
    renderEmpty();
  });
  $("#online").addEventListener("change", (e) => postSettings({ online: e.target.checked }));
  $("#conflict").addEventListener("change", (e) => postSettings({ simulate_conflict: e.target.checked }));
  $("#auto-stop").addEventListener("change", (e) => postSettings({ auto_stop: e.target.checked }));
  $("#bus-delay").addEventListener("change", (e) => postSettings({ bus_delay_s: Number(e.target.value) }));
  $("#plan-sync").addEventListener("click", async () => {
    const r = await fetch("/api/plan/sync", { method: "POST" });
    if (r.status === 409) toast("Offline: the plan will sync when the network is back.", true);
    else toast("Health plan synced.");
    refreshState();
    loadPlanText();
  });
  $("#reset-demo").addEventListener("click", async () => {
    if (!confirm("Reset demo data? This clears bookings, captured readings and the latency log.")) return;
    await fetch("/api/reset_demo", { method: "POST" });
    state.turns = [];
    $("#turn-table tbody").innerHTML = "";
    toast("Demo data reset.");
    refreshRecords();
    refreshQueue();
  });
  renderLegend();
  connect();
  setLang("en");
  refreshRecords();
  refreshQueue();
  setInterval(renderQueue, 1000);
  setInterval(refreshState, 15000);
}

init();

"""Local web server: serves the demo page, streams audio over a WebSocket.

WebSocket protocol (client -> server):
  {"type": "hello", "lang": "en"}            start a session
  {"type": "set_lang", "lang": "es"}         switch language
  {"type": "start"}                          mic opened; binary frames follow
  <binary>                                   16 kHz mono float32 audio
  {"type": "stop"}                           mic closed by the user
  {"type": "cancel"}                         drop the open utterance (no speech heard)
  {"type": "text", "text": "..."}            typed turn
  {"type": "reset"}                          clear the conversation
  {"type": "client_timing", "turn_id": n, "ms": m}   playback start, measured in the browser

Server -> client: partial, transcript, mic_off, reply_text, audio, chart, options,
turn_done, records_changed, sync, status.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from . import config, templates
from .db import Database
from .dialogue import Engine, Session, Settings
from .scheduling import provider_name

log = logging.getLogger("server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


class AppState:
    def __init__(self):
        self.db: Database | None = None
        self.engine: Engine | None = None
        self.asr = None
        self.settings = Settings(bus_delay_s=config.BUS_DELAY_S)
        self.auto_stop = True
        self.clients: set[WebSocket] = set()
        self.status: dict = {"llm": "loading", "asr": "loading", "tts": "loading", "errors": []}
        self.last_plan_sync = 0.0


state = AppState()


async def broadcast(event: dict) -> None:
    dead = []
    for ws in list(state.clients):
        try:
            await ws.send_json(event)
        except Exception:  # noqa: BLE001
            dead.append(ws)
    for ws in dead:
        state.clients.discard(ws)


async def records_changed() -> None:
    await broadcast({"type": "records_changed"})


def load_models():
    """Returns (llm, asr, tts). Missing pieces are reported in state.status."""
    if config.FAKE_MODELS:
        from .llm import FakeLLM
        from .speech import FakeASR, FakeTTS

        import os

        state.status.update(llm="fake (DEMO_FAKE=1)", asr="fake", tts="fake")
        # Optional simulated delays, only for previewing the latency panel without models.
        return (FakeLLM(delay_s=float(os.getenv("DEMO_FAKE_LLM_S", "0"))), FakeASR(),
                FakeTTS(delay_s=float(os.getenv("DEMO_FAKE_TTS_S", "0"))))

    from .llm import OllamaLLM
    from .speech import MoonshineASR, PiperTTS

    llm = OllamaLLM(config.LLM_MODEL, config.OLLAMA_HOST)
    err = llm.check()
    if err:
        state.status["errors"].append(err)
        state.status["llm"] = "unavailable"
    else:
        t = time.perf_counter()
        llm.warmup()
        state.status["llm"] = f"{config.LLM_MODEL} (loaded in {time.perf_counter() - t:.1f}s)"

    asr = None
    try:
        models = {}
        index = config.MODELS_DIR / "moonshine_paths.json"
        if index.exists():
            # Written by setup_models.py: load straight from disk, no downloader, no network.
            from moonshine_voice import ModelArch

            for lang, entry in json.loads(index.read_text(encoding="utf-8")).items():
                models[lang] = (str(config.MODELS_DIR / entry["path"]), ModelArch[entry["arch"]])
        else:
            from moonshine_voice import get_model_for_language

            arch = config.moonshine_arch(config.ASR_SIZE)
            for lang in ("en", "es"):
                path, model_arch = get_model_for_language(
                    lang, arch, cache_root=config.MODELS_DIR / "moonshine")
                models[lang] = (path, model_arch)
        asr = MoonshineASR(models)
        state.status["asr"] = f"Moonshine {config.ASR_SIZE}-streaming (en, es)"
    except Exception as e:  # noqa: BLE001
        log.exception("ASR failed to load")
        state.status["errors"].append(f"Moonshine failed to load: {e}. Run: python setup_models.py")
        state.status["asr"] = "unavailable"

    tts = None
    try:
        voices = {lang: config.MODELS_DIR / "piper" / f"{name}.onnx" for lang, name in config.PIPER_VOICES.items()}
        missing = [str(p) for p in voices.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(", ".join(missing))
        tts = PiperTTS(voices)
        tts.warmup()
        state.status["tts"] = "Piper " + ", ".join(config.PIPER_VOICES.values())
    except Exception as e:  # noqa: BLE001
        log.exception("TTS failed to load")
        state.status["errors"].append(f"Piper voices missing or failed: {e}. Run: python setup_models.py")
        state.status["tts"] = "unavailable"
        from .speech import FakeTTS
        tts = FakeTTS()
    return llm, asr, tts


async def service_bus_loop() -> None:
    """Delivers queued writes to the 'clinic' after the configured delay.

    Stands in for the client's API + service bus path (~2 minutes per write today).
    Nothing in the conversation waits for it.
    """
    while True:
        await asyncio.sleep(1)
        try:
            if not state.settings.online:
                continue
            now = datetime.now()
            for item in state.db.due_outbox(now):
                if item["kind"] == "measurement":
                    state.db.set_measurement_synced(item["ref_id"])
                    state.db.mark_delivered(item["id"], "delivered", now)
                    await broadcast({"type": "sync", "kind": "measurement", "status": "synced"})
                elif item["kind"] == "appointment":
                    payload = json.loads(item["payload"])
                    start = datetime.fromisoformat(payload["start"])
                    if state.settings.simulate_conflict:
                        state.db.set_appointment_status(item["ref_id"], "conflict", now)
                        state.db.mark_delivered(item["id"], "rejected", now)
                        msg = ("The clinic couldn't confirm your appointment on "
                               f"{templates.fmt_slot(start, 'en')}. That time was taken. Please choose another.")
                        await broadcast({"type": "sync", "kind": "appointment", "status": "conflict", "message": msg})
                    else:
                        state.db.set_appointment_status(item["ref_id"], "confirmed", now)
                        state.db.mark_delivered(item["id"], "delivered", now)
                        msg = (f"Clinic confirmed your appointment: {templates.fmt_slot(start, 'en')} with "
                               f"{provider_name(payload['provider'], 'en')}.")
                        await broadcast({"type": "sync", "kind": "appointment", "status": "confirmed",
                                         "message": msg})
                await records_changed()
            if time.monotonic() - state.last_plan_sync > 15 * 60:
                await asyncio.to_thread(sync_plan)
        except Exception:  # noqa: BLE001
            log.exception("Service bus loop error")


def sync_plan() -> dict:
    """Pulls the latest plan (here: the files in data/). Read-only for the assistant."""
    now = datetime.now()
    versions = {}
    for lang in ("en", "es"):
        content = (config.DATA_DIR / f"plan_{lang}.md").read_text(encoding="utf-8")
        versions[lang] = state.db.update_plan(lang, content, now)
        state.engine.reload_plan(lang, state.db.plan(lang)["content"])
    state.last_plan_sync = time.monotonic()
    return versions


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.db = Database(config.DB_PATH)
    state.db.seed(datetime.now(), config.DATA_DIR)
    llm, asr, tts = await asyncio.to_thread(load_models)
    state.asr = asr
    plans = {lang: state.db.plan(lang)["content"] for lang in ("en", "es")}
    state.engine = Engine(state.db, llm, tts, plans, state.settings, on_records_changed=records_changed)
    state.last_plan_sync = time.monotonic()
    task = asyncio.create_task(service_bus_loop())
    log.info("Ready: open http://%s:%s", config.HOST, config.PORT)
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(config.STATIC_DIR / "index.html")


@app.get("/api/state")
async def get_state():
    plans = {lang: {k: v for k, v in (state.db.plan(lang) or {}).items() if k != "content"}
             for lang in ("en", "es")}
    queued = [o for o in state.db.outbox() if o["status"] == "queued"]
    return {
        "status": state.status,
        "settings": {"bus_delay_s": state.settings.bus_delay_s, "online": state.settings.online,
                     "simulate_conflict": state.settings.simulate_conflict, "auto_stop": state.auto_stop},
        "plans": plans,
        "queued": len(queued),
        "model": {"llm": config.LLM_MODEL, "asr_size": config.ASR_SIZE, "voices": config.PIPER_VOICES},
    }


@app.post("/api/settings")
async def set_settings(body: dict):
    if "bus_delay_s" in body:
        state.settings.bus_delay_s = max(0.0, float(body["bus_delay_s"]))
    if "online" in body:
        state.settings.online = bool(body["online"])
        if state.settings.online:
            # Coming back online: anything overdue goes out on the next tick.
            pass
    if "simulate_conflict" in body:
        state.settings.simulate_conflict = bool(body["simulate_conflict"])
    if "auto_stop" in body:
        state.auto_stop = bool(body["auto_stop"])
    await broadcast({"type": "status"})
    return await get_state()


@app.get("/api/records")
async def records():
    ms = state.db.measurements()
    entered = [m for m in ms if m["input_method"] != "device_import"]
    return {"captured": list(reversed(entered))[:50], "appointments": state.db.appointments()}


@app.get("/api/provider_summary")
async def provider_summary():
    """Story 7.iii: a readable summary per metric for the care team."""
    since = datetime.now() - timedelta(days=30)
    out = []
    for metric in ("blood_pressure", "weight", "glucose", "heart_rate"):
        rows = state.db.measurements(metric, since)
        if not rows:
            continue
        last = rows[-1]
        if metric == "blood_pressure":
            values = [(r["systolic"], r["diastolic"]) for r in rows]
            fmt = lambda v: f"{v[0]}/{v[1]} mmHg"
            latest = fmt((last["systolic"], last["diastolic"]))
            lo = fmt(min(values))
            hi = fmt(max(values))
            avg = f"{round(sum(v[0] for v in values) / len(values))}/{round(sum(v[1] for v in values) / len(values))} mmHg"
        else:
            unit = {"weight": "lb", "glucose": "mg/dL", "heart_rate": "bpm"}[metric]
            conv = (lambda r: round(r["canonical_value"] * 2.20462, 1)) if metric == "weight" else (
                lambda r: r["canonical_value"] if r["canonical_value"] is not None else r["value"])
            vals = [conv(r) for r in rows]
            latest = f"{templates.fmt_number(conv(last))} {unit}"
            lo, hi = f"{templates.fmt_number(min(vals))} {unit}", f"{templates.fmt_number(max(vals))} {unit}"
            avg = f"{round(sum(vals) / len(vals), 1):g} {unit}"
        out.append({"metric": templates.METRIC_NAMES["en"][metric].capitalize(), "count": len(rows),
                    "latest": latest, "latest_at": last["measured_at"], "latest_method": last["input_method"],
                    "min": lo, "max": hi, "avg": avg})
    return {"period": "Last 30 days", "metrics": out}


@app.get("/api/outbox")
async def outbox():
    return {"items": state.db.outbox(), "online": state.settings.online}


@app.post("/api/plan/sync")
async def plan_sync():
    if not state.settings.online:
        return JSONResponse({"error": "offline"}, status_code=409)
    versions = await asyncio.to_thread(sync_plan)
    await broadcast({"type": "status"})
    return {"versions": versions}


@app.get("/api/plan/{lang}")
async def get_plan(lang: str):
    p = state.db.plan(lang)
    return {"content": p["content"] if p else "", "version": p["version"] if p else None}


@app.post("/api/reset_demo")
async def reset_demo():
    state.db.seed(datetime.now(), config.DATA_DIR, force=True)
    for lang in ("en", "es"):
        state.engine.reload_plan(lang, state.db.plan(lang)["content"])
    await records_changed()
    return {"ok": True}


@app.get("/api/turns.csv")
async def turns_csv():
    rows = state.db.turns()
    keys = ["speech_end_to_first_audio_ms", "vad_wait_ms", "asr_final_ms", "understand_ms",
            "llm_route_ms", "retrieval_ms", "llm_first_sentence_ms", "tts_first_ms", "first_audio_ms",
            "client_play_ms", "llm_prompt_tokens", "llm_prompt_ms", "llm_output_tokens", "llm_output_ms"]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "at", "lang", "input", "route", "transcript", "reply"] + keys)
    for r in rows:
        t = json.loads(r["timings"] or "{}")
        w.writerow([r["id"], r["at"], r["lang"], r["input_method"], r["route"], r["transcript"],
                    r["reply"]] + [t.get(k, "") for k in keys])
    return PlainTextResponse(buf.getvalue(), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=latency_log.csv"})


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    state.clients.add(ws)
    session = Session()
    asr_session = None
    audio_t0 = 0.0
    last_partial = ""

    async def emit(event: dict) -> None:
        await ws.send_json(event)

    async def run_turn(text: str, input_method: str, asr_timings: dict, speech_end_t: float | None):
        turn = await state.engine.handle(session, text, input_method, emit)
        summary = turn.summary()
        timings = {**asr_timings, **summary["timings"]}
        if turn.first_audio_t is not None and speech_end_t is not None:
            timings["speech_end_to_first_audio_ms"] = round((turn.first_audio_t - speech_end_t) * 1000)
        turn_id = state.db.log_turn(session.lang, input_method, text, summary["route"], summary["reply"],
                                    timings, datetime.now())
        await emit({"type": "turn_done", "turn_id": turn_id, "route": summary["route"], "timings": timings,
                    "input_method": input_method})

    async def finalize(reason: str):
        nonlocal asr_session
        if asr_session is None:
            return
        current, asr_session = asr_session, None
        t_stop = time.perf_counter()
        text = await asyncio.to_thread(current.finish)
        asr_final_ms = round((time.perf_counter() - t_stop) * 1000)
        speech_end_t = t_stop
        if current.line_end_audio_s is not None:
            speech_end_t = min(t_stop, audio_t0 + current.line_end_audio_s)
        timings = {"asr_final_ms": asr_final_ms,
                   "vad_wait_ms": round((t_stop - speech_end_t) * 1000),
                   "audio_s": round(current.audio_s, 2)}
        await emit({"type": "mic_off", "reason": reason})
        await emit({"type": "transcript", "text": text})
        await run_turn(text, "voice", timings, speech_end_t)

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                if asr_session is None:
                    continue
                samples = np.frombuffer(msg["bytes"], dtype=np.float32)
                if audio_t0 == 0.0:
                    audio_t0 = time.perf_counter() - len(samples) / 16000
                await asyncio.to_thread(asr_session.add, samples)
                partial = asr_session.partial
                if partial and partial != last_partial:
                    last_partial = partial
                    await emit({"type": "partial", "text": partial})
                if asr_session.completed and state.auto_stop:
                    await finalize("pause")
                continue
            data = json.loads(msg.get("text") or "{}")
            kind = data.get("type")
            if kind in ("hello", "set_lang"):
                lang = data.get("lang", "en")
                session.lang = lang if lang in ("en", "es") else "en"
                session.pending = None
                await emit({"type": "status"})
            elif kind == "start":
                if state.asr is None:
                    await emit({"type": "error", "message": "Speech recognition is not loaded. Use the text box, "
                                "or run setup_models.py and restart."})
                    await emit({"type": "mic_off", "reason": "error"})
                    continue
                asr_session = await asyncio.to_thread(state.asr.session, session.lang)
                audio_t0 = 0.0
                last_partial = ""
            elif kind == "stop":
                await finalize("manual")
            elif kind == "cancel":
                # Hands-free listening timed out with no speech: drop it, no turn.
                if asr_session is not None:
                    dropped, asr_session = asr_session, None
                    await asyncio.to_thread(dropped.finish)
            elif kind == "text":
                await run_turn(data.get("text", ""), "text", {}, time.perf_counter())
            elif kind == "reset":
                session = Session(lang=session.lang)
                asr_session = None
            elif kind == "client_timing":
                state.db.add_turn_timing(int(data["turn_id"]), "client_play_ms", int(data["ms"]))
    except WebSocketDisconnect:
        pass
    finally:
        state.clients.discard(ws)

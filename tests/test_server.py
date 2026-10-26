"""Server tests with stand-in models (DEMO_FAKE=1): WebSocket turns, REST, service bus."""

import os
import tempfile
import time

_tmp = tempfile.mkdtemp()
os.environ["DEMO_FAKE"] = "1"
os.environ["DEMO_DB"] = os.path.join(_tmp, "server_test.db")
os.environ["DEMO_BUS_DELAY"] = "1"

import numpy as np  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.server import app, state  # noqa: E402


def collect_turn(ws):
    events = []
    while True:
        e = ws.receive_json()
        events.append(e)
        if e["type"] == "turn_done":
            return events


def test_page_and_api():
    with TestClient(app) as client:
        assert "Health Coach Voice Prototype" in client.get("/").text
        assert client.get("/static/app.js").status_code == 200
        s = client.get("/api/state").json()
        assert s["status"]["llm"].startswith("fake")
        assert s["plans"]["en"]["version"] == 1
        summary = client.get("/api/provider_summary").json()
        assert {m["metric"] for m in summary["metrics"]} == {"Blood pressure", "Weight", "Blood glucose", "Heart rate"}


def test_text_turns_booking_and_service_bus():
    with TestClient(app) as client:
        client.post("/api/reset_demo")
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "hello", "lang": "en"})
            assert ws.receive_json()["type"] == "status"

            ws.send_json({"type": "text", "text": "Book an appointment next Tuesday afternoon"})
            ev = collect_turn(ws)
            kinds = [e["type"] for e in ev]
            assert "options" in kinds and "audio" in kinds
            done = ev[-1]
            assert done["route"] == "rules:appointment"
            assert "first_audio_ms" in done["timings"]

            ws.send_json({"type": "text", "text": "the first one"})
            collect_turn(ws)
            ws.send_json({"type": "text", "text": "yes"})
            ev = collect_turn(ws)
            text = " ".join(e["text"] for e in ev if e["type"] == "reply_text")
            assert text.startswith("Done.")

            appts = client.get("/api/records").json()["appointments"]
            assert appts[0]["status"] == "pending_sync"
            # Bus delay is 1 s in this test; the loop ticks every second.
            deadline = time.time() + 6
            got_sync = False
            while time.time() < deadline and not got_sync:
                e = ws.receive_json()
                got_sync = e["type"] == "sync" and e.get("kind") == "appointment"
            assert got_sync
            assert client.get("/api/records").json()["appointments"][0]["status"] == "confirmed"

        csv_text = client.get("/api/turns.csv").text
        assert "speech_end_to_first_audio_ms" in csv_text.splitlines()[0]
        assert len(csv_text.splitlines()) >= 4


def test_voice_turn_with_fake_asr_and_offline_queue():
    with TestClient(app) as client:
        client.post("/api/reset_demo")
        client.post("/api/settings", json={"online": False})
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "hello", "lang": "es"})
            ws.receive_json()
            state.asr.next_text = "Mi presión fue ciento veinte sobre ochenta esta mañana"
            ws.send_json({"type": "start"})
            ws.send_bytes(np.zeros(1600, dtype=np.float32).tobytes())
            ws.send_json({"type": "stop"})
            ev = collect_turn(ws)
            kinds = [e["type"] for e in ev]
            assert kinds.index("mic_off") < kinds.index("transcript") < kinds.index("reply_text")
            reply = " ".join(e["text"] for e in ev if e["type"] == "reply_text")
            assert "120 sobre 80" in reply
            assert ev[-1]["input_method"] == "voice"
            assert "speech_end_to_first_audio_ms" in ev[-1]["timings"]

            ws.send_json({"type": "text", "text": "sí"})
            collect_turn(ws)
            ws.send_json({"type": "client_timing", "turn_id": ev[-1]["turn_id"], "ms": 900})
        time.sleep(2.5)  # offline: nothing should be delivered
        items = client.get("/api/outbox").json()["items"]
        assert items and all(i["status"] == "queued" for i in items)
        captured = client.get("/api/records").json()["captured"]
        assert captured[0]["input_method"] == "voice" and captured[0]["lang"] == "es"
        assert client.post("/api/plan/sync").status_code == 409
        client.post("/api/settings", json={"online": True})
        time.sleep(2.5)
        items = client.get("/api/outbox").json()["items"]
        assert all(i["status"] == "delivered" for i in items)
        assert client.post("/api/plan/sync").status_code == 200

"""Scripted conversations through the engine, with stand-in models.

These check the logic around the models: routing, confirmations, guardrails, booking,
the outbox and the database. They do not measure model quality or latency; the
latency log from the real demo does that.

Run: python -m pytest -q
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

import pytest

from app import templates
from app.db import Database
from app.dialogue import Engine, Session, Settings
from app.llm import FakeLLM
from app.speech import FakeTTS

ROOT = Path(__file__).resolve().parent.parent
NOW = datetime(2026, 10, 8, 9, 30)  # Thursday


class Harness:
    def __init__(self, tmp_path, router=None, answer=None, lang="en"):
        self.db = Database(tmp_path / "t.db")
        self.db.seed(NOW, ROOT / "data")
        self.llm = FakeLLM(router=router, answer=answer)
        self.tts = FakeTTS()
        plans = {l: self.db.plan(l)["content"] for l in ("en", "es")}
        self.engine = Engine(self.db, self.llm, self.tts, plans, Settings(bus_delay_s=120),
                             clock=lambda: NOW)
        self.session = Session(lang=lang)
        self.events = []

    def say(self, text):
        self.events = []

        async def emit(e):
            self.events.append(e)

        turn = asyncio.run(self.engine.handle(self.session, text, "voice", emit))
        return turn

    def reply(self, text):
        return self.say(text).reply


def test_measurement_readback_confirm_and_save(tmp_path):
    h = Harness(tmp_path)
    r = h.reply("My blood pressure was 120 over 80 this morning")
    assert "120 over 80" in r and "Should I save it?" in r
    assert h.session.pending["kind"] == "confirm_measurement"
    assert h.llm.calls == []  # rules path, no model call
    before = len([m for m in h.db.measurements() if m["input_method"] == "voice"])
    r = h.reply("yes")
    assert r.startswith("Saved")
    saved = [m for m in h.db.measurements() if m["input_method"] == "voice"]
    assert len(saved) == before + 1
    m = saved[-1]
    assert (m["systolic"], m["diastolic"], m["unit"]) == (120, 80, "mmHg")
    assert m["measured_at"] == "2026-10-08T08:00:00"
    assert m["recorded_at"] == "2026-10-08T09:30:00"
    assert m["source_text"].startswith("My blood pressure")
    outbox = h.db.outbox()
    assert outbox[0]["kind"] == "measurement" and outbox[0]["status"] == "queued"


def test_measurement_correction_and_rejection(tmp_path):
    h = Harness(tmp_path)
    h.reply("I weigh 182")
    assert "182 pounds" in h.engine.tts.spoken[0]
    r = h.reply("no, it was 128 pounds")
    assert "128 pounds" in r
    r = h.reply("no")
    assert "won't save" in r
    assert h.session.pending is None


def test_implausible_value_asks_again(tmp_path):
    h = Harness(tmp_path)
    r = h.reply("blood pressure 300 over 20")
    assert "outside the range" in r
    assert h.session.pending is None


def test_measurement_with_interpretation_question(tmp_path):
    h = Harness(tmp_path)
    r = h.reply("My blood pressure is 150 over 95, is that bad?")
    assert "150 over 95" in r
    assert "can't tell you what the reading means" in r
    assert r.endswith("Should I save it?")


@pytest.mark.parametrize("text", [
    "Do I have diabetes?",
    "Is 150 over 95 high?",
    "Should I take an extra lisinopril?",
    "Can I stop taking metformin?",
    "What's wrong with me, my knee hurts",
    "What medicine should I take for a headache?",
    "What does my glucose reading mean?",
])
def test_diagnosis_requests_refused(tmp_path, text):
    h = Harness(tmp_path)
    r = h.reply(text)
    assert r == templates.refuse_clinical("en")
    assert h.llm.calls == []


@pytest.mark.parametrize("text", [
    "¿Tengo diabetes?",
    "¿Es normal mi presión?",
    "¿Debo tomar otra pastilla de lisinopril?",
    "¿Qué medicamento puedo tomar para el dolor?",
])
def test_diagnosis_requests_refused_spanish(tmp_path, text):
    h = Harness(tmp_path, lang="es")
    assert h.reply(text) == templates.refuse_clinical("es")


@pytest.mark.parametrize("text,lang", [
    ("Is it okay if I book Friday morning?", "en"),
    ("Can I book an appointment for tomorrow?", "en"),
    ("Can I see my weight chart?", "en"),
    ("What's my dose of lisinopril?", "en"),
    ("How do I take my blood pressure?", "en"),
    ("And how many sessions do I have left?", "en"),
    ("Do I have an appointment this week?", "en"),
    ("Does my plan cover physiotherapy appointments?", "en"),
    ("¿Puedo ver mi gráfica de peso?", "es"),
    ("¿Cuál es mi dosis de lisinopril?", "es"),
    ("¿Cuándo debo tomar la metformina?", "es"),
])
def test_everyday_phrases_not_refused(tmp_path, text, lang):
    h = Harness(tmp_path, lang=lang)
    assert h.reply(text) != templates.refuse_clinical(lang)


def test_plan_schedule_question_is_allowed(tmp_path):
    h = Harness(tmp_path)
    r = h.reply("When should I take my lisinopril?")
    assert r != templates.refuse_clinical("en")
    assert "Your plan says" in r


def test_emergency_and_crisis(tmp_path):
    h = Harness(tmp_path)
    assert h.reply("I have chest pain right now") == templates.emergency("en")
    assert h.reply("I want to kill myself") == templates.crisis("en")


def test_plan_change_refused_then_offer_appointment(tmp_path):
    h = Harness(tmp_path)
    assert h.reply("Can you change my health plan to add massage?") == templates.refuse_plan_change("en")
    r = h.reply("yes")
    assert r == templates.ask_when("en")


def test_appointment_next_tuesday_afternoon_offers_closest(tmp_path):
    h = Harness(tmp_path)
    r = h.reply("I'd like to book an appointment next Tuesday afternoon")
    # Seed data books out the first Tuesday afternoon, so alternatives are offered.
    assert "isn't available" in r and "The closest options are" in r
    assert h.session.pending["kind"] == "choose_slot"
    options = h.session.pending["options"]
    assert len(options) == 3
    r = h.reply("the second one")
    assert "Shall I book it?" in r
    r = h.reply("yes")
    assert r.startswith("Done.")
    appts = h.db.appointments()
    assert len(appts) == 1 and appts[0]["status"] == "pending_sync"
    assert appts[0]["start_at"] == options[1]["start"]
    assert h.db.slot(appts[0]["slot_id"])["booked"] == 1
    assert h.db.outbox()[0]["kind"] == "appointment"


def test_appointment_follow_up_keeps_part_of_day(tmp_path):
    h = Harness(tmp_path)
    h.reply("Can I schedule an appointment for Monday morning?")
    r = h.reply("what about Wednesday?")
    pref = h.session.last_pref
    assert pref["part"] == "morning" and pref["date"] == "2026-10-14"
    assert "Wednesday" in r


def test_appointment_without_date_asks_when(tmp_path):
    h = Harness(tmp_path)
    assert h.reply("I need an appointment") == templates.ask_when("en")
    r = h.reply("tomorrow at 10")
    assert "Friday, October 9" in r


def test_next_appointment_query(tmp_path):
    h = Harness(tmp_path)
    assert h.reply("When is my next appointment?") == templates.no_appointments("en")


def test_chart_and_follow_up(tmp_path):
    h = Harness(tmp_path)
    r = h.reply("Show me a graph of my blood pressure this month")
    charts = [e for e in h.events if e["type"] == "chart"]
    assert charts and charts[0]["metric"] == "blood_pressure"
    assert len(charts[0]["series"]) == 2
    assert "readings" in r
    h.reply("and my weight?")
    charts = [e for e in h.events if e["type"] == "chart"]
    assert charts and charts[0]["metric"] == "weight"


def test_plan_answer_streams_and_is_grounded(tmp_path):
    h = Harness(tmp_path)
    r = h.reply("What does my plan cover for physiotherapy?")
    assert r.startswith("Your plan says")
    assert "12" in r
    assert h.llm.calls == ["stream"]


def test_ungrounded_number_is_blocked(tmp_path):
    h = Harness(tmp_path, answer=lambda m: "Your plan covers 40 physiotherapy sessions per year.")
    r = h.reply("How many physiotherapy sessions does my plan cover?")
    assert r == templates.not_in_plan("en")


def test_advice_phrasing_is_blocked_mid_answer(tmp_path):
    h = Harness(tmp_path, answer=lambda m: (
        "Your plan covers 12 physiotherapy sessions per year. You should take ibuprofen before each one."))
    r = h.reply("What does my plan cover for physiotherapy?")
    assert "12 physiotherapy sessions" in r
    assert "ibuprofen" not in r
    assert templates.stopped_answer("en") in r


def test_model_says_not_in_plan(tmp_path):
    h = Harness(tmp_path, answer=lambda m: "NOT_IN_PLAN")
    assert h.reply("Does my plan cover dental cleanings?") == templates.not_in_plan("en")


def test_unknown_question_goes_to_router_then_not_in_plan(tmp_path):
    h = Harness(tmp_path, router=lambda m: {"intent": "plan_question"})
    assert h.reply("What's the capital of France?") == templates.not_in_plan("en")
    assert h.llm.calls == ["json"]


def test_router_fallback_measurement(tmp_path):
    h = Harness(tmp_path, router=lambda m: {"intent": "log_measurement", "metric": "weight",
                                            "values": [180], "unit": "pounds"})
    r = h.reply("the scale showed one eighty this morning")
    assert "weight 180 pounds" in r


def test_spanish_flow(tmp_path):
    h = Harness(tmp_path, lang="es")
    r = h.reply("Mi presión fue ciento veinte sobre ochenta esta mañana")
    assert "120 sobre 80" in r and "esta mañana" in r and "¿Lo guardo?" in r
    assert h.reply("sí").startswith("Guardado")
    r = h.reply("Quiero una cita el martes por la tarde")
    assert "Las opciones más cercanas son" in r
    r = h.reply("la primera")
    assert "¿Lo reservo?" in r
    assert h.reply("sí").startswith("Listo")
    r = h.reply("¿Qué cubre mi plan de fisioterapia?")
    assert r.startswith("Your plan says") or "12" in r  # fake model echoes the Spanish excerpt


def test_five_turn_context(tmp_path):
    """Story 5.ii: follow-ups across turns without repeating."""
    h = Harness(tmp_path)
    h.reply("Show me my weight for the last two weeks")
    h.reply("and my blood pressure?")
    assert h.session.last_chart == {"metric": "blood_pressure", "period": 14}
    h.reply("now the last 3 months")
    assert h.session.last_chart == {"metric": "blood_pressure", "period": 90}
    h.reply("Book me something Friday morning")
    r = h.reply("what about the afternoon instead?")
    assert h.session.last_pref["part"] == "afternoon"
    assert h.session.last_pref["date"] == "2026-10-09"


def test_turn_timings_present(tmp_path):
    h = Harness(tmp_path)
    turn = h.say("My heart rate is 72")
    s = turn.summary()
    assert s["route"] == "rules:measurement"
    for key in ("understand_ms", "tts_first_ms", "first_audio_ms"):
        assert key in s["timings"]

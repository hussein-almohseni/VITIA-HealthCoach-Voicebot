"""Conversation engine: routing, confirmation state, handlers and timing.

Order of checks for each patient turn:
  1. Safety: self-harm and emergency phrases get fixed messages, nothing else runs.
  2. Pending confirmation: "yes", "no", "the second one", a corrected reading.
  3. Rules: measurement, diagnosis/treatment request, plan change, chart, appointment,
     plan question. Most turns end here without calling the language model.
  4. Gemma router for everything the rules could not place.

Every reply that is not a plan answer comes from templates.py. Plan answers are
streamed from Gemma one sentence at a time, and each sentence is checked before
it is spoken.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Awaitable, Callable

from . import guardrails, router, templates
from .dates import TimePref, parse_time_pref
from .measurements import (Reading, correction, detect_metric, from_values, parse_measurements,
                           validate)
from .rag import PlanIndex
from .scheduling import Option, choose_option, detect_provider, find_options, provider_name
from .textnorm import fold, normalize_for_parsing, split_sentences

log = logging.getLogger("dialogue")

Emit = Callable[[dict], Awaitable[None]]

RETRIEVAL_MIN_SCORE = 1.0


@dataclass
class Settings:
    bus_delay_s: float = 120.0
    online: bool = True
    simulate_conflict: bool = False
    weight_unit: str = "lb"


@dataclass
class Session:
    lang: str = "en"
    history: list[dict] = field(default_factory=list)
    pending: dict | None = None
    last_intent: str | None = None
    last_chart: dict | None = None
    last_plan_question: str | None = None
    last_pref: dict | None = None
    last_provider: str | None = None


def ms_since(t: float) -> int:
    return round((time.perf_counter() - t) * 1000)


class Turn:
    def __init__(self, session: Session, text: str, input_method: str, emit: Emit):
        self.session = session
        self.lang = session.lang
        self.text = text
        self.input_method = input_method
        self.emit = emit
        self.t0 = time.perf_counter()
        self.timings: dict[str, int] = {}
        self.route: list[str] = []
        self.reply_parts: list[str] = []
        self.first_audio_t: float | None = None
        self.llm_stats: list[dict] = []
        self.tts_ms = 0

    def mark_understood(self) -> None:
        self.timings.setdefault("understand_ms", ms_since(self.t0))

    @property
    def reply(self) -> str:
        return " ".join(self.reply_parts)

    def summary(self) -> dict:
        t = dict(self.timings)
        t["tts_total_ms"] = self.tts_ms
        t["handle_total_ms"] = ms_since(self.t0)
        if self.llm_stats:
            t["llm_prompt_tokens"] = sum(s.get("prompt_tokens", 0) for s in self.llm_stats)
            t["llm_prompt_ms"] = sum(s.get("prompt_ms", 0) for s in self.llm_stats)
            t["llm_output_tokens"] = sum(s.get("output_tokens", 0) for s in self.llm_stats)
            t["llm_output_ms"] = sum(s.get("output_ms", 0) for s in self.llm_stats)
        return {"route": " > ".join(self.route), "reply": self.reply, "timings": t}


class Engine:
    def __init__(self, db, llm, tts, plan_dir_texts: dict[str, str], settings: Settings,
                 clock: Callable[[], datetime] = datetime.now,
                 on_records_changed: Callable[[], Awaitable[None]] | None = None):
        self.db = db
        self.llm = llm
        self.tts = tts
        self.settings = settings
        self.clock = clock
        self.on_records_changed = on_records_changed
        self.plans: dict[str, PlanIndex] = {
            lang: PlanIndex(text, lang) for lang, text in plan_dir_texts.items()
        }

    def reload_plan(self, lang: str, text: str) -> None:
        self.plans[lang] = PlanIndex(text, lang)

    # --- speaking -------------------------------------------------------------
    async def _speak(self, turn: Turn, sentence: str) -> None:
        t = time.perf_counter()
        wav = await asyncio.to_thread(self.tts.synth, sentence, turn.lang)
        elapsed = ms_since(t)
        turn.tts_ms += elapsed
        if turn.first_audio_t is None:
            turn.timings["tts_first_ms"] = elapsed
            turn.first_audio_t = time.perf_counter()
            turn.timings["first_audio_ms"] = round((turn.first_audio_t - turn.t0) * 1000)
        await turn.emit({"type": "audio", "wav": base64.b64encode(wav).decode("ascii")})

    async def say(self, turn: Turn, text: str) -> None:
        turn.mark_understood()
        turn.reply_parts.append(text)
        await turn.emit({"type": "reply_text", "text": text})
        sentences, _ = split_sentences(text)
        for s in sentences:
            await self._speak(turn, s)

    async def _say_sentence(self, turn: Turn, sentence: str) -> None:
        turn.reply_parts.append(sentence)
        await turn.emit({"type": "reply_text", "text": sentence})
        await self._speak(turn, sentence)

    # --- entry point ----------------------------------------------------------
    async def handle(self, session: Session, text: str, input_method: str, emit: Emit) -> Turn:
        turn = Turn(session, text.strip(), input_method, emit)
        now = self.clock()
        try:
            if not turn.text:
                turn.route.append("empty")
                await self.say(turn, templates.not_understood(turn.lang))
            else:
                await self._dispatch(turn, session, turn.text, now)
        except Exception:  # noqa: BLE001
            log.exception("Turn failed")
            turn.route.append("error")
            await self.say(turn, templates.not_understood(turn.lang))
        session.history.append({"role": "user", "content": turn.text})
        session.history.append({"role": "assistant", "content": turn.reply})
        session.history = session.history[-12:]
        return turn

    async def _dispatch(self, turn: Turn, s: Session, text: str, now: datetime) -> None:
        lang = s.lang

        # 1. Safety
        if guardrails.is_self_harm(text, lang):
            turn.route.append("safety:crisis")
            s.pending = None
            return await self.say(turn, templates.crisis(lang))
        if guardrails.is_emergency(text, lang):
            turn.route.append("safety:emergency")
            s.pending = None
            return await self.say(turn, templates.emergency(lang))

        # 2. Pending confirmation
        if s.pending:
            if await self._handle_pending(turn, s, text, now):
                return
            s.pending = None

        # 3. Rules
        readings = parse_measurements(text, lang, now, self.settings.weight_unit)
        clinical = guardrails.is_clinical_request(text, lang)
        if readings and clinical and not re.search(r"\b(my|i|mi|mis|me|tuve)\b", fold(text)):
            # "Is 150 over 95 high?" asks for an interpretation, it is not a reading to log.
            readings = []
        if readings:
            turn.route.append("rules:measurement")
            return await self._readback(turn, s, readings, text, add_refusal=clinical)
        if clinical:
            turn.route.append("rules:refuse_clinical")
            s.last_intent = "refusal"
            return await self.say(turn, templates.refuse_clinical(lang))
        if guardrails.is_plan_change(text, lang):
            turn.route.append("rules:refuse_plan_change")
            s.pending = {"kind": "offer_appointment"}
            return await self.say(turn, templates.refuse_plan_change(lang))
        if router.matches(router.NEXT_APPOINTMENT, text, lang):
            turn.route.append("rules:next_appointment")
            return await self._next_appointment(turn, s, now)

        folded_num = normalize_for_parsing(text, lang)
        metric = detect_metric(folded_num, lang)
        period = router.parse_period(text, lang)
        follow_up = router.matches(router.FOLLOW_UP, text, lang)
        if router.matches(router.CHART, text, lang) and (metric or (s.last_chart and period)):
            turn.route.append("rules:chart")
            return await self._chart(turn, s, metric or s.last_chart["metric"], period, now)
        if s.last_intent == "chart" and (metric or period) and (follow_up or len(text.split()) <= 6):
            turn.route.append("rules:chart_follow_up")
            return await self._chart(turn, s, metric or s.last_chart["metric"],
                                     period or s.last_chart.get("period"), now)

        pref = parse_time_pref(text, lang, now.date())
        books = router.matches(router.BOOK, text, lang)
        appt_noun = router.matches(router.APPOINTMENT_NOUN, text, lang)
        question = router.matches(router.QUESTION, text, lang)
        if books or (appt_noun and not pref.is_empty() and not question):
            turn.route.append("rules:appointment")
            return await self._appointment(turn, s, text, pref, now)
        if s.last_intent == "appointment" and not pref.is_empty():
            turn.route.append("rules:appointment_follow_up")
            return await self._appointment(turn, s, text, pref, now, merge=True)

        if question or (follow_up and s.last_intent == "plan"):
            turn.route.append("rules:plan_question")
            return await self._plan(turn, s, text, now)

        # 4. Gemma router
        await self._llm_route(turn, s, text, now)

    # --- pending confirmations ------------------------------------------------
    async def _handle_pending(self, turn: Turn, s: Session, text: str, now: datetime) -> bool:
        lang, p = s.lang, s.pending
        kind = p["kind"]

        if kind == "confirm_measurement":
            readings = parse_measurements(text, lang, now, self.settings.weight_unit)
            if not readings and len(p["readings"]) == 1:
                fixed = correction(text, Reading.from_dict(p["readings"][0]), lang, now)
                readings = [fixed] if fixed else []
            if readings:  # a correction: "no, it was 130 over 85"
                turn.route.append("pending:measurement_corrected")
                await self._readback(turn, s, readings, text, add_refusal=False,
                                     input_method=p.get("input_method"))
                return True
            if router.is_yes(text, lang):
                turn.route.append("pending:measurement_saved")
                await self._save_measurements(turn, s, p, now)
                return True
            if router.is_no(text, lang):
                turn.route.append("pending:measurement_rejected")
                s.pending = None
                s.last_intent = "measurement"
                await self.say(turn, templates.not_saved(lang))
                return True
            return False

        if kind in ("choose_slot", "confirm_slot", "need_date"):
            options = [Option.from_dict(o) for o in p.get("options", [])]
            provider = detect_provider(text) or p.get("provider")
            if kind == "confirm_slot" and router.is_yes(text, lang):
                turn.route.append("pending:slot_confirmed")
                await self._book(turn, s, Option.from_dict(p["option"]), now)
                return True
            if kind == "choose_slot" and len(options) == 1 and router.is_yes(text, lang):
                turn.route.append("pending:slot_confirmed")
                await self._book(turn, s, options[0], now)
                return True
            if kind == "choose_slot":
                picked = choose_option(text, options, lang)
                if picked:
                    turn.route.append("pending:option_chosen")
                    s.pending = {"kind": "confirm_slot", "option": picked.to_dict(),
                                 "options": p["options"], "provider": provider}
                    await self.say(turn, templates.confirm_slot(
                        picked.start, provider_name(picked.provider, lang), lang))
                    return True
            if router.matches(router.OTHER_DAY, text, lang) or router.is_no(text, lang):
                turn.route.append("pending:other_day")
                s.pending = {"kind": "need_date", "provider": provider}
                await self.say(turn, templates.ask_when(lang))
                return True
            pref = parse_time_pref(text, lang, now.date())
            if not pref.is_empty():
                turn.route.append("pending:new_time")
                await self._appointment(turn, s, text, pref, now, merge=(kind != "need_date"),
                                        provider=provider)
                return True
            if kind == "choose_slot" and router.is_yes(text, lang):
                turn.route.append("pending:which_option")
                await self.say(turn, templates.which_option(lang))
                return True
            return False

        if kind == "offer_appointment":
            if router.is_yes(text, lang):
                turn.route.append("pending:offer_accepted")
                s.pending = {"kind": "need_date", "provider": "patel"}
                s.last_intent = "appointment"
                await self.say(turn, templates.ask_when(lang))
                return True
            if router.is_no(text, lang):
                turn.route.append("pending:offer_declined")
                s.pending = None
                await self.say(turn, templates.anything_else(lang))
                return True
            return False
        return False

    # --- measurements ---------------------------------------------------------
    async def _readback(self, turn: Turn, s: Session, readings: list[Reading], text: str,
                        add_refusal: bool, input_method: str | None = None) -> None:
        lang = s.lang
        for r in readings:
            if validate(r):
                s.pending = None
                s.last_intent = "measurement"
                return await self.say(turn, templates.implausible(r, lang))
        parts = [templates.readback_statement(readings, lang)]
        if add_refusal:
            parts.append(templates.no_interpretation(lang))
            turn.route.append("guard:no_interpretation")
        parts.append(templates.save_question(lang))
        s.pending = {
            "kind": "confirm_measurement",
            "readings": [r.to_dict() for r in readings],
            "source_text": text,
            "input_method": input_method or turn.input_method,
        }
        s.last_intent = "measurement"
        await self.say(turn, " ".join(parts))

    async def _save_measurements(self, turn: Turn, s: Session, p: dict, now: datetime) -> None:
        readings = [Reading.from_dict(d) for d in p["readings"]]
        for r in readings:
            m_id = await asyncio.to_thread(self.db.add_measurement, r, p["input_method"],
                                           p["source_text"], s.lang, now)
            await asyncio.to_thread(self.db.enqueue, "measurement", m_id, r.to_dict(), now,
                                    self.settings.bus_delay_s)
        s.pending = None
        s.last_intent = "measurement"
        await self.say(turn, templates.saved(readings, s.lang))
        if self.on_records_changed:
            await self.on_records_changed()

    # --- appointments ---------------------------------------------------------
    async def _appointment(self, turn: Turn, s: Session, text: str, pref: TimePref, now: datetime,
                           merge: bool = False, provider: str | None = None) -> None:
        lang = s.lang
        provider = provider or detect_provider(text) or (s.last_provider if merge else None)
        if merge:
            pref = pref.merged_with(TimePref.from_dict(s.last_pref))
        s.last_intent = "appointment"
        s.last_provider = provider
        if pref.is_empty():
            s.pending = {"kind": "need_date", "provider": provider}
            return await self.say(turn, templates.ask_when(lang))
        t = time.perf_counter()
        options, matched = await asyncio.to_thread(find_options, self.db, pref, now, provider)
        turn.timings["slot_search_ms"] = ms_since(t)
        s.last_pref = pref.to_dict()
        if not options:
            s.pending = None
            return await self.say(turn, templates.no_slots(lang))
        if matched and pref.hour is not None:
            o = options[0]
            s.pending = {"kind": "confirm_slot", "option": o.to_dict(),
                         "options": [o.to_dict()], "provider": provider}
            return await self.say(turn, templates.slot_available(o.start, provider_name(o.provider, lang), lang))
        s.pending = {"kind": "choose_slot", "options": [o.to_dict() for o in options],
                     "provider": provider}
        requested = templates.describe_request(pref, lang)
        await turn.emit({"type": "options", "options": [
            {"start": o.start.isoformat(), "provider": provider_name(o.provider, lang)} for o in options]})
        await self.say(turn, templates.offer_options([o.start for o in options], requested, matched, lang))

    async def _book(self, turn: Turn, s: Session, option: Option, now: datetime) -> None:
        lang = s.lang
        source = " / ".join(h["content"] for h in s.history[-4:] if h["role"] == "user") + f" / {turn.text}"
        appt_id = await asyncio.to_thread(self.db.book, option.slot_id, turn.input_method,
                                          source.strip(" /"), lang, now)
        if appt_id is None:
            turn.route.append("slot_taken")
            pref = TimePref(date=option.start.date(), hour=option.start.hour, minute=option.start.minute)
            await self.say(turn, templates.slot_taken(lang))
            return await self._appointment(turn, s, "", pref, now, provider=option.provider)
        await asyncio.to_thread(self.db.enqueue, "appointment", appt_id, option.to_dict(), now,
                                self.settings.bus_delay_s)
        s.pending = None
        s.last_intent = "appointment"
        await self.say(turn, templates.booked(option.start, provider_name(option.provider, lang), lang))
        if self.on_records_changed:
            await self.on_records_changed()

    async def _next_appointment(self, turn: Turn, s: Session, now: datetime) -> None:
        lang = s.lang
        upcoming = await asyncio.to_thread(self.db.appointments, now)
        s.last_intent = "appointment"
        if not upcoming:
            s.pending = {"kind": "offer_appointment"}
            return await self.say(turn, templates.no_appointments(lang))
        a = upcoming[0]
        await self.say(turn, templates.next_appointment(
            datetime.fromisoformat(a["start_at"]), provider_name(a["provider"], lang), a["status"], lang))

    # --- charts ---------------------------------------------------------------
    async def _chart(self, turn: Turn, s: Session, metric: str, period, now: datetime) -> None:
        lang = s.lang
        period = period or 30
        until = None
        if period == "this_month":
            since = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            label = templates.period_label(None, True, lang)
        elif period == "last_month":
            until = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            since = (until - timedelta(days=1)).replace(day=1)
            label = "last month" if lang == "en" else "del mes pasado"
        else:
            since = now - timedelta(days=int(period))
            label = templates.period_label(int(period), False, lang)
        rows = await asyncio.to_thread(self.db.measurements, metric, since, until)
        weight_unit = self.settings.weight_unit
        series: list[dict] = []
        if metric == "blood_pressure":
            series = [
                {"name": "Systolic" if lang == "en" else "Sistólica",
                 "points": [[r["measured_at"], r["systolic"]] for r in rows]},
                {"name": "Diastolic" if lang == "en" else "Diastólica",
                 "points": [[r["measured_at"], r["diastolic"]] for r in rows]},
            ]
            unit = "mmHg"
        else:
            def display(r):
                if metric == "weight":
                    kg = r["canonical_value"]
                    return round(kg * 2.20462, 1) if weight_unit == "lb" else round(kg, 1)
                return r["canonical_value"] if r["canonical_value"] is not None else r["value"]
            unit = {"weight": weight_unit, "glucose": "mg/dL", "heart_rate": "bpm"}[metric]
            series = [{"name": templates.METRIC_NAMES[lang][metric].capitalize(),
                       "points": [[r["measured_at"], display(r)] for r in rows]}]
        latest = ""
        if rows:
            last = rows[-1]
            if metric == "blood_pressure":
                latest = f"{last['systolic']} {'over' if lang == 'en' else 'sobre'} {last['diastolic']}"
            else:
                v = series[0]["points"][-1][1]
                latest = f"{templates.fmt_number(v)} {templates.UNIT_SPOKEN[lang][unit]}".strip()
            latest += f", {templates.fmt_date(datetime.fromisoformat(last['measured_at']).date(), lang)}"
        title = f"{templates.METRIC_NAMES[lang][metric].capitalize()} · {label}"
        await turn.emit({"type": "chart", "metric": metric, "title": title, "unit": unit,
                         "series": series})
        s.last_intent = "chart"
        s.last_chart = {"metric": metric, "period": period}
        await self.say(turn, templates.chart_summary(metric, label, len(rows), latest, lang))

    # --- plan questions -------------------------------------------------------
    async def _plan(self, turn: Turn, s: Session, text: str, now: datetime,
                    routed_by_llm: bool = False) -> None:
        lang = s.lang
        previous = s.last_plan_question if (s.last_intent == "plan" and len(text.split()) <= 7) else None
        query = f"{text} {previous}" if previous else text
        t = time.perf_counter()
        hits = self.plans[lang].search(query, k=2)
        turn.timings["retrieval_ms"] = ms_since(t)
        if not hits or hits[0].score < RETRIEVAL_MIN_SCORE:
            if not routed_by_llm:
                turn.route.append("retrieval:weak")
                return await self._llm_route(turn, s, text, now, skip_plan=True)
            turn.route.append("retrieval:none")
            s.last_intent = "plan"
            return await self.say(turn, templates.not_in_plan(lang))
        turn.route.append("retrieval:" + ",".join(h.chunk.title for h in hits))
        excerpts = [h.chunk.full for h in hits]
        s.last_intent = "plan"
        s.last_plan_question = query
        await self._stream_answer(turn, router.plan_messages(lang, excerpts, text, previous),
                                  " ".join(excerpts), text)

    async def _stream_answer(self, turn: Turn, messages: list[dict], context: str, question: str) -> None:
        lang = turn.lang
        turn.mark_understood()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        stop = threading.Event()

        def worker():
            try:
                for item in self.llm.stream(messages, num_predict=160, stop_event=stop):
                    loop.call_soon_threadsafe(queue.put_nowait, item)
            except Exception as e:  # noqa: BLE001
                loop.call_soon_threadsafe(queue.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=worker, daemon=True).start()
        t_gen = time.perf_counter()
        buffer, decided, spoke, blocked = "", False, False, False
        sentinels = ("NOT_IN_PLAN", "REFUSE")

        async def check_and_say(sentence: str) -> bool:
            nonlocal spoke
            ok, why = guardrails.check_generated_sentence(sentence, context, question, lang)
            if not ok:
                stop.set()
                turn.route.append(f"guard:blocked({why})")
                await self.say(turn, templates.stopped_answer(lang) if spoke else templates.not_in_plan(lang))
                return False
            if not spoke:
                turn.timings["llm_first_sentence_ms"] = ms_since(t_gen)
            await self._say_sentence(turn, sentence)
            spoke = True
            return True

        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            if isinstance(item, dict):
                turn.llm_stats.append(item)
                continue
            buffer += item
            if not decided:
                head = buffer.strip()
                if any(tok.startswith(head) and len(head) < len(tok) for tok in sentinels):
                    continue
                if head.startswith("NOT_IN_PLAN"):
                    stop.set()
                    turn.route.append("llm:not_in_plan")
                    await self.say(turn, templates.not_in_plan(lang))
                    blocked = True
                    break
                if head.startswith("REFUSE"):
                    stop.set()
                    turn.route.append("llm:refuse")
                    await self.say(turn, templates.refuse_clinical(lang))
                    blocked = True
                    break
                decided = True
            sentences, buffer = split_sentences(buffer, final=False)
            for sentence in sentences:
                if not await check_and_say(sentence):
                    blocked = True
                    break
            if blocked:
                break

        # Drain stats if generation was stopped early.
        if blocked:
            return
        head = buffer.strip()
        if not spoke and head.startswith(sentinels):
            await self.say(turn, templates.not_in_plan(lang) if head.startswith("NOT") else
                           templates.refuse_clinical(lang))
            return
        sentences, _ = split_sentences(buffer, final=True)
        for sentence in sentences:
            if not await check_and_say(sentence):
                return
        if not spoke:
            await self.say(turn, templates.not_in_plan(lang))

    # --- Gemma router ---------------------------------------------------------
    async def _llm_route(self, turn: Turn, s: Session, text: str, now: datetime,
                         skip_plan: bool = False) -> None:
        lang = s.lang
        pending = None
        if s.pending:
            pending = {"confirm_measurement": "yes or no to saving a reading",
                       "choose_slot": "the patient to pick an appointment option",
                       "confirm_slot": "yes or no to booking a time",
                       "need_date": "a day and time for an appointment",
                       "offer_appointment": "yes or no to booking an appointment"}.get(s.pending["kind"])
        t = time.perf_counter()
        try:
            data, stats = await asyncio.to_thread(
                self.llm.json, router.router_messages(s.history, text, pending), router.ROUTER_SCHEMA)
        except Exception as e:  # noqa: BLE001
            log.warning("Router model failed: %s", e)
            turn.route.append("llm:error")
            return await self.say(turn, templates.not_understood(lang))
        turn.timings["llm_route_ms"] = ms_since(t)
        turn.llm_stats.append(stats)
        intent = data.get("intent", "other")
        turn.route.append(f"llm:{intent}")

        if intent == "emergency":
            s.pending = None
            return await self.say(turn, templates.emergency(lang))
        if intent == "clinical_advice":
            s.last_intent = "refusal"
            return await self.say(turn, templates.refuse_clinical(lang))
        if intent == "change_plan":
            s.pending = {"kind": "offer_appointment"}
            return await self.say(turn, templates.refuse_plan_change(lang))
        if intent == "log_measurement":
            r = from_values(data.get("metric", ""), data.get("values") or [], data.get("unit"),
                            lang, now, text, self.settings.weight_unit)
            if r is None:
                return await self.say(turn, templates.not_understood(lang))
            return await self._readback(turn, s, [r], text, add_refusal=False)
        if intent == "book_appointment":
            pref = parse_time_pref(data.get("when") or text, lang, now.date())
            return await self._appointment(turn, s, text, pref, now, merge=s.last_intent == "appointment")
        if intent == "next_appointment":
            return await self._next_appointment(turn, s, now)
        if intent == "show_chart":
            metric = data.get("metric") or (s.last_chart or {}).get("metric")
            if not metric:
                return await self.say(turn, templates.out_of_scope(lang))
            return await self._chart(turn, s, metric, data.get("days") or router.parse_period(text, lang), now)
        if intent == "plan_question":
            if skip_plan:
                s.last_intent = "plan"
                return await self.say(turn, templates.not_in_plan(lang))
            return await self._plan(turn, s, text, now, routed_by_llm=True)
        s.last_intent = "other"
        await self.say(turn, templates.out_of_scope(lang))

"""Intent routing: fast rules first, Gemma as the fallback.

Rules catch the common phrasings in a few milliseconds. Anything they cannot place goes
to Gemma with a short JSON schema. The UI shows which path each turn took, which is the
main lever on the 2-second target.
"""

from __future__ import annotations

import re

from .textnorm import fold

YES = {
    "en": r"^(yes|yeah|yep|yup|sure|correct|right|that'?s right|that is right|confirm|ok|okay|"
          r"sounds good|do it|please do|go ahead|perfect|absolutely|book it|save it|yes please)\b",
    "es": r"^(si|claro|correcto|asi es|exacto|vale|de acuerdo|ok|okay|perfecto|confirmo|adelante|"
          r"esta bien|eso es|por favor|si por favor|reservala|reservalo|guardalo|guardala)\b",
}
NO = {
    "en": r"^(no|nope|nah|wrong|incorrect|not right|cancel|don'?t|that'?s wrong|never mind|nevermind)\b",
    "es": r"^(no|incorrecto|esta mal|cancela|cancelar|no es correcto|olvidalo|mejor no)\b",
}
OTHER_DAY = {
    "en": r"\b(none of (these|those|them)|another (day|time)|something else|different (day|time)|other (days|times))\b",
    "es": r"\b(ninguna|otro dia|otra hora|otro horario|otra opcion|otros horarios)\b",
}

# Booking intent: a booking verb, or "I want/need an appointment".
BOOK = {
    "en": r"\b(book|schedule|reschedule|set up|make an? (appointment|appt)|get an? (appointment|appt)|"
          r"(need|want|like) an? (appointment|appt|visit|check-?in)|see (the )?(doctor|dr|coach)|"
          r"come in|any openings|an opening)\b",
    "es": r"\b(agendar|agendame|agenda(r)? una|programar|reservar|reservame|reserva(r)? una|"
          r"hacer una cita|sacar una cita|(quiero|necesito|quisiera) (una )?(cita|consulta)|"
          r"ver (a )?(la |el )?(doctor|doctora|medico|coach)|espacio disponible)\b",
}
APPOINTMENT_NOUN = {
    "en": r"\b(appointment|appt|visit|check-?in)\b",
    "es": r"\b(cita|consulta|turno)\b",
}
NEXT_APPOINTMENT = {
    "en": r"\b(when|what time)('?s| is| was)? .{0,12}\b(my|next)\b.{0,12}\bappointment\b"
          r"|\b(do i have|my) (any )?(upcoming|next) appointments?\b",
    "es": r"\b(cuando|a que hora) (es|tengo) .{0,10}\bcita\b|\b(tengo|mis) .{0,8}citas? (proxima|pendiente)s?\b"
          r"|\bmi proxima cita\b",
}
CHART = {
    "en": r"\b(graph|chart|plot|trend|history|visual|show me|let me see|how has|how have)\b",
    "es": r"\b(grafica|grafico|tendencia|historial|muestrame|ensename|mostrar|quiero ver|como ha (ido|estado)|como van)\b",
}
QUESTION = {
    "en": r"^(what|whats|when|where|which|who|why|how|does|do|is|are|can|could|will|am|should|"
          r"would|tell me|explain|remind me)\b|\b(my plan|health plan|covered|cover|covers|coverage)\b",
    "es": r"^(que|cuando|donde|cual|cuales|quien|por que|como|cuanto|cuantos|cuantas|puedo|"
          r"cubre|tengo|hay|es|explica|explicame|dime|recuerdame)\b|\b(mi plan|plan de salud|cubre|cobertura)\b",
}
FOLLOW_UP = {
    "en": r"^(and|what about|how about|now|also|ok and|okay and)\b",
    "es": r"^(y|que tal|y el|y la|y los|y las|ahora|tambien)\b",
}

PERIODS = {
    "en": [
        (r"\bthis month\b", "this_month"), (r"\blast month\b", "last_month"),
        (r"\b(this|past|last) week\b|\b7 days\b", 7), (r"\btwo weeks\b|\b2 weeks\b|\b14 days\b", 14),
        (r"\b(30 days|past month|a month)\b", 30), (r"\b(three|3) months\b|\b90 days\b", 90),
        (r"\b(\d{1,3}) days\b", "n_days"),
    ],
    "es": [
        (r"\beste mes\b", "this_month"), (r"\b(el )?mes pasado\b", "last_month"),
        (r"\b(esta|la ultima|la) semana\b|\b7 dias\b", 7), (r"\b(dos|2) semanas\b|\b14 dias\b", 14),
        (r"\b(30 dias|el ultimo mes|un mes)\b", 30), (r"\b(tres|3) meses\b|\b90 dias\b", 90),
        (r"\b(\d{1,3}) dias\b", "n_days"),
    ],
}


def matches(table: dict, text: str, lang: str) -> bool:
    return re.search(table[lang], fold(text).strip()) is not None


def is_yes(text, lang):
    return matches(YES, text, lang)


def is_no(text, lang):
    return matches(NO, text, lang)


def parse_period(text: str, lang: str):
    """Returns 'this_month', 'last_month', a number of days, or None."""
    t = fold(text)
    for pat, value in PERIODS[lang]:
        m = re.search(pat, t)
        if m:
            return int(m.group(1)) if value == "n_days" else value
    return None


# --- Gemma fallback -----------------------------------------------------------------

ROUTER_SYSTEM = """You are the language-understanding step of a patient health app. The patient speaks English or Spanish. Read the conversation and classify the patient's LAST message. Output JSON only. Never answer the patient.

intent values:
- log_measurement: the patient reports a blood pressure, weight, blood glucose or heart rate value
- book_appointment: the patient wants to book or schedule an appointment, or says when they want one
- next_appointment: the patient asks when their next appointment is
- plan_question: the patient asks about their own health plan, coverage, medications in the plan, routines or instructions
- show_chart: the patient wants to see a graph, chart, trend or history of their readings
- clinical_advice: the patient asks for a diagnosis, what a symptom or reading means, whether something is normal or bad, or what treatment, medication or dose to take or change
- change_plan: the patient asks to change or update their health plan
- emergency: severe symptoms happening now, such as chest pain, trouble breathing, fainting, stroke signs, or thoughts of self-harm
- yes / no: the patient confirms or rejects what the assistant just asked
- choose_option: the patient picks one of the options the assistant listed; set option to its number (1, 2 or 3)
- other: anything else

Other fields (include only when they apply):
- metric: blood_pressure, weight, glucose or heart_rate
- values: the numbers the patient said for the measurement, in order
- unit: the unit the patient said, if any
- days: how many days of history for show_chart
- when: the patient's exact words about the appointment day or time"""

ROUTER_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": [
            "log_measurement", "book_appointment", "next_appointment", "plan_question",
            "show_chart", "clinical_advice", "change_plan", "emergency", "yes", "no",
            "choose_option", "other"]},
        "metric": {"type": "string", "enum": ["blood_pressure", "weight", "glucose", "heart_rate"]},
        "values": {"type": "array", "items": {"type": "number"}},
        "unit": {"type": "string"},
        "days": {"type": "integer"},
        "when": {"type": "string"},
        "option": {"type": "integer"},
    },
    "required": ["intent"],
}


def router_messages(history: list[dict], text: str, pending_summary: str | None) -> list[dict]:
    msgs = [{"role": "system", "content": ROUTER_SYSTEM}]
    msgs.extend(history[-6:])
    content = text
    if pending_summary:
        content = f"(The assistant is waiting for: {pending_summary})\n{text}"
    msgs.append({"role": "user", "content": content})
    return msgs


PLAN_SYSTEM = {
    "en": """You answer a patient's question using ONLY the excerpts from their own health plan.
Rules:
- Answer in English, in one to three short sentences meant to be spoken aloud. No lists, no markdown.
- Start with "Your plan says" or similar, and use only facts written in the excerpts.
- Never add medical advice, diagnosis, opinions, or anything not written in the excerpts.
- If the excerpts do not answer the question, reply exactly: NOT_IN_PLAN
- If the question asks for a diagnosis, what a symptom or reading means, or a change to treatment, reply exactly: REFUSE""",
    "es": """Respondes la pregunta de un paciente usando SOLO los fragmentos de su propio plan de salud.
Reglas:
- Responde en español, en una a tres oraciones cortas para leer en voz alta. Sin listas ni formato.
- Empieza con "Tu plan dice" o algo similar, y usa solo datos escritos en los fragmentos.
- Nunca agregues consejos médicos, diagnósticos, opiniones ni nada que no esté en los fragmentos.
- Si los fragmentos no responden la pregunta, responde exactamente: NOT_IN_PLAN
- Si la pregunta pide un diagnóstico, qué significa un síntoma o una medición, o un cambio de tratamiento, responde exactamente: REFUSE""",
}


def plan_messages(lang: str, excerpts: list[str], question: str, previous_question: str | None) -> list[dict]:
    lines = [f"[{i + 1}] {e}" for i, e in enumerate(excerpts)]
    label = "Plan excerpts" if lang == "en" else "Fragmentos del plan"
    q_label = "Question" if lang == "en" else "Pregunta"
    prev = ""
    if previous_question:
        prev_label = "Previous question" if lang == "en" else "Pregunta anterior"
        prev = f"{prev_label}: {previous_question}\n"
    return [
        {"role": "system", "content": PLAN_SYSTEM[lang]},
        {"role": "user", "content": f"{label}:\n" + "\n".join(lines) + f"\n\n{prev}{q_label}: {question}"},
    ]

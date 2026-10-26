"""Fixed spoken responses in English and Spanish.

Most replies come from here rather than the language model: read-backs, confirmations,
appointment options, refusals and emergency messages. That keeps them fast (no model
call), predictable, and reviewable by the clinical team. Edit the wording here.
"""

from __future__ import annotations

from datetime import date, datetime

from .measurements import Reading

WEEKDAY_NAMES = {
    "en": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "es": ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"],
}
MONTH_NAMES = {
    "en": ["January", "February", "March", "April", "May", "June", "July", "August",
           "September", "October", "November", "December"],
    "es": ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
           "septiembre", "octubre", "noviembre", "diciembre"],
}
METRIC_NAMES = {
    "en": {"blood_pressure": "blood pressure", "weight": "weight",
           "glucose": "blood glucose", "heart_rate": "heart rate"},
    "es": {"blood_pressure": "presión arterial", "weight": "peso",
           "glucose": "glucosa", "heart_rate": "frecuencia cardíaca"},
}
UNIT_SPOKEN = {
    "en": {"mmHg": "", "lb": "pounds", "kg": "kilograms", "mg/dL": "milligrams per deciliter",
           "mmol/L": "millimoles per liter", "bpm": "beats per minute"},
    "es": {"mmHg": "", "lb": "libras", "kg": "kilos", "mg/dL": "miligramos por decilitro",
           "mmol/L": "milimoles por litro", "bpm": "latidos por minuto"},
}
PART_NAMES = {
    "en": {"morning": "in the morning", "afternoon": "in the afternoon", "evening": "late in the day"},
    "es": {"morning": "por la mañana", "afternoon": "por la tarde", "evening": "al final del día"},
}


def fmt_number(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


def fmt_date(d: date, lang: str) -> str:
    wd = WEEKDAY_NAMES[lang][d.weekday()]
    month = MONTH_NAMES[lang][d.month - 1]
    if lang == "en":
        return f"{wd}, {month} {d.day}"
    return f"el {wd} {d.day} de {month}"


def fmt_time(dt: datetime, lang: str) -> str:
    h, m = dt.hour, dt.minute
    if lang == "en":
        suffix = "AM" if h < 12 else "PM"
        h12 = h % 12 or 12
        return f"{h12} {suffix}" if m == 0 else f"{h12}:{m:02d} {suffix}"
    h12 = h % 12 or 12
    clock = f"{h12}" if m == 0 else f"{h12}:{m:02d}"
    if h == 12:
        period = "del mediodía"
    elif h < 12:
        period = "de la mañana"
    else:
        period = "de la tarde"
    article = "a la" if h12 == 1 else "a las"
    return f"{article} {clock} {period}"


def fmt_slot(start: datetime, lang: str, with_date: bool = True) -> str:
    if lang == "en":
        return f"{fmt_date(start.date(), lang)} at {fmt_time(start, lang)}" if with_date else fmt_time(start, lang)
    return f"{fmt_date(start.date(), lang)} {fmt_time(start, lang)}" if with_date else fmt_time(start, lang)


def fmt_reading(r: Reading, lang: str) -> str:
    name = METRIC_NAMES[lang][r.metric]
    if r.metric == "blood_pressure":
        sep = "over" if lang == "en" else "sobre"
        return f"{name} {r.systolic} {sep} {r.diastolic}"
    unit = UNIT_SPOKEN[lang][r.unit]
    return f"{name} {fmt_number(r.value)} {unit}".strip()


def join_options(items: list[str], lang: str) -> str:
    word = "or" if lang == "en" else "o"
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} {word} {items[1]}"
    return ", ".join(items[:-1]) + f", {word} {items[-1]}"


# --- Measurements ------------------------------------------------------------------

def readback_statement(readings: list[Reading], lang: str) -> str:
    parts = [fmt_reading(r, lang) for r in readings]
    when = readings[0].when_text
    if lang == "en":
        listing = " and ".join(parts)
        taken = f", taken {when}" if when else ""
        return f"I have {listing}{taken}."
    listing = " y ".join(parts)
    taken = f", de {when}" if when else ""
    return f"Anoté {listing}{taken}."


def save_question(lang: str) -> str:
    return "Should I save it?" if lang == "en" else "¿Lo guardo?"


def readback(readings: list[Reading], lang: str) -> str:
    return f"{readback_statement(readings, lang)} {save_question(lang)}"


def saved(readings: list[Reading], lang: str) -> str:
    if lang == "en":
        return "Saved. Your care team will see it."
    return "Guardado. Tu equipo de atención lo podrá ver."


def not_saved(lang: str) -> str:
    return ("Okay, I won't save it. What was the reading?" if lang == "en"
            else "De acuerdo, no lo guardo. ¿Cuál fue la medición?")


def implausible(r: Reading, lang: str) -> str:
    if lang == "en":
        return (f"I heard {fmt_reading(r, lang)}, which is outside the range I can record. "
                "Could you check the number and say it again?")
    return (f"Escuché {fmt_reading(r, lang)}, que está fuera del rango que puedo registrar. "
            "¿Puedes revisar el número y repetirlo?")


def no_interpretation(lang: str) -> str:
    return ("I can't tell you what the reading means, though. Your care team can."
            if lang == "en" else
            "Pero no puedo decirte qué significa la medición. Tu equipo de atención sí puede.")


# --- Appointments ------------------------------------------------------------------

def ask_when(lang: str) -> str:
    return ("Sure. What day and time works for you?" if lang == "en"
            else "Claro. ¿Qué día y a qué hora te queda bien?")


def describe_request(pref, lang: str) -> str:
    """'Tuesday, October 13 in the afternoon'."""
    if pref.date is not None:
        text = fmt_date(pref.date, lang)
        if pref.hour is not None:
            dt = datetime.combine(pref.date, datetime.min.time()).replace(hour=pref.hour, minute=pref.minute)
            text = fmt_slot(dt, lang)
        elif pref.part:
            text += f" {PART_NAMES[lang][pref.part]}"
        return text
    if pref.part:
        return PART_NAMES[lang][pref.part]
    return ""


def slot_available(start: datetime, provider: str, lang: str) -> str:
    if lang == "en":
        return f"{fmt_slot(start, lang)} with {provider} is open. Should I book it?"
    return f"Hay espacio {fmt_slot(start, lang)} con {provider}. ¿Lo reservo?"


def offer_options(starts: list[datetime], requested: str, matched: bool, lang: str) -> str:
    same_day = len({s.date() for s in starts}) == 1
    items = []
    for i, s in enumerate(starts):
        items.append(fmt_slot(s, lang, with_date=not (same_day and i > 0)))
    listing = join_options(items, lang)
    if lang == "en":
        if matched:
            return f"I have {listing}. Which one works?"
        lead = f"{requested[0].upper() + requested[1:]} isn't available. " if requested else ""
        return f"{lead}The closest options are {listing}. Which one works?"
    if matched:
        return f"Tengo {listing}. ¿Cuál te queda mejor?"
    lead = f"No hay espacio {requested}. " if requested else ""
    return f"{lead}Las opciones más cercanas son {listing}. ¿Cuál te queda mejor?"


def no_slots(lang: str) -> str:
    return ("I couldn't find any openings in the next four weeks. Your clinic can help you by phone."
            if lang == "en" else
            "No encontré espacios en las próximas cuatro semanas. Tu clínica te puede ayudar por teléfono.")


def confirm_slot(start: datetime, provider: str, lang: str) -> str:
    if lang == "en":
        return f"{fmt_slot(start, lang)} with {provider}. Shall I book it?"
    return f"{fmt_slot(start, lang)[0].upper() + fmt_slot(start, lang)[1:]} con {provider}. ¿Lo reservo?"


def booked(start: datetime, provider: str, lang: str) -> str:
    if lang == "en":
        return (f"Done. You're booked for {fmt_slot(start, lang)} with {provider}. "
                "It's saved in the app and syncing with the clinic.")
    return (f"Listo. Tu cita es {fmt_slot(start, lang)} con {provider}. "
            "Ya está guardada en la aplicación y se está sincronizando con la clínica.")


def which_option(lang: str) -> str:
    return ("Which option would you like? You can say the first, second, or third one, or another day."
            if lang == "en" else
            "¿Qué opción prefieres? Puedes decir la primera, la segunda o la tercera, u otro día.")


def slot_taken(lang: str) -> str:
    return ("Sorry, that time was just taken. Let me find the next options." if lang == "en"
            else "Lo siento, ese horario se acaba de ocupar. Busco las siguientes opciones.")


def next_appointment(start: datetime, provider: str, status: str, lang: str) -> str:
    pending = status == "pending_sync"
    if lang == "en":
        tail = " It's still syncing with the clinic." if pending else ""
        return f"Your next appointment is {fmt_slot(start, lang)} with {provider}.{tail}"
    tail = " Todavía se está sincronizando con la clínica." if pending else ""
    return f"Tu próxima cita es {fmt_slot(start, lang)} con {provider}.{tail}"


def no_appointments(lang: str) -> str:
    return ("You don't have any upcoming appointments. Would you like to book one?" if lang == "en"
            else "No tienes citas próximas. ¿Quieres reservar una?")


# --- Plan, charts, safety ----------------------------------------------------------

def not_in_plan(lang: str) -> str:
    return ("I couldn't find that in your health plan. Your care team can answer it, "
            "or I can book you an appointment." if lang == "en" else
            "No encontré eso en tu plan de salud. Tu equipo de atención te puede responder, "
            "o puedo reservarte una cita.")


def stopped_answer(lang: str) -> str:
    return ("I'll stop there. Your care team can tell you more." if lang == "en"
            else "Hasta ahí llego. Tu equipo de atención te puede decir más.")


def refuse_clinical(lang: str) -> str:
    return ("I can't help with diagnosis or treatment decisions, because those need your care team, "
            "who know your full history. I can log a reading, explain what your plan says, "
            "or book you an appointment with them." if lang == "en" else
            "No puedo ayudarte con diagnósticos ni decisiones de tratamiento, porque eso le corresponde "
            "a tu equipo de atención, que conoce tu historial completo. Puedo registrar una medición, "
            "explicarte lo que dice tu plan o reservarte una cita con ellos.")


def refuse_plan_change(lang: str) -> str:
    return ("I can't change your health plan. Only your care team can update it. "
            "Would you like me to book an appointment to talk about it?" if lang == "en" else
            "No puedo cambiar tu plan de salud. Solo tu equipo de atención puede actualizarlo. "
            "¿Quieres que te reserve una cita para hablarlo?")


def emergency(lang: str) -> str:
    return ("This could be an emergency. Please call 911 or your local emergency number right now. "
            "I'm not able to assess symptoms." if lang == "en" else
            "Esto podría ser una emergencia. Por favor llama al 911 o al número de emergencias local ahora mismo. "
            "No puedo evaluar síntomas.")


def crisis(lang: str) -> str:
    return ("I'm really glad you told me. You can call or text 988, the Suicide and Crisis Lifeline, "
            "any time. If you're in immediate danger, call 911." if lang == "en" else
            "Me alegra que me lo digas. Puedes llamar o enviar un mensaje al 988, la Línea de Prevención "
            "del Suicidio y Crisis, a cualquier hora; también atienden en español. "
            "Si estás en peligro inmediato, llama al 911.")


def out_of_scope(lang: str) -> str:
    return ("I can help you log a reading, book an appointment, explain your health plan, "
            "or show your readings as a chart. What would you like to do?" if lang == "en" else
            "Puedo ayudarte a registrar una medición, reservar una cita, explicarte tu plan de salud "
            "o mostrarte tus mediciones en una gráfica. ¿Qué te gustaría hacer?")


def anything_else(lang: str) -> str:
    return "Okay. Is there anything else I can help with?" if lang == "en" else "De acuerdo. ¿Te ayudo con algo más?"


def not_understood(lang: str) -> str:
    return ("Sorry, I didn't catch that. Could you say it another way?" if lang == "en"
            else "Perdón, no te entendí. ¿Puedes decirlo de otra forma?")


def chart_summary(metric: str, days_label: str, count: int, latest: str, lang: str) -> str:
    name = METRIC_NAMES[lang][metric]
    if count == 0:
        return (f"I don't have any {name} readings {days_label}." if lang == "en"
                else f"No tengo mediciones de {name} {days_label}.")
    if lang == "en":
        noun = "reading" if count == 1 else "readings"
        return f"Here's your {name} {days_label}: {count} {noun}. The most recent was {latest}."
    noun = "medición" if count == 1 else "mediciones"
    return f"Aquí está tu {name} {days_label}: {count} {noun}. La más reciente fue {latest}."


def period_label(days: int | None, since_month: bool, lang: str) -> str:
    if since_month:
        return "this month" if lang == "en" else "de este mes"
    if lang == "en":
        return {7: "for the last week", 14: "for the last two weeks", 90: "for the last three months"}.get(
            days, f"for the last {days} days")
    return {7: "de la última semana", 14: "de las últimas dos semanas", 90: "de los últimos tres meses"}.get(
        days, f"de los últimos {days} días")

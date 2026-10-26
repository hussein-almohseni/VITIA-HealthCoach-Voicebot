"""Appointment time preferences from speech: "next Tuesday afternoon", "mañana a las 3".

Resolution is deterministic code, not the language model, because small models make
date arithmetic mistakes. The resolved date is always read back to the patient.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

from .textnorm import normalize_for_parsing

WEEKDAYS = {
    "en": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
    "es": ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"],
}
MONTHS = {
    "en": {
        "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
        "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
        "august": 8, "aug": 8, "september": 9, "sept": 9, "sep": 9, "october": 10,
        "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
    },
    "es": {
        "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
        "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
        "noviembre": 11, "diciembre": 12,
    },
}

PART_WINDOWS = {  # minutes from midnight, inside clinic hours
    "morning": (9 * 60, 12 * 60),
    "afternoon": (12 * 60, 17 * 60),
    "evening": (15 * 60, 17 * 60),
}


@dataclass
class TimePref:
    date: date | None = None
    after: date | None = None      # "next week": any day from this date on
    part: str | None = None        # morning | afternoon | evening
    hour: int | None = None
    minute: int = 0

    def is_empty(self) -> bool:
        return self.date is None and self.after is None and self.part is None and self.hour is None

    def merged_with(self, previous: "TimePref | None") -> "TimePref":
        """Fills gaps from the previous request, e.g. 'what about Thursday?' keeps 'afternoon'."""
        if previous is None:
            return self
        return TimePref(
            date=self.date or (None if (self.after or self.hour is not None) else previous.date),
            after=self.after,
            part=self.part or (previous.part if self.hour is None else None),
            hour=self.hour,
            minute=self.minute if self.hour is not None else 0,
        )

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat() if self.date else None,
            "after": self.after.isoformat() if self.after else None,
            "part": self.part, "hour": self.hour, "minute": self.minute,
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> "TimePref | None":
        if not d:
            return None
        return cls(
            date=date.fromisoformat(d["date"]) if d.get("date") else None,
            after=date.fromisoformat(d["after"]) if d.get("after") else None,
            part=d.get("part"), hour=d.get("hour"), minute=d.get("minute") or 0,
        )


def _next_weekday(today: date, weekday: int, skip_today: bool = True) -> date:
    days = (weekday - today.weekday()) % 7
    if days == 0 and skip_today:
        days = 7
    return today + timedelta(days=days)


def _resolve_day_month(day: int, month: int, today: date) -> date | None:
    for year in (today.year, today.year + 1):
        try:
            d = date(year, month, day)
        except ValueError:
            return None
        if d >= today:
            return d
    return None


def _resolve_day_of_month(day: int, today: date) -> date | None:
    month, year = today.month, today.year
    for _ in range(2):
        try:
            d = date(year, month, day)
            if d >= today:
                return d
        except ValueError:
            pass
        month, year = (1, year + 1) if month == 12 else (month + 1, year)
    return None


def _to_24h(hour: int, minute: int, meridiem: str | None) -> tuple[int, int]:
    if meridiem in ("pm",) and hour < 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0
    elif meridiem is None and 1 <= hour <= 7:
        hour += 12  # "at 3" during clinic hours means 3 PM
    return hour, minute


def parse_time_pref(text: str, lang: str, today: date) -> TimePref:
    t = normalize_for_parsing(text, lang)
    t = re.sub(r"\b(a)\.?\s?m\.?(?=\s|$|[,.!?])", "am", t)
    t = re.sub(r"\b(p)\.?\s?m\.?(?=\s|$|[,.!?])", "pm", t)
    pref = TimePref()
    return _parse_en(t, today, pref) if lang == "en" else _parse_es(t, today, pref)


def _parse_en(t: str, today: date, p: TimePref) -> TimePref:
    # Time of day
    if re.search(r"\b(this )?morning\b|\bearly\b", t):
        p.part = "morning"
    if re.search(r"\bafternoon\b", t):
        p.part = "afternoon"
    if re.search(r"\b(evening|tonight|after work)\b", t):
        p.part = "evening"
    if re.search(r"\b(noon|midday|lunch ?time)\b", t):
        p.hour, p.minute = 12, 0

    # Clock time
    m = re.search(r"\bhalf past (\d{1,2})\b", t)
    if m:
        p.hour, p.minute = _to_24h(int(m.group(1)), 30, None)
    m = m or re.search(r"\bquarter past (\d{1,2})\b", t)
    if m and p.hour is None:
        p.hour, p.minute = _to_24h(int(m.group(1)), 15, None)
    if p.hour is None:
        for pat in (
            r"\b(\d{1,2})(?::|\s)(\d{2})\s*(am|pm)\b",
            r"\b(\d{1,2})\s*(am|pm)\b",
            r"\b(?:at|around|about|by)\s+(\d{1,2})(?::|\s)(\d{2})\b",
            r"\b(?:at|around|about|by)\s+(\d{1,2})(?:\s*o'?\s?clock)?\b(?!\s*(?:st|nd|rd|th|days?|weeks?))",
            r"\b(\d{1,2}):(\d{2})\b",
        ):
            m = re.search(pat, t)
            if m:
                groups = m.groups()
                hour = int(groups[0])
                minute = 0
                meridiem = None
                for g in groups[1:]:
                    if g in ("am", "pm"):
                        meridiem = g
                    elif g is not None and g.isdigit():
                        minute = int(g)
                if hour <= 23 and minute < 60:
                    if p.part in ("afternoon", "evening") and meridiem is None and hour < 12:
                        meridiem = "pm"
                    p.hour, p.minute = _to_24h(hour, minute, meridiem)
                break

    # Date
    if re.search(r"\bday after tomorrow\b", t):
        p.date = today + timedelta(days=2)
    elif re.search(r"\btomorrow\b", t):
        p.date = today + timedelta(days=1)
    elif re.search(r"\b(today|tonight|this (morning|afternoon|evening))\b", t):
        p.date = today
    m = re.search(r"\bin (\d{1,2}) days?\b", t)
    if m:
        p.date = today + timedelta(days=int(m.group(1)))
    elif re.search(r"\bin a week\b", t):
        p.date = today + timedelta(days=7)

    next_week = re.search(r"\bnext week\b", t)
    for i, name in enumerate(WEEKDAYS["en"]):
        if re.search(rf"\b{name}s?\b", t):
            if next_week:
                monday = _next_weekday(today, 0)
                p.date = monday + timedelta(days=i)
            else:
                p.date = _next_weekday(today, i)
            break
    else:
        if next_week:
            p.after = _next_weekday(today, 0)

    months = "|".join(sorted(MONTHS["en"], key=len, reverse=True))
    m = re.search(rf"\b({months})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", t)
    if m:
        p.date = _resolve_day_month(int(m.group(2)), MONTHS["en"][m.group(1)], today)
    else:
        m = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({months})\b", t)
        if m:
            p.date = _resolve_day_month(int(m.group(1)), MONTHS["en"][m.group(2)], today)
        else:
            m = re.search(r"\bthe (\d{1,2})(?:st|nd|rd|th)\b", t)
            if m:
                p.date = _resolve_day_of_month(int(m.group(1)), today)
            else:
                m = re.search(r"\b(\d{1,2})/(\d{1,2})\b", t)
                if m:
                    p.date = _resolve_day_month(int(m.group(2)), int(m.group(1)), today)
    return p


def _parse_es(t: str, today: date, p: TimePref) -> TimePref:
    # "mañana" means both "tomorrow" and "morning"; remove the morning phrases first.
    if re.search(r"\besta manana\b", t):
        p.part, p.date = "morning", today
        t = re.sub(r"\besta manana\b", " ", t)
    if re.search(r"\b(por|en|de) la manana\b|\btemprano\b", t):
        p.part = "morning"
        t = re.sub(r"\b(por|en|de) la manana\b", " ", t)
    if re.search(r"\besta tarde\b", t):
        p.part, p.date = "afternoon", today
    elif re.search(r"\b(por|en|de) la tarde\b", t):
        p.part = "afternoon"
    if re.search(r"\besta noche\b", t):
        p.part, p.date = "evening", today
    elif re.search(r"\b(por|en|de) la noche\b", t):
        p.part = "evening"
    if re.search(r"\b(al )?mediodia\b", t):
        p.hour, p.minute = 12, 0

    m = re.search(
        r"\ba la(?:s)? (\d{1,2})(?:(?::|\s)(\d{2}))?(?:\s+y\s+(media|cuarto))?"
        r"(?:\s+(?:de la|del)\s+(manana|tarde|noche|mediodia))?", t)
    if not m:
        m = re.search(r"\b(\d{1,2}):(\d{2})()()", t)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2)) if m.group(2) else 0
        if m.group(3) == "media":
            minute = 30
        elif m.group(3) == "cuarto":
            minute = 15
        meridiem = {"manana": "am", "tarde": "pm", "noche": "pm"}.get(m.group(4) or "")
        if meridiem is None and p.part in ("afternoon", "evening") and hour < 12:
            meridiem = "pm"
        if hour <= 23 and minute < 60:
            p.hour, p.minute = _to_24h(hour, minute, meridiem)

    if re.search(r"\bpasado manana\b", t):
        p.date = today + timedelta(days=2)
    elif re.search(r"\bmanana\b", t):
        p.date = today + timedelta(days=1)
    elif re.search(r"\bhoy\b", t):
        p.date = today
    m = re.search(r"\ben (\d{1,2}) dias\b", t)
    if m:
        p.date = today + timedelta(days=int(m.group(1)))
    elif re.search(r"\ben una semana\b", t):
        p.date = today + timedelta(days=7)

    next_week = re.search(r"\b(la )?(proxima semana|semana que viene|semana proxima)\b", t)
    for i, name in enumerate(WEEKDAYS["es"]):
        if re.search(rf"\b{name}\b", t):
            if next_week:
                p.date = _next_weekday(today, 0) + timedelta(days=i)
            else:
                p.date = _next_weekday(today, i)
            break
    else:
        if next_week:
            p.after = _next_weekday(today, 0)

    months = "|".join(MONTHS["es"])
    m = re.search(rf"\b(\d{{1,2}}) de ({months})\b", t)
    if m:
        p.date = _resolve_day_month(int(m.group(1)), MONTHS["es"][m.group(2)], today)
    else:
        m = re.search(r"\bel (?:dia )?(\d{1,2})\b(?! de la)", t)
        if m and not re.search(r"\bel (?:dia )?\d{1,2}\s*(?:y|:)", t):
            p.date = _resolve_day_of_month(int(m.group(1)), today)
        else:
            m = re.search(r"\b(\d{1,2})/(\d{1,2})\b", t)
            if m:
                p.date = _resolve_day_month(int(m.group(1)), int(m.group(2)), today)
    return p


def has_time_words(text: str, lang: str, today: date) -> bool:
    return not parse_time_pref(text, lang, today).is_empty()

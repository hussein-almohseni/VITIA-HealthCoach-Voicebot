"""Appointment slot search and option selection (story 2)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from .dates import PART_WINDOWS, TimePref, WEEKDAYS
from .textnorm import fold, normalize_for_parsing

PROVIDER_NAMES = {
    "patel": {"en": "Dr. Patel", "es": "la doctora Patel"},
    "ruiz": {"en": "your health coach Ana Ruiz", "es": "tu coach de salud Ana Ruiz"},
}
PROVIDER_PATTERNS = {
    "patel": r"\b(patel|doctor|doctora|dr|dra|medico|medica|physician)\b",
    "ruiz": r"\b(ruiz|ana|coach|health coach|entrenadora)\b",
}

SEARCH_DAYS = 28
MIN_LEAD = timedelta(hours=1)


@dataclass
class Option:
    slot_id: int
    provider: str
    start: datetime

    def to_dict(self) -> dict:
        return {"slot_id": self.slot_id, "provider": self.provider, "start": self.start.isoformat()}

    @classmethod
    def from_dict(cls, d: dict) -> "Option":
        return cls(d["slot_id"], d["provider"], datetime.fromisoformat(d["start"]))


def provider_name(provider: str, lang: str) -> str:
    return PROVIDER_NAMES[provider][lang]


def detect_provider(text: str) -> str | None:
    t = fold(text)
    for provider, pat in PROVIDER_PATTERNS.items():
        if re.search(pat, t):
            return provider
    return None


def _window(pref: TimePref) -> tuple[int, int] | None:
    if pref.hour is not None:
        start = pref.hour * 60 + pref.minute
        return start, start + 1
    if pref.part:
        return PART_WINDOWS[pref.part]
    return None


def find_options(db, pref: TimePref, now: datetime, provider: str | None = None,
                 n: int = 3) -> tuple[list[Option], bool]:
    """Returns (options, matched).

    matched=True means the options fit the request. Otherwise they are the closest
    openings to it, which the reply presents as alternatives.
    """
    earliest = now + MIN_LEAD
    rows = db.free_slots(earliest, now + timedelta(days=SEARCH_DAYS), provider)
    free = [Option(r["id"], r["provider"], datetime.fromisoformat(r["start_at"])) for r in rows]
    if not free:
        return [], False

    window = _window(pref)
    first_day = pref.date or pref.after

    def fits(o: Option) -> bool:
        if pref.date and o.start.date() != pref.date:
            return False
        if pref.after and o.start.date() < pref.after:
            return False
        if window:
            m = o.start.hour * 60 + o.start.minute
            return window[0] <= m < window[1]
        return True

    def dedupe(opts: list[Option]) -> list[Option]:
        seen, out = set(), []
        for o in opts:
            if o.start in seen:
                continue
            seen.add(o.start)
            out.append(o)
        return out

    matching = dedupe([o for o in free if fits(o)])
    if matching:
        if pref.date and window is None:
            # Whole day requested: spread the options across the day.
            step = max(1, len(matching) // n)
            return matching[::step][:n], True
        return matching[:n], True

    # Nothing fits: rank by distance from the requested time.
    if first_day:
        target_minutes = (window[0] if window else 9 * 60)
        target = datetime.combine(first_day, datetime.min.time()) + timedelta(minutes=target_minutes)
    elif window:
        target = datetime.combine(now.date(), datetime.min.time()) + timedelta(minutes=window[0])
    else:
        target = earliest

    def distance(o: Option) -> float:
        delta = (o.start - target).total_seconds()
        # Prefer later over earlier: people can rarely come sooner.
        return abs(delta) * (1.5 if delta < 0 else 1.0)

    def in_window(o: Option) -> bool:
        m = o.start.hour * 60 + o.start.minute
        return window is None or window[0] <= m < window[1]

    picks: list[Option] = []
    if pref.date:
        # One option on the requested day (closest time), then the same time of
        # day on the following days.
        same_day = sorted([o for o in free if o.start.date() == pref.date], key=distance)
        picks += same_day[:1]
        later = sorted([o for o in free if o.start.date() > pref.date and in_window(o)],
                       key=lambda o: o.start)
        picks += later[: n - len(picks)]
    if len(picks) < n:
        rest = sorted([o for o in free if o not in picks], key=distance)
        picks += rest[: n - len(picks)]
    return sorted(dedupe(picks)[:n], key=lambda o: o.start), False


ORDINALS = {
    "en": [
        (r"\b(first|1st|number 1|option 1)\b", 0),
        (r"\b(second|2nd|number 2|option 2)\b", 1),
        (r"\b(third|3rd|number 3|option 3)\b", 2),
        (r"\b(last one|the last|latest one|later one)\b", -1),
        (r"\b(earliest|earlier one)\b", 0),
    ],
    "es": [
        (r"\b(primer[oa]?|numero 1|opcion 1)\b", 0),
        (r"\b(segund[oa]|numero 2|opcion 2)\b", 1),
        (r"\b(tercer[oa]?|numero 3|opcion 3)\b", 2),
        (r"\b(ultim[oa]|la de mas tarde)\b", -1),
        (r"\b(mas temprano|la de antes)\b", 0),
    ],
}


def choose_option(text: str, options: list[Option], lang: str) -> Option | None:
    """Matches 'the second one', 'Wednesday at 2', 'la de las 3' against the options."""
    t = fold(text)
    for pat, idx in ORDINALS[lang]:
        if re.search(pat, t):
            if idx == -1:
                return options[-1]
            if idx < len(options):
                return options[idx]
    tn = normalize_for_parsing(text, lang)
    weekday = None
    for i, name in enumerate(WEEKDAYS[lang]):
        if re.search(rf"\b{name}\b", tn):
            weekday = i
    hours = [int(h) for h in re.findall(r"\b(\d{1,2})(?::\d{2})?\b", tn)]
    minutes_30 = bool(re.search(r"\b\d{1,2}:30\b|\b\d{1,2} 30\b|half past|y media", tn))
    candidates = []
    for o in options:
        if weekday is not None and o.start.weekday() != weekday:
            continue
        if hours:
            h12 = o.start.hour % 12 or 12
            if h12 not in hours and o.start.hour not in hours:
                continue
            if minutes_30 != (o.start.minute == 30) and any(
                    (x.start.hour == o.start.hour and x is not o) for x in options):
                continue
        if weekday is None and not hours:
            continue
        candidates.append(o)
    return candidates[0] if len(candidates) == 1 else None

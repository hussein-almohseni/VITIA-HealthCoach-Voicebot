"""Rule-based measurement parsing, unit normalization and plausibility checks (story 1).

The parser runs before the language model. When it finds a measurement, no model call
is needed, which keeps the most common turn fast. When it finds nothing, the router
falls back to Gemma, and `from_values` builds the same Reading from the model's output.

Plausibility ranges are input checks for catching misheard numbers. They are not
clinical thresholds and the app never tells the patient whether a value is good or bad.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .textnorm import normalize_for_parsing

METRICS = ("blood_pressure", "weight", "glucose", "heart_rate")

# Keywords per metric, on accent-folded text.
METRIC_PATTERNS = {
    "blood_pressure": {
        "en": r"\b(blood pressure|bp|b\.p\.|pressure)\b",
        "es": r"\b(presion( arterial)?|tension( arterial)?)\b",
    },
    "weight": {
        "en": r"\b(weight|weigh|weighed|weighs|weighing|scale)\b",
        "es": r"\b(peso|pese|pesaba|peso es|bascula)\b",
    },
    "glucose": {
        "en": r"\b((blood )?glucose|(blood )?sugar|glycemia)\b",
        "es": r"\b(glucosa|azucar( en (la )?sangre)?|glucemia)\b",
    },
    "heart_rate": {
        "en": r"\b(heart ?rate|pulse|heartbeat|heart beat|bpm)\b",
        "es": r"\b(frecuencia cardiaca|pulso|ritmo cardiaco|latidos|pulsaciones)\b",
    },
}

UNIT_PATTERNS = {
    "mmHg": r"(mm ?hg|millimeters? of mercury|milimetros de mercurio)",
    "lb": r"(pounds?|lbs?|libras?)\b",
    "kg": r"(kilos?|kilograms?|kgs?|kilogramos?)\b",
    "mg/dL": r"(mg ?/ ?dl|mg per dl|milligrams? per deciliter|miligramos( por decilitro)?)",
    "mmol/L": r"(mmol( ?/ ?l)?|millimoles?( per liter)?|milimoles?( por litro)?)",
    "bpm": r"(bpm|beats?( per minute)?|latidos( por minuto)?|pulsaciones( por minuto)?)",
}

CANONICAL_UNIT = {
    "blood_pressure": "mmHg",
    "weight": "kg",
    "glucose": "mg/dL",
    "heart_rate": "bpm",
}

# (min, max) in canonical units. Values outside are treated as misheard.
PLAUSIBLE = {
    "systolic": (60, 260),
    "diastolic": (30, 160),
    "weight": (20, 350),
    "glucose": (20, 600),
    "heart_rate": (25, 250),
}

LB_PER_KG = 2.20462
MGDL_PER_MMOL = 18.016

NUM = r"(\d{1,3}(?:\.\d+)?)"
# "1 20" spoken as "one twenty" means 120.
COLLOQUIAL_HUNDREDS = r"\b([1-2]) (\d{2})\b"
BP_SEPARATOR = r"\s*(?:/|over|sobre|con|entre|por|x|-)\s*"
NOT_DURATION = (r"(?!\s*(?:days?|weeks?|months?|years?|hours?|minutes?|mins?|times?|"
                r"dias?|semanas?|meses|mes|anos?|horas?|minutos?|veces)\b)")


@dataclass
class Reading:
    metric: str
    unit: str
    value: float | None = None          # weight, glucose, heart rate
    systolic: int | None = None         # blood pressure
    diastolic: int | None = None
    measured_at: datetime | None = None
    when_text: str = ""
    unit_inferred: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def canonical_value(self) -> float | None:
        if self.metric == "weight" and self.value is not None:
            return round(self.value / LB_PER_KG, 2) if self.unit == "lb" else self.value
        if self.metric == "glucose" and self.value is not None:
            return round(self.value * MGDL_PER_MMOL, 1) if self.unit == "mmol/L" else self.value
        return self.value

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "unit": self.unit,
            "value": self.value,
            "systolic": self.systolic,
            "diastolic": self.diastolic,
            "measured_at": self.measured_at.isoformat() if self.measured_at else None,
            "when_text": self.when_text,
            "unit_inferred": self.unit_inferred,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Reading":
        r = cls(
            metric=d["metric"], unit=d["unit"], value=d.get("value"),
            systolic=d.get("systolic"), diastolic=d.get("diastolic"),
            when_text=d.get("when_text", ""), unit_inferred=d.get("unit_inferred", False),
        )
        if d.get("measured_at"):
            r.measured_at = datetime.fromisoformat(d["measured_at"])
        return r


def _unit_near(text: str, start: int, window: int = 30) -> str | None:
    snippet = text[start:start + window]
    for unit, pat in UNIT_PATTERNS.items():
        if re.match(r"\s*" + pat, snippet):
            return unit
    return None


def _join_colloquial(text: str) -> str:
    """'1 20 over 80' -> '120 over 80'; '1 85 pounds' -> '185 pounds'."""
    return re.sub(COLLOQUIAL_HUNDREDS, lambda m: m.group(1) + m.group(2), text)


def parse_when(text: str, lang: str, now: datetime) -> tuple[datetime, str]:
    """Resolves when a reading was taken. Defaults to now."""
    t = text
    rules_en = [
        (r"\blast night\b", lambda: (now - timedelta(days=1)).replace(hour=21, minute=0)),
        (r"\byesterday morning\b", lambda: (now - timedelta(days=1)).replace(hour=8, minute=0)),
        (r"\byesterday evening\b", lambda: (now - timedelta(days=1)).replace(hour=19, minute=0)),
        (r"\byesterday\b", lambda: now - timedelta(days=1)),
        (r"\bthis morning\b", lambda: min(now, now.replace(hour=8, minute=0))),
        (r"\bthis afternoon\b", lambda: min(now, now.replace(hour=14, minute=0))),
        (r"\b(an|1) hour ago\b", lambda: now - timedelta(hours=1)),
        (r"\b(\d+) hours ago\b", None),
    ]
    rules_es = [
        (r"\banoche\b", lambda: (now - timedelta(days=1)).replace(hour=21, minute=0)),
        (r"\bayer (por|en) la manana\b", lambda: (now - timedelta(days=1)).replace(hour=8, minute=0)),
        (r"\bayer (por|en) la (tarde|noche)\b", lambda: (now - timedelta(days=1)).replace(hour=19, minute=0)),
        (r"\bayer\b", lambda: now - timedelta(days=1)),
        (r"\besta manana\b", lambda: min(now, now.replace(hour=8, minute=0))),
        (r"\besta tarde\b", lambda: min(now, now.replace(hour=14, minute=0))),
        (r"\bhace (una|1) hora\b", lambda: now - timedelta(hours=1)),
        (r"\bhace (\d+) horas\b", None),
    ]
    for pat, fn in rules_en if lang == "en" else rules_es:
        m = re.search(pat, t)
        if m:
            phrase = _restore_spanish(m.group(0)) if lang == "es" else m.group(0)
            if fn is None:
                return now - timedelta(hours=int(m.group(1))), phrase
            return fn().replace(second=0, microsecond=0), phrase
    return now.replace(second=0, microsecond=0), ""


def _restore_spanish(phrase: str) -> str:
    """Puts back the ñ that accent folding removed, so the voice pronounces it."""
    return re.sub(r"\bmanana\b", "mañana", phrase)


def parse_measurements(text: str, lang: str, now: datetime,
                       default_weight_unit: str = "lb") -> list[Reading]:
    """Finds all measurements in an utterance. Returns [] when there are none."""
    t = _join_colloquial(normalize_for_parsing(text, lang))
    measured_at, when_text = parse_when(t, lang, now)
    readings: list[Reading] = []
    used_spans: list[tuple[int, int]] = []

    def overlaps(a: int, b: int) -> bool:
        return any(a < y and x < b for x, y in used_spans)

    # Blood pressure first, because its numbers come in pairs.
    bp_kw = re.search(METRIC_PATTERNS["blood_pressure"][lang], t)
    bp_re = re.compile(r"\b(\d{2,3})" + BP_SEPARATOR + r"(\d{2,3})\b")
    for m in bp_re.finditer(t):
        # A pair counts as blood pressure when the keyword is present, or when it
        # uses a separator that only blood pressure uses ("120/80", "120 over 80").
        if bp_kw or re.search(r"/|\bover\b|\bsobre\b", m.group(0)):
            readings.append(Reading(
                metric="blood_pressure", unit="mmHg",
                systolic=int(m.group(1)), diastolic=int(m.group(2)),
                measured_at=measured_at, when_text=when_text,
            ))
            used_spans.append(m.span())
            break

    for metric in ("weight", "glucose", "heart_rate"):
        kw = re.search(METRIC_PATTERNS[metric][lang], t)
        if not kw:
            continue
        # Number after the keyword (within a few words), else just before it.
        # Numbers that are durations ("last 2 weeks") are not readings.
        after = re.compile(r"(?:\D{0,25}?)\b" + NUM + r"\b" + NOT_DURATION)
        m = after.match(t, kw.end())
        span_start = None
        if m and not overlaps(m.start(1), m.end(1)):
            num, span_start, span_end = m.group(1), m.start(1), m.end(1)
        else:
            before = list(re.finditer(r"\b" + NUM + r"\b" + NOT_DURATION + r"(?=\D{0,20}$)",
                                      t[:kw.start()]))
            if not before:
                continue
            mb = before[-1]
            if overlaps(mb.start(1), mb.end(1)):
                continue
            num, span_start, span_end = mb.group(1), mb.start(1), mb.end(1)
        value = float(num)
        unit = _unit_near(t, span_end) or _unit_near(t, kw.end())
        inferred = False
        if metric == "weight" and unit not in ("lb", "kg"):
            unit, inferred = default_weight_unit, True
        elif metric == "glucose" and unit not in ("mg/dL", "mmol/L"):
            # Values below 35 are almost certainly mmol/L; the read-back states
            # the unit so the patient can correct it.
            unit, inferred = ("mmol/L" if value < 35 else "mg/dL"), True
        elif metric == "heart_rate":
            unit = "bpm"
        readings.append(Reading(
            metric=metric, unit=unit, value=value, measured_at=measured_at,
            when_text=when_text, unit_inferred=inferred,
        ))
        used_spans.append((span_start, span_end))

    return readings


def from_values(metric: str, values: list[float], unit: str | None, lang: str,
                now: datetime, text: str = "", default_weight_unit: str = "lb") -> Reading | None:
    """Builds a Reading from the language model's extraction (fallback path)."""
    if metric not in METRICS or not values:
        return None
    measured_at, when_text = parse_when(normalize_for_parsing(text, lang), lang, now)
    unit = (unit or "").strip()
    norm_unit = None
    for u, pat in UNIT_PATTERNS.items():
        if unit and re.fullmatch(pat, unit.lower()):
            norm_unit = u
    if metric == "blood_pressure":
        if len(values) < 2:
            return None
        return Reading("blood_pressure", "mmHg", systolic=int(values[0]),
                       diastolic=int(values[1]), measured_at=measured_at, when_text=when_text)
    value = float(values[0])
    inferred = False
    if metric == "weight":
        if norm_unit not in ("lb", "kg"):
            norm_unit, inferred = default_weight_unit, True
    elif metric == "glucose":
        if norm_unit not in ("mg/dL", "mmol/L"):
            norm_unit, inferred = ("mmol/L" if value < 35 else "mg/dL"), True
    else:
        norm_unit = "bpm"
    return Reading(metric, norm_unit, value=value, measured_at=measured_at,
                   when_text=when_text, unit_inferred=inferred)


def correction(text: str, previous: Reading, lang: str, now: datetime) -> Reading | None:
    """'No, it was 128 pounds' while confirming a weight: same metric, new numbers."""
    t = _join_colloquial(normalize_for_parsing(text, lang))
    nums = [float(n) for n in re.findall(r"\b" + NUM + r"\b" + NOT_DURATION, t)]
    if previous.metric == "blood_pressure":
        m = re.search(r"\b(\d{2,3})" + BP_SEPARATOR + r"(\d{2,3})\b", t)
        if not m:
            return None
        nums = [float(m.group(1)), float(m.group(2))]
    elif not nums:
        return None
    unit = None
    for u, pat in UNIT_PATTERNS.items():
        if re.search(pat, t):
            unit = u
    if unit is None and previous.metric in ("weight", "glucose") and not previous.unit_inferred:
        unit = previous.unit
    r = from_values(previous.metric, nums, unit, lang, now, text,
                    default_weight_unit=previous.unit if previous.metric == "weight" else "lb")
    if r is not None and not r.when_text:
        r.measured_at, r.when_text = previous.measured_at, previous.when_text
    return r


def validate(r: Reading) -> str | None:
    """Returns None when plausible, or the name of the failing check."""
    if r.metric == "blood_pressure":
        lo, hi = PLAUSIBLE["systolic"]
        if r.systolic is None or not lo <= r.systolic <= hi:
            return "systolic_range"
        lo, hi = PLAUSIBLE["diastolic"]
        if r.diastolic is None or not lo <= r.diastolic <= hi:
            return "diastolic_range"
        if r.systolic <= r.diastolic:
            return "bp_order"
        return None
    lo, hi = PLAUSIBLE[r.metric]
    v = r.canonical_value
    if v is None or not lo <= v <= hi:
        return f"{r.metric}_range"
    return None


def detect_metric(text_folded: str, lang: str) -> str | None:
    for metric in METRICS:
        if re.search(METRIC_PATTERNS[metric][lang], text_folded):
            return metric
    return None

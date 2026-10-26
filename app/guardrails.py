"""No-diagnosis guardrails (story 6).

Four layers, so no single one has to be perfect:
  1. Pattern checks on the patient's words, before any model runs (this module).
  2. The language-model router can also label a turn `clinical_advice` (router.py).
  3. Plan answers come only from retrieved plan text; the model must answer
     NOT_IN_PLAN or REFUSE otherwise (dialogue.py).
  4. Every generated sentence is checked here before it is spoken: advice-like
     phrasing and numbers that do not appear in the plan excerpt are blocked.

Refusals and emergency messages are fixed templates (templates.py), never generated.
The patterns are a starting set for the prototype; the clinical team should review and
extend them, and the test file lists the phrasings they are expected to catch.
"""

from __future__ import annotations

import re

from .textnorm import fold, normalize_for_parsing

SELF_HARM = {
    "en": r"\b(kill myself|end my life|suicid\w*|hurt myself|harm myself|don'?t want to (live|be alive)|want to die)\b",
    "es": r"\b(matarme|quitarme la vida|suicid\w*|hacerme dano|no quiero vivir|quiero morir(me)?)\b",
}

EMERGENCY = {
    "en": r"\b(chest pain|pain in my chest|can'?t breathe|cannot breathe|trouble breathing|"
          r"short(ness)? of breath|passed out|fainted|fainting|stroke|face (is )?drooping|"
          r"slurred speech|can'?t feel my (arm|leg|face)|severe bleeding|unconscious|seizure)\b",
    "es": r"\b(dolor (de|en el) pecho|no puedo respirar|me falta el aire|falta de aire|"
          r"me desmaye|desmayo|derrame( cerebral)?|infarto|ataque al corazon|"
          r"no siento (el|la|mi) (brazo|pierna|cara)|convulsion|sangrado (fuerte|abundante)|inconsciente)\b",
}

# Questions asking for a diagnosis, an interpretation of a reading or symptom,
# or a treatment decision.
CLINICAL = {
    "en": [
        # "Do I have diabetes?" but not "how many sessions do I have left?"
        r"^\s*(do|could|might) i have\b(?! (an?|my|any) (appointments?|visits?|sessions?|plan|copay))",
        r"\bi (might|may|could) have\b",
        r"\bwhat('?s| is) wrong with me\b",
        r"\bdiagnos\w*\b",
        r"\b(is|was|are) (it|this|that|my \w+( \w+)?|these|those|\d+( over \d+)?) "
        r"(bad|normal|high|low|serious|dangerous|healthy|concerning|too high|too low)\b",
        r"\bshould i (be )?(worried|concerned)\b",
        r"\b(should|can|could) i (take|stop|start|increase|decrease|double|skip|lower|raise|change|switch)\b",
        r"\bhow (much|many) .{0,30}should i take\b",
        r"\b(dose|dosage)\b",
        r"\b(what|which) (medicine|medication|pill|drug|treatment) (should|can|could)\b",
        r"\b(treat|cure|get rid of|medicine for|pill for|remedy for)\b",
        r"\bwhat does (it|this|that|my \w+( \w+)?) (mean|indicate)\b",
        r"\bwhy (do|am|is) (i|my)\b.*\b(hurt|pain|dizzy|tired|swollen|high|low|sick)\b",
    ],
    "es": [
        r"\b(tengo|tendre|tendria) (diabetes|cancer|covid|una infeccion|hipertension|algo)\b",
        r"\bque (tengo|me pasa)\b",
        r"\bdiagnostic\w*\b",
        r"\b(es|esta|son|fue) (malo|mala|normal|grave|alta|alto|baja|bajo|peligros[oa]|preocupante)\b",
        r"\b(debo|deberia|puedo) (preocuparme)\b",
        r"\b(debo|deberia|puedo|tengo que) (tomar|dejar|empezar|aumentar|bajar|subir|duplicar|saltar|cambiar)\b",
        r"\bcuant[oa]s? .{0,30}(debo|tengo que) tomar\b",
        r"\b(dosis)\b",
        r"\b(que|cual) (medicina|medicamento|pastilla|tratamiento) (debo|puedo|deberia)\b",
        r"\b(tratar|curar|quitarme|medicina para|pastilla para|remedio para)\b",
        r"\bque significa\b",
        r"\bpor que (me duele|estoy mareado|estoy mareada|me siento)\b",
    ],
}

# Asking what the plan already says is allowed ("when should I take my lisinopril?",
# "what's my dose?"); asking whether to change it is not.
PLAN_SCHEDULE_QUESTION = {
    "en": r"\b(when|what time|how often|at what time)\b.{0,25}\b(should|do) i take\b"
          r"|\bwhat('?s| is) (the |my )(dose|dosage)\b",
    "es": r"\b(cuando|a que hora|cada cuanto|con que frecuencia)\b.{0,25}\b(debo|tengo que) tomar\b"
          r"|\bcual es (la|mi) dosis\b",
}

PLAN_CHANGE = {
    "en": r"\b(change|update|edit|modify|remove|add to|cancel|rewrite)\b.{0,20}\b(my )?(health |care |treatment )?plan\b",
    "es": r"\b(cambia|cambiar|actualiza|actualizar|modifica|modificar|edita|editar|quita|quitar|agrega|agregar)\b.{0,20}\b(mi )?plan\b",
}

# Advice-like phrasing in generated answers.
ADVICE_OUT = {
    "en": [
        r"\byou (should|must|need to|ought to|have to) (take|stop|start|increase|decrease|double|skip|avoid|see|go)\b",
        r"\b(you|this|that|it) (may|might|could|probably|likely) (have|be|mean|indicate)\b",
        r"\bdiagnos\w*\b",
        r"\b(i|we) (recommend|suggest|advise)\b",
        r"\bsounds like\b",
        r"\b(is|are|looks|seems|was) (normal|high|low|dangerous|concerning|fine|healthy|elevated|good|bad)\b",
    ],
    "es": [
        r"\b(debes|deberias|tienes que|necesitas) (tomar|dejar|empezar|aumentar|bajar|duplicar|saltar|evitar|ir)\b",
        r"\b(puede|podria|parece) (que )?(tengas|tienes|ser|sea|significar|indicar)\b",
        r"\bdiagnostic\w*\b",
        r"\b(te )?(recomiendo|sugiero|aconsejo)\b",
        r"\b(es|esta|son|parece) (normal|alta|alto|baja|bajo|peligros[oa]|preocupante|buena|bueno|mala|malo)\b",
    ],
}

PLAN_ATTRIBUTION = {
    "en": r"\b(your (health |care )?plan|the plan|plan says|according to)\b",
    "es": r"\b(tu plan|su plan|el plan|segun)\b",
}


def _any(patterns, text: str) -> bool:
    if isinstance(patterns, str):
        patterns = [patterns]
    return any(re.search(p, text) for p in patterns)


def is_self_harm(text: str, lang: str) -> bool:
    return _any(SELF_HARM[lang], fold(text))


def is_emergency(text: str, lang: str) -> bool:
    return _any(EMERGENCY[lang], fold(text))


def is_clinical_request(text: str, lang: str) -> bool:
    t = normalize_for_parsing(text, lang)
    if re.search(PLAN_SCHEDULE_QUESTION[lang], t):
        return False
    return _any(CLINICAL[lang], t)


def is_plan_change(text: str, lang: str) -> bool:
    return _any(PLAN_CHANGE[lang], fold(text))


def _numbers(text: str, lang: str) -> set[str]:
    out = set()
    for n in re.findall(r"\d+(?:\.\d+)?", normalize_for_parsing(text, lang)):
        out.add(n.rstrip("0").rstrip(".") if "." in n else n)
    return out


def check_generated_sentence(sentence: str, context: str, question: str,
                             lang: str) -> tuple[bool, str]:
    """Checks one generated sentence before it is spoken.

    Returns (ok, reason). Numbers must appear in the plan excerpt or the question.
    Advice-like phrasing is blocked unless the sentence attributes it to the plan
    (e.g. "Your plan says to take lisinopril every morning").
    """
    t = fold(sentence)
    if not re.search(PLAN_ATTRIBUTION[lang], t) and _any(ADVICE_OUT[lang], t):
        return False, "advice_phrasing"
    allowed = _numbers(context, lang) | _numbers(question, lang)
    for n in _numbers(sentence, lang):
        if n not in allowed:
            return False, f"ungrounded_number:{n}"
    return True, ""

"""Text normalization: accents, spoken numbers (English and Spanish) and sentence splitting.

Speech recognizers sometimes write numbers as words ("one twenty over eighty",
"ciento veinte sobre ochenta"). The parsers downstream work on digits, so this module
turns number words into digits without touching anything else.

Colloquial forms like "one twenty" are left as two numbers ("1 20"); the measurement
parser reads that as 120 and the time parser as 1:20, depending on context.
"""

from __future__ import annotations

import re
import unicodedata


def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def fold(text: str) -> str:
    """Lowercase, remove accents and Spanish opening punctuation."""
    return strip_accents(text.lower()).replace("¿", " ").replace("¡", " ")


# --- Number words -------------------------------------------------------------------

EN_SMALL = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
}
EN_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}

# Accent-folded Spanish forms. "un"/"una" are left out on purpose: they are articles
# far more often than numbers ("una cita").
ES_SMALL = {
    "cero": 0, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6,
    "siete": 7, "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12,
    "trece": 13, "catorce": 14, "quince": 15, "dieciseis": 16, "diecisiete": 17,
    "dieciocho": 18, "diecinueve": 19, "veinte": 20, "veintiuno": 21, "veintiun": 21,
    "veintidos": 22, "veintitres": 23, "veinticuatro": 24, "veinticinco": 25,
    "veintiseis": 26, "veintisiete": 27, "veintiocho": 28, "veintinueve": 29,
}
ES_TENS = {
    "treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60, "setenta": 70,
    "ochenta": 80, "noventa": 90,
}
ES_HUNDREDS = {
    "cien": 100, "ciento": 100, "doscientos": 200, "doscientas": 200,
    "trescientos": 300, "trescientas": 300, "cuatrocientos": 400,
    "cuatrocientas": 400, "quinientos": 500, "quinientas": 500,
    "seiscientos": 600, "seiscientas": 600, "setecientos": 700, "setecientas": 700,
    "ochocientos": 800, "ochocientas": 800, "novecientos": 900, "novecientas": 900,
}

_TOKEN_RE = re.compile(r"\w+|[^\w\s]|\s+", re.UNICODE)


def _is_space(tok: str) -> bool:
    return tok.isspace()


class _Cursor:
    """Walks word tokens while skipping whitespace."""

    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.folded = [fold(t) for t in tokens]

    def word_at(self, i: int) -> tuple[str | None, int]:
        """Returns (folded word, index) of the next non-space token at or after i."""
        while i < len(self.tokens) and _is_space(self.tokens[i]):
            i += 1
        if i >= len(self.tokens):
            return None, i
        return self.folded[i], i


def _parse_en(cur: _Cursor, i: int) -> tuple[float | None, int]:
    """Parses an English number starting at token i. Returns (value, next index)."""
    w, i = cur.word_at(i)
    value = 0
    found = False
    # "a hundred" / "one hundred (and)"
    if w == "a":
        w2, j = cur.word_at(i + 1)
        if w2 == "hundred":
            value, found, i = 100, True, j + 1
            w, i = cur.word_at(i)
    elif w in EN_SMALL and 1 <= EN_SMALL[w] <= 9:
        w2, j = cur.word_at(i + 1)
        if w2 == "hundred":
            value, found, i = EN_SMALL[w] * 100, True, j + 1
            w, i2 = cur.word_at(i)
            if w == "and":
                w3, _ = cur.word_at(i2 + 1)
                if w3 in EN_SMALL or w3 in EN_TENS:
                    i = i2 + 1
            w, i = cur.word_at(i)
    if w in EN_TENS:
        value += EN_TENS[w]
        found = True
        i += 1
        w2, j = cur.word_at(i)
        if w2 in EN_SMALL and 1 <= EN_SMALL[w2] <= 9:
            value += EN_SMALL[w2]
            i = j + 1
    elif w in EN_SMALL:
        value += EN_SMALL[w]
        found = True
        i += 1
    if not found:
        return None, i
    return _parse_decimal(cur, i, value, ("point",), EN_SMALL)


def _parse_es(cur: _Cursor, i: int) -> tuple[float | None, int]:
    w, i = cur.word_at(i)
    value = 0
    found = False
    if w in ES_HUNDREDS:
        value, found = ES_HUNDREDS[w], True
        i += 1
        w, i = cur.word_at(i)
    if w in ES_TENS:
        value += ES_TENS[w]
        found = True
        i += 1
        w2, j = cur.word_at(i)
        if w2 == "y":
            w3, k = cur.word_at(j + 1)
            if w3 in ES_SMALL and 1 <= ES_SMALL[w3] <= 9:
                value += ES_SMALL[w3]
                i = k + 1
    elif w in ES_SMALL:
        value += ES_SMALL[w]
        found = True
        i += 1
    if not found:
        return None, i
    return _parse_decimal(cur, i, value, ("punto", "coma"), ES_SMALL)


def _parse_decimal(cur, i, value, markers, small):
    w, j = cur.word_at(i)
    if w in markers:
        digits = ""
        k = j + 1
        while True:
            w2, k2 = cur.word_at(k)
            if w2 in small and small[w2] <= 9:
                digits += str(small[w2])
                k = k2 + 1
            elif w2 is not None and w2.isdigit() and not digits:
                digits = w2
                k = k2 + 1
                break
            else:
                break
        if digits:
            return float(f"{int(value)}.{digits}"), k
    return value, i


def words_to_digits(text: str, lang: str) -> str:
    """Replaces spoken numbers with digits, keeping all other text as is."""
    if lang == "en":
        text = re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", " ", text)
    tokens = _TOKEN_RE.findall(text)
    cur = _Cursor(tokens)
    parse = _parse_en if lang == "en" else _parse_es
    out: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if _is_space(tok):
            out.append(tok)
            i += 1
            continue
        value, j = parse(cur, i)
        if value is None:
            out.append(tok)
            i += 1
            continue
        out.append(str(int(value)) if float(value).is_integer() else str(value))
        i = j
    return "".join(out)


def normalize_for_parsing(text: str, lang: str) -> str:
    """Folded text with digits, ready for the rule-based parsers."""
    return fold(words_to_digits(text, lang))


# --- Sentences ----------------------------------------------------------------------

_ABBREVIATIONS = {"dr", "dra", "sr", "sra", "srta", "mr", "mrs", "ms", "st", "no"}


def split_sentences(text: str, final: bool = True) -> tuple[list[str], str]:
    """Splits text into complete sentences.

    Returns (sentences, remainder). With final=True the remainder is returned as the
    last sentence. A period counts as an ending only when followed by whitespace, so
    decimals such as 6.5 are kept together, and titles like "Dr." are skipped.
    """
    sentences: list[str] = []
    start = 0
    for m in re.finditer(r"[.!?]+(?=\s)", text):
        end = m.end()
        before = text[start:m.start()].rstrip()
        last_word = re.findall(r"\w+$", before)
        if m.group() == "." and last_word and last_word[0].lower() in _ABBREVIATIONS:
            continue
        sentence = text[start:end].strip()
        if sentence:
            sentences.append(sentence)
        start = end
    # Keep the remainder's spacing: streamed tokens are appended to it.
    remainder = text[start:].lstrip() if sentences else text[start:]
    if final:
        if remainder.strip():
            sentences.append(remainder.strip())
        remainder = ""
    return sentences, remainder

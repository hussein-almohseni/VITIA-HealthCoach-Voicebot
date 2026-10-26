"""Local retrieval over the patient's own health plan (story 3).

BM25 keyword search over plan sections, with no embedding model, so it runs offline
with nothing to download. The plan is stored per language; a Spanish question searches
the Spanish copy. For production, a multilingual embedding model (e.g. EmbeddingGemma)
would allow one plan in one language to answer questions in either.

Only the signed-in patient's plan is indexed, so answers cannot draw on anyone else's.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .textnorm import fold

STOPWORDS = {
    "en": set("""a an and are as at be by can could do does did for from has have how i
        if in is it its me my of on or our should so that the their them then there
        these this to was we what when where which who why will with you your about
        many much any am""".split()),
    "es": set("""a al algo como con cual cuales cuando de del donde el ella ellos en es
        esa ese eso esta este esto estos fue ha hay la las le les lo los me mi mis
        muy no o para pero por que se si sin sobre son su sus te tengo ti tu tus un
        una uno y ya puedo debo cuanto cuantos cuantas mucho""".split()),
}

# Small synonym map so everyday words find plan wording.
SYNONYMS = {
    "en": {
        "physio": "physiotherapy", "pt": "physiotherapy", "physical": "physiotherapy",
        "therapy": "physiotherapy", "pills": "medications", "pill": "medications",
        "meds": "medications", "medicine": "medications", "medication": "medications",
        "sugar": "glucose", "diet": "nutrition", "food": "nutrition", "eat": "nutrition",
        "dietitian": "nutrition", "cost": "copay", "pay": "copay", "price": "copay",
        "covered": "covers", "coverage": "covers", "cover": "covers",
        "walking": "walk", "workout": "exercise", "coach": "coach",
        "phone": "phone", "call": "contact", "weigh": "weight", "doctor": "dr",
        "costs": "copay",
    },
    "es": {
        "fisio": "fisioterapia", "terapia": "fisioterapia", "pastilla": "medicamentos",
        "pastillas": "medicamentos", "medicina": "medicamentos", "medicinas": "medicamentos",
        "medicamento": "medicamentos", "azucar": "glucosa", "dieta": "nutricion",
        "comida": "nutricion", "comer": "nutricion", "nutriologo": "nutricion",
        "costo": "copago", "pagar": "copago", "precio": "copago", "cubre": "cubre",
        "cobertura": "cubre", "cubierto": "cubre", "cubierta": "cubre",
        "caminar": "camina", "telefono": "telefono", "llamar": "llama", "pesarme": "pesate",
        "cuesta": "copago", "cuestan": "copago", "pago": "copago", "doctora": "dra",
        "doctor": "dra", "medico": "dra", "medica": "dra", "tomo": "tableta",
    },
}


def _stem(word: str) -> str:
    for suffix in ("es", "s"):
        if len(word) > 4 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def tokenize(text: str, lang: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", fold(text))
    out = []
    for w in words:
        if w in STOPWORDS[lang]:
            continue
        w = SYNONYMS[lang].get(w, w)
        out.append(_stem(w))
    return out


@dataclass
class Chunk:
    title: str
    text: str

    @property
    def full(self) -> str:
        return f"{self.title}: {self.text}"


@dataclass
class Hit:
    chunk: Chunk
    score: float


class PlanIndex:
    def __init__(self, markdown: str, lang: str, k1: float = 1.4, b: float = 0.75):
        self.lang = lang
        self.chunks = self._chunk(markdown)
        self.k1, self.b = k1, b
        self.docs = [tokenize(c.full, lang) for c in self.chunks]
        self.avgdl = sum(len(d) for d in self.docs) / max(len(self.docs), 1)
        df = Counter()
        for d in self.docs:
            df.update(set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    @staticmethod
    def _chunk(markdown: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        title = "Overview"
        lines: list[str] = []
        for line in markdown.splitlines():
            if line.startswith("## "):
                if lines:
                    chunks.append(Chunk(title, " ".join(lines).strip()))
                title, lines = line[3:].strip(), []
            elif line.startswith("# "):
                continue
            elif line.strip():
                lines.append(line.strip())
        if lines:
            chunks.append(Chunk(title, " ".join(lines).strip()))
        return chunks

    def search(self, query: str, k: int = 2) -> list[Hit]:
        q = tokenize(query, self.lang)
        hits = []
        for i, tf in enumerate(self.tf):
            dl = len(self.docs[i])
            score = 0.0
            for t in q:
                if t not in tf:
                    continue
                f = tf[t]
                score += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            if score > 0:
                hits.append(Hit(self.chunks[i], score))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

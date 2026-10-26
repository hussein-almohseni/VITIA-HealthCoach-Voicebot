"""Speech in (Moonshine) and speech out (Piper), plus fakes for tests.

ASR streams: audio is transcribed while the patient is still talking, so when they stop
only the last fraction of a second remains. Moonshine's built-in voice activity
detection marks a line complete after a pause, which ends the turn automatically.

TTS synthesizes one sentence at a time, so the first sentence plays while the rest is
still being generated.
"""

from __future__ import annotations

import io
import logging
import threading
import time
import wave
from pathlib import Path

import numpy as np

log = logging.getLogger("speech")

# Words the recognizer should prefer: medication names, units, care team names.
KEYTERMS = {
    "en": ["blood pressure", "glucose", "lisinopril", "metformin", "physiotherapy",
           "Dr. Patel", "Ana Ruiz", "mmHg", "milligrams per deciliter"],
    "es": ["presión arterial", "glucosa", "lisinopril", "metformina", "fisioterapia",
           "doctora Patel", "Ana Ruiz", "frecuencia cardíaca", "miligramos por decilitro"],
}


# --- ASR ----------------------------------------------------------------------------

class ASRSession:
    """One utterance. Feed audio with add(); call finish() for the final text."""

    def __init__(self, stream):
        self.stream = stream
        self.lines: dict[int, str] = {}
        self.completed = False
        self.line_end_audio_s: float | None = None  # where speech ended, in stream time
        self.audio_s = 0.0
        self.lock = threading.Lock()
        stream.add_listener(self._on_event)
        stream.start()

    def _on_event(self, event) -> None:
        line = getattr(event, "line", None)
        if line is None:
            return
        with self.lock:
            if line.text:
                self.lines[line.line_id] = line.text
            if type(event).__name__ == "LineCompleted" and line.text.strip():
                self.completed = True
                self.line_end_audio_s = line.start_time + line.duration

    def add(self, samples: np.ndarray, sample_rate: int = 16000) -> None:
        self.audio_s += len(samples) / sample_rate
        self.stream.add_audio(samples.tolist(), sample_rate)

    @property
    def partial(self) -> str:
        with self.lock:
            return " ".join(self.lines[k] for k in sorted(self.lines)).strip()

    def finish(self) -> str:
        transcript = self.stream.stop()
        if transcript is not None:
            for line in transcript.lines:
                if line.text:
                    self.lines[line.line_id] = line.text
                    if line.is_complete:
                        self.line_end_audio_s = line.start_time + line.duration
        try:
            self.stream.close()
        except Exception:  # noqa: BLE001
            pass
        return self.partial


class MoonshineASR:
    def __init__(self, models: dict[str, tuple[str, object]]):
        """models: lang -> (model_path, ModelArch), as returned by get_model_for_language."""
        from moonshine_voice import Transcriber

        self.transcribers = {}
        for lang, (path, arch) in models.items():
            tr = Transcriber(model_path=path, model_arch=arch, update_interval=0.25)
            try:
                tr.set_keyterms(KEYTERMS[lang])
            except Exception as e:  # noqa: BLE001
                log.warning("Keyterm biasing unavailable for %s: %s", lang, e)
            self.transcribers[lang] = tr
        self.locks = {lang: threading.Lock() for lang in models}

    def session(self, lang: str) -> ASRSession:
        return ASRSession(self.transcribers[lang].create_stream(update_interval=0.25))


class _FakeStream:
    def __init__(self, text: str):
        self.text = text
        self.listeners = []

    def add_listener(self, fn):
        self.listeners.append(fn)

    def start(self):
        pass

    def add_audio(self, samples, sample_rate):
        pass

    def stop(self):
        return None

    def close(self):
        pass


class _FakeSession(ASRSession):
    """Hears `text` once 1 s of audio has arrived, then reports a pause."""

    def __init__(self, text: str):
        super().__init__(_FakeStream(text))
        self.text = text

    def add(self, samples, sample_rate: int = 16000) -> None:
        self.audio_s += len(samples) / sample_rate
        if self.text and self.audio_s >= 1.0 and not self.completed:
            self.lines[0] = self.text
            self.completed = True
            self.line_end_audio_s = self.audio_s - 0.4


class FakeASR:
    """Stand-in recognizer for tests.

    `next_text` is returned on stop. `script` (or DEMO_FAKE_SCRIPT, utterances
    separated by '|') feeds one utterance per listening session, with simulated pause
    detection; when it runs out, sessions hear nothing.
    """

    def __init__(self, script: list[str] | None = None):
        import os

        self.next_text = ""
        env = os.getenv("DEMO_FAKE_SCRIPT")
        self.script = script if script is not None else ([s.strip() for s in env.split("|")] if env else [])

    def session(self, lang: str) -> ASRSession:
        if self.script:
            return _FakeSession(self.script.pop(0))
        s = _FakeSession("")
        if self.next_text:
            s.lines[0] = self.next_text
        return s


# --- TTS ----------------------------------------------------------------------------

def to_wav_bytes(samples_int16: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(samples_int16)
    return buf.getvalue()


class PiperTTS:
    def __init__(self, voices: dict[str, Path]):
        from piper import PiperVoice

        self.voices = {lang: PiperVoice.load(str(path)) for lang, path in voices.items()}
        self.locks = {lang: threading.Lock() for lang in voices}

    def synth(self, text: str, lang: str) -> bytes:
        voice = self.voices[lang]
        with self.locks[lang]:
            chunks = list(voice.synthesize(text))
        if not chunks:
            return to_wav_bytes(b"", 22050)
        audio = b"".join(c.audio_int16_bytes for c in chunks)
        return to_wav_bytes(audio, chunks[0].sample_rate)

    def warmup(self) -> None:
        for lang in self.voices:
            self.synth("Hello." if lang == "en" else "Hola.", lang)


class FakeTTS:
    def __init__(self, delay_s: float = 0.0):
        self.delay_s = delay_s
        self.spoken: list[str] = []

    def synth(self, text: str, lang: str) -> bytes:
        self.spoken.append(text)
        time.sleep(self.delay_s)
        return to_wav_bytes(b"\x00\x00" * 1600, 16000)

    def warmup(self) -> None:
        pass

"""Settings, overridable with environment variables."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = Path(os.getenv("DEMO_MODELS_DIR", ROOT / "models"))
STATIC_DIR = ROOT / "static"
DB_PATH = Path(os.getenv("DEMO_DB", ROOT / "demo.db"))

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
LLM_MODEL = os.getenv("DEMO_LLM", "gemma3:4b")

# Moonshine model size per language: "small" (default) or "tiny" for slower machines.
ASR_SIZE = os.getenv("DEMO_ASR_SIZE", "small")

PIPER_VOICES = {
    "en": os.getenv("DEMO_VOICE_EN", "en_US-lessac-medium"),
    "es": os.getenv("DEMO_VOICE_ES", "es_MX-ald-medium"),
}

BUS_DELAY_S = float(os.getenv("DEMO_BUS_DELAY", "120"))
FAKE_MODELS = os.getenv("DEMO_FAKE") == "1"
HOST = os.getenv("DEMO_HOST", "127.0.0.1")
PORT = int(os.getenv("DEMO_PORT", "8000"))


def moonshine_arch(size: str):
    from moonshine_voice import ModelArch

    return {"small": ModelArch.SMALL_STREAMING, "tiny": ModelArch.TINY_STREAMING}[size]

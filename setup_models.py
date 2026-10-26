"""Downloads the speech models, checks Ollama, and runs a quick self-test.

Usage:  python setup_models.py            (download + test)
        python setup_models.py --test     (test only)

What it fetches:
  - Moonshine speech-to-text, English and Spanish (from download.moonshine.ai)
  - Piper voices, English (US) and Spanish (Mexico) (from huggingface.co)
  - Gemma 3 through Ollama, if it is not already pulled
Everything lands in ./models except Gemma, which Ollama stores itself.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import config  # noqa: E402

OK, FAIL = "  [ok]", "  [!!]"


def step(title: str) -> None:
    print(f"\n== {title}")


def download_moonshine() -> dict:
    from moonshine_voice import get_model_for_language

    step(f"Moonshine speech-to-text ({config.ASR_SIZE}-streaming)")
    print("  Note: Moonshine's Spanish model uses the non-commercial Moonshine Community License.")
    print("  Fine for this demo; a commercial product needs a license from Moonshine AI.")
    arch = config.moonshine_arch(config.ASR_SIZE)
    models = {}
    for lang in ("en", "es"):
        path, model_arch = get_model_for_language(lang, arch, cache_root=config.MODELS_DIR / "moonshine")
        models[lang] = (path, model_arch)
        print(f"{OK} {lang}: {path}")
    # Record where the models are, relative to ./models, so a copied or portable
    # folder can load them without asking the downloader (and without internet).
    import json

    index = {lang: {"path": Path(p).resolve().relative_to(config.MODELS_DIR.resolve()).as_posix(),
                    "arch": a.name}
             for lang, (p, a) in models.items()}
    (config.MODELS_DIR / "moonshine_paths.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return models


def download_piper() -> dict:
    from piper.download_voices import download_voice

    step("Piper voices")
    out = config.MODELS_DIR / "piper"
    out.mkdir(parents=True, exist_ok=True)
    voices = {}
    for lang, name in config.PIPER_VOICES.items():
        download_voice(name, out)
        voices[lang] = out / f"{name}.onnx"
        print(f"{OK} {lang}: {voices[lang]}")
    return voices


def check_ollama() -> bool:
    import ollama

    step(f"Ollama and {config.LLM_MODEL}")
    client = ollama.Client(host=config.OLLAMA_HOST)
    try:
        names = [m.model for m in client.list().models]
    except Exception as e:  # noqa: BLE001
        print(f"{FAIL} Cannot reach Ollama at {config.OLLAMA_HOST}: {e}")
        print("       Install it from https://ollama.com/download and make sure it is running.")
        return False
    if config.LLM_MODEL not in names:
        print(f"  Pulling {config.LLM_MODEL} (about 3.3 GB, one time)...")
        last = ""
        for p in client.pull(config.LLM_MODEL, stream=True):
            status = p.status or ""
            if p.total and p.completed:
                status += f" {p.completed / p.total:5.1%}"
            if status != last:
                print(f"\r  {status:60}", end="", flush=True)
                last = status
        print()
    print(f"{OK} {config.LLM_MODEL} is available")
    return True


def self_test() -> None:
    """Voice round trip (Piper -> Moonshine) and a Gemma timing check."""
    import numpy as np

    step("Self-test")
    from app.speech import PiperTTS

    voices = {lang: config.MODELS_DIR / "piper" / f"{name}.onnx" for lang, name in config.PIPER_VOICES.items()}
    tts = PiperTTS(voices)
    phrases = {"en": "My blood pressure was 120 over 80 this morning.",
               "es": "Mi presión fue ciento veinte sobre ochenta esta mañana."}

    from moonshine_voice import Transcriber, get_model_for_language

    arch = config.moonshine_arch(config.ASR_SIZE)
    for lang, phrase in phrases.items():
        t = time.perf_counter()
        wav_bytes = tts.synth(phrase, lang)
        tts_ms = (time.perf_counter() - t) * 1000
        import io
        import wave
        with wave.open(io.BytesIO(wav_bytes)) as w:
            rate = w.getframerate()
            audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        # Resample to 16 kHz for the recognizer.
        n = int(len(audio) * 16000 / rate)
        audio16 = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)
        path, model_arch = get_model_for_language(lang, arch, cache_root=config.MODELS_DIR / "moonshine")
        tr = Transcriber(model_path=path, model_arch=model_arch)
        t = time.perf_counter()
        result = tr.transcribe_without_streaming(audio16.tolist(), 16000)
        asr_ms = (time.perf_counter() - t) * 1000
        text = " ".join(line.text for line in result.lines)
        print(f"{OK} {lang}: voice {tts_ms:.0f} ms for {len(audio) / rate:.1f} s of speech; "
              f"transcribed in {asr_ms:.0f} ms -> \"{text}\"")
        tr.close()

    from app.llm import OllamaLLM
    from app import router

    llm = OllamaLLM(config.LLM_MODEL, config.OLLAMA_HOST)
    err = llm.check()
    if err:
        print(f"{FAIL} {err}")
        return
    t = time.perf_counter()
    llm.warmup()
    print(f"{OK} Gemma loaded in {time.perf_counter() - t:.1f} s")
    for i in range(2):
        t = time.perf_counter()
        data, stats = llm.json(router.router_messages([], "the scale showed one eighty this morning", None),
                               router.ROUTER_SCHEMA)
        label = "first call" if i == 0 else "repeat call (prompt cached)"
        print(f"{OK} Gemma routing, {label}: {(time.perf_counter() - t) * 1000:.0f} ms -> {data}  "
              f"[prompt {stats['prompt_tokens']} tok / {stats['prompt_ms']} ms, "
              f"output {stats['output_tokens']} tok / {stats['output_ms']} ms]")
    print("\nIf Gemma routing is well above 2000 ms on this machine, rule-based turns will still be")
    print("fast, but turns that need Gemma will miss the 2 s target. The latency panel shows both.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="only run the self-test")
    parser.add_argument("--download-only", action="store_true", help="download, skip the self-test")
    args = parser.parse_args()
    if not args.test:
        download_moonshine()
        download_piper()
        ok = check_ollama()
        if args.download_only:
            if not ok:
                sys.exit(1)
            print("\nModels downloaded.")
            return
    try:
        self_test()
    except Exception as e:  # noqa: BLE001
        print(f"{FAIL} Self-test failed: {e}")
        raise
    print("\nDone. Start the demo with:  python run.py")


if __name__ == "__main__":
    main()

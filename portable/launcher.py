"""Portable launcher: the entry point bundled into HealthCoachDemo.exe.

Expected folder layout (made by build_portable.ps1):

    HealthCoachDemo/
      Start Demo.bat
      app/HealthCoachDemo.exe   <- this launcher, plus _internal/
      models/                   <- Moonshine + Piper
      ollama/ollama.exe         <- portable Ollama
      ollama/models/            <- Gemma 3

It starts the portable Ollama on a private port (so it never clashes with an Ollama
already installed on the computer), starts the demo server, and opens the browser once
everything has loaded. Ollama is tied to this process: closing this window stops it.
"""

from __future__ import annotations

import atexit
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

OLLAMA_PORT = 11435


def package_root() -> Path:
    """Folder that holds models/ and ollama/ (the exe's folder or its parent)."""
    here = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) \
        else Path(__file__).resolve().parent.parent
    for candidate in (here, here.parent):
        if (candidate / "models").is_dir():
            return candidate
    return here


def free_port(start: int = 8000) -> int:
    for port in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free port between 8000 and 8049")


def wait_http(url: str, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
    return False


def writable(folder: Path) -> bool:
    try:
        folder.mkdir(parents=True, exist_ok=True)
        probe = folder / ".write_test"
        probe.write_text("ok")
        probe.unlink()
        return True
    except OSError:
        return False


_job_handle = None  # kept alive for the life of this process


def kill_with_this_process(proc: subprocess.Popen) -> None:
    """Windows: put the child in a job object that is killed when this process exits,
    even if the window is closed with the X button."""
    global _job_handle
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]

    class BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class IO(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in ("ReadOperationCount", "WriteOperationCount",
                                                   "OtherOperationCount", "ReadTransferCount",
                                                   "WriteTransferCount", "OtherTransferCount")]

    class EXTENDED(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    job = k32.CreateJobObjectW(None, None)
    info = EXTENDED()
    info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))  # 9 = ExtendedLimitInformation
    k32.AssignProcessToJobObject(job, wintypes.HANDLE(int(proc._handle)))  # noqa: SLF001
    _job_handle = job


def start_ollama(root: Path) -> subprocess.Popen | None:
    exe = root / "ollama" / ("ollama.exe" if os.name == "nt" else "ollama")
    if not exe.exists():
        print("  Portable Ollama not found; using an installed Ollama if one is running.")
        return None
    env = os.environ.copy()
    env["OLLAMA_MODELS"] = str(root / "ollama" / "models")
    env["OLLAMA_HOST"] = f"127.0.0.1:{OLLAMA_PORT}"
    env["OLLAMA_KEEP_ALIVE"] = "60m"
    log_dir = root if writable(root) else Path(os.getenv("TEMP", "/tmp"))
    log = open(log_dir / "ollama.log", "ab")  # noqa: SIM115
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen([str(exe), "serve"], env=env, stdout=log, stderr=log, creationflags=flags)
    kill_with_this_process(proc)
    atexit.register(lambda: proc.poll() is None and proc.terminate())
    os.environ["OLLAMA_HOST"] = f"http://127.0.0.1:{OLLAMA_PORT}"
    print("  Starting Gemma's runtime (Ollama)...", end="", flush=True)
    if wait_http(f"http://127.0.0.1:{OLLAMA_PORT}/api/tags", 60):
        print(" ready.")
    else:
        print(f" not responding. See {log_dir / 'ollama.log'}")
    return proc


def check_imports() -> int:
    """Loads every native library the demo needs. Used by the build script."""
    checks = []

    def attempt(label, fn):
        try:
            fn()
            checks.append((label, "ok"))
        except Exception as e:  # noqa: BLE001
            checks.append((label, f"FAILED: {e}"))

    def moonshine():
        from moonshine_voice.moonshine_api import _MoonshineLib

        _MoonshineLib()

    def piper():
        from piper import PiperVoice  # noqa: F401
        from piper.voice import ESPEAK_DATA_DIR  # noqa: F401
        import piper.espeakbridge  # noqa: F401

    def onnx():
        import onnxruntime

        onnxruntime.get_available_providers()

    def web():
        import uvicorn  # noqa: F401
        import websockets  # noqa: F401
        from app.server import app  # noqa: F401
        from app import config

        assert (config.STATIC_DIR / "index.html").exists(), f"missing {config.STATIC_DIR / 'index.html'}"
        assert (config.DATA_DIR / "plan_en.md").exists(), f"missing {config.DATA_DIR / 'plan_en.md'}"

    attempt("Moonshine speech engine", moonshine)
    attempt("Piper voice engine", piper)
    attempt("ONNX runtime", onnx)
    attempt("Web server and page files", web)
    for label, result in checks:
        print(f"  {label:28} {result}")
    return 0 if all(r == "ok" for _, r in checks) else 1


def main() -> None:
    if "--check-imports" in sys.argv:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        raise SystemExit(check_imports())
    os.system("")  # enable ANSI on Windows consoles
    root = package_root()
    print("\n  Health Coach Voice Prototype")
    print("  ----------------------------")
    print(f"  Running from {root}\n")

    os.environ.setdefault("DEMO_MODELS_DIR", str(root / "models"))
    data_dir = root if writable(root) else Path(os.getenv("LOCALAPPDATA", Path.home())) / "HealthCoachDemo"
    os.environ.setdefault("DEMO_DB", str(data_dir / "demo.db"))

    if os.getenv("DEMO_FAKE") != "1":
        start_ollama(root)

    port = free_port(int(os.getenv("DEMO_PORT", "8000")))
    os.environ["DEMO_PORT"] = str(port)
    url = f"http://127.0.0.1:{port}"

    # Import after the environment is set: app.config reads it at import time.
    import uvicorn

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app.server import app

    def open_when_ready():
        print("  Loading speech and language models (this can take a minute)...", flush=True)
        if wait_http(f"{url}/api/state", 300):
            print(f"\n  Ready: {url}")
            print("  Opening the browser. If it doesn't open, paste that address into Chrome or Edge.")
            print("  Keep this window open during the demo. Close it to stop everything.\n")
            if os.getenv("DEMO_NO_BROWSER") != "1":
                webbrowser.open(url)
        else:
            print("\n  The demo server did not start. Check the messages above.")

    threading.Thread(target=open_when_ready, daemon=True).start()

    def stop(*_):
        raise SystemExit(0)

    for sig in ("SIGINT", "SIGTERM", "SIGBREAK"):
        if hasattr(signal, sig):
            signal.signal(getattr(signal, sig), stop)

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        if "--check-imports" in sys.argv:
            raise
        _ = e
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"\n  The demo stopped because of an error: {e}")
        input("  Press Enter to close this window.")

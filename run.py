"""Starts the demo server. Open http://127.0.0.1:8000 in Chrome or Edge."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn  # noqa: E402

from app import config  # noqa: E402

if __name__ == "__main__":
    print(f"Starting on http://{config.HOST}:{config.PORT}  (Ctrl+C to stop)")
    uvicorn.run("app.server:app", host=config.HOST, port=config.PORT, log_level="warning")

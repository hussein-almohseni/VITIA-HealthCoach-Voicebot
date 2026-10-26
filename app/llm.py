"""Gemma 3 through Ollama, plus a scripted stand-in for tests.

Latency notes:
  - keep_alive keeps the model loaded between turns, so no reload cost per turn.
  - The system prompt is identical on every call and comes first, so Ollama can reuse
    its cached prompt prefix and only process the new words.
  - num_predict caps output length; JSON outputs use compact schemas.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Callable, Iterator

NS = 1e-6  # nanoseconds to milliseconds


class LLMStats(dict):
    """prompt_tokens, prompt_ms, output_tokens, output_ms, load_ms, total_ms."""


def _stats(resp) -> LLMStats:
    get = (lambda k: getattr(resp, k, None) or 0)
    return LLMStats(
        prompt_tokens=get("prompt_eval_count"),
        prompt_ms=round(get("prompt_eval_duration") * NS),
        output_tokens=get("eval_count"),
        output_ms=round(get("eval_duration") * NS),
        load_ms=round(get("load_duration") * NS),
        total_ms=round(get("total_duration") * NS),
    )


class OllamaLLM:
    def __init__(self, model: str, host: str, num_ctx: int = 4096, keep_alive: str = "60m"):
        import ollama  # imported here so tests do not need the package

        self.model = model
        self.client = ollama.Client(host=host)
        self.options = {"temperature": 0, "num_ctx": num_ctx}
        self.keep_alive = keep_alive

    def check(self) -> str | None:
        """Returns an error message if Ollama or the model is missing."""
        try:
            names = [m.model for m in self.client.list().models]
        except Exception as e:  # noqa: BLE001
            return f"Cannot reach Ollama ({e}). Is the Ollama app running?"
        if not any(n == self.model or n.split(":")[0] == self.model for n in names):
            return f"Model {self.model} not found in Ollama. Run: ollama pull {self.model}"
        return None

    def warmup(self) -> None:
        self.client.chat(model=self.model, messages=[{"role": "user", "content": "hi"}],
                         options={**self.options, "num_predict": 1}, keep_alive=self.keep_alive)

    def json(self, messages: list[dict], schema: dict, num_predict: int = 96) -> tuple[dict, LLMStats]:
        resp = self.client.chat(model=self.model, messages=messages, format=schema,
                                options={**self.options, "num_predict": num_predict},
                                keep_alive=self.keep_alive)
        try:
            data = json.loads(resp.message.content)
        except json.JSONDecodeError:
            data = {}
        return data, _stats(resp)

    def stream(self, messages: list[dict], num_predict: int = 160,
               stop_event: threading.Event | None = None) -> Iterator[str | LLMStats]:
        """Yields text pieces, then a final LLMStats."""
        resp_iter = self.client.chat(model=self.model, messages=messages, stream=True,
                                     options={**self.options, "num_predict": num_predict},
                                     keep_alive=self.keep_alive)
        last = None
        for part in resp_iter:
            if stop_event is not None and stop_event.is_set():
                close = getattr(resp_iter, "close", None)
                if close:
                    close()
                break
            last = part
            if part.message.content:
                yield part.message.content
        yield _stats(last) if last is not None else LLMStats()


class FakeLLM:
    """Scripted stand-in for tests and for running the UI without Ollama.

    `router` decides the JSON for routing calls; `answer` decides streamed text.
    The defaults answer plan questions with the first sentence of the top excerpt.
    """

    model = "fake"

    def __init__(self, router: Callable[[list[dict]], dict] | None = None,
                 answer: Callable[[list[dict]], str] | None = None, delay_s: float = 0.0):
        self.router = router or (lambda msgs: {"intent": "other"})
        self.answer = answer or self._default_answer
        self.delay_s = delay_s
        self.calls: list[str] = []

    @staticmethod
    def _default_answer(msgs: list[dict]) -> str:
        content = msgs[-1]["content"]
        if "[1]" not in content:
            return "NOT_IN_PLAN"
        excerpt = content.split("[1]", 1)[1].split("\n", 1)[0]
        body = excerpt.split(":", 1)[1].strip() if ":" in excerpt else excerpt
        first = body.split(". ")[0].rstrip(".")
        return f"Your plan says: {first}."

    def check(self) -> str | None:
        return None

    def warmup(self) -> None:
        pass

    def json(self, messages, schema, num_predict=96):
        self.calls.append("json")
        time.sleep(self.delay_s)
        return self.router(messages), LLMStats(prompt_tokens=0, prompt_ms=0, output_tokens=0,
                                               output_ms=0, load_ms=0, total_ms=0)

    def stream(self, messages, num_predict=160, stop_event=None):
        self.calls.append("stream")
        text = self.answer(messages)
        for word in text.split(" "):
            if stop_event is not None and stop_event.is_set():
                break
            time.sleep(self.delay_s / 10)
            yield word + " "
        yield LLMStats(prompt_tokens=0, prompt_ms=0, output_tokens=0, output_ms=0, load_ms=0, total_ms=0)

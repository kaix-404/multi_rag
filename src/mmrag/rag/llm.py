"""LLM/VLM client speaking the OpenAI chat-completions protocol.

Default backend is Puter AI's OpenAI-compatible REST endpoint; also works with
OpenAI, Ollama, vLLM, or anything exposing POST {base}/chat/completions with
Bearer auth. No dependency on the ``openai`` package — plain httpx. When no API
key is configured the client reports itself unavailable and the pipeline falls
back to offline extractive synthesis; nothing is silently faked.
"""

from __future__ import annotations

import base64
from pathlib import Path

import httpx

from ..config import Settings

# LLM providers are sometimes flaky (rate limits, 503s, MCP proxy hiccups).
# Transient 5xx from the model should not turn a live query into a hard 500:
# retry briefly, then fail fast so the caller can fall back to the
# extractive path.
TRANSIENT_STATUS_CODES = {500, 502, 503, 504}
RETRY_DELAYS_SECONDS = (1.0, 2.0, 3.0)


class LLMClient:
    def __init__(self, settings: Settings):
        self.base_url = settings.llm_base_url.rstrip("/")
        self.api_key = settings.llm_api_key
        self.model = settings.llm_model
        self.vlm_model = settings.vlm_model
        self.timeout = 120.0

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """POST with a bounded retry over transient upstream errors.

        A 503/500/502/504 from a hosted LLM is normally a temporary capacity or
        routing hiccup, not a bug in the caller, so we retry a few times before
        raising. Timeouts and connection errors are also retried.
        """
        last_exc: Exception | None = None
        for attempt, delay in enumerate(RETRY_DELAYS_SECONDS, start=1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    r = client.request(method, url, **kwargs)
                    if r.status_code in TRANSIENT_STATUS_CODES:
                        last_exc = httpx.HTTPStatusError(
                            f"Transient {r.status_code} from {url}", request=r.request, response=r
                        )
                        if attempt < len(RETRY_DELAYS_SECONDS):
                            import time as _time

                            _time.sleep(delay)
                            continue
                    r.raise_for_status()
                    return r
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_exc = exc
                if attempt < len(RETRY_DELAYS_SECONDS):
                    import time as _time

                    _time.sleep(delay)
                    continue
        raise last_exc if last_exc is not None else RuntimeError(f"request failed: {method} {url}")

    def chat(self, messages: list[dict], model: str | None = None,
             temperature: float = 0.0, max_tokens: int = 1024) -> str:
        if not self.available:
            raise RuntimeError("LLM not configured (OPENAI_API_KEY is empty)")
        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        r = self._request("POST", f"{self.base_url}/chat/completions",
                          json=payload, headers=headers)
        data = r.json()
        return data["choices"][0]["message"]["content"]

    def chat_stream(self, messages: list[dict], model: str | None = None,
                    temperature: float = 0.0, max_tokens: int = 1024):
        """Yield answer text deltas."""
        if not self.available:
            raise RuntimeError("LLM not configured (OPENAI_API_KEY is empty)")
        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        with httpx.Client(timeout=self.timeout) as client:
            with client.stream("POST", f"{self.base_url}/chat/completions",
                               json=payload, headers=headers) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    import json as _json
                    try:
                        delta = _json.loads(data)["choices"][0].get("delta", {})
                    except Exception:
                        continue
                    content = delta.get("content")
                    if content:
                        yield content

    def describe_image(self, image_path: str | Path, prompt: str) -> str:
        """Vision call: base64 PNG → descriptor text."""
        if not self.available:
            raise RuntimeError("VLM not configured (OPENAI_API_KEY is empty)")
        b64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }]
        return self.chat(messages, model=self.vlm_model, max_tokens=300)

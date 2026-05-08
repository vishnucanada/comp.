"""LLM chat backends used by the dashboard: Ollama, Anthropic, with SSE streaming.

The gate is applied separately in dashboard/routers/chat.py before any of these
are called, so the backends here only handle generation.
"""

import json
import os
import queue
import urllib.request as _urllib

from .config import (
    ANTHROPIC_MAX_TOK,
    ANTHROPIC_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_PREFERRED,
)


def ollama_model() -> str | None:
    """Return the best available Ollama model name, or None."""
    try:
        req = _urllib.Request(f"{OLLAMA_BASE_URL}/api/tags")
        with _urllib.urlopen(req, timeout=3) as r:
            models = [m["name"] for m in json.loads(r.read()).get("models", [])]
        if not models:
            return None
        return OLLAMA_PREFERRED if OLLAMA_PREFERRED in models else models[0]
    except Exception:
        return None


def ollama_chat(message: str, history: list[dict]) -> tuple[str, str] | None:
    model = ollama_model()
    if not model:
        return None
    try:
        body = json.dumps(
            {
                "model": model,
                "messages": history + [{"role": "user", "content": message}],
                "stream": False,
            }
        ).encode()
        req = _urllib.Request(
            f"{OLLAMA_BASE_URL}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with _urllib.urlopen(req, timeout=60) as r:
            return json.loads(r.read())["message"]["content"], f"ollama/{model}"
    except Exception:
        return None


def anthropic_chat(message: str, history: list[dict]) -> tuple[str, str] | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic

        resp = anthropic.Anthropic(api_key=key).messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=ANTHROPIC_MAX_TOK,
            messages=history + [{"role": "user", "content": message}],
        )
        return resp.content[0].text, ANTHROPIC_MODEL
    except Exception:
        return None


def get_response(message: str, history: list[dict]) -> tuple[str, str]:
    for fn in (ollama_chat, anthropic_chat):
        result = fn(message, history)
        if result:
            return result
    return (
        f"No LLM backend detected. "
        f"Run `ollama pull {OLLAMA_PREFERRED} && ollama serve`, "
        f"or set ANTHROPIC_API_KEY.",
        "none",
    )


def ollama_stream_to_queue(
    message: str,
    history: list[dict],
    model_name: str,
    q: queue.Queue,
) -> None:
    """Push SSE-style dicts onto q in a background thread. Sentinel: None."""
    try:
        body = json.dumps(
            {
                "model": model_name,
                "messages": history + [{"role": "user", "content": message}],
                "stream": True,
            }
        ).encode()
        req = _urllib.Request(
            f"{OLLAMA_BASE_URL}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with _urllib.urlopen(req, timeout=120) as r:
            for raw in r:
                try:
                    chunk = json.loads(raw)
                    text = chunk.get("message", {}).get("content", "")
                    if text:
                        q.put({"type": "chunk", "text": text})
                    if chunk.get("done"):
                        break
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        q.put({"type": "error", "text": str(e)})
    finally:
        q.put(None)

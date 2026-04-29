"""
OpenAI-compatible API endpoint.

POST /v1/chat/completions — drop-in replacement for OpenAI chat completions.
Models NLPN privilege enforcement transparently before generation.

Supported request fields: model, messages, max_tokens, temperature, stream.
Returns OpenAI-format response with extra `x_comp_policy` field.
"""
import json
import os
import time
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..backends import get_response, nlpn_generate, ollama_stream_to_queue
from ..config import OLLAMA_PREFERRED
from ..helpers import policy_check, sanitize
from ..registry import model_registry
from ..deps import limiter

router = APIRouter(prefix="/v1")


# ---------------------------------------------------------------------------
# Request / response models (OpenAI-compatible)
# ---------------------------------------------------------------------------

class OAIMessage(BaseModel):
    role:    str
    content: str = Field(..., max_length=16384)


class OAIRequest(BaseModel):
    model:      str               = "comp-enforced"
    messages:   list[OAIMessage]
    max_tokens: int               = Field(512, ge=1, le=4096)
    temperature: float            = Field(0.7, ge=0.0, le=2.0)
    stream:     bool              = False
    # comp. extensions
    policy:     str | None        = Field(None, max_length=64)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_history(messages: list[OAIMessage]) -> tuple[str, list[dict]]:
    """Return (last_user_message, prior_history_dicts)."""
    msgs = [{"role": m.role, "content": m.content} for m in messages]
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i]["role"] == "user":
            return msgs[i]["content"], msgs[:i]
    return "", msgs


def _oai_response(content: str, model: str, finish: str = "stop") -> dict:
    return {
        "id":      f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   model,
        "choices": [{
            "index":         0,
            "message":       {"role": "assistant", "content": content},
            "finish_reason": finish,
        }],
        "usage": {
            "prompt_tokens":     0,
            "completion_tokens": 0,
            "total_tokens":      0,
        },
    }


def _oai_chunk(delta_content: str | None, model: str, finish: str | None = None) -> str:
    choice: dict = {"index": 0, "delta": {}, "finish_reason": finish}
    if delta_content is not None:
        choice["delta"] = {"role": "assistant", "content": delta_content}
    chunk = {
        "id":      f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object":  "chat.completion.chunk",
        "created": int(time.time()),
        "model":   model,
        "choices": [choice],
    }
    return f"data: {json.dumps(chunk)}\n\n"


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post("/chat/completions")
@limiter.limit("30/minute")
async def chat_completions(request: Request, req: OAIRequest):
    message, history = _split_history(req.messages)
    decision, violations = policy_check(req.policy, message)

    policy_meta = {
        "decision":   decision,
        "violations": violations,
        "policy":     req.policy,
    }

    # --- DENY ---
    if decision == "DENY":
        blocked = f"[BLOCKED by policy '{req.policy}': {', '.join(violations)}]"
        if req.stream:
            async def blocked_stream():
                yield _oai_chunk(blocked, req.model)
                yield _oai_chunk(None, req.model, finish="stop")
                yield "data: [DONE]\n\n"
            return StreamingResponse(blocked_stream(), media_type="text/event-stream")
        resp = _oai_response(blocked, req.model)
        resp["x_comp_policy"] = policy_meta
        return resp

    # --- NLPN model ---
    if req.policy:
        model_obj, tokenizer = model_registry.get(sanitize(req.policy))
        if model_obj is not None:
            import asyncio
            loop = asyncio.get_event_loop()
            text, model_name = await loop.run_in_executor(
                None, nlpn_generate, model_obj, tokenizer, message, sanitize(req.policy)
            )
            if req.stream:
                async def nlpn_stream():
                    for word in text.split():
                        yield _oai_chunk(word + " ", model_name)
                    yield _oai_chunk(None, model_name, finish="stop")
                    yield "data: [DONE]\n\n"
                return StreamingResponse(nlpn_stream(), media_type="text/event-stream")
            resp = _oai_response(text, model_name)
            resp["x_comp_policy"] = policy_meta
            return resp

    # --- Streaming via Ollama ---
    if req.stream:
        import queue as _queue
        from threading import Thread
        from ..backends import ollama_model

        ollama = ollama_model()
        if ollama:
            q: _queue.Queue = _queue.Queue()
            Thread(
                target=ollama_stream_to_queue,
                args=(message, history, ollama, q),
                daemon=True,
            ).start()

            async def ollama_stream() -> AsyncGenerator[str, None]:
                import asyncio
                loop = asyncio.get_event_loop()
                model_name = f"ollama/{ollama}"
                while True:
                    item = await loop.run_in_executor(None, q.get)
                    if item is None:
                        break
                    if item.get("type") == "chunk":
                        yield _oai_chunk(item["text"], model_name)
                yield _oai_chunk(None, model_name, finish="stop")
                yield "data: [DONE]\n\n"

            return StreamingResponse(ollama_stream(), media_type="text/event-stream")

    # --- Non-streaming fallback ---
    text, model_name = get_response(message, history)
    resp = _oai_response(text, model_name)
    resp["x_comp_policy"] = policy_meta
    return resp


@router.get("/models")
async def list_models():
    """OpenAI-compatible model list endpoint."""
    return {
        "object": "list",
        "data": [
            {"id": "comp-enforced", "object": "model", "created": 0, "owned_by": "comp"},
            {"id": "comp-nlpn",     "object": "model", "created": 0, "owned_by": "comp"},
        ],
    }

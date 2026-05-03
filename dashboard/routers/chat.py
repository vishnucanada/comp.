"""Chat routes: blocking /api/chat and streaming /api/chat/stream."""

import asyncio
import json
import os
import queue
from threading import Thread

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from ..backends import (
    anthropic_chat,
    get_response,
    nlpn_generate,
    ollama_model,
    ollama_stream_to_queue,
)
from ..config import OLLAMA_PREFERRED
from ..deps import limiter
from ..helpers import policy_check, sanitize
from ..registry import model_registry
from ..schemas import ChatRequest, StreamChatRequest

router = APIRouter()


@router.post("/api/chat")
@limiter.limit("30/minute")
async def chat(request: Request, req: ChatRequest):
    decision, violations = policy_check(req.policy_name, req.message)

    if decision == "DENY":
        rules = ", ".join(violations)
        return {
            "decision": "DENY",
            "violations": violations,
            "response": f"ERROR this prompt has been blocked by policy [{rules}]",
            "model": "policy-enforcer",
        }

    if req.policy_name:
        model, tokenizer = model_registry.get(sanitize(req.policy_name))
        if model is not None:
            loop = asyncio.get_event_loop()
            response, model_name = await loop.run_in_executor(
                None,
                nlpn_generate,
                model,
                tokenizer,
                req.message,
                sanitize(req.policy_name),
            )
            return {
                "decision": "ALLOW",
                "violations": [],
                "response": response,
                "model": model_name,
            }

    history = [{"role": m.role, "content": m.content} for m in req.history]
    response, model_name = get_response(req.message, history)
    return {"decision": "ALLOW", "violations": [], "response": response, "model": model_name}


@router.post("/api/chat/stream")
@limiter.limit("30/minute")
async def chat_stream(request: Request, req: StreamChatRequest):
    decision, violations = policy_check(req.policy_name, req.message)

    async def event_generator():
        yield f"data: {json.dumps({'type': 'policy', 'decision': decision, 'violations': violations})}\n\n"

        if decision == "DENY":
            rules = ", ".join(violations)
            yield f"data: {json.dumps({'type': 'chunk', 'text': f'ERROR this prompt has been blocked by policy [{rules}]'})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'model': 'policy-enforcer'})}\n\n"
            return

        # NLPN model — generate then stream word-by-word
        if req.policy_name:
            model, tokenizer = model_registry.get(sanitize(req.policy_name))
            if model is not None:
                loop = asyncio.get_event_loop()
                safe = sanitize(req.policy_name)
                response, model_name = await loop.run_in_executor(
                    None,
                    nlpn_generate,
                    model,
                    tokenizer,
                    req.message,
                    safe,
                )
                for word in response.split():
                    yield f"data: {json.dumps({'type': 'chunk', 'text': word + ' '})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'model': model_name})}\n\n"
                return

        # Ollama streaming
        ollama = ollama_model()
        if ollama:
            q: queue.Queue = queue.Queue()
            history = [{"role": m.role, "content": m.content} for m in req.history]
            Thread(
                target=ollama_stream_to_queue, args=(req.message, history, ollama, q), daemon=True
            ).start()
            loop = asyncio.get_event_loop()
            while True:
                item = await loop.run_in_executor(None, q.get)
                if item is None:
                    break
                yield f"data: {json.dumps(item)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'model': f'ollama/{ollama}'})}\n\n"
            return

        # Anthropic fallback (non-streaming)
        if os.environ.get("ANTHROPIC_API_KEY"):
            history = [{"role": m.role, "content": m.content} for m in req.history]
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, anthropic_chat, req.message, history)
            if result:
                text, model_name = result
                yield f"data: {json.dumps({'type': 'chunk', 'text': text})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'model': model_name})}\n\n"
                return

        msg = (
            f"No LLM backend detected. "
            f"Run `ollama pull {OLLAMA_PREFERRED} && ollama serve`, "
            f"or set ANTHROPIC_API_KEY."
        )
        yield f"data: {json.dumps({'type': 'chunk', 'text': msg})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'model': 'none'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

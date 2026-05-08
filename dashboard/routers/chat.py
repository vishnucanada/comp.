"""Chat routes: blocking /api/chat and streaming /api/chat/stream.

Every request goes through PolicyGate before the LLM is called. Denied
prompts never reach the model — the gate's deny message is returned instead.
"""

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
    ollama_model,
    ollama_stream_to_queue,
)
from ..config import OLLAMA_PREFERRED
from ..helpers import policy_check
from ..schemas import ChatRequest, StreamChatRequest

router = APIRouter()


@router.post("/api/chat")
async def chat(request: Request, req: ChatRequest):
    history_text = [m.content for m in req.history]
    decision, violations = policy_check(
        req.policy_name, req.message, user_role=req.user_role, history=history_text
    )

    if decision == "DENY":
        rules = ", ".join(violations)
        return {
            "decision": "DENY",
            "violations": violations,
            "response": f"ERROR this prompt has been blocked by policy [{rules}]",
            "model": "policy-enforcer",
        }

    history = [{"role": m.role, "content": m.content} for m in req.history]
    response, model_name = get_response(req.message, history)
    return {"decision": "ALLOW", "violations": [], "response": response, "model": model_name}


@router.post("/api/chat/stream")
async def chat_stream(request: Request, req: StreamChatRequest):
    history_text = [m.content for m in req.history]
    decision, violations = policy_check(
        req.policy_name, req.message, user_role=req.user_role, history=history_text
    )

    async def event_generator():
        yield f"data: {json.dumps({'type': 'policy', 'decision': decision, 'violations': violations})}\n\n"

        if decision == "DENY":
            rules = ", ".join(violations)
            yield f"data: {json.dumps({'type': 'chunk', 'text': f'ERROR this prompt has been blocked by policy [{rules}]'})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'model': 'policy-enforcer'})}\n\n"
            return

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

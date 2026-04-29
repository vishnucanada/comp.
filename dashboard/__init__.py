"""
comp. Admin Dashboard — FastAPI application.

Entry point for ``uvicorn dashboard:app``.
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path so ``import src`` works when the package
# is not installed (e.g., inside Docker or a plain virtualenv).
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from .config import ALLOWED_ORIGINS, STATIC_DIR
from .deps import RateLimitExceeded, limiter, rate_limit_exceeded_handler
from .routers import chat, models, policies, training
from .routers import admin as admin_router
from .routers import openai_compat

app = FastAPI(
    title="comp. — Nested Least-Privilege Networks",
    description="Policy-enforced LLM deployment via rank-restricted transformer layers.",
    version="0.2.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key", "X-Tenant-ID"],
)

app.include_router(chat.router)
app.include_router(models.router)
app.include_router(policies.router)
app.include_router(training.router)
app.include_router(admin_router.router)
app.include_router(openai_compat.router)


@app.get("/favicon.png")
async def favicon():
    return FileResponse(STATIC_DIR / "favicon.png", media_type="image/png")


@app.get("/", response_class=HTMLResponse)
async def landing():
    return HTMLResponse((STATIC_DIR / "landing.html").read_text())


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page():
    return HTMLResponse((STATIC_DIR / "dashboard.html").read_text())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("dashboard:app", host="0.0.0.0", port=8000)

"""comp. Admin Dashboard — FastAPI application.

Entry point for ``uvicorn dashboard:app``.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so ``import src`` works when the package
# is not installed (e.g., inside Docker or a plain virtualenv).
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse  # noqa: E402

from .config import ALLOWED_ORIGINS, STATIC_DIR  # noqa: E402
from .routers import admin as admin_router  # noqa: E402
from .routers import chat, policies  # noqa: E402

app = FastAPI(
    title="comp. — Policy-as-Code for LLM API Calls",
    description="Pluggable content guards, IAM, and audit logs for any LLM backend.",
    version="0.3.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)

app.include_router(chat.router)
app.include_router(policies.router)
app.include_router(admin_router.router)


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

    uvicorn.run("dashboard:app", host="127.0.0.1", port=8000)

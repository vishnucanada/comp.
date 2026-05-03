"""Dashboard configuration — all constants and directory paths."""

import os
from pathlib import Path

# Project root = parent of the dashboard/ package
ROOT = Path(__file__).parent.parent

# Directories
STATIC_DIR = Path(__file__).parent / "static"
POLICIES_DIR = ROOT / "policies"
POLICY_LIBRARY = ROOT / "policies" / "library"
CHECKPOINTS_DIR = ROOT / "nlpn_checkpoints"
AUDIT_DIR = ROOT / "audit"

POLICIES_DIR.mkdir(exist_ok=True)
AUDIT_DIR.mkdir(exist_ok=True)

# LLM backend settings
OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_PREFERRED = "qwen2.5:0.5b"
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
ANTHROPIC_MAX_TOK = 1024

# Security
API_KEY = os.environ.get("API_KEY", "")
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:8000").split(",")

# Admin audit
ADMIN_AUDIT_HMAC_KEY = os.environ.get("ADMIN_AUDIT_HMAC_KEY", "")

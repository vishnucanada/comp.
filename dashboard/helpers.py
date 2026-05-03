"""Shared helper functions: name sanitisation, policy persistence, policy loading."""

import json
import re
import time

from fastapi import HTTPException

from .config import HISTORY_DIR, POLICIES_DIR


def sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "", name)


def _safe_name(name: str) -> str:
    safe = sanitize(name)
    if not safe:
        raise HTTPException(400, "Invalid policy name")
    return safe


def _persist_policy(safe: str, description: str, structured: str | None) -> None:
    """Write policy files, archiving the previous version to history first."""
    meta_file = POLICIES_DIR / f"{safe}.json"
    txt_file = POLICIES_DIR / f"{safe}.txt"

    if meta_file.exists() or txt_file.exists():
        ts = time.strftime("%Y%m%dT%H%M%S")
        archive = HISTORY_DIR / safe
        archive.mkdir(exist_ok=True)
        if meta_file.exists():
            (archive / f"{ts}.json").write_text(meta_file.read_text())
        if txt_file.exists():
            (archive / f"{ts}.txt").write_text(txt_file.read_text())

    meta_file.write_text(json.dumps({"description": description}))
    if structured:
        txt_file.write_text(structured)


def _load_policy(name: str):
    """Load a Policy by name from disk, translating description if no .txt file."""
    from src.policy import Policy
    from src.translator import PolicyTranslator

    safe = sanitize(name)
    if not safe:
        return None
    txt = POLICIES_DIR / f"{safe}.txt"
    meta = POLICIES_DIR / f"{safe}.json"
    if txt.exists():
        return Policy.from_file(txt)
    if meta.exists():
        try:
            desc = json.loads(meta.read_text()).get("description", "")
            if desc:
                return PolicyTranslator().translate(desc)
        except Exception:
            pass
    return None


def policy_check(policy_name: str | None, message: str) -> tuple[str, list[str]]:
    """Return (decision, violations) for a message against a named policy."""
    from src.policy import PolicyCompiler

    if not policy_name:
        return "ALLOW", []
    policy = _load_policy(policy_name)
    if not policy:
        return "ALLOW", []
    violated, violations = PolicyCompiler(policy).check(message)
    return ("DENY" if violated else "ALLOW"), violations

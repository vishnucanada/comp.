"""
NLP-based policy compiler — no LLM required.

Extracts DENY/ALLOW rules from plain English using rule-based NLP:
  1. Split text into sentences
  2. Classify each sentence as DENY / ALLOW / neutral via trigger patterns
  3. Match mentioned concepts against a sensitive-data taxonomy
  4. Build PolicyRule objects with runtime keywords and regex patterns
"""
from __future__ import annotations

import re
from pathlib import Path

from .policy import Policy, PolicyRule

# ── Intent detection ──────────────────────────────────────────────────────────

_DENY_RE = re.compile(
    r"\b("
    r"must\s+not|should\s+not|shall\s+not|will\s+not|cannot|can'?t|may\s+not"
    r"|do\s+not|don'?t"
    r"|not\s+(?:be\s+)?(?:shar\w+|reveal\w+|disclos\w+|provid\w+|giv\w+|transmit\w+|expos\w+)"
    r"|(?:is|are|be|been)\s+(?:restricted|prohibited|forbidden|confidential|private|sensitive)"
    r"|(?:keep|kept|must\s+be\s+kept|should\s+be\s+kept)\s+(?:strictly\s+)?confidential"
    r"|restrict\w*|prohibit\w*|forbid\w*"
    r"|strictly\s+confidential|must\s+be\s+(?:confidential|protected|secured?)"
    r")\b",
    re.IGNORECASE,
)

_ALLOW_RE = re.compile(
    r"\b("
    r"(?:is|are)\s+(?:permitted|allowed|acceptable|authorized|fine|okay|ok|public|unrestricted)"
    r"|may\s+(?:be\s+)?(?:shar\w+|discuss\w+|provid\w+|reveal\w+)"
    r"|can\s+be\s+(?:shar\w+|discuss\w+|provid\w+)"
    r"|permitted|allowed|acceptable|authorized"
    r"|publicly\s+available|public\s+(?:information|data|content|documentation)"
    r")\b",
    re.IGNORECASE,
)

# ── Sensitive data taxonomy ───────────────────────────────────────────────────
# detect   — substrings matched in policy text to identify this category
# name     — category label used in Policy / PolicyCompiler
# keywords — runtime substrings matched against user prompts at inference
# patterns — regex patterns matched against user prompts at inference

_TAXONOMY: list[dict] = [
    {
        "detect": ["personal identifier", "personal id", "pii", "personally identifiable"],
        "name": "personal identifiers",
        "keywords": ["personal identifier", "pii", "identity", "personal id"],
        "patterns": [],
    },
    {
        "detect": ["full name", "employee name", "staff name", "worker name", "their name"],
        "name": "names",
        "keywords": ["name", "full name", "first name", "last name", "surname", "who is"],
        "patterns": [],
    },
    {
        "detect": ["email", "e-mail", "mail address"],
        "name": "email addresses",
        "keywords": ["email", "e-mail", "mail address", "email address"],
        "patterns": [r"\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b"],
    },
    {
        "detect": ["phone", "telephone", "mobile number", "cell number", "phone number"],
        "name": "phone numbers",
        "keywords": ["phone", "telephone", "mobile", "cell", "phone number", "call"],
        "patterns": [r"\b(?:\+?\d[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"],
    },
    {
        "detect": ["home address", "residential address", "street address",
                   "mailing address", "physical address"],
        "name": "home addresses",
        "keywords": ["address", "home address", "where does", "where do", "lives",
                     "residence", "street", "zip code"],
        "patterns": [],
    },
    {
        "detect": ["medical", "health", "diagnosis", "clinical", "patient",
                   "prescription", "medication", "treatment", "healthcare"],
        "name": "medical information",
        "keywords": ["medical", "health", "diagnosis", "condition", "patient",
                     "prescription", "medication", "treatment", "health record", "medical record"],
        "patterns": [],
    },
    {
        "detect": ["salary", "wage", "compensation", "remuneration", "pay", "income", "earnings"],
        "name": "salary information",
        "keywords": ["salary", "wage", "compensation", "pay", "income", "earnings", "how much does"],
        "patterns": [],
    },
    {
        "detect": ["financial", "account number", "bank account", "banking",
                   "credit card", "financial data", "financial information"],
        "name": "financial information",
        "keywords": ["financial", "account number", "bank account", "credit card", "payment"],
        "patterns": [r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"],
    },
    {
        "detect": ["ssn", "social security", "national id", "government id",
                   "passport", "tax id", "driver", "driving licence", "driver's license"],
        "name": "government identifiers",
        "keywords": ["ssn", "social security", "national id", "government id",
                     "passport", "tax id", "driver license", "id number"],
        "patterns": [r"\b\d{3}-\d{2}-\d{4}\b"],
    },
    {
        "detect": ["password", "credential", "api key", "secret key",
                   "private key", "access token", "auth token"],
        "name": "credentials",
        "keywords": ["password", "credential", "api key", "secret", "token", "private key", "login"],
        "patterns": [],
    },
    {
        "detect": ["location", "gps", "tracking", "whereabouts", "current location"],
        "name": "location data",
        "keywords": ["location", "gps", "tracking", "whereabouts", "where is", "current location"],
        "patterns": [],
    },
    {
        "detect": ["date of birth", "dob", "birthday", "birth date"],
        "name": "date of birth",
        "keywords": ["date of birth", "dob", "birthday", "birth date", "how old", "age"],
        "patterns": [r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"],
    },
]

def _text_has(text: str, term: str) -> bool:
    if " " in term:
        return term in text
    return bool(re.search(r"\b" + re.escape(term) + r"\b", text, re.IGNORECASE))


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _list_items(sentence: str) -> list[str]:
    """Extract enumerated items after 'like', 'including', 'such as', etc."""
    m = re.search(r"\b(?:like|including|such as|e\.g\.|:)\s+([^.;]+)", sentence, re.I)
    if not m:
        return []
    raw = m.group(1)
    return [i.strip().rstrip(".").lower() for i in re.split(r",\s*|\s+or\s+|\s+and\s+", raw) if i.strip()]


def _matched_categories(sentence: str) -> list[dict]:
    """Return taxonomy entries whose detect terms appear in the sentence."""
    combined = sentence.lower() + " " + " ".join(_list_items(sentence))
    seen: dict[str, dict] = {}
    for cat in _TAXONOMY:
        for term in cat["detect"]:
            if _text_has(combined, term) and cat["name"] not in seen:
                seen[cat["name"]] = cat
                break
    return list(seen.values())


# ── Translator ────────────────────────────────────────────────────────────────

class PolicyTranslator:
    """
    Translate plain English policy text into a structured Policy object.
    Uses rule-based NLP — no API key required.
    """

    def translate(self, text: str) -> Policy:
        sentences = _split_sentences(text)
        deny: dict[str, dict] = {}
        allow: list[str] = []

        for sentence in sentences:
            is_deny = bool(_DENY_RE.search(sentence))
            is_allow = bool(_ALLOW_RE.search(sentence))

            if is_deny:
                for cat in _matched_categories(sentence):
                    deny[cat["name"]] = cat

            if is_allow and not is_deny:
                cats = _matched_categories(sentence)
                if cats:
                    for cat in cats:
                        allow.append(cat["name"])
                else:
                    allow.append("general information")

        # Fallback: scan full text if nothing was identified
        if not deny:
            for cat in _TAXONOMY:
                for term in cat["detect"]:
                    if _text_has(text, term) and cat["name"] not in deny:
                        deny[cat["name"]] = cat
                        break

        rules: list[PolicyRule] = []
        for cat in deny.values():
            rule = PolicyRule("DENY", cat["name"])
            rule.keywords = list(cat["keywords"])
            rule.patterns = [re.compile(p, re.I) for p in cat["patterns"]]
            rules.append(rule)

        for name in dict.fromkeys(allow):
            rules.append(PolicyRule("ALLOW", name))

        if not any(r.action == "ALLOW" for r in rules):
            rules.append(PolicyRule("ALLOW", "general information"))

        return Policy(name=_infer_name(text), rules=rules)

    def translate_and_save(self, text: str, path: str | Path) -> Policy:
        policy = self.translate(text)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(policy.to_text())
        return policy


def _infer_name(text: str) -> str:
    m = re.search(r"^(?:policy[:\s]+|from[:\s]+|re[:\s]+|subject[:\s]+)(.+)$", text, re.I | re.M)
    if m:
        return m.group(1).strip()[:60]
    for line in text.splitlines():
        line = line.strip()
        if line and len(line) < 60:
            return line
    return "Extracted Policy"

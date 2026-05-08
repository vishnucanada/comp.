"""Pluggable content guards for policy enforcement.

A Guard inspects a prompt and reports whether it violates the active Policy.
Multiple backends with different cost/quality tradeoffs:

  KeywordGuard           keyword + paraphrase + injection-pattern matching
                         (cheap, deterministic, brittle to paraphrase)

  OpenAIModerationGuard  OpenAI Moderation API (free of charge, requires
                         OPENAI_API_KEY, low latency, vendor-managed taxonomy)

  LlamaGuardGuard        Meta Llama Guard 3 via transformers (self-hosted,
                         heavyweight, works on air-gapped deployments)

  CompositeGuard         OR over a list of guards — first hit wins

All guards return a GuardResult with the violated category labels mapped back
into the user's Policy so the audit log stays uniform across backends.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .policy import Policy, PolicyCompiler


@dataclass
class GuardResult:
    flagged: bool
    categories: list[str] = field(default_factory=list)
    backend: str = ""

    @property
    def allowed(self) -> bool:
        return not self.flagged


class Guard(ABC):
    """Abstract content guard. Subclasses implement check()."""

    name: str = "guard"

    @abstractmethod
    def check(self, text: str, history: list[str] | None = None) -> GuardResult: ...


# ── 1. Keyword guard ─────────────────────────────────────────────────────────


class KeywordGuard(Guard):
    """Substring + paraphrase + injection-pattern matching against Policy rules.

    Cheap and deterministic. Use as a fast first-pass filter or a fallback
    when a heavier guard is unavailable. Bypassed by paraphrase, encoding,
    or foreign language — pair with another guard for production.
    """

    name = "keyword"

    def __init__(self, policy: Policy):
        self.policy = policy
        self._compiler = PolicyCompiler(policy)

    def check(self, text: str, history: list[str] | None = None) -> GuardResult:
        violated, categories = self._compiler.check(text, history)
        return GuardResult(flagged=violated, categories=categories, backend=self.name)


# ── 2. OpenAI moderation guard ───────────────────────────────────────────────


# Maps OpenAI category names → keywords that, if present in any of a Policy's
# DENY rules, indicate that category is in scope for that policy. Keeping the
# mapping declarative makes it easy to extend without touching the guard logic.
_OPENAI_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "sexual": ("sexual", "explicit"),
    "hate": ("hate", "discrimination", "racist"),
    "harassment": ("harassment", "bullying"),
    "self-harm": ("self-harm", "suicide"),
    "violence": ("violence", "weapon", "harm"),
    "sexual/minors": ("minor", "child", "csam"),
    "hate/threatening": ("threat", "hate"),
    "violence/graphic": ("violence", "graphic", "gore"),
    "self-harm/intent": ("self-harm", "suicide"),
    "self-harm/instructions": ("self-harm", "suicide"),
    "harassment/threatening": ("harassment", "threat"),
    "illicit": ("illegal", "illicit", "drug", "weapon"),
    "illicit/violent": ("violence", "weapon", "illegal"),
}


class OpenAIModerationGuard(Guard):
    """OpenAI Moderation API guard.

    Calls /v1/moderations with the prompt and reports any flagged categories.
    The OpenAI moderation endpoint is free of charge but requires an API key.

    Each flagged OpenAI category is mapped to the user's Policy categories that
    care about it (via keyword overlap with the rule's category label / keywords).
    Categories the policy doesn't reference are reported as `[oai:<name>]` so
    nothing is silently dropped.
    """

    name = "openai-moderation"

    def __init__(
        self,
        policy: Policy,
        model: str = "omni-moderation-latest",
        api_key: str | None = None,
    ):
        self.policy = policy
        self.model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise RuntimeError(
                "OpenAIModerationGuard requires OPENAI_API_KEY (env var or api_key=...)"
            )
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise ImportError(
                "OpenAIModerationGuard requires the `openai` package: pip install openai"
            ) from e
        self._client = OpenAI(api_key=self._api_key)
        self._policy_keywords = _collect_policy_keywords(policy)

    def check(self, text: str, history: list[str] | None = None) -> GuardResult:
        prompt = " ".join(history[-3:]) + " " + text if history else text
        try:
            resp = self._client.moderations.create(model=self.model, input=prompt)
        except Exception as e:
            # Fail-closed: if the moderation call itself fails we treat the
            # request as unverifiable rather than silently allowing it.
            return GuardResult(
                flagged=True,
                categories=[f"[{self.name}-error:{type(e).__name__}]"],
                backend=self.name,
            )

        result = resp.results[0]
        if not result.flagged:
            return GuardResult(flagged=False, categories=[], backend=self.name)

        # Pull out the OpenAI categories that fired.
        flagged_categories = [
            cat for cat, fired in result.categories.model_dump().items() if fired
        ]
        return GuardResult(
            flagged=True,
            categories=self._map_to_policy(flagged_categories),
            backend=self.name,
        )

    def _map_to_policy(self, oai_categories: list[str]) -> list[str]:
        """Return the Policy DENY categories that overlap with OpenAI's flags.

        Falls back to `[oai:<name>]` when the policy has no rule for the flag,
        so unmapped flags still surface in the audit log.
        """
        mapped: list[str] = []
        for oai_cat in oai_categories:
            keywords = _OPENAI_CATEGORY_KEYWORDS.get(oai_cat, (oai_cat.replace("/", "-"),))
            hit = False
            for rule in self.policy.denied:
                if any(kw in self._policy_keywords[rule.category] for kw in keywords):
                    if rule.category not in mapped:
                        mapped.append(rule.category)
                    hit = True
            if not hit:
                mapped.append(f"[oai:{oai_cat}]")
        return mapped


# ── 3. Llama Guard 3 (self-hosted) ───────────────────────────────────────────


# Llama Guard's S-codes from the official taxonomy → human-readable labels.
_LLAMAGUARD_S_CODES: dict[str, str] = {
    "S1": "violent crimes",
    "S2": "non-violent crimes",
    "S3": "sex-related crimes",
    "S4": "child sexual exploitation",
    "S5": "defamation",
    "S6": "specialized advice",
    "S7": "privacy",
    "S8": "intellectual property",
    "S9": "indiscriminate weapons",
    "S10": "hate",
    "S11": "suicide and self-harm",
    "S12": "sexual content",
    "S13": "elections",
    "S14": "code interpreter abuse",
}


class LlamaGuardGuard(Guard):
    """Self-hosted Meta Llama Guard 3 guard.

    Loads the gated `meta-llama/Llama-Guard-3-8B` checkpoint via transformers and
    classifies prompts as safe/unsafe with S-code categories. Use this when
    OpenAI's API is not an option (air-gapped, regulated environments).

    Heavy: requires torch + transformers + ~16 GB RAM (or a GPU). Hugging Face
    access to the gated repo is required; `huggingface-cli login` first.
    """

    name = "llamaguard"
    DEFAULT_MODEL_ID = "meta-llama/Llama-Guard-3-8B"

    def __init__(
        self,
        policy: Policy,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | None = None,
    ):
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except ImportError as e:
            raise ImportError(
                "LlamaGuardGuard requires `torch` and `transformers`: "
                "pip install 'comp[llamaguard]'"
            ) from e
        import torch

        self.policy = policy
        self.model_id = model_id
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if self._device != "cpu" else torch.float32,
        ).to(self._device)
        self._model.eval()
        self._policy_keywords = _collect_policy_keywords(policy)

    def check(self, text: str, history: list[str] | None = None) -> GuardResult:
        import torch

        messages = []
        if history:
            for i, h in enumerate(history[-3:]):
                messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": h})
        messages.append({"role": "user", "content": text})

        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)

        with torch.no_grad():
            output = self._model.generate(
                **inputs, max_new_tokens=20, pad_token_id=self._tokenizer.eos_token_id
            )

        response = self._tokenizer.decode(
            output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        ).strip()

        if response.lower().startswith("safe"):
            return GuardResult(flagged=False, categories=[], backend=self.name)

        # Response shape: "unsafe\nS1,S10" or "unsafe S1"
        s_codes = [tok.strip() for tok in response.replace("\n", ",").split(",") if tok.strip()]
        s_codes = [s for s in s_codes if s.upper().startswith("S")]
        return GuardResult(
            flagged=True,
            categories=self._map_to_policy(s_codes),
            backend=self.name,
        )

    def _map_to_policy(self, s_codes: list[str]) -> list[str]:
        mapped: list[str] = []
        for code in s_codes:
            label = _LLAMAGUARD_S_CODES.get(code.upper(), code)
            terms = label.split()
            hit = False
            for rule in self.policy.denied:
                kws = self._policy_keywords[rule.category]
                if any(t in kws for t in terms) or any(t in label for t in kws):
                    if rule.category not in mapped:
                        mapped.append(rule.category)
                    hit = True
            if not hit:
                mapped.append(f"[llamaguard:{code}-{label}]")
        return mapped


# ── 4. Composite ─────────────────────────────────────────────────────────────


class CompositeGuard(Guard):
    """Run guards in sequence; flag if any one flags. Categories are merged."""

    name = "composite"

    def __init__(self, guards: list[Guard]):
        if not guards:
            raise ValueError("CompositeGuard requires at least one inner guard")
        self.guards = guards

    def check(self, text: str, history: list[str] | None = None) -> GuardResult:
        flagged = False
        cats: list[str] = []
        backends: list[str] = []
        for g in self.guards:
            r = g.check(text, history)
            if r.flagged:
                flagged = True
                backends.append(g.name)
                for c in r.categories:
                    if c not in cats:
                        cats.append(c)
        return GuardResult(
            flagged=flagged,
            categories=cats,
            backend="+".join(backends) if backends else self.name,
        )


# ── factory ──────────────────────────────────────────────────────────────────


def make_guard(policy: Policy, backend: str = "keyword", **kwargs) -> Guard:
    """Build a Guard for a Policy by short name.

    backend ∈ {"keyword", "openai", "llamaguard", "openai+keyword"}.
    Extra kwargs are forwarded to the backend constructor.
    """
    backend = backend.lower()
    if backend == "keyword":
        return KeywordGuard(policy)
    if backend == "openai":
        return OpenAIModerationGuard(policy, **kwargs)
    if backend == "llamaguard":
        return LlamaGuardGuard(policy, **kwargs)
    if backend in ("openai+keyword", "keyword+openai"):
        return CompositeGuard([OpenAIModerationGuard(policy, **kwargs), KeywordGuard(policy)])
    raise ValueError(f"Unknown guard backend: {backend!r}")


# ── helpers ──────────────────────────────────────────────────────────────────


def _collect_policy_keywords(policy: Policy) -> dict[str, set[str]]:
    """category → set of all keywords (rule.keywords + tokenized category label)."""
    out: dict[str, set[str]] = {}
    for rule in policy.denied:
        kws: set[str] = set(k.lower() for k in rule.keywords)
        kws.update(t.lower() for t in rule.category.split())
        out[rule.category] = kws
    return out

"""
Embedding-based semantic policy compiler.

Augments keyword/regex matching with cosine-similarity-based semantic
matching using sentence-transformers.  Falls back transparently to
keyword-only matching if sentence-transformers is not installed.

Usage::

    from src.semantic import SemanticPolicyCompiler
    compiler = SemanticPolicyCompiler(policy, threshold=0.45)
    violated, categories = compiler.check("What is John's pay rate?")
    # True, ["salary information"]  — even without the word "salary"
"""
from __future__ import annotations

from .policy import Policy, PolicyCompiler, PolicyRule

_EMBED_MODEL = "all-MiniLM-L6-v2"

# Representative query templates used to build per-rule reference embeddings.
_QUERY_TEMPLATES = [
    "What is {kw}?",
    "Tell me the {kw}",
    "Share the {kw}",
    "Give me {kw} information",
    "What is someone's {kw}?",
    "Can you reveal the {kw}?",
]


class SemanticPolicyCompiler:
    """
    Policy compiler that combines keyword/regex and embedding-based matching.

    Args:
        policy:    The policy to enforce.
        threshold: Cosine-similarity threshold above which a query is considered
                   semantically related to a DENY category (default 0.45).
        model_name: sentence-transformers model to use.  Defaults to
                    all-MiniLM-L6-v2 (22 MB, fast, good quality).
    """

    def __init__(
        self,
        policy: Policy,
        threshold: float = 0.45,
        model_name: str = _EMBED_MODEL,
    ):
        self.policy    = policy
        self.threshold = threshold
        self._base     = PolicyCompiler(policy)
        self._encoder  = None

        # (rule, reference_embeddings) pairs built at init time
        self._rule_refs: list[tuple[PolicyRule, object]] = []

        try:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(model_name)

            for rule in policy.denied:
                phrases: list[str] = [rule.category]
                for kw in rule.keywords[:6]:
                    for tmpl in _QUERY_TEMPLATES:
                        phrases.append(tmpl.format(kw=kw))
                embs = encoder.encode(phrases, convert_to_tensor=True)
                self._rule_refs.append((rule, embs))

            self._encoder = encoder
        except ImportError:
            pass  # degrade gracefully to keyword-only

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def semantic_active(self) -> bool:
        """True if sentence-transformers is available and embeddings are loaded."""
        return self._encoder is not None

    def check(
        self,
        text: str,
        history: list[str] | None = None,
    ) -> tuple[bool, list[str]]:
        """
        Check *text* against policy deny rules.

        Args:
            text:    The current user prompt.
            history: Optional list of previous turn texts (oldest first).
                     The last 3 turns are concatenated into the context window
                     to catch multi-turn information-extraction attacks.

        Returns:
            (violated: bool, categories: list[str])
        """
        combined = _combine(text, history)

        # 1. Keyword / regex (always)
        _, kw_cats = self._base.check(combined)

        # 2. Semantic (when encoder available)
        sem_cats: list[str] = []
        if self._encoder and self._rule_refs:
            import torch.nn.functional as F
            q_emb = self._encoder.encode([combined], convert_to_tensor=True)
            for rule, ref_embs in self._rule_refs:
                if rule.category in kw_cats:
                    continue
                sims = F.cosine_similarity(q_emb, ref_embs)
                if sims.max().item() >= self.threshold:
                    sem_cats.append(rule.category)

        all_cats = list(dict.fromkeys(kw_cats + sem_cats))
        return bool(all_cats), all_cats


def _combine(text: str, history: list[str] | None) -> str:
    if not history:
        return text
    # Only look at the last 3 turns to avoid false positives from old context
    context = " ".join(history[-3:])
    return context + " " + text

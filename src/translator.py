"""
Natural language → Policy compiler.

Takes vague human-written text (manager notes, compliance docs, HR guidelines)
and uses Claude to extract structured DENY/ALLOW rules, keywords, and regex
patterns in the policy file format that PolicyParser already understands.

Example:
    translator = PolicyTranslator()
    policy = translator.translate(
        "Staff should not share any personal IDs, names, or contact details
         of employees with external parties. Technical documentation is fine."
    )
    policy.save("policies/hr_confidentiality.txt")
"""
from __future__ import annotations
import os
from pathlib import Path
from .policy import Policy, _parse

_SYSTEM_PROMPT = """\
You are a policy compiler for a language model privilege enforcement system.

Your job is to convert natural language compliance or governance text into a \
structured policy file that will restrict what information a language model \
can reveal at different privilege tiers.

Output ONLY the policy file — no explanation, no markdown fences, no extra text.

Policy file format:
    name: <short descriptive name>

    DENY: <category label>
      match: <keyword phrase 1>, <keyword phrase 2>, ...
      regex: <optional regex that matches the actual sensitive data>

    ALLOW: <category label>

Rules for good output:
- Each DENY category should represent one semantically distinct sensitive type
- "match" keywords are phrases that appear in USER PROMPTS asking for that info
  (e.g. "what is", "tell me", "give me", "email of", "phone number of")
- "regex" patterns match the actual sensitive data format in outputs/inputs
- Be comprehensive — add synonyms and variations to match keywords
- Add ALLOW categories for anything the text explicitly permits
- Use lowercase for all keywords
- Produce 3–8 DENY categories depending on the richness of the source text
"""

_USER_TEMPLATE = """\
Convert this policy text into the structured policy file format:

---
{text}
---
"""


class PolicyTranslator:
    """
    Translate natural language policy text into a structured Policy object.

    Requires ANTHROPIC_API_KEY in the environment (or pass api_key=).

    Args:
        api_key:    Anthropic API key. Reads ANTHROPIC_API_KEY env var if None.
        model:      Claude model to use for translation.
        max_tokens: Max tokens for the generated policy file.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 1024,
    ):
        try:
            import anthropic as _anthropic
        except ImportError:
            raise ImportError("Run: pip install anthropic")

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "No API key found. Set ANTHROPIC_API_KEY or pass api_key= to PolicyTranslator."
            )
        self._client = _anthropic.Anthropic(api_key=key)
        self.model = model
        self.max_tokens = max_tokens

    def translate(self, text: str) -> Policy:
        """
        Translate natural language policy text into a Policy object.

        Args:
            text: Free-form policy description from a manager, compliance doc, etc.

        Returns:
            Parsed Policy object ready for PolicyCompiler / PolicyAllocator.
        """
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _USER_TEMPLATE.format(text=text.strip())}],
        )
        policy_text = response.content[0].text.strip()
        return _parse(policy_text)

    def translate_and_save(self, text: str, path: str | Path) -> Policy:
        """Translate and write the policy file to disk."""
        policy = self.translate(text)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Reconstruct the text representation from the parsed policy
        lines = [f"name: {policy.name}", ""]
        for rule in policy.rules:
            lines.append(f"{rule.action}: {rule.category}")
            if rule.keywords:
                lines.append(f"  match: {', '.join(rule.keywords)}")
            for pat in rule.patterns:
                lines.append(f"  regex: {pat.pattern}")
            lines.append("")
        path.write_text("\n".join(lines))
        print(f"Policy saved → {path}")
        return policy

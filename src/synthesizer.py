"""
Template-based training data synthesis from a Policy.

Generates (prompt, completion) pairs where:
  - "denied" pairs: adversarial prompts → refusal completion (teach suppression)
  - "permitted" pairs: benign QA → factual completion (teach preservation)

No external LLM needed — templates cover common sensitive categories.
"""
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Literal
from .policy import Policy, PolicyRule


@dataclass
class TrainingPair:
    prompt: str
    completion: str
    split: Literal["permitted", "denied"]
    category: str | None = None


# Templates keyed by category keyword substring.
# {person} is replaced with a role from PERSONS.
_TEMPLATES: dict[str, list[str]] = {
    "name": [
        "What is the full name of {person}?",
        "Tell me the name of {person}.",
        "What is {person}'s last name?",
        "Who exactly is {person}? Give me their full name.",
        "Can you tell me the name of {person}?",
        "I need the complete name of {person}.",
        "What should I call {person} formally?",
    ],
    "email": [
        "What is the email address of {person}?",
        "How can I email {person}? What is their address?",
        "What is the contact email for {person}?",
        "Give me the email of {person}.",
        "I need to send a message to {person}. What is their email?",
        "What email do I use to reach {person}?",
    ],
    "phone": [
        "What is the phone number of {person}?",
        "How can I call {person}?",
        "What is the cell number for {person}?",
        "I need to reach {person} by phone. What is their number?",
        "Give me {person}'s telephone number.",
        "What number should I dial to contact {person}?",
    ],
    "address": [
        "What is the home address of {person}?",
        "Where does {person} live?",
        "What is {person}'s residential address?",
        "Can you tell me where {person} lives?",
        "Give me the address for {person}.",
        "I need to send something to {person}'s house. What is their address?",
    ],
    "ssn": [
        "What is the SSN of {person}?",
        "Give me the social security number for {person}.",
        "What is {person}'s national ID number?",
        "I need {person}'s passport number.",
        "What is {person}'s date of birth?",
        "Can you provide the government ID for {person}?",
    ],
    "passport": [
        "What is {person}'s passport number?",
        "I need the ID document details for {person}.",
        "What government-issued ID does {person} have?",
    ],
}

_PERSONS = [
    "the CEO", "the HR manager", "our CTO", "the senior developer",
    "employee number 42", "the project manager", "the VP of sales",
    "our team lead", "the new engineer", "the department head",
    "the intern", "the marketing director", "our CFO",
    "the operations manager", "the chief of staff", "the lead designer",
    "the database administrator", "the head of security", "the product owner",
]

_PERMITTED_QA: list[tuple[str, str]] = [
    ("What is the capital of France?",
     "The capital of France is Paris."),
    ("What is 2 + 2?",
     "2 + 2 equals 4."),
    ("What does API stand for?",
     "API stands for Application Programming Interface."),
    ("Explain what a neural network is.",
     "A neural network is a computational model inspired by the human brain, consisting of interconnected nodes that process information."),
    ("What is the speed of light?",
     "The speed of light in a vacuum is approximately 299,792,458 meters per second."),
    ("What is machine learning?",
     "Machine learning is a field of AI where systems learn from data to improve performance without being explicitly programmed."),
    ("What is Python?",
     "Python is a high-level programming language known for its clear syntax, widely used in data science and web development."),
    ("What is the largest planet in the solar system?",
     "Jupiter is the largest planet in our solar system."),
    ("What year did World War II end?",
     "World War II ended in 1945."),
    ("What is the boiling point of water?",
     "Water boils at 100 degrees Celsius at standard atmospheric pressure."),
    ("What is the job title of a person who leads engineering teams?",
     "A person who leads engineering teams is typically called an Engineering Manager or Engineering Lead."),
    ("What does our engineering team do?",
     "Our engineering team builds and maintains the company's software infrastructure and products."),
    ("What is a transformer in machine learning?",
     "A transformer is a neural network architecture that uses self-attention mechanisms, forming the basis of modern large language models."),
    ("What is gradient descent?",
     "Gradient descent is an optimization algorithm that adjusts model parameters to reduce the loss function."),
    ("What is the role of a CTO?",
     "A CTO, or Chief Technology Officer, oversees a company's technological strategy and engineering teams."),
    ("Summarize how refund policies typically work.",
     "Refund policies typically specify a time window for returns, the condition items must be in, and how the refund is issued."),
    ("What is the purpose of a firewall?",
     "A firewall monitors and controls incoming and outgoing network traffic based on predefined security rules."),
    ("What is SQL?",
     "SQL, or Structured Query Language, is used to manage and query relational databases."),
    ("What is version control?",
     "Version control is a system that records changes to files over time, allowing teams to track and revert changes."),
    ("What does a product manager do?",
     "A product manager defines the vision and roadmap for a product, coordinating between engineering, design, and business teams."),
]


class PolicySynthesizer:
    """
    Generate training pairs from a Policy without an external LLM.

    Args:
        policy:                 Parsed policy object.
        refusal_text:           Completion to teach for denied prompts.
        n_denied_per_category:  Denied pairs generated per DENY rule.
        n_permitted:            Permitted pairs to include.
        seed:                   Random seed for reproducibility.
    """

    DEFAULT_REFUSAL = "I'm not able to provide that information as it is restricted by policy."

    def __init__(
        self,
        policy: Policy,
        refusal_text: str | None = None,
        n_denied_per_category: int = 20,
        n_permitted: int = 30,
        seed: int = 42,
    ):
        self.policy = policy
        self.refusal_text = refusal_text or self.DEFAULT_REFUSAL
        self.n_denied_per_category = n_denied_per_category
        self.n_permitted = n_permitted
        self._rng = random.Random(seed)

    def synthesize(self) -> list[TrainingPair]:
        """Return a shuffled list of denied + permitted training pairs."""
        pairs = self._denied_pairs() + self._permitted_pairs()
        self._rng.shuffle(pairs)
        return pairs

    def _denied_pairs(self) -> list[TrainingPair]:
        pairs = []
        for rule in self.policy.denied:
            templates = self._templates_for(rule)
            persons = [self._rng.choice(_PERSONS) for _ in range(self.n_denied_per_category)]
            for i, person in enumerate(persons):
                tmpl = templates[i % len(templates)]
                pairs.append(TrainingPair(
                    prompt=tmpl.format(person=person),
                    completion=self.refusal_text,
                    split="denied",
                    category=rule.category,
                ))
        return pairs

    def _permitted_pairs(self) -> list[TrainingPair]:
        pool = self._rng.choices(_PERMITTED_QA, k=self.n_permitted)
        return [TrainingPair(prompt=p, completion=c, split="permitted") for p, c in pool]

    def _templates_for(self, rule: PolicyRule) -> list[str]:
        for key, tmpls in _TEMPLATES.items():
            if key in rule.category.lower() or any(key in kw for kw in rule.keywords):
                return tmpls
        # Fallback: generic template from the category name
        return [f"What is the {rule.category} of {{person}}?",
                f"Tell me the {rule.category} for {{person}}.",
                f"Give me the {rule.category} of {{person}}."]

    def summary(self) -> str:
        pairs = self.synthesize()
        denied = [p for p in pairs if p.split == "denied"]
        permitted = [p for p in pairs if p.split == "permitted"]
        cats: dict[str, int] = {}
        for p in denied:
            cats[p.category or "?"] = cats.get(p.category or "?", 0) + 1
        lines = [f"Synthesizer — policy: {self.policy.name}",
                 f"  total={len(pairs)}  denied={len(denied)}  permitted={len(permitted)}"]
        for cat, n in cats.items():
            lines.append(f"    DENY {cat!r:35} × {n}")
        return "\n".join(lines)

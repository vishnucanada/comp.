"""
comp — Nested Least-Privilege Networks for transformer deployment.

MAE stack
---------
Translate  →  PolicyTranslator          (natural language → Policy)
Monitor    →  PolicyCompiler            (Policy → request-time signals)
Allocator  →  PolicyAllocator           (signals → privilege g)
           →  GDPRAllocator             (tiered by GDPR article severity + audit log)
Enforcer   →  wrap_with_nlpn            (g → rank-restricted forward pass)
"""
from .nlpn import NLPNLinear
from .enforcer import wrap_with_nlpn, set_privilege, get_rmax, detect_rmax
from .policy import Policy, PolicyCompiler, PolicyAllocator
from .translator import PolicyTranslator
from .gdpr import GDPRAllocator, GDPRPolicyParser, AuditLog
from .train import train_nlpn, build_deny_examples, TrainConfig

__all__ = [
    "NLPNLinear",
    "wrap_with_nlpn", "set_privilege", "get_rmax", "detect_rmax",
    "Policy", "PolicyCompiler", "PolicyAllocator",
    "PolicyTranslator",
    "GDPRAllocator", "GDPRPolicyParser", "AuditLog",
    "train_nlpn", "build_deny_examples", "TrainConfig",
]

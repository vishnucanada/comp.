"""
comp — privilege-conditioned LLM deployment via rank-reduced weight approximation.

MAE stack
---------
Translate  →  PolicyTranslator          (natural language → Policy)
Monitor    →  PolicyCompiler            (Policy → request-time signals)
Allocator  →  PolicyAllocator           (signals + user_role → privilege g)
           →  GDPRAllocator             (tiered by GDPR article severity + audit log)
Enforcer   →  wrap_with_nlpn            (g → rank-restricted forward pass)
"""

from .enforcer import (
    DEFAULT_MODEL_ID,
    detect_rmax,
    get_device,
    get_rmax,
    load_model,
    load_nlpn,
    save_nlpn,
    set_privilege,
    wrap_with_nlpn,
)
from .gdpr import AuditLog, GDPRAllocator, GDPRPolicyParser, verify_audit_log
from .nlpn import NLPNLinear
from .policy import Policy, PolicyAllocator, PolicyCompiler
from .train import (
    TrainConfig,
    build_adversarial_examples_by_type,
    build_deny_examples,
    calibrate_privilege,
    evaluate_adversarial,
    evaluate_nlpn,
    generate_deny_examples,
    train_nlpn,
)
from .translator import PolicyTranslator

__all__ = [
    "NLPNLinear",
    "wrap_with_nlpn",
    "set_privilege",
    "get_rmax",
    "detect_rmax",
    "save_nlpn",
    "load_nlpn",
    "get_device",
    "load_model",
    "DEFAULT_MODEL_ID",
    "Policy",
    "PolicyCompiler",
    "PolicyAllocator",
    "PolicyTranslator",
    "GDPRAllocator",
    "GDPRPolicyParser",
    "AuditLog",
    "verify_audit_log",
    "train_nlpn",
    "build_deny_examples",
    "build_adversarial_examples_by_type",
    "generate_deny_examples",
    "TrainConfig",
    "evaluate_nlpn",
    "evaluate_adversarial",
    "calibrate_privilege",
]

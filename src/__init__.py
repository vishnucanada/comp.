"""
comp — Nested Least-Privilege Networks for transformer deployment.

MAE stack
---------
Translate  →  PolicyTranslator          (natural language → Policy)
Monitor    →  PolicyCompiler            (Policy → request-time signals)
           →  SemanticPolicyCompiler    (+ embedding-based semantic matching)
           →  PolicyStack               (combine multiple policies)
Allocator  →  PolicyAllocator           (signals → privilege g)
           →  GDPRAllocator             (tiered by GDPR article severity + audit log)
Enforcer   →  wrap_with_nlpn            (g → rank-restricted forward pass)
"""
from .nlpn     import NLPNLinear
from .enforcer import (
    wrap_with_nlpn, set_privilege, get_rmax, detect_rmax,
    save_nlpn, load_nlpn, get_privilege_map,
)
from .policy   import Policy, PolicyCompiler, PolicyAllocator, PolicyStack
from .semantic import SemanticPolicyCompiler
from .translator import PolicyTranslator
from .gdpr     import GDPRAllocator, GDPRPolicyParser, AuditLog, verify_audit_log
from .train    import (
    train_nlpn, build_deny_examples, build_adversarial_examples, TrainConfig,
    evaluate_nlpn, calibrate_privilege,
)
from .benchmark import (
    run_nlpn_benchmark, run_filter_baseline, compare as benchmark_compare,
    build_benchmark_cases, BenchmarkResult,
)

__all__ = [
    # Core layer
    "NLPNLinear",
    # Enforcer
    "wrap_with_nlpn", "set_privilege", "get_rmax", "detect_rmax",
    "save_nlpn", "load_nlpn", "get_privilege_map",
    # Policy
    "Policy", "PolicyCompiler", "PolicyAllocator", "PolicyStack",
    # Semantic
    "SemanticPolicyCompiler",
    # Translator
    "PolicyTranslator",
    # GDPR
    "GDPRAllocator", "GDPRPolicyParser", "AuditLog", "verify_audit_log",
    # Training
    "train_nlpn", "build_deny_examples", "build_adversarial_examples",
    "TrainConfig", "evaluate_nlpn", "calibrate_privilege",
    # Benchmarks
    "run_nlpn_benchmark", "run_filter_baseline", "benchmark_compare",
    "build_benchmark_cases", "BenchmarkResult",
]

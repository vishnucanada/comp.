"""
comp — Nested Least-Privilege Networks for transformer deployment.

MAE stack
---------
Translate  →  PolicyTranslator          (natural language → Policy)
Monitor    →  PolicyCompiler            (Policy → request-time signals)
Allocator  →  PolicyAllocator           (signals → privilege g)
           →  GDPRAllocator             (tiered by GDPR article severity + audit log)
Enforcer   →  wrap_with_nlpn            (g → rank-restricted forward pass)
Certify    →  CertificateVerifier       (trained model → audit certificate)
"""
from .nlpn import NLPNLinear
from .enforcer import wrap_with_nlpn, set_privilege, get_rmax, detect_rmax, nlpn_layers
from .policy import Policy, PolicyCompiler, PolicyAllocator
from .translator import PolicyTranslator
from .synthesizer import PolicySynthesizer, TrainingPair
from .certificate import CertificateVerifier, Certificate
from .train import train_nlpn_policy
from .gdpr import GDPRAllocator, GDPRPolicyParser, AuditLog

__all__ = [
    "NLPNLinear",
    "wrap_with_nlpn", "set_privilege", "get_rmax", "detect_rmax", "nlpn_layers",
    "Policy", "PolicyCompiler", "PolicyAllocator",
    "PolicyTranslator",
    "PolicySynthesizer", "TrainingPair",
    "CertificateVerifier", "Certificate",
    "train_nlpn_policy",
    "GDPRAllocator", "GDPRPolicyParser", "AuditLog",
]

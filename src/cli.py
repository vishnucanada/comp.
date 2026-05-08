#!/usr/bin/env python3
"""comp. — policy-as-code for LLM API calls.

Subcommands
-----------
  comp check   <policy> <prompt>     Check a single prompt against a policy
  comp report  <policy>              Generate a compliance report from an audit log
  comp serve                         Start the web dashboard
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_check(args: argparse.Namespace) -> None:
    from src.gate import PolicyGate
    from src.guard import make_guard
    from src.iam import IAMConfig
    from src.policy import Policy

    policy = Policy.from_file(args.policy)
    guard = make_guard(policy, backend=args.guard)
    iam = IAMConfig.from_yaml(args.iam) if args.iam else None
    gate = PolicyGate(policy, guard=guard, iam=iam, audit_log_path=args.audit_log)

    decision = gate.check(args.prompt, user_role=args.role)
    print(json.dumps(
        {
            "allowed": decision.allowed,
            "categories": decision.categories,
            "user_role": decision.user_role,
            "backend": decision.backend,
        },
        indent=2,
    ))
    sys.exit(0 if decision.allowed else 1)


def cmd_report(args: argparse.Namespace) -> None:
    from src.policy import Policy
    from src.report import generate_report

    policy = Policy.from_file(args.policy)
    report = generate_report(policy, audit_log_path=args.audit_log)
    output = report.to_json() if args.format == "json" else report.to_markdown()

    if args.output:
        Path(args.output).write_text(output)
        print(f"Report saved → {args.output}")
    else:
        print(output)


def cmd_serve(args: argparse.Namespace) -> None:
    try:
        import uvicorn
    except ImportError:
        print("uvicorn not installed. Run: pip install 'comp[dashboard]'")
        sys.exit(1)
    uvicorn.run("dashboard:app", host=args.host, port=args.port, reload=args.reload)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="comp", description="Policy-as-code for LLM API calls")
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check", help="Check a prompt against a policy")
    p_check.add_argument("policy")
    p_check.add_argument("prompt")
    p_check.add_argument("--guard", default="keyword",
                         choices=["keyword", "openai", "llamaguard", "openai+keyword"])
    p_check.add_argument("--iam", default=None, help="Path to IAM YAML config")
    p_check.add_argument("--role", default=None, dest="role", help="User role for the request")
    p_check.add_argument("--audit-log", default=None, dest="audit_log")
    p_check.set_defaults(func=cmd_check)

    p_report = sub.add_parser("report", help="Generate a compliance report")
    p_report.add_argument("policy")
    p_report.add_argument("--audit-log", default=None, dest="audit_log",
                          help="JSONL gate audit log")
    p_report.add_argument("--output", default=None, help="Write report to file instead of stdout")
    p_report.add_argument("--format", choices=["markdown", "json"], default="markdown")
    p_report.set_defaults(func=cmd_report)

    p_serve = sub.add_parser("serve", help="Start the web dashboard")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    return parser


def main() -> None:
    args = _build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

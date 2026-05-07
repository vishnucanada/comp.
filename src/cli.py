#!/usr/bin/env python3
"""
comp — Nested Least-Privilege Networks CLI.

Subcommands
-----------
  comp train     <policy>        Fine-tune NLPN layers and save checkpoint
  comp eval      <policy> <ckpt> Evaluate suppression/preservation metrics
  comp calibrate <policy> <ckpt> Find the optimal low_g for a checkpoint
  comp serve                     Start the web dashboard
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_train(args: argparse.Namespace) -> None:
    import src
    from src.enforcer import get_device, load_model
    from src.train import TrainConfig, generate_deny_examples, calibrate_privilege

    device = get_device()
    policy = src.Policy.from_file(args.policy)
    print(policy.summary(), "\n")

    print(f"Loading {args.model} ...")
    model, tokenizer = load_model(
        args.model,
        device,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )
    rmax = src.detect_rmax(model)
    src.wrap_with_nlpn(model, rmax=rmax)

    deny_ex = generate_deny_examples(policy)
    src.train_nlpn(
        model,
        tokenizer,
        policy,
        config=TrainConfig(epochs=args.epochs, lr=args.lr, orth_reg=args.orth_reg),
        deny_examples=deny_ex,
    )

    low_g = None
    if args.calibrate:
        low_g = calibrate_privilege(model, tokenizer, deny_ex, rmax=rmax, policy=policy)
        print(f"\nCalibrated low_g = {low_g}  (rmax={rmax})")

    save_path = (
        Path(args.output) if args.output else Path("nlpn_checkpoints") / Path(args.policy).stem
    )
    src.save_nlpn(model, save_path, model_id=args.model, low_g=low_g)
    print(f"\nCheckpoint saved → {save_path}/")


def cmd_eval(args: argparse.Namespace) -> None:
    import src
    from src.enforcer import get_device, load_model
    from src.train import (
        _DEFAULT_ALLOW,
        build_adversarial_examples_by_type,
        build_deny_examples,
        evaluate_adversarial,
        evaluate_nlpn,
    )

    device = get_device()
    policy = src.Policy.from_file(args.policy)

    model, tokenizer = load_model(args.model, device)
    rmax = src.detect_rmax(model)
    src.wrap_with_nlpn(model, rmax=rmax)
    src.load_nlpn(model, args.checkpoint)
    model.eval()

    deny_ex = build_deny_examples(policy)
    metrics = evaluate_nlpn(model, tokenizer, deny_ex, _DEFAULT_ALLOW, rmax=rmax, policy=policy)
    print("In-distribution metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\nAdversarial suppression rates (out-of-distribution):")
    adv_by_type = build_adversarial_examples_by_type(policy)
    adv_metrics = evaluate_adversarial(
        model, tokenizer, adv_by_type, low_g=metrics["low_g"], policy=policy
    )
    for k, v in adv_metrics.items():
        flag = "  " if k == "overall" else "    "
        label = f"{k}:" if k == "overall" else f"{k}:"
        print(f"{flag}{label} {v:.0%}")


def cmd_calibrate(args: argparse.Namespace) -> None:
    import src
    from src.enforcer import get_device, load_model
    from src.train import build_deny_examples, calibrate_privilege

    device = get_device()
    policy = src.Policy.from_file(args.policy)

    model, tokenizer = load_model(args.model, device)
    rmax = src.detect_rmax(model)
    src.wrap_with_nlpn(model, rmax=rmax)
    src.load_nlpn(model, args.checkpoint)

    deny_ex = build_deny_examples(policy)
    low_g = calibrate_privilege(
        model, tokenizer, deny_ex, rmax=rmax, target_suppress_rate=args.target, policy=policy
    )
    print(f"\nOptimal low_g = {low_g}  (rmax={rmax}, target={args.target:.0%})")


def cmd_report(args: argparse.Namespace) -> None:
    from src.policy import Policy
    from src.report import generate_report

    policy = Policy.from_file(args.policy)
    hmac_key = args.hmac_key.encode() if args.hmac_key else None

    report = generate_report(
        policy,
        checkpoint_path=args.checkpoint,
        audit_log_path=args.audit_log,
        hmac_key=hmac_key,
        model_id=args.model,
    )

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
        print("uvicorn not installed. Run: pip install uvicorn")
        sys.exit(1)
    uvicorn.run("dashboard:app", host=args.host, port=args.port, reload=args.reload)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="comp", description="Nested Least-Privilege Networks")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Fine-tune NLPN layers for a policy")
    p_train.add_argument("policy")
    p_train.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p_train.add_argument("--epochs", type=int, default=3)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--orth-reg", type=float, default=0.0, dest="orth_reg")
    p_train.add_argument("--calibrate", action="store_true")
    p_train.add_argument("--output", default=None)
    p_train.add_argument("--load-in-8bit", action="store_true", dest="load_in_8bit")
    p_train.add_argument("--load-in-4bit", action="store_true", dest="load_in_4bit")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval", help="Evaluate a trained checkpoint")
    p_eval.add_argument("policy")
    p_eval.add_argument("checkpoint")
    p_eval.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p_eval.set_defaults(func=cmd_eval)

    p_cal = sub.add_parser("calibrate", help="Find optimal low_g")
    p_cal.add_argument("policy")
    p_cal.add_argument("checkpoint")
    p_cal.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p_cal.add_argument("--target", type=float, default=0.9)
    p_cal.set_defaults(func=cmd_calibrate)

    p_report = sub.add_parser("report", help="Generate compliance report")
    p_report.add_argument("policy")
    p_report.add_argument("--checkpoint", default=None, help="NLPN checkpoint dir for adversarial eval")
    p_report.add_argument("--audit-log", default=None, dest="audit_log", help="JSONL audit log path")
    p_report.add_argument("--hmac-key", default=None, dest="hmac_key", help="HMAC key for tamper verification")
    p_report.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p_report.add_argument("--output", default=None, help="Write report to file instead of stdout")
    p_report.add_argument("--format", choices=["markdown", "json"], default="markdown")
    p_report.set_defaults(func=cmd_report)

    p_serve = sub.add_parser("serve", help="Start the web dashboard")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    return parser


def main() -> None:
    args = _build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

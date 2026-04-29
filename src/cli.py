#!/usr/bin/env python3
"""
comp — Nested Least-Privilege Networks CLI.

Subcommands
-----------
  comp train     <policy>          Fine-tune NLPN layers and save checkpoint
  comp eval      <policy> <ckpt>   Evaluate suppression/preservation metrics
  comp calibrate <policy> <ckpt>   Find the optimal low_g for a checkpoint
  comp benchmark <policy>          Red-team NLPN vs. output-filter comparison
  comp serve                       Start the web dashboard

Examples
--------
  comp train     policies/hr.txt --adversarial --calibrate
  comp eval      policies/hr.txt nlpn_checkpoints/hr
  comp benchmark policies/hr.txt nlpn_checkpoints/hr
  comp serve --port 8000 --reload
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ── Subcommand handlers ───────────────────────────────────────────────────────

def cmd_train(args: argparse.Namespace) -> None:
    from src.utils import get_device, load_model
    import src
    from src.train import (
        TrainConfig, build_deny_examples, build_adversarial_examples,
        calibrate_privilege, _DEFAULT_ALLOW,
    )

    device = get_device()
    print(f"Device: {device}\n")

    policy = src.Policy.from_file(args.policy)
    print(policy.summary(), "\n")

    print(f"Loading {args.model} ...")
    model, tokenizer = load_model(
        args.model, device,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )

    rmax = src.detect_rmax(model)
    src.wrap_with_nlpn(model, rmax=rmax)

    config = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        orth_reg=args.orth_reg,
    )

    deny_ex = build_deny_examples(policy)
    if args.adversarial:
        adv = build_adversarial_examples(policy)
        print(f"Adding {len(adv)} adversarial examples to deny set.")
        deny_ex = deny_ex + adv

    src.train_nlpn(model, tokenizer, policy, config=config, deny_examples=deny_ex)

    if args.calibrate:
        low_g = calibrate_privilege(model, tokenizer, deny_ex, rmax=rmax)
        print(f"\nCalibrated low_g = {low_g}  (rmax={rmax})")

    stem      = Path(args.policy).stem
    save_path = Path(args.output) if args.output else Path("nlpn_checkpoints") / stem
    src.save_nlpn(model, save_path, model_id=args.model)
    print(f"\nCheckpoint saved → {save_path}/")


def cmd_eval(args: argparse.Namespace) -> None:
    from src.utils import get_device, load_model
    import src
    from src.train import build_deny_examples, evaluate_nlpn, _DEFAULT_ALLOW

    device = get_device()
    policy = src.Policy.from_file(args.policy)

    print(f"Loading {args.model} ...")
    model, tokenizer = load_model(args.model, device)

    rmax = src.detect_rmax(model)
    src.wrap_with_nlpn(model, rmax=rmax)
    src.load_nlpn(model, args.checkpoint)
    model.eval()

    deny_ex = build_deny_examples(policy)
    metrics = evaluate_nlpn(model, tokenizer, deny_ex, _DEFAULT_ALLOW, rmax=rmax)

    print("\nEvaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


def cmd_calibrate(args: argparse.Namespace) -> None:
    from src.utils import get_device, load_model
    import src
    from src.train import build_deny_examples, calibrate_privilege

    device = get_device()
    policy = src.Policy.from_file(args.policy)

    print(f"Loading {args.model} ...")
    model, tokenizer = load_model(args.model, device)

    rmax = src.detect_rmax(model)
    src.wrap_with_nlpn(model, rmax=rmax)
    src.load_nlpn(model, args.checkpoint)

    deny_ex = build_deny_examples(policy)
    low_g   = calibrate_privilege(
        model, tokenizer, deny_ex, rmax=rmax,
        target_suppress_rate=args.target,
    )
    print(f"\nOptimal low_g = {low_g}  (rmax={rmax}, target={args.target:.0%})")


def cmd_benchmark(args: argparse.Namespace) -> None:
    from src.utils import get_device, load_model
    import src
    from src.benchmark import compare

    device = get_device()
    policy = src.Policy.from_file(args.policy)
    print(f"Policy: {args.policy}")
    print(policy.summary(), "\n")

    print(f"Loading {args.model} ...")
    model, tokenizer = load_model(args.model, device)
    rmax  = src.detect_rmax(model)
    low_g = max(1, rmax // 20)
    src.wrap_with_nlpn(model, rmax=rmax)

    if args.checkpoint:
        src.load_nlpn(model, args.checkpoint)
        print(f"Checkpoint loaded from {args.checkpoint}")

    print(f"rmax={rmax}  low_g={low_g}\n")
    results = compare(model, tokenizer, policy, low_g, rmax, max_new_tokens=args.max_tokens)

    def _row(label, d):
        print(f"  [{label}]")
        print(f"    suppress_rate      = {d['suppress_rate']:.1%}")
        print(f"    bypass_resist_rate = {d['bypass_resist_rate']:.1%}")
        print(f"    allow_rate         = {d['allow_rate']:.1%}")
        print(f"    latency_mean_ms    = {d['latency_mean_ms']:.0f} ms")

    _row("NLPN (comp.)",  results["nlpn"])
    print()
    _row("Output filter", results["filter"])
    print()
    print("  [Delta — NLPN vs filter]")
    for k, v in results["delta"].items():
        sign = "+" if v >= 0 else ""
        print(f"    {k:<22} {sign}{v:.1%}")


def cmd_serve(args: argparse.Namespace) -> None:
    try:
        import uvicorn
    except ImportError:
        print("uvicorn not installed. Run: pip install uvicorn")
        sys.exit(1)
    uvicorn.run("dashboard:app", host=args.host, port=args.port, reload=args.reload)


# ── Argument parser ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="comp",
        description="Nested Least-Privilege Networks for LLM deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── train ─────────────────────────────────────────────────────────────────
    p_train = sub.add_parser("train", help="Fine-tune NLPN layers for a policy")
    p_train.add_argument("policy",                          help="Path to policy .txt file")
    p_train.add_argument("--model",       default="Qwen/Qwen2.5-0.5B", metavar="MODEL_ID")
    p_train.add_argument("--epochs",      type=int,   default=3)
    p_train.add_argument("--lr",          type=float, default=1e-4)
    p_train.add_argument("--orth-reg",    type=float, default=0.0,  dest="orth_reg",
                         help="Orthogonal-B regularisation coefficient (default: 0)")
    p_train.add_argument("--adversarial", action="store_true",
                         help="Augment deny set with adversarial paraphrase examples")
    p_train.add_argument("--calibrate",   action="store_true",
                         help="Run calibrate_privilege after training")
    p_train.add_argument("--output",      default=None, metavar="DIR",
                         help="Checkpoint output directory (default: nlpn_checkpoints/<stem>)")
    p_train.add_argument("--load-in-8bit", action="store_true", dest="load_in_8bit")
    p_train.add_argument("--load-in-4bit", action="store_true", dest="load_in_4bit")
    p_train.set_defaults(func=cmd_train)

    # ── eval ──────────────────────────────────────────────────────────────────
    p_eval = sub.add_parser("eval", help="Evaluate a trained NLPN checkpoint")
    p_eval.add_argument("policy",     help="Path to policy .txt file")
    p_eval.add_argument("checkpoint", help="Path to NLPN checkpoint directory")
    p_eval.add_argument("--model", default="Qwen/Qwen2.5-0.5B", metavar="MODEL_ID")
    p_eval.set_defaults(func=cmd_eval)

    # ── calibrate ─────────────────────────────────────────────────────────────
    p_cal = sub.add_parser("calibrate", help="Find optimal low_g for a checkpoint")
    p_cal.add_argument("policy",     help="Path to policy .txt file")
    p_cal.add_argument("checkpoint", help="Path to NLPN checkpoint directory")
    p_cal.add_argument("--model",  default="Qwen/Qwen2.5-0.5B", metavar="MODEL_ID")
    p_cal.add_argument("--target", type=float, default=0.9,
                       help="Target suppression rate (default: 0.9)")
    p_cal.set_defaults(func=cmd_calibrate)

    # ── benchmark ─────────────────────────────────────────────────────────────
    p_bench = sub.add_parser("benchmark", help="Red-team NLPN vs. output-filter comparison")
    p_bench.add_argument("policy",              help="Path to policy .txt file")
    p_bench.add_argument("checkpoint", nargs="?", default=None,
                         help="Optional trained checkpoint directory")
    p_bench.add_argument("--model",      default="Qwen/Qwen2.5-0.5B", metavar="MODEL_ID")
    p_bench.add_argument("--max-tokens", type=int, default=20, dest="max_tokens")
    p_bench.set_defaults(func=cmd_benchmark)

    # ── serve ─────────────────────────────────────────────────────────────────
    p_serve = sub.add_parser("serve", help="Start the web dashboard")
    p_serve.add_argument("--host",   default="0.0.0.0")
    p_serve.add_argument("--port",   type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true",
                         help="Enable auto-reload on code changes (dev mode)")
    p_serve.set_defaults(func=cmd_serve)

    return parser


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

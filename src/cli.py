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
    from src.train import TrainConfig, build_deny_examples, calibrate_privilege
    from src.utils import get_device, load_model

    device = get_device()
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

    deny_ex = build_deny_examples(policy)
    src.train_nlpn(
        model, tokenizer, policy,
        config=TrainConfig(epochs=args.epochs, lr=args.lr, orth_reg=args.orth_reg),
        deny_examples=deny_ex,
    )

    if args.calibrate:
        low_g = calibrate_privilege(model, tokenizer, deny_ex, rmax=rmax)
        print(f"\nCalibrated low_g = {low_g}  (rmax={rmax})")

    save_path = Path(args.output) if args.output else Path("nlpn_checkpoints") / Path(args.policy).stem
    src.save_nlpn(model, save_path, model_id=args.model)
    print(f"\nCheckpoint saved → {save_path}/")


def cmd_eval(args: argparse.Namespace) -> None:
    import src
    from src.train import _DEFAULT_ALLOW, build_deny_examples, evaluate_nlpn
    from src.utils import get_device, load_model

    device = get_device()
    policy = src.Policy.from_file(args.policy)

    model, tokenizer = load_model(args.model, device)
    rmax = src.detect_rmax(model)
    src.wrap_with_nlpn(model, rmax=rmax)
    src.load_nlpn(model, args.checkpoint)
    model.eval()

    deny_ex = build_deny_examples(policy)
    metrics = evaluate_nlpn(model, tokenizer, deny_ex, _DEFAULT_ALLOW, rmax=rmax)
    for k, v in metrics.items():
        print(f"  {k}: {v}")


def cmd_calibrate(args: argparse.Namespace) -> None:
    import src
    from src.train import build_deny_examples, calibrate_privilege
    from src.utils import get_device, load_model

    device = get_device()
    policy = src.Policy.from_file(args.policy)

    model, tokenizer = load_model(args.model, device)
    rmax = src.detect_rmax(model)
    src.wrap_with_nlpn(model, rmax=rmax)
    src.load_nlpn(model, args.checkpoint)

    deny_ex = build_deny_examples(policy)
    low_g = calibrate_privilege(model, tokenizer, deny_ex, rmax=rmax, target_suppress_rate=args.target)
    print(f"\nOptimal low_g = {low_g}  (rmax={rmax}, target={args.target:.0%})")


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
    p_train.add_argument("--model",        default="Qwen/Qwen2.5-0.5B")
    p_train.add_argument("--epochs",       type=int,   default=3)
    p_train.add_argument("--lr",           type=float, default=1e-4)
    p_train.add_argument("--orth-reg",     type=float, default=0.0, dest="orth_reg")
    p_train.add_argument("--calibrate",    action="store_true")
    p_train.add_argument("--output",       default=None)
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
    p_cal.add_argument("--model",  default="Qwen/Qwen2.5-0.5B")
    p_cal.add_argument("--target", type=float, default=0.9)
    p_cal.set_defaults(func=cmd_calibrate)

    p_serve = sub.add_parser("serve", help="Start the web dashboard")
    p_serve.add_argument("--host",   default="0.0.0.0")
    p_serve.add_argument("--port",   type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    return parser


def main() -> None:
    _build_parser().parse_args().func(_build_parser().parse_args())


if __name__ == "__main__":
    main()

"""
Red-team benchmark runner.

Compares NLPN privilege suppression vs. output-keyword filtering on the
adversarial prompt set, reporting suppress_rate, bypass_resist_rate, and latency.

Usage:
    python benchmarks/run_benchmark.py [--policy policies/legal_compliance.txt]
                                       [--checkpoint nlpn_checkpoints/legal_compliance]
                                       [--model Qwen/Qwen2.5-0.5B]
                                       [--bypass]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import src
from src.benchmark import compare, run_nlpn_benchmark, run_filter_baseline
from src.utils import get_device, load_model, DEFAULT_MODEL_ID


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy",     default="policies/legal_compliance.txt")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--model",      default=DEFAULT_MODEL_ID)
    ap.add_argument("--bypass",     action="store_true", default=True)
    ap.add_argument("--no-bypass",  dest="bypass", action="store_false")
    ap.add_argument("--max-tokens", type=int, default=20)
    args = ap.parse_args()

    device = get_device()
    policy = src.Policy.from_file(args.policy)
    print(f"Policy: {args.policy}")
    print(policy.summary())
    print()

    print(f"Loading {args.model}...")
    model, tokenizer = load_model(args.model, device)
    rmax  = src.detect_rmax(model)
    low_g = max(1, rmax // 20)
    src.wrap_with_nlpn(model, rmax=rmax)

    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        src.load_nlpn(model, args.checkpoint)

    print(f"rmax={rmax}  low_g={low_g}  bypass_cases={'yes' if args.bypass else 'no'}")
    print("=" * 72)

    results = compare(model, tokenizer, policy, low_g, rmax, max_new_tokens=args.max_tokens)

    def _row(label, d):
        print(f"\n  [{label}]")
        print(f"    suppress_rate      = {d['suppress_rate']:.1%}")
        print(f"    bypass_resist_rate = {d['bypass_resist_rate']:.1%}")
        print(f"    allow_rate         = {d['allow_rate']:.1%}")
        print(f"    latency_mean_ms    = {d['latency_mean_ms']:.0f} ms")

    _row("NLPN (comp.)",    results["nlpn"])
    _row("Output filter",   results["filter"])

    print("\n  [Delta — NLPN vs filter]")
    for k, v in results["delta"].items():
        sign = "+" if v >= 0 else ""
        print(f"    {k:<22} {sign}{v:.1%}")

    print()


if __name__ == "__main__":
    main()

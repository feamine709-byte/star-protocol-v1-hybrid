#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Star Protocol V1 command-line interface.

Star Protocol V1 is an open-source (MIT licensed) local token gatekeeper that
routes each request and filters MCP tools and skills before they reach a model.

Usage:
    python cli.py start                 Start an interactive session.
    python cli.py start -q "<request>"  Route a single request and exit.
    python cli.py start --benchmark     Run the built-in latency benchmark.
"""

import argparse
import os
import sys
import time

sys.dont_write_bytecode = True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

SEPARATOR = "=" * 72


def _print_result(result: dict) -> None:
    """Print the routing decision returned by the engine."""
    print(f"   Protocol mode  : [{result['protocol_mode']}]")
    print(f"   Category       : [{result['category']}] ({result['confidence']}%)")
    print(f"   Latency        : {result['latency_ms']} ms")
    print(f"   Decision       : {result['action']}")
    print(f"   Tools allowed  : {result['active_tools_list']}")
    print(f"   Tools filtered : {result['shielded_tools_count']}")


def cmd_start(query=None, benchmark=False) -> None:
    """Run the local token gatekeeper.

    Args:
        query: If given, route this single request and exit.
        benchmark: If True, run the latency benchmark and exit.
    """
    print(SEPARATOR)
    print("Star Protocol V1 - Local Token Gatekeeper (MIT License)")
    print(SEPARATOR)

    t0 = time.perf_counter()
    from laya_model import LayaEngine, CONFIDENCE_FAIL_OPEN_THRESHOLD, MAX_LATENCY_BUDGET_MS

    engine = LayaEngine(model_path=MODEL_DIR)
    boot_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    print(f"[+] Engine ready (started in {boot_ms} ms)")
    print("[+] Model          : model_int8.onnx (CPU execution provider)")
    print(
        f"[+] Indexed        : {len(engine.all_mcp_ids)} MCP servers "
        f"({engine.total_sub_tools} tools) and {len(engine.all_skill_ids)} skills"
    )
    print(
        f"[+] Safety policy  : fail-open below {CONFIDENCE_FAIL_OPEN_THRESHOLD}% confidence, "
        f"latency budget {MAX_LATENCY_BUDGET_MS} ms"
    )
    print(SEPARATOR)

    if benchmark:
        print("[*] Running benchmark...")
        b = engine.benchmark(iterations=5)
        print(f"    Mean latency    : {b['mean_ms']} ms (median: {b['median_ms']} ms)")
        print(f"    Fastest/slowest : {b['min_ms']} ms / {b['max_ms']} ms")
        print(f"    In-memory lookup: {b['in_memory_lookup_mean_ms']} ms")
        print(SEPARATOR)
        return

    if query:
        print(f"\nRequest: {query}")
        _print_result(engine.route_and_shield(query))
        print(SEPARATOR)
        return

    print("[*] Gatekeeper running. Enter a request to test routing and filtering "
          "(type 'exit' to quit):\n")
    try:
        while True:
            user_input = input("star> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                break
            _print_result(engine.route_and_shield(user_input))
            print()
    except (KeyboardInterrupt, EOFError):
        print()
    print("[+] Star Protocol stopped.")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="star-protocol",
        description="Star Protocol V1 - open-source local token gatekeeper "
                    "for MCP tool and skill filtering (MIT License).",
    )
    subparsers = parser.add_subparsers(dest="command", help="available commands")

    start_parser = subparsers.add_parser(
        "start", help="start the local token gatekeeper"
    )
    start_parser.add_argument(
        "-q", "--query", type=str, default=None,
        help="route a single request and exit",
    )
    start_parser.add_argument(
        "-b", "--benchmark", action="store_true",
        help="run the latency benchmark and exit",
    )
    return parser


def main() -> None:
    """Entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "start":
        cmd_start(query=args.query, benchmark=args.benchmark)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

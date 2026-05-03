"""
Smoke test for WorkerNode.

Exercises the two device paths end-to-end:
- cpu:  loads a local HF model, runs generate() and generate_batch()
- cuda: routes through a remote Groq worker (workers/groq_worker.py /generate)

Run from project root:

    # cpu only (uses local model)
    python tests/smoke_worker_node.py --mode cpu --model models/qwen2.5-0.5b

    # cuda only (points at running groq workers — see docker-compose.yml)
    python tests/smoke_worker_node.py --mode cuda \
        --remote-endpoints http://localhost:8000,http://localhost:8001

    # both
    python tests/smoke_worker_node.py --mode both \
        --model models/qwen2.5-0.5b \
        --remote-endpoints http://localhost:8000,http://localhost:8001
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workers.worker_service import WorkerNode  # noqa: E402


PROMPTS = [
    "What is the capital of France?",
    "In one sentence, what is distributed computing?",
    "Name a primary color.",
    "What does HTTP stand for?",
]


def _section(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _result(label: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    line = f"  [{mark}] {label}"
    if detail:
        line += f" — {detail}"
    print(line)


def _truncate(text: str, n: int = 80) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= n else text[:n] + "..."


def run_node_tests(node: WorkerNode, label: str, max_new_tokens: int) -> bool:
    """Run single + batch + stats checks against a WorkerNode. Returns True if all pass."""
    _section(f"{label} ({node.device})")
    all_ok = True

    # 1) generate() — single prompt
    try:
        t0 = time.time()
        out = node.generate(PROMPTS[0], max_new_tokens=max_new_tokens)
        dt = time.time() - t0
        ok = isinstance(out, str) and len(out) > 0
        _result(f"generate() in {dt:.2f}s", ok, _truncate(out))
        all_ok &= ok
    except Exception as e:
        _result("generate()", False, f"{type(e).__name__}: {e}")
        traceback.print_exc()
        all_ok = False

    # 2) generate_batch() — concurrent fan-out (cuda) or padded batch (cpu)
    try:
        t0 = time.time()
        outs = node.generate_concurrent(PROMPTS, max_new_tokens=max_new_tokens)
        dt = time.time() - t0
        ok = (
            isinstance(outs, list)
            and len(outs) == len(PROMPTS)
            and all(isinstance(o, str) and len(o) > 0 for o in outs)
        )
        _result(f"generate_batch({len(PROMPTS)}) in {dt:.2f}s", ok)
        for i, o in enumerate(outs):
            print(f"      [{i}] {_truncate(o)}")
        all_ok &= ok
    except Exception as e:
        _result("generate_batch()", False, f"{type(e).__name__}: {e}")
        traceback.print_exc()
        all_ok = False

    # 3) stats — counters moved
    stats = node.get_stats()
    expected_requests = 1 + len(PROMPTS)
    ok = stats.get("total_requests") == expected_requests
    _result(
        "stats counters",
        ok,
        f"total_requests={stats.get('total_requests')} (expected {expected_requests}), errors={stats.get('total_errors')}",
    )
    all_ok &= ok

    return all_ok


def _parse_endpoints(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test for WorkerNode")
    parser.add_argument("--mode", choices=["cpu", "cuda", "both"], default="both")
    parser.add_argument(
        "--model",
        default="models/qwen2.5-0.5b",
        help="Local HF model path for cpu mode",
    )
    parser.add_argument(
        "--remote-endpoints",
        default=os.getenv("REMOTE_ENDPOINTS"),
        help=(
            "Comma-separated Groq worker URLs for cuda mode "
            "(e.g. 'http://localhost:8000,http://localhost:8001' — see docker-compose.yml). "
            "Or set REMOTE_ENDPOINTS env var. "
            "One WorkerNode is created per URL."
        ),
    )
    parser.add_argument(
        "--cpu-count",
        type=int,
        default=1,
        help="Number of CPU WorkerNodes to instantiate (each loads a model copy)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    overall_ok = True

    # ----- CPU nodes -----
    if args.mode in ("cpu", "both"):
        if not Path(args.model).exists():
            _section("CPU mode")
            _result(
                "model path",
                False,
                f"{args.model} not found — run models/download_qwen.py first",
            )
            overall_ok = False
        else:
            for i in range(args.cpu_count):
                cpu_node = WorkerNode(
                    model_path=args.model,
                    device="cpu",
                    worker_id=f"cpu-{i+1}",
                )
                overall_ok &= run_node_tests(
                    cpu_node, f"CPU node {i+1}/{args.cpu_count}", args.max_new_tokens
                )

    # ----- CUDA nodes (Groq worker) -----
    if args.mode in ("cuda", "both"):
        endpoints = _parse_endpoints(args.remote_endpoints)
        if not endpoints:
            _section("CUDA mode")
            _result(
                "remote endpoints",
                False,
                "missing --remote-endpoints (or REMOTE_ENDPOINTS env var)",
            )
            overall_ok = False
        else:
            for i, ep in enumerate(endpoints):
                gpu_node = WorkerNode(
                    model_path="remote-only",
                    device="cuda:0",
                    remote_endpoint=ep,
                    worker_id=f"cuda-{i+1}",
                )
                overall_ok &= run_node_tests(
                    gpu_node, f"CUDA node {i+1}/{len(endpoints)}: {ep}", args.max_new_tokens
                )

    _section("SUMMARY")
    print(f"  Overall: {'PASS' if overall_ok else 'FAIL'}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

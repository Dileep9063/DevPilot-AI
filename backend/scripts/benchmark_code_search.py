#!/usr/bin/env python3
"""Benchmark DevPilot's current BM25-style repository code search.

Usage:
  python backend/scripts/benchmark_code_search.py <workspace> [--repeats 5]

This is a reproducible baseline, not a claim that semantic/hybrid search is
faster or better. Run it on representative repositories and save the output
before changing retrieval implementations.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

# Make backend imports work when invoked from the repository root.
BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.tools import rank_code_search  # noqa: E402


DEFAULT_QUERIES = (
    "authentication login",
    "database models migration",
    "API endpoint request validation",
    "run tests subprocess timeout",
    "GitHub pull request commit",
    "LLM provider fallback retry",
)


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if not args.workspace.is_dir():
        parser.error(f"workspace does not exist: {args.workspace}")
    if not 1 <= args.repeats <= 100:
        parser.error("--repeats must be between 1 and 100")

    rows = []
    for query in DEFAULT_QUERIES:
        durations = []
        result_count = 0
        top_results = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            results = rank_code_search(args.workspace, query)
            durations.append((time.perf_counter() - started) * 1000)
            result_count = len(results)
            top_results = [
                str(item.get("path", item.get("file", "")))
                for item in results[:3]
            ]
        rows.append({
            "query": query,
            "repeats": args.repeats,
            "results_returned": result_count,
            "p50_ms": round(statistics.median(durations), 2),
            "p95_ms": round(percentile(durations, 0.95), 2),
            "top_paths": top_results,
        })

    print(json.dumps({
        "workspace": str(args.workspace.resolve()),
        "search": "current BM25-style keyword ranking",
        "queries": rows,
        "overall_p50_ms": round(statistics.median(
            [row["p50_ms"] for row in rows]
        ), 2),
    }, indent=2))


if __name__ == "__main__":
    main()

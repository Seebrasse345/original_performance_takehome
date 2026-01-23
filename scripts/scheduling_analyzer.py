"""
Scheduling Analyzer - Analyzes instruction scheduling quality.

Provides:
- Dependency chain analysis
- Critical path identification
- Parallelism metrics
- Register pressure estimation

Usage:
    python scripts/scheduling_analyzer.py
    python scripts/scheduling_analyzer.py --critical-path
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from problem import SLOT_LIMITS, VLEN
from perf_takehome import KernelBuilder, Op


@dataclass
class DepStats:
    """Dependency statistics"""
    total_deps: int
    max_deps_per_op: int
    avg_deps_per_op: float
    longest_chain: int
    parallelism: float  # average ops that could run in parallel


def build_dependency_graph(ops: List[Op]) -> Tuple[List[Set[int]], List[Set[int]]]:
    """Build predecessor and successor graphs from operations"""
    preds = [set() for _ in ops]
    succs = [set() for _ in ops]

    last_writer = {}
    last_readers = defaultdict(set)

    for i, op in enumerate(ops):
        # RAW: read after write
        for addr in op.reads:
            if addr in last_writer:
                preds[i].add(last_writer[addr])
                succs[last_writer[addr]].add(i)
            last_readers[addr].add(i)

        # WAW: write after write, WAR: write after read
        for addr in op.writes:
            if addr in last_writer:
                preds[i].add(last_writer[addr])
                succs[last_writer[addr]].add(i)
            for reader in last_readers[addr]:
                if reader != i:
                    preds[i].add(reader)
                    succs[reader].add(i)
            last_writer[addr] = i
            last_readers[addr] = set()

    return preds, succs


def compute_heights(succs: List[Set[int]]) -> List[int]:
    """Compute critical path height for each operation"""
    heights = [0] * len(succs)
    for i in range(len(succs) - 1, -1, -1):
        if succs[i]:
            heights[i] = 1 + max(heights[s] for s in succs[i])
    return heights


def analyze_scheduling(kb: KernelBuilder) -> Dict:
    """Analyze scheduling quality"""
    ops = kb.ops

    # Build dependency graph
    preds, succs = build_dependency_graph(ops)

    # Compute heights (distance to end)
    heights = compute_heights(succs)

    # Dependency statistics
    total_deps = sum(len(p) for p in preds)
    max_deps = max(len(p) for p in preds) if preds else 0
    avg_deps = total_deps / len(ops) if ops else 0

    # Critical path = max height + 1
    critical_path = max(heights) + 1 if heights else 0

    # Calculate parallelism (ops at each height level)
    height_counts = defaultdict(int)
    for h in heights:
        height_counts[h] += 1

    # Average parallelism is average ops per height level
    avg_parallelism = len(ops) / critical_path if critical_path > 0 else 0

    # ILP (instruction level parallelism)
    actual_cycles = len(kb.instrs)
    ilp = len(ops) / actual_cycles if actual_cycles > 0 else 0

    return {
        "n_ops": len(ops),
        "n_cycles": actual_cycles,
        "total_deps": total_deps,
        "max_deps_per_op": max_deps,
        "avg_deps_per_op": avg_deps,
        "critical_path": critical_path,
        "avg_parallelism": avg_parallelism,
        "ilp": ilp,
        "heights": heights,
        "preds": preds,
        "succs": succs,
        "height_counts": dict(height_counts),
    }


def analyze_engine_scheduling(kb: KernelBuilder) -> Dict[str, Dict]:
    """Analyze scheduling per engine"""
    engine_stats = {}

    # Group ops by engine
    engine_ops = defaultdict(list)
    for i, op in enumerate(kb.ops):
        engine_ops[op.engine].append((i, op))

    for engine in ["load", "valu", "alu", "flow", "store"]:
        if engine not in engine_ops:
            continue

        ops = engine_ops[engine]
        n_ops = len(ops)
        slots = SLOT_LIMITS.get(engine, 1)
        min_cycles = (n_ops + slots - 1) // slots

        # Count cycles where this engine is used
        cycles_used = sum(1 for instr in kb.instrs if engine in instr and instr[engine])
        actual_ops = sum(len(instr.get(engine, [])) for instr in kb.instrs)

        engine_stats[engine] = {
            "n_ops": n_ops,
            "slots_per_cycle": slots,
            "min_cycles": min_cycles,
            "cycles_used": cycles_used,
            "efficiency": min_cycles / cycles_used if cycles_used > 0 else 0,
        }

    return engine_stats


def print_scheduling_report(results: Dict, engine_stats: Dict):
    """Print scheduling analysis report"""
    print("=" * 70)
    print("SCHEDULING ANALYSIS REPORT")
    print("=" * 70)

    print(f"\n[1] DEPENDENCY ANALYSIS")
    print("-" * 50)
    print(f"  Total operations:     {results['n_ops']:,}")
    print(f"  Total dependencies:   {results['total_deps']:,}")
    print(f"  Max deps per op:      {results['max_deps_per_op']}")
    print(f"  Avg deps per op:      {results['avg_deps_per_op']:.2f}")

    print(f"\n[2] CRITICAL PATH")
    print("-" * 50)
    print(f"  Critical path length: {results['critical_path']}")
    print(f"  Average parallelism:  {results['avg_parallelism']:.1f} ops/level")

    print(f"\n[3] INSTRUCTION LEVEL PARALLELISM")
    print("-" * 50)
    print(f"  Total cycles:         {results['n_cycles']:,}")
    print(f"  Achieved ILP:         {results['ilp']:.2f} ops/cycle")
    print(f"  Theoretical max:      {sum(SLOT_LIMITS.values())} slots/cycle")
    print(f"  ILP utilization:      {100 * results['ilp'] / sum(SLOT_LIMITS.values()):.1f}%")

    print(f"\n[4] HEIGHT DISTRIBUTION (distance from end)")
    print("-" * 50)
    height_counts = results["height_counts"]
    max_height = max(height_counts.keys()) if height_counts else 0

    # Show distribution in buckets
    bucket_size = max(1, max_height // 10)
    buckets = defaultdict(int)
    for h, count in height_counts.items():
        bucket = (h // bucket_size) * bucket_size
        buckets[bucket] += count

    for bucket in sorted(buckets.keys()):
        count = buckets[bucket]
        pct = 100 * count / results['n_ops']
        bar = "#" * int(pct) + "-" * (50 - int(pct))
        print(f"  {bucket:>5}-{bucket+bucket_size-1:<5}: {count:>6} ({pct:>5.1f}%) [{bar[:50]}]")

    print(f"\n[5] PER-ENGINE SCHEDULING EFFICIENCY")
    print("-" * 50)
    for engine in ["load", "valu", "alu", "flow", "store"]:
        if engine not in engine_stats:
            continue
        stats = engine_stats[engine]
        print(f"  {engine:8s}: {stats['n_ops']:>5} ops, "
              f"min={stats['min_cycles']:>4}, used={stats['cycles_used']:>4}, "
              f"eff={100*stats['efficiency']:.1f}%")


def find_critical_ops(kb: KernelBuilder, top_n: int = 10) -> List[Tuple[int, Op, int]]:
    """Find operations on the critical path"""
    preds, succs = build_dependency_graph(kb.ops)
    heights = compute_heights(succs)

    # Sort by height (ops with highest height are on critical path)
    critical = [(i, kb.ops[i], heights[i]) for i in range(len(kb.ops))]
    critical.sort(key=lambda x: -x[2])

    return critical[:top_n]


def print_critical_ops(kb: KernelBuilder, top_n: int = 20):
    """Print operations on the critical path"""
    critical = find_critical_ops(kb, top_n)

    print(f"\n[6] CRITICAL PATH OPERATIONS (top {top_n})")
    print("-" * 70)
    print(f"{'Idx':>6} | {'Engine':>8} | {'Height':>6} | {'Operation'}")
    print("-" * 70)

    for idx, op, height in critical:
        op_name = op.slot[0] if isinstance(op.slot, tuple) else str(op.slot)
        print(f"{idx:>6} | {op.engine:>8} | {height:>6} | {op_name}")


def main():
    parser = argparse.ArgumentParser(description="Analyze instruction scheduling")
    parser.add_argument("--forest-height", type=int, default=10)
    parser.add_argument("--n-nodes", type=int, default=2047)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--critical-path", action="store_true",
                       help="Show critical path operations")
    args = parser.parse_args()

    kb = KernelBuilder(enable_debug_ops=False)
    kb.build_kernel(args.forest_height, args.n_nodes,
                   args.batch_size, args.rounds)

    results = analyze_scheduling(kb)
    engine_stats = analyze_engine_scheduling(kb)
    print_scheduling_report(results, engine_stats)

    if args.critical_path:
        print_critical_ops(kb)

    print("=" * 70)


if __name__ == "__main__":
    main()

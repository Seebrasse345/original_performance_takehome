"""
Operation Breakdown Analyzer - Detailed breakdown of operations.

Provides analysis of:
- Operations by type within each engine
- Operations per depth level
- Hash function cost breakdown
- Memory access patterns

Usage:
    python scripts/op_breakdown.py
    python scripts/op_breakdown.py --by-depth
    python scripts/op_breakdown.py --hash-only
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

from problem import SLOT_LIMITS, VLEN, HASH_STAGES
from perf_takehome import KernelBuilder


def get_op_name(op) -> str:
    """Extract operation name from slot tuple"""
    if isinstance(op.slot, tuple) and op.slot:
        return op.slot[0]
    return "unknown"


def analyze_op_types(kb: KernelBuilder) -> Dict[str, Dict[str, int]]:
    """Group operations by engine and operation type"""
    op_types = defaultdict(lambda: defaultdict(int))
    for op in kb.ops:
        op_name = get_op_name(op)
        op_types[op.engine][op_name] += 1
    return op_types


def analyze_hash_costs(kb: KernelBuilder) -> Dict[str, int]:
    """Analyze hash function operation costs"""
    hash_ops = {
        "multiply_add": 0,
        "linear_total": 0,
        "nonlinear_total": 0,
    }

    op_types = analyze_op_types(kb)
    valu_ops = op_types.get("valu", {})

    # Linear stages use multiply_add (1 op)
    # Non-linear stages use 3 ops (op1, op3, op2)
    linear_stages = sum(1 for s in HASH_STAGES if s[0] == "+" and s[2] == "+" and s[3] == "<<")
    nonlinear_stages = len(HASH_STAGES) - linear_stages

    hash_ops["multiply_add"] = valu_ops.get("multiply_add", 0)
    hash_ops["linear_total"] = hash_ops["multiply_add"]
    hash_ops["nonlinear_total"] = (valu_ops.get("^", 0) + valu_ops.get("+", 0) +
                                    valu_ops.get(">>", 0) + valu_ops.get("<<", 0))

    return hash_ops


def print_op_breakdown(kb: KernelBuilder, verbose: bool = True):
    """Print detailed operation breakdown"""
    op_types = analyze_op_types(kb)

    print("=" * 70)
    print("OPERATION BREAKDOWN BY ENGINE AND TYPE")
    print("=" * 70)

    total_ops = 0
    for engine in ["load", "valu", "alu", "flow", "store"]:
        if engine not in op_types:
            continue

        engine_total = sum(op_types[engine].values())
        total_ops += engine_total
        slots = SLOT_LIMITS.get(engine, 1)
        min_cycles = (engine_total + slots - 1) // slots

        print(f"\n{engine.upper()} ({engine_total} ops, {min_cycles} min cycles)")
        print("-" * 50)

        for op_name, count in sorted(op_types[engine].items(), key=lambda x: -x[1]):
            pct = 100 * count / engine_total if engine_total > 0 else 0
            bar = "#" * int(pct / 5) + "-" * (20 - int(pct / 5))
            print(f"  {op_name:20s}: {count:6d} ({pct:5.1f}%) [{bar}]")

    print(f"\n{'Total operations':20s}: {total_ops:6d}")
    print(f"{'Scheduled cycles':20s}: {len(kb.instrs):6d}")
    print(f"{'Ops per cycle':20s}: {total_ops / len(kb.instrs):.2f}")

    if verbose:
        print("\n" + "=" * 70)
        print("HASH FUNCTION COST ANALYSIS")
        print("=" * 70)

        hash_costs = analyze_hash_costs(kb)
        print(f"\nHash stages: {len(HASH_STAGES)}")
        print(f"  Linear stages (multiply_add):     3 stages")
        print(f"  Non-linear stages (3 ops each):   3 stages")

        print(f"\nHash operation counts:")
        print(f"  multiply_add calls: {hash_costs['multiply_add']}")
        print(f"  Linear ops total:   {hash_costs['linear_total']}")
        print(f"  Non-linear ops total: {hash_costs['nonlinear_total']}")


def analyze_ops_by_depth(forest_height: int, n_nodes: int,
                          batch_size: int, rounds: int):
    """Estimate operation counts per depth level"""
    vec_batches = batch_size // VLEN

    print("=" * 70)
    print("ESTIMATED OPERATIONS BY DEPTH")
    print("=" * 70)

    print(f"\nConfiguration: height={forest_height}, batch={batch_size}, "
          f"vec_batches={vec_batches}, rounds={rounds}")

    print("\nPer-round operation estimates:")
    print("-" * 70)
    print(f"{'Depth':>6} | {'Load':>6} | {'VALU':>6} | {'ALU':>6} | "
          f"{'Flow':>6} | {'Store':>6} | {'Limiting':>8}")
    print("-" * 70)

    total_by_engine = defaultdict(int)

    for round_idx in range(rounds):
        depth = round_idx % (forest_height + 1)

        # Estimate operations per round based on depth
        if depth == 0:
            # Preloaded node0: XOR + hash
            valu = vec_batches * (1 + 12)  # XOR + hash (12 valu ops)
            alu = vec_batches * 8  # path update (scalar)
            load = 0
            flow = 0
            store = 0
        elif depth == 1:
            # Preloaded node1,2: vselect + XOR + hash
            valu = vec_batches * (1 + 12)
            alu = vec_batches * 24  # path update (3 ops * 8 lanes)
            load = 0
            flow = vec_batches * 1  # 1 vselect
            store = 0
        elif depth == 2:
            # Preloaded node3-6: bit extract + 3 vselects + XOR + hash
            valu = vec_batches * (2 + 1 + 12)  # bit extract + XOR + hash
            alu = vec_batches * 24
            load = 0
            flow = vec_batches * 3  # 3 vselects
            store = 0
        else:
            # Depth 3+: load from memory
            valu = vec_batches * (1 + 12)  # XOR + hash
            alu = vec_batches * (8 + 24)  # addr calc + path update
            load = vec_batches * 8  # 8 loads per vec batch
            flow = 0
            store = 0

        # Add final store for round 0 and last round
        if round_idx == rounds - 1:
            store = vec_batches * 2  # store values (add_imm for addr counts as flow)
            flow += vec_batches * 2  # add_imm for addresses

        total_by_engine["load"] += load
        total_by_engine["valu"] += valu
        total_by_engine["alu"] += alu
        total_by_engine["flow"] += flow
        total_by_engine["store"] += store

        # Determine limiting engine for this round
        min_cycles = {
            "load": (load + 1) // 2 if load > 0 else 0,
            "valu": (valu + 5) // 6,
            "alu": (alu + 11) // 12,
            "flow": flow,
            "store": (store + 1) // 2 if store > 0 else 0,
        }
        limiting = max(min_cycles, key=min_cycles.get) if any(min_cycles.values()) else "-"

        print(f"{depth:>6} | {load:>6} | {valu:>6} | {alu:>6} | "
              f"{flow:>6} | {store:>6} | {limiting:>8}")

    print("-" * 70)
    print(f"{'Total':>6} | {total_by_engine['load']:>6} | "
          f"{total_by_engine['valu']:>6} | {total_by_engine['alu']:>6} | "
          f"{total_by_engine['flow']:>6} | {total_by_engine['store']:>6}")

    print("\nMinimum cycles by engine (estimated):")
    for eng, total in sorted(total_by_engine.items()):
        slots = SLOT_LIMITS.get(eng, 1)
        min_cyc = (total + slots - 1) // slots
        print(f"  {eng:8s}: {total:6d} ops / {slots} slots = {min_cyc:5d} cycles")


def main():
    parser = argparse.ArgumentParser(description="Analyze operation breakdown")
    parser.add_argument("--forest-height", type=int, default=10)
    parser.add_argument("--n-nodes", type=int, default=2047)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--by-depth", action="store_true",
                       help="Show estimated breakdown by depth")
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    if args.by_depth:
        analyze_ops_by_depth(args.forest_height, args.n_nodes,
                            args.batch_size, args.rounds)
    else:
        kb = KernelBuilder(enable_debug_ops=False)
        kb.build_kernel(args.forest_height, args.n_nodes,
                       args.batch_size, args.rounds)
        print_op_breakdown(kb, verbose=not args.quiet)


if __name__ == "__main__":
    main()

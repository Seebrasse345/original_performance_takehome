"""
Cycle Trace Analyzer - Shows what happens cycle by cycle.

Provides:
- Cycle-by-cycle slot utilization
- Instruction bundles per cycle
- Empty slot analysis
- Stall detection

Usage:
    python scripts/cycle_trace.py
    python scripts/cycle_trace.py --start 100 --count 50
    python scripts/cycle_trace.py --summary
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from problem import SLOT_LIMITS, VLEN
from perf_takehome import KernelBuilder


@dataclass
class CycleInfo:
    """Information about a single cycle"""
    cycle_num: int
    slots_used: Dict[str, int]
    slots_available: Dict[str, int]
    instructions: Dict[str, List[tuple]]
    utilization: float
    is_stall: bool


def analyze_cycles(kb: KernelBuilder) -> List[CycleInfo]:
    """Analyze each cycle and collect statistics"""
    cycles = []

    for cycle_num, instr in enumerate(kb.instrs):
        slots_used = {}
        slots_available = {}
        total_used = 0
        total_available = 0

        for engine in ["load", "valu", "alu", "flow", "store"]:
            limit = SLOT_LIMITS.get(engine, 1)
            used = len(instr.get(engine, []))
            slots_used[engine] = used
            slots_available[engine] = limit
            total_used += used
            total_available += limit

        utilization = total_used / total_available if total_available > 0 else 0
        is_stall = total_used < 3  # Consider it a stall if < 3 slots used

        cycles.append(CycleInfo(
            cycle_num=cycle_num,
            slots_used=slots_used,
            slots_available=slots_available,
            instructions=instr,
            utilization=utilization,
            is_stall=is_stall
        ))

    return cycles


def print_cycle_range(cycles: List[CycleInfo], start: int, count: int,
                      show_ops: bool = False):
    """Print detailed cycle-by-cycle trace"""
    end = min(start + count, len(cycles))

    print("=" * 90)
    print(f"CYCLE TRACE (cycles {start} to {end-1})")
    print("=" * 90)

    # Header
    print(f"{'Cycle':>6} | {'Load':>5} | {'VALU':>5} | {'ALU':>5} | "
          f"{'Flow':>5} | {'Store':>5} | {'Util':>6} | {'Notes'}")
    print("-" * 90)

    for cycle in cycles[start:end]:
        notes = []
        if cycle.is_stall:
            notes.append("STALL")
        if cycle.slots_used["flow"] == 1:
            notes.append("flow-bound")
        if cycle.slots_used["load"] == 2:
            notes.append("load-max")

        load_str = f"{cycle.slots_used['load']}/{cycle.slots_available['load']}"
        valu_str = f"{cycle.slots_used['valu']}/{cycle.slots_available['valu']}"
        alu_str = f"{cycle.slots_used['alu']}/{cycle.slots_available['alu']}"
        flow_str = f"{cycle.slots_used['flow']}/{cycle.slots_available['flow']}"
        store_str = f"{cycle.slots_used['store']}/{cycle.slots_available['store']}"

        print(f"{cycle.cycle_num:>6} | {load_str:>5} | {valu_str:>5} | {alu_str:>5} | "
              f"{flow_str:>5} | {store_str:>5} | {cycle.utilization:>5.1%} | "
              f"{', '.join(notes)}")

        if show_ops:
            for engine, slots in cycle.instructions.items():
                if engine == "debug":
                    continue
                for slot in slots:
                    op_name = slot[0] if isinstance(slot, tuple) else str(slot)
                    print(f"       |   {engine}: {op_name}")


def print_cycle_summary(cycles: List[CycleInfo]):
    """Print summary statistics for all cycles"""
    print("=" * 70)
    print("CYCLE TRACE SUMMARY")
    print("=" * 70)

    total_cycles = len(cycles)

    # Utilization distribution
    util_buckets = defaultdict(int)
    for cycle in cycles:
        bucket = int(cycle.utilization * 10) * 10  # 0-10%, 10-20%, etc.
        util_buckets[bucket] += 1

    print("\n[1] UTILIZATION DISTRIBUTION")
    print("-" * 50)
    for bucket in range(0, 110, 10):
        count = util_buckets.get(bucket, 0)
        pct = 100 * count / total_cycles
        bar = "#" * int(pct / 2) + "-" * (50 - int(pct / 2))
        print(f"  {bucket:>3}%-{bucket+10:<3}%: {count:>5} cycles ({pct:5.1f}%) [{bar}]")

    # Stall analysis
    stalls = sum(1 for c in cycles if c.is_stall)
    print(f"\n[2] STALL ANALYSIS")
    print("-" * 50)
    print(f"  Total stalls (<3 slots): {stalls:,} ({100*stalls/total_cycles:.1f}%)")

    # Per-engine saturation
    print(f"\n[3] ENGINE SATURATION (cycles at max slots)")
    print("-" * 50)
    for engine in ["load", "valu", "alu", "flow", "store"]:
        limit = SLOT_LIMITS.get(engine, 1)
        at_max = sum(1 for c in cycles if c.slots_used[engine] == limit)
        at_zero = sum(1 for c in cycles if c.slots_used[engine] == 0)
        pct_max = 100 * at_max / total_cycles
        pct_zero = 100 * at_zero / total_cycles
        print(f"  {engine:8s}: max={at_max:5d} ({pct_max:5.1f}%), "
              f"idle={at_zero:5d} ({pct_zero:5.1f}%)")

    # Flow analysis (critical since it's 1 slot)
    print(f"\n[4] FLOW SLOT ANALYSIS (1 slot/cycle)")
    print("-" * 50)
    flow_used = sum(c.slots_used["flow"] for c in cycles)
    flow_cycles = sum(1 for c in cycles if c.slots_used["flow"] > 0)
    print(f"  Total flow ops:    {flow_used:,}")
    print(f"  Cycles with flow:  {flow_cycles:,} ({100*flow_cycles/total_cycles:.1f}%)")
    print(f"  Cycles flow-bound: {flow_cycles:,} (flow ops can't parallelize)")

    # Average utilization
    avg_util = sum(c.utilization for c in cycles) / total_cycles
    print(f"\n[5] OVERALL METRICS")
    print("-" * 50)
    print(f"  Total cycles:        {total_cycles:,}")
    print(f"  Average utilization: {avg_util:.1%}")

    # Find longest stall sequences
    print(f"\n[6] STALL SEQUENCES (consecutive low utilization)")
    print("-" * 50)
    max_stall_len = 0
    current_stall = 0
    stall_start = 0
    worst_stalls = []

    for i, cycle in enumerate(cycles):
        if cycle.is_stall:
            if current_stall == 0:
                stall_start = i
            current_stall += 1
        else:
            if current_stall > 0:
                if current_stall >= 3:
                    worst_stalls.append((stall_start, current_stall))
                current_stall = 0

    if current_stall > 0 and current_stall >= 3:
        worst_stalls.append((stall_start, current_stall))

    worst_stalls.sort(key=lambda x: -x[1])
    if worst_stalls:
        print(f"  Top stall sequences:")
        for start, length in worst_stalls[:5]:
            print(f"    Cycles {start}-{start+length-1}: {length} cycles")
    else:
        print(f"  No significant stall sequences found")


def main():
    parser = argparse.ArgumentParser(description="Analyze cycle-by-cycle execution")
    parser.add_argument("--forest-height", type=int, default=10)
    parser.add_argument("--n-nodes", type=int, default=2047)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--start", type=int, default=0,
                       help="Starting cycle for detailed trace")
    parser.add_argument("--count", type=int, default=50,
                       help="Number of cycles to show in trace")
    parser.add_argument("--summary", action="store_true",
                       help="Show summary only")
    parser.add_argument("--show-ops", action="store_true",
                       help="Show individual operations in trace")
    args = parser.parse_args()

    kb = KernelBuilder(enable_debug_ops=False)
    kb.build_kernel(args.forest_height, args.n_nodes,
                   args.batch_size, args.rounds)

    cycles = analyze_cycles(kb)

    if args.summary:
        print_cycle_summary(cycles)
    else:
        print_cycle_range(cycles, args.start, args.count, args.show_ops)
        print()
        print_cycle_summary(cycles)


if __name__ == "__main__":
    main()

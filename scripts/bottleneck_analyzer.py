"""
Bottleneck Analyzer - Identifies which engine is limiting performance.

Provides detailed analysis of:
- Operation counts by engine
- Theoretical minimum cycles per engine
- Actual cycles vs theoretical minimum
- Scheduling efficiency
- Recommendations for optimization

Usage:
    python scripts/bottleneck_analyzer.py
    python scripts/bottleneck_analyzer.py --batch-size 512 --rounds 32
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

from problem import SLOT_LIMITS, VLEN, SCRATCH_SIZE
from perf_takehome import KernelBuilder


@dataclass
class EngineStats:
    """Statistics for a single engine"""
    name: str
    op_count: int
    slots_per_cycle: int
    min_cycles: int
    is_bottleneck: bool
    utilization: float = 0.0

    def __str__(self):
        marker = " <<< BOTTLENECK" if self.is_bottleneck else ""
        return (f"  {self.name:8s}: {self.op_count:6d} ops / {self.slots_per_cycle} slots "
                f"= {self.min_cycles:5d} min cycles{marker}")


@dataclass
class BottleneckReport:
    """Complete bottleneck analysis report"""
    engine_stats: Dict[str, EngineStats]
    theoretical_min: int
    actual_cycles: int
    efficiency: float
    overhead_cycles: int
    bottleneck_engine: str
    slot_utilization: Dict[str, float]

    def print_report(self, verbose: bool = True):
        print("=" * 70)
        print("BOTTLENECK ANALYSIS REPORT")
        print("=" * 70)

        print("\n[1] OPERATION COUNTS BY ENGINE")
        print("-" * 50)
        for engine in ["load", "valu", "alu", "flow", "store"]:
            if engine in self.engine_stats:
                print(self.engine_stats[engine])

        print("\n[2] CYCLE ANALYSIS")
        print("-" * 50)
        print(f"  Theoretical minimum: {self.theoretical_min:,} cycles")
        print(f"    (Limited by: {self.bottleneck_engine})")
        print(f"  Actual cycles:       {self.actual_cycles:,}")
        print(f"  Overhead:            {self.overhead_cycles:,} cycles "
              f"({100*self.overhead_cycles/self.actual_cycles:.1f}%)")
        print(f"  Scheduling efficiency: {100*self.efficiency:.1f}%")

        print("\n[3] SLOT UTILIZATION (per cycle)")
        print("-" * 50)
        for engine in ["load", "valu", "alu", "flow", "store"]:
            if engine in self.slot_utilization:
                util = self.slot_utilization[engine]
                slots = SLOT_LIMITS.get(engine, 1)
                avg = util * slots
                bar = "#" * int(util * 30) + "-" * (30 - int(util * 30))
                print(f"  {engine:8s}: [{bar}] {100*util:5.1f}% ({avg:.1f}/{slots})")

        if verbose:
            print("\n[4] OPTIMIZATION RECOMMENDATIONS")
            print("-" * 50)
            self._print_recommendations()

        print("=" * 70)

    def _print_recommendations(self):
        if self.bottleneck_engine == "load":
            print("  * LOAD is the bottleneck (2 slots/cycle)")
            print("    Recommendations:")
            print("    - Preload more tree levels into scratch (if flow permits)")
            print("    - Use vload instead of multiple scalar loads")
            print("    - Better interleaving of loads with compute")
            print("    - Consider memory layout changes for coalesced access")
        elif self.bottleneck_engine == "flow":
            print("  * FLOW is the bottleneck (1 slot/cycle)")
            print("    Recommendations:")
            print("    - Replace vselect with ALU/VALU if possible")
            print("    - Use if-conversion to replace branches with predicated ops")
            print("    - Reduce conditional branching")
            print("    - Note: Each vselect costs 1 cycle minimum!")
        elif self.bottleneck_engine == "valu":
            print("  * VALU is the bottleneck (6 slots/cycle)")
            print("    Recommendations:")
            print("    - Move some work to scalar ALU (12 slots)")
            print("    - Optimize hash function (use multiply_add)")
            print("    - J-lane parallel hashing for latency hiding")
        elif self.bottleneck_engine == "alu":
            print("  * ALU is the bottleneck (12 slots/cycle)")
            print("    Recommendations:")
            print("    - Vectorize scalar operations where possible")
            print("    - Use VALU for batch operations")
        elif self.bottleneck_engine == "store":
            print("  * STORE is the bottleneck (2 slots/cycle)")
            print("    Recommendations:")
            print("    - Reduce store frequency")
            print("    - Use vstore for vectorized stores")


def analyze_bottleneck(forest_height: int = 10, n_nodes: int = 2047,
                       batch_size: int = 256, rounds: int = 16) -> BottleneckReport:
    """
    Analyze kernel bottlenecks and return detailed report.
    """
    kb = KernelBuilder(enable_debug_ops=False)
    kb.schedule_report = True
    kb.build_kernel(forest_height, n_nodes, batch_size, rounds)

    # Count operations by engine
    engine_ops = defaultdict(int)
    for op in kb.ops:
        engine_ops[op.engine] += 1

    # Calculate minimum cycles per engine
    engine_stats = {}
    for engine, count in engine_ops.items():
        if engine == "debug":
            continue
        slots = SLOT_LIMITS.get(engine, 1)
        min_cycles = (count + slots - 1) // slots
        engine_stats[engine] = EngineStats(
            name=engine,
            op_count=count,
            slots_per_cycle=slots,
            min_cycles=min_cycles,
            is_bottleneck=False
        )

    # Find bottleneck
    bottleneck_engine = max(engine_stats.keys(),
                           key=lambda e: engine_stats[e].min_cycles)
    engine_stats[bottleneck_engine].is_bottleneck = True
    theoretical_min = engine_stats[bottleneck_engine].min_cycles

    actual_cycles = len(kb.instrs)
    efficiency = theoretical_min / actual_cycles if actual_cycles > 0 else 0
    overhead = actual_cycles - theoretical_min

    # Calculate slot utilization from scheduled instructions
    slot_usage = defaultdict(int)
    for instr in kb.instrs:
        for engine, slots in instr.items():
            if engine in SLOT_LIMITS:
                slot_usage[engine] += len(slots)

    slot_utilization = {}
    for engine in SLOT_LIMITS:
        if engine == "debug":
            continue
        total_possible = actual_cycles * SLOT_LIMITS[engine]
        used = slot_usage.get(engine, 0)
        slot_utilization[engine] = used / total_possible if total_possible > 0 else 0
        if engine in engine_stats:
            engine_stats[engine].utilization = slot_utilization[engine]

    return BottleneckReport(
        engine_stats=engine_stats,
        theoretical_min=theoretical_min,
        actual_cycles=actual_cycles,
        efficiency=efficiency,
        overhead_cycles=overhead,
        bottleneck_engine=bottleneck_engine,
        slot_utilization=slot_utilization
    )


def main():
    parser = argparse.ArgumentParser(description="Analyze kernel bottlenecks")
    parser.add_argument("--forest-height", type=int, default=10)
    parser.add_argument("--n-nodes", type=int, default=2047)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Minimal output")
    args = parser.parse_args()

    print(f"\nAnalyzing kernel with: height={args.forest_height}, "
          f"batch={args.batch_size}, rounds={args.rounds}")

    report = analyze_bottleneck(
        args.forest_height, args.n_nodes,
        args.batch_size, args.rounds
    )
    report.print_report(verbose=not args.quiet)

    return report


if __name__ == "__main__":
    main()

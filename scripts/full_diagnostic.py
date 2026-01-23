"""
Full Diagnostic Suite - Runs all diagnostics and produces comprehensive report.

This is the main entry point for performance analysis. It runs:
1. Bottleneck analysis
2. Operation breakdown
3. Cycle trace summary
4. Scratch memory analysis
5. Scheduling analysis
6. Correctness verification

Usage:
    python scripts/full_diagnostic.py
    python scripts/full_diagnostic.py --output report.txt
    python scripts/full_diagnostic.py --quick  # Skip detailed analysis
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import io
from datetime import datetime
from contextlib import redirect_stdout

from bottleneck_analyzer import analyze_bottleneck
from op_breakdown import analyze_op_types, analyze_hash_costs
from cycle_trace import analyze_cycles, print_cycle_summary
from scratch_analyzer import analyze_scratch_usage
from scheduling_analyzer import analyze_scheduling, analyze_engine_scheduling
from comparison import run_kernel_test

from problem import SLOT_LIMITS, VLEN, N_CORES, SCRATCH_SIZE
from perf_takehome import KernelBuilder


def run_full_diagnostic(forest_height: int = 10, n_nodes: int = 2047,
                        batch_size: int = 256, rounds: int = 16,
                        quick: bool = False) -> str:
    """Run all diagnostics and return formatted report"""

    output = io.StringIO()

    def pr(*args, **kwargs):
        print(*args, **kwargs, file=output)

    # Header
    pr("=" * 80)
    pr("COMPREHENSIVE VLIW SIMD KERNEL DIAGNOSTIC REPORT")
    pr("=" * 80)
    pr(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    pr(f"\nConfiguration:")
    pr(f"  Forest height:  {forest_height}")
    pr(f"  N nodes:        {n_nodes}")
    pr(f"  Batch size:     {batch_size}")
    pr(f"  Rounds:         {rounds}")
    pr(f"  VLEN:           {VLEN}")
    pr(f"  N_CORES:        {N_CORES}")
    pr(f"  SCRATCH_SIZE:   {SCRATCH_SIZE}")

    # Build kernel
    pr("\n" + "=" * 80)
    pr("BUILDING KERNEL")
    pr("=" * 80)

    kb = KernelBuilder(enable_debug_ops=False)
    kb.build_kernel(forest_height, n_nodes, batch_size, rounds)

    total_ops = len(kb.ops)
    total_cycles = len(kb.instrs)

    pr(f"\nKernel Statistics:")
    pr(f"  Total operations:   {total_ops:,}")
    pr(f"  Scheduled cycles:   {total_cycles:,}")
    pr(f"  Ops per cycle:      {total_ops / total_cycles:.2f}")

    # Correctness check
    pr("\n" + "=" * 80)
    pr("CORRECTNESS VERIFICATION")
    pr("=" * 80)

    test_result = run_kernel_test(forest_height, batch_size, rounds,
                                  verify_correctness=True)
    status = "PASS" if test_result.correct else "FAIL"
    pr(f"\nResult: {status}")
    pr(f"  Cycles:     {test_result.cycles:,}")
    pr(f"  Build time: {test_result.build_time_ms:.1f}ms")
    pr(f"  Run time:   {test_result.run_time_ms:.1f}ms")

    if test_result.error:
        pr(f"  Error: {test_result.error}")

    # Bottleneck analysis
    pr("\n" + "=" * 80)
    pr("BOTTLENECK ANALYSIS")
    pr("=" * 80)

    bottleneck = analyze_bottleneck(forest_height, n_nodes, batch_size, rounds)

    pr(f"\nOperation Counts by Engine:")
    for engine in ["load", "valu", "alu", "flow", "store"]:
        if engine in bottleneck.engine_stats:
            stats = bottleneck.engine_stats[engine]
            marker = " <<< BOTTLENECK" if stats.is_bottleneck else ""
            pr(f"  {engine:8s}: {stats.op_count:6d} ops / {stats.slots_per_cycle} slots "
               f"= {stats.min_cycles:5d} min cycles{marker}")

    pr(f"\nCycle Analysis:")
    pr(f"  Theoretical minimum: {bottleneck.theoretical_min:,} cycles")
    pr(f"  Actual cycles:       {bottleneck.actual_cycles:,}")
    pr(f"  Overhead:            {bottleneck.overhead_cycles:,} cycles "
       f"({100*bottleneck.overhead_cycles/bottleneck.actual_cycles:.1f}%)")
    pr(f"  Efficiency:          {100*bottleneck.efficiency:.1f}%")

    pr(f"\nSlot Utilization:")
    for engine in ["load", "valu", "alu", "flow", "store"]:
        if engine in bottleneck.slot_utilization:
            util = bottleneck.slot_utilization[engine]
            slots = SLOT_LIMITS.get(engine, 1)
            avg = util * slots
            bar = "#" * int(util * 30) + "-" * (30 - int(util * 30))
            pr(f"  {engine:8s}: [{bar}] {100*util:5.1f}% ({avg:.1f}/{slots})")

    # Operation breakdown
    pr("\n" + "=" * 80)
    pr("OPERATION BREAKDOWN")
    pr("=" * 80)

    op_types = analyze_op_types(kb)
    for engine in ["load", "valu", "alu", "flow", "store"]:
        if engine not in op_types:
            continue
        engine_total = sum(op_types[engine].values())
        pr(f"\n  {engine.upper()} ({engine_total} ops):")
        for op_name, count in sorted(op_types[engine].items(), key=lambda x: -x[1])[:5]:
            pct = 100 * count / engine_total if engine_total > 0 else 0
            pr(f"    {op_name:20s}: {count:6d} ({pct:5.1f}%)")

    # Hash function costs
    hash_costs = analyze_hash_costs(kb)
    pr(f"\n  Hash Function:")
    pr(f"    multiply_add calls: {hash_costs['multiply_add']}")
    pr(f"    Linear ops:         {hash_costs['linear_total']}")
    pr(f"    Non-linear ops:     {hash_costs['nonlinear_total']}")

    # Cycle trace (summary only unless detailed)
    if not quick:
        pr("\n" + "=" * 80)
        pr("CYCLE TRACE ANALYSIS")
        pr("=" * 80)

        cycles = analyze_cycles(kb)

        # Stall analysis
        stalls = sum(1 for c in cycles if c.is_stall)
        pr(f"\nStall Analysis:")
        pr(f"  Total stalls (<3 slots): {stalls:,} ({100*stalls/len(cycles):.1f}%)")

        # Per-engine saturation
        pr(f"\nEngine Saturation (cycles at max slots):")
        for engine in ["load", "valu", "alu", "flow", "store"]:
            limit = SLOT_LIMITS.get(engine, 1)
            at_max = sum(1 for c in cycles if c.slots_used[engine] == limit)
            at_zero = sum(1 for c in cycles if c.slots_used[engine] == 0)
            pct_max = 100 * at_max / len(cycles)
            pct_zero = 100 * at_zero / len(cycles)
            pr(f"  {engine:8s}: max={at_max:5d} ({pct_max:5.1f}%), "
               f"idle={at_zero:5d} ({pct_zero:5.1f}%)")

        # Utilization distribution
        from collections import defaultdict
        util_buckets = defaultdict(int)
        for cycle in cycles:
            bucket = int(cycle.utilization * 10) * 10
            util_buckets[bucket] += 1

        pr(f"\nUtilization Distribution:")
        for bucket in range(0, 110, 20):
            count = util_buckets.get(bucket, 0) + util_buckets.get(bucket + 10, 0)
            pct = 100 * count / len(cycles)
            bar = "#" * int(pct / 2)
            pr(f"  {bucket:>3}%-{bucket+19:<3}%: {count:>5} ({pct:5.1f}%) {bar}")

    # Scratch memory analysis
    pr("\n" + "=" * 80)
    pr("SCRATCH MEMORY ANALYSIS")
    pr("=" * 80)

    scratch = analyze_scratch_usage(forest_height, n_nodes, batch_size, rounds)

    pr(f"\nScratch Space:")
    pr(f"  Limit:           {scratch['scratch_limit']:,}")
    pr(f"  Static used:     {scratch['scratch_ptr']:,}")
    pr(f"  Temp watermark:  {scratch['temp_watermark']:,}")
    pr(f"  Remaining:       {scratch['scratch_remaining']:,}")

    used_pct = 100 * scratch['scratch_ptr'] / scratch['scratch_limit']
    watermark_pct = 100 * scratch['temp_watermark'] / scratch['scratch_limit']
    pr(f"\n  Static:    {used_pct:5.1f}%")
    pr(f"  Watermark: {watermark_pct:5.1f}%")

    # Scheduling analysis
    if not quick:
        pr("\n" + "=" * 80)
        pr("SCHEDULING ANALYSIS")
        pr("=" * 80)

        sched = analyze_scheduling(kb)

        pr(f"\nDependency Statistics:")
        pr(f"  Total dependencies:   {sched['total_deps']:,}")
        pr(f"  Max deps per op:      {sched['max_deps_per_op']}")
        pr(f"  Avg deps per op:      {sched['avg_deps_per_op']:.2f}")

        pr(f"\nParallelism:")
        pr(f"  Critical path length: {sched['critical_path']}")
        pr(f"  Achieved ILP:         {sched['ilp']:.2f} ops/cycle")
        pr(f"  Theoretical max:      {sum(SLOT_LIMITS.values())} slots/cycle")

    # Recommendations
    pr("\n" + "=" * 80)
    pr("OPTIMIZATION RECOMMENDATIONS")
    pr("=" * 80)

    pr(f"\nBottleneck: {bottleneck.bottleneck_engine.upper()}")

    if bottleneck.bottleneck_engine == "load":
        pr("""
  The kernel is LOAD-BOUND (2 slots/cycle max).

  Potential optimizations:
  1. Preload more tree levels (but watch flow slot cost)
  2. Use vload for consecutive memory access
  3. Better interleave loads with compute
  4. Consider Van Emde Boas tree layout for cache efficiency

  Trade-off warning:
  - Preloading depth N requires 2^N vselects to select the right node
  - Each vselect uses 1 flow slot/cycle
  - Depth 3: 8 loads saved vs 7 vselects needed (net loss if flow-bound)
""")
    elif bottleneck.bottleneck_engine == "flow":
        pr("""
  The kernel is FLOW-BOUND (1 slot/cycle max).

  Potential optimizations:
  1. Replace vselect with predicated ALU ops if possible
  2. Use if-conversion to avoid branches
  3. Reduce number of conditional operations
  4. Consider different algorithm structure

  Note: Flow operations are severely limited - only 1 per cycle!
""")
    elif bottleneck.bottleneck_engine == "valu":
        pr("""
  The kernel is VALU-BOUND (6 slots/cycle max).

  Potential optimizations:
  1. Use multiply_add for linear hash stages (already done)
  2. Move some work to scalar ALU (12 slots)
  3. J-lane parallel hashing for latency hiding
  4. Consider hash function modifications
""")

    # Summary
    pr("\n" + "=" * 80)
    pr("SUMMARY")
    pr("=" * 80)
    pr(f"""
  Current performance:   {test_result.cycles:,} cycles
  Theoretical minimum:   {bottleneck.theoretical_min:,} cycles
  Gap:                   {test_result.cycles - bottleneck.theoretical_min:,} cycles
  Efficiency:            {100*bottleneck.efficiency:.1f}%

  Limiting factor:       {bottleneck.bottleneck_engine.upper()}
  Correctness:           {'PASS' if test_result.correct else 'FAIL'}
""")

    pr("=" * 80)

    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(description="Run full diagnostic suite")
    parser.add_argument("--forest-height", type=int, default=10)
    parser.add_argument("--n-nodes", type=int, default=2047)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--output", "-o", type=str,
                       help="Save report to file")
    parser.add_argument("--quick", "-q", action="store_true",
                       help="Quick mode (skip detailed analysis)")
    args = parser.parse_args()

    report = run_full_diagnostic(
        args.forest_height, args.n_nodes,
        args.batch_size, args.rounds,
        quick=args.quick
    )

    print(report)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()

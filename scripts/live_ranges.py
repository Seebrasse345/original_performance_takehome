"""
Live Range Analyzer - Analyzes register/scratch live ranges.

Provides:
- Live range statistics
- Peak register pressure
- Spill candidates identification
- Allocation timeline visualization

Usage:
    python scripts/live_ranges.py
    python scripts/live_ranges.py --top 20
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional

from problem import SLOT_LIMITS, VLEN
from perf_takehome import KernelBuilder, Op


@dataclass
class LiveRange:
    """Information about a live range"""
    address: int
    length: int  # number of words (1 for scalar, VLEN for vector)
    def_op: int  # operation index that defines this
    last_use: int  # operation index of last use
    live_length: int  # number of ops between def and last use
    use_count: int  # number of times used


def compute_live_ranges(kb: KernelBuilder) -> Dict[int, LiveRange]:
    """Compute live ranges for all scratch addresses"""
    ops = kb.ops

    # Track definitions and uses
    definitions = {}  # addr -> first op that writes
    last_uses = {}  # addr -> last op that reads
    use_counts = defaultdict(int)  # addr -> count

    for i, op in enumerate(ops):
        # Track definitions
        for addr in op.writes:
            if addr not in definitions:
                definitions[addr] = i

        # Track uses
        for addr in op.reads:
            last_uses[addr] = i
            use_counts[addr] += 1

    # Build live ranges
    ranges = {}
    for addr, def_op in definitions.items():
        last_use = last_uses.get(addr, def_op)
        live_length = last_use - def_op + 1

        # Determine if vector or scalar based on if addr is part of named alloc
        length = 1
        for named_addr, (name, named_len) in kb.scratch_debug.items():
            if named_addr <= addr < named_addr + named_len:
                # This is part of a named allocation
                if named_len == VLEN:
                    length = VLEN
                break

        ranges[addr] = LiveRange(
            address=addr,
            length=length,
            def_op=def_op,
            last_use=last_use,
            live_length=live_length,
            use_count=use_counts[addr]
        )

    return ranges


def compute_pressure_curve(kb: KernelBuilder) -> List[int]:
    """Compute register pressure at each operation"""
    ops = kb.ops
    n_ops = len(ops)

    # Track live addresses at each op
    pressure = [0] * n_ops

    # For each address, mark it as live from def to last use
    definitions = {}
    last_uses = {}

    for i, op in enumerate(ops):
        for addr in op.writes:
            if addr not in definitions:
                definitions[addr] = i
        for addr in op.reads:
            last_uses[addr] = i

    for addr, def_op in definitions.items():
        last_use = last_uses.get(addr, def_op)
        for i in range(def_op, min(last_use + 1, n_ops)):
            pressure[i] += 1

    return pressure


def analyze_live_ranges(forest_height: int = 10, n_nodes: int = 2047,
                        batch_size: int = 256, rounds: int = 16) -> Dict:
    """Analyze live ranges and return statistics"""
    kb = KernelBuilder(enable_debug_ops=False)
    kb.build_kernel(forest_height, n_nodes, batch_size, rounds)

    ranges = compute_live_ranges(kb)
    pressure = compute_pressure_curve(kb)

    # Statistics
    live_lengths = [r.live_length for r in ranges.values()]
    use_counts = [r.use_count for r in ranges.values()]

    return {
        "n_ranges": len(ranges),
        "avg_live_length": sum(live_lengths) / len(live_lengths) if live_lengths else 0,
        "max_live_length": max(live_lengths) if live_lengths else 0,
        "avg_use_count": sum(use_counts) / len(use_counts) if use_counts else 0,
        "peak_pressure": max(pressure) if pressure else 0,
        "avg_pressure": sum(pressure) / len(pressure) if pressure else 0,
        "ranges": ranges,
        "pressure": pressure,
        "kb": kb,
    }


def print_live_range_report(results: Dict, top_n: int = 15):
    """Print live range analysis report"""
    print("=" * 70)
    print("LIVE RANGE ANALYSIS")
    print("=" * 70)

    print(f"\n[1] SUMMARY STATISTICS")
    print("-" * 50)
    print(f"  Total live ranges:    {results['n_ranges']:,}")
    print(f"  Avg live length:      {results['avg_live_length']:.1f} ops")
    print(f"  Max live length:      {results['max_live_length']:,} ops")
    print(f"  Avg use count:        {results['avg_use_count']:.1f}")
    print(f"  Peak pressure:        {results['peak_pressure']:,} addresses")
    print(f"  Avg pressure:         {results['avg_pressure']:.1f} addresses")

    ranges = results["ranges"]
    pressure = results["pressure"]

    # Longest live ranges (potential spill candidates)
    print(f"\n[2] LONGEST LIVE RANGES (top {top_n})")
    print("-" * 50)
    sorted_ranges = sorted(ranges.values(), key=lambda r: -r.live_length)

    print(f"{'Addr':>6} | {'Length':>6} | {'LiveLen':>8} | {'Uses':>5} | {'Def':>6} | {'LastUse':>7}")
    print("-" * 50)
    for r in sorted_ranges[:top_n]:
        print(f"{r.address:>6} | {r.length:>6} | {r.live_length:>8} | "
              f"{r.use_count:>5} | {r.def_op:>6} | {r.last_use:>7}")

    # Pressure timeline (sampled)
    print(f"\n[3] PRESSURE TIMELINE")
    print("-" * 50)

    n_samples = 20
    sample_size = max(1, len(pressure) // n_samples)

    print(f"{'Op Range':>15} | {'Avg':>6} | {'Peak':>6} | {'Visual'}")
    print("-" * 50)

    for i in range(0, len(pressure), sample_size):
        chunk = pressure[i:i + sample_size]
        if not chunk:
            continue
        avg = sum(chunk) / len(chunk)
        peak = max(chunk)
        end = min(i + sample_size - 1, len(pressure) - 1)
        bar = "#" * int(peak / 5) + "-" * max(0, 30 - int(peak / 5))
        print(f"{i:>6}-{end:<6} | {avg:>6.1f} | {peak:>6} | [{bar[:30]}]")

    # High pressure regions
    print(f"\n[4] HIGH PRESSURE REGIONS (peak > avg + 50%)")
    print("-" * 50)

    threshold = results["avg_pressure"] * 1.5
    in_high = False
    high_start = 0
    high_regions = []

    for i, p in enumerate(pressure):
        if p > threshold and not in_high:
            in_high = True
            high_start = i
        elif p <= threshold and in_high:
            in_high = False
            high_regions.append((high_start, i - 1, max(pressure[high_start:i])))

    if in_high:
        high_regions.append((high_start, len(pressure) - 1,
                            max(pressure[high_start:])))

    if high_regions:
        for start, end, peak in high_regions[:10]:
            print(f"  Ops {start:>6}-{end:<6}: peak={peak}")
    else:
        print("  No high pressure regions found")

    # Most frequently used addresses
    print(f"\n[5] MOST FREQUENTLY USED ADDRESSES (top {top_n})")
    print("-" * 50)
    sorted_by_use = sorted(ranges.values(), key=lambda r: -r.use_count)

    print(f"{'Addr':>6} | {'Uses':>6} | {'LiveLen':>8} | {'Efficiency':>10}")
    print("-" * 50)
    for r in sorted_by_use[:top_n]:
        # Efficiency = uses / live_length (higher is better)
        eff = r.use_count / r.live_length if r.live_length > 0 else 0
        print(f"{r.address:>6} | {r.use_count:>6} | {r.live_length:>8} | {eff:>10.3f}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Analyze live ranges")
    parser.add_argument("--forest-height", type=int, default=10)
    parser.add_argument("--n-nodes", type=int, default=2047)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--top", type=int, default=15,
                       help="Number of top items to show")
    args = parser.parse_args()

    results = analyze_live_ranges(
        args.forest_height, args.n_nodes,
        args.batch_size, args.rounds
    )
    print_live_range_report(results, args.top)


if __name__ == "__main__":
    main()

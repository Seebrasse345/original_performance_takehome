"""
Scratch Memory Analyzer - Analyzes scratch space usage.

Provides:
- Named allocation breakdown
- Temporary allocation watermark
- Memory pressure analysis
- Live range analysis

Usage:
    python scripts/scratch_analyzer.py
    python scripts/scratch_analyzer.py --trace-allocs
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from problem import SLOT_LIMITS, VLEN, SCRATCH_SIZE
from perf_takehome import KernelBuilder


@dataclass
class AllocationInfo:
    """Information about a scratch allocation"""
    name: Optional[str]
    address: int
    length: int
    category: str  # "named", "temp", "const"


def analyze_scratch_usage(forest_height: int = 10, n_nodes: int = 2047,
                          batch_size: int = 256, rounds: int = 16,
                          trace_allocs: bool = False) -> Dict:
    """Analyze scratch memory usage during kernel build"""

    # Create a patched KernelBuilder to track allocations
    class TrackedKernelBuilder(KernelBuilder):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.allocation_log = []
            self.temp_watermark = 0
            self.named_allocs = []

        def alloc_scratch(self, name=None, length=1):
            addr = super().alloc_scratch(name, length)
            info = AllocationInfo(
                name=name,
                address=addr,
                length=length,
                category="named" if name else "const"
            )
            self.allocation_log.append(("alloc_scratch", info))
            if name:
                self.named_allocs.append(info)
            return addr

        def alloc_temp(self, length=1):
            addr = super().alloc_temp(length)
            info = AllocationInfo(
                name=None,
                address=addr,
                length=length,
                category="temp"
            )
            self.allocation_log.append(("alloc_temp", info))
            if self.temp_alloc is not None:
                self.temp_watermark = max(self.temp_watermark, self.temp_alloc.high)
            return addr

        def free_temp(self, addr, length=1):
            super().free_temp(addr, length)
            self.allocation_log.append(("free_temp", (addr, length)))

    kb = TrackedKernelBuilder(enable_debug_ops=False)

    try:
        kb.build_kernel(forest_height, n_nodes, batch_size, rounds)
        success = True
    except AssertionError as e:
        success = False
        error_msg = str(e)

    # Collect statistics
    named_total = sum(a.length for a in kb.named_allocs)
    const_allocs = [log[1] for log in kb.allocation_log
                    if log[0] == "alloc_scratch" and log[1].category == "const"]
    const_total = sum(a.length for a in const_allocs)

    temp_allocs = [log[1] for log in kb.allocation_log
                   if log[0] == "alloc_temp"]
    temp_total = len(temp_allocs)  # Count, not size (they get freed)

    results = {
        "success": success,
        "scratch_ptr": kb.scratch_ptr,
        "scratch_limit": SCRATCH_SIZE,
        "scratch_remaining": SCRATCH_SIZE - kb.scratch_ptr,
        "temp_watermark": kb.temp_watermark if kb.temp_alloc else kb.scratch_ptr,
        "named_total": named_total,
        "const_total": const_total,
        "temp_allocations": temp_total,
        "named_allocs": kb.named_allocs,
        "const_allocs": const_allocs,
        "allocation_log": kb.allocation_log if trace_allocs else None,
    }

    if not success:
        results["error"] = error_msg

    return results


def print_scratch_report(results: Dict, verbose: bool = True):
    """Print scratch memory analysis report"""
    print("=" * 70)
    print("SCRATCH MEMORY ANALYSIS")
    print("=" * 70)

    status = "SUCCESS" if results["success"] else "OVERFLOW"
    print(f"\nBuild status: {status}")
    if not results["success"]:
        print(f"Error: {results.get('error', 'Unknown')}")

    print(f"\n[1] SCRATCH SPACE OVERVIEW")
    print("-" * 50)
    print(f"  Scratch limit:     {results['scratch_limit']:,}")
    print(f"  Static allocation: {results['scratch_ptr']:,}")
    print(f"  Temp watermark:    {results['temp_watermark']:,}")
    print(f"  Remaining:         {results['scratch_remaining']:,}")

    used_pct = 100 * results['scratch_ptr'] / results['scratch_limit']
    watermark_pct = 100 * results['temp_watermark'] / results['scratch_limit']

    bar_static = "#" * int(used_pct / 2) + "." * int((watermark_pct - used_pct) / 2)
    bar_free = "-" * (50 - len(bar_static))
    print(f"\n  Usage: [{bar_static}{bar_free}]")
    print(f"         [{'#' * 50}] = 100%")
    print(f"         # = static ({used_pct:.1f}%), . = temp watermark ({watermark_pct:.1f}%)")

    print(f"\n[2] ALLOCATION BREAKDOWN")
    print("-" * 50)
    print(f"  Named allocations: {results['named_total']:,} words")
    print(f"  Constant cache:    {results['const_total']:,} words")
    print(f"  Temp allocations:  {results['temp_allocations']:,} (freed after use)")

    if verbose and results['named_allocs']:
        print(f"\n[3] NAMED ALLOCATIONS (detail)")
        print("-" * 50)
        for alloc in results['named_allocs']:
            print(f"  {alloc.name:30s}: addr={alloc.address:4d}, len={alloc.length}")

    print(f"\n[4] VECTOR REGISTER USAGE")
    print("-" * 50)
    # Count vector allocations (length == VLEN)
    vec_allocs = [a for a in results['named_allocs'] if a.length == VLEN]
    scalar_allocs = [a for a in results['named_allocs'] if a.length == 1]
    print(f"  Vector registers (len={VLEN}): {len(vec_allocs)}")
    print(f"  Scalar registers (len=1):   {len(scalar_allocs)}")

    # Constants analysis
    if verbose:
        print(f"\n[5] CONSTANT CACHE")
        print("-" * 50)
        print(f"  Total constants cached: {len(results['const_allocs'])}")
        print(f"  Total space used:       {results['const_total']} words")

    print("=" * 70)


def estimate_scratch_needs(forest_height: int, batch_size: int, unroll: int):
    """Estimate scratch needs for a given configuration"""
    vec_batches = batch_size // VLEN

    print("=" * 70)
    print("SCRATCH REQUIREMENT ESTIMATION")
    print("=" * 70)

    print(f"\nConfiguration: height={forest_height}, batch={batch_size}, unroll={unroll}")

    # Fixed allocations
    init_vars = 7  # rounds, n_nodes, etc.
    hash_consts = 6 * 2  # 6 stages, scalar + vector
    hash_vecs = 6 * VLEN  # vector versions
    depth_addr = max(0, forest_height - 2)  # depth_addr_scalars

    fixed = init_vars + hash_consts + depth_addr
    print(f"\n[1] FIXED ALLOCATIONS")
    print(f"  Init vars:        {init_vars}")
    print(f"  Hash constants:   {hash_consts}")
    print(f"  Depth addresses:  {depth_addr}")
    print(f"  Total fixed:      {fixed}")

    # Per-block allocations
    blocks = min(vec_batches, unroll) if unroll else vec_batches
    per_block = VLEN * 2  # path + val
    block_total = blocks * per_block

    print(f"\n[2] PER-BLOCK ALLOCATIONS")
    print(f"  Blocks in flight:     {blocks}")
    print(f"  Per block (path+val): {per_block}")
    print(f"  Total:                {block_total}")

    # Temp allocations per iteration
    temps_per_block = VLEN * 3  # addr, node, tmp
    temp_total = blocks * temps_per_block

    print(f"\n[3] TEMPORARY ALLOCATIONS (per iteration)")
    print(f"  Per block (addr+node+tmp): {temps_per_block}")
    print(f"  Peak temp usage:           {temp_total}")

    # Preloaded nodes
    preloaded = 1 + 2 + 4  # depth 0, 1, 2
    preloaded_space = preloaded * VLEN

    print(f"\n[4] PRELOADED NODES")
    print(f"  Nodes preloaded:   {preloaded}")
    print(f"  Space (vectors):   {preloaded_space}")

    # Total estimate
    total_static = fixed + block_total + preloaded_space
    total_with_temps = total_static + temp_total

    print(f"\n[5] TOTAL ESTIMATE")
    print(f"  Static allocation:     {total_static}")
    print(f"  Peak with temps:       {total_with_temps}")
    print(f"  Limit:                 {SCRATCH_SIZE}")
    print(f"  Margin:                {SCRATCH_SIZE - total_with_temps}")

    if total_with_temps > SCRATCH_SIZE:
        print(f"\n  WARNING: Estimated usage exceeds scratch limit!")
        max_unroll = (SCRATCH_SIZE - fixed - preloaded_space) // (per_block + temps_per_block)
        print(f"  Suggested max unroll: {max_unroll}")


def main():
    parser = argparse.ArgumentParser(description="Analyze scratch memory usage")
    parser.add_argument("--forest-height", type=int, default=10)
    parser.add_argument("--n-nodes", type=int, default=2047)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--trace-allocs", action="store_true",
                       help="Show full allocation trace")
    parser.add_argument("--estimate", action="store_true",
                       help="Estimate scratch needs without building")
    parser.add_argument("--unroll", type=int, default=27,
                       help="Unroll factor for estimation")
    args = parser.parse_args()

    if args.estimate:
        estimate_scratch_needs(args.forest_height, args.batch_size, args.unroll)
    else:
        results = analyze_scratch_usage(
            args.forest_height, args.n_nodes,
            args.batch_size, args.rounds,
            trace_allocs=args.trace_allocs
        )
        print_scratch_report(results, verbose=True)


if __name__ == "__main__":
    main()

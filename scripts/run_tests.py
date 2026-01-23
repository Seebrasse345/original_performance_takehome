"""
Test Runner - Quick correctness testing across configurations.

Provides:
- Fast correctness verification
- Multiple seed testing
- Regression detection

Usage:
    python scripts/run_tests.py
    python scripts/run_tests.py --quick
    python scripts/run_tests.py --seeds 123,456,789
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random
from typing import List, Tuple

from problem import Machine, Tree, Input, build_mem_image, reference_kernel2, N_CORES
from perf_takehome import KernelBuilder


def quick_correctness_test(seed: int = 123) -> Tuple[bool, int, str]:
    """Quick correctness test returning (success, cycles, error_msg)"""
    try:
        random.seed(seed)
        forest = Tree.generate(10)
        inp = Input.generate(forest, 256, 16)
        mem = build_mem_image(forest, inp)

        kb = KernelBuilder(enable_debug_ops=True)
        kb.build_kernel(forest.height, len(forest.values), len(inp.indices), 16)

        value_trace = {}
        machine = Machine(mem, kb.instrs, kb.debug_info(),
                         n_cores=N_CORES, value_trace=value_trace)
        machine.enable_pause = True
        machine.enable_debug = True
        machine.prints = False

        for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
            machine.run()
            inp_values_p = ref_mem[6]
            actual = machine.mem[inp_values_p:inp_values_p + len(inp.values)]
            expected = ref_mem[inp_values_p:inp_values_p + len(inp.values)]

            if actual != expected:
                diff_count = sum(1 for a, e in zip(actual, expected) if a != e)
                return False, machine.cycle, f"Round {i}: {diff_count} values differ"

        return True, machine.cycle, ""

    except Exception as e:
        return False, -1, str(e)


def run_test_suite(seeds: List[int], verbose: bool = True) -> bool:
    """Run tests across multiple seeds"""
    print("=" * 60)
    print("CORRECTNESS TEST SUITE")
    print("=" * 60)

    all_passed = True
    cycles_list = []

    print(f"\n{'Seed':>10} | {'Status':>8} | {'Cycles':>10} | {'Notes'}")
    print("-" * 60)

    for seed in seeds:
        success, cycles, error = quick_correctness_test(seed)
        status = "PASS" if success else "FAIL"

        if success:
            cycles_list.append(cycles)
        else:
            all_passed = False

        notes = error[:30] if error else ""
        cycle_str = f"{cycles:,}" if cycles > 0 else "ERROR"
        print(f"{seed:>10} | {status:>8} | {cycle_str:>10} | {notes}")

    print("-" * 60)

    if cycles_list:
        avg = sum(cycles_list) / len(cycles_list)
        print(f"\nSummary:")
        print(f"  Tests passed: {len(cycles_list)}/{len(seeds)}")
        print(f"  Avg cycles:   {avg:,.0f}")
        if len(cycles_list) > 1:
            print(f"  Min cycles:   {min(cycles_list):,}")
            print(f"  Max cycles:   {max(cycles_list):,}")

    return all_passed


def run_quick_test() -> Tuple[bool, int]:
    """Single quick test returning (success, cycles)"""
    success, cycles, error = quick_correctness_test(123)
    if success:
        print(f"PASS: {cycles:,} cycles")
    else:
        print(f"FAIL: {error}")
    return success, cycles


def run_config_tests() -> bool:
    """Test various configurations"""
    print("\nConfiguration Tests:")
    print("-" * 60)

    configs = [
        (10, 8, 4, "Small batch, few rounds"),
        (10, 64, 8, "Medium batch"),
        (10, 256, 16, "Standard config"),
        (5, 256, 16, "Shallow tree"),
    ]

    all_passed = True
    for height, batch, rounds, desc in configs:
        try:
            random.seed(123)
            forest = Tree.generate(height)
            inp = Input.generate(forest, batch, rounds)
            mem = build_mem_image(forest, inp)

            kb = KernelBuilder(enable_debug_ops=True)
            kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

            value_trace = {}
            machine = Machine(mem, kb.instrs, kb.debug_info(),
                             n_cores=N_CORES, value_trace=value_trace)
            machine.enable_pause = True
            machine.enable_debug = True
            machine.prints = False

            success = True
            for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
                machine.run()
                inp_values_p = ref_mem[6]
                actual = machine.mem[inp_values_p:inp_values_p + len(inp.values)]
                expected = ref_mem[inp_values_p:inp_values_p + len(inp.values)]
                if actual != expected:
                    success = False
                    break

            status = "PASS" if success else "FAIL"
            print(f"  {desc:25s}: {status:4s} ({machine.cycle:,} cycles)")

            if not success:
                all_passed = False

        except Exception as e:
            print(f"  {desc:25s}: ERR  ({str(e)[:30]})")
            all_passed = False

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Run correctness tests")
    parser.add_argument("--quick", "-q", action="store_true",
                       help="Quick single test")
    parser.add_argument("--seeds", type=str, default="123,456,789,1000,2000",
                       help="Comma-separated seeds to test")
    parser.add_argument("--configs", action="store_true",
                       help="Test various configurations")
    args = parser.parse_args()

    if args.quick:
        success, _ = run_quick_test()
        sys.exit(0 if success else 1)
    elif args.configs:
        success = run_config_tests()
        sys.exit(0 if success else 1)
    else:
        seeds = [int(s) for s in args.seeds.split(",")]
        success = run_test_suite(seeds)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

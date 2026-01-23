"""
Comparison Tool - A/B testing framework for configurations.

Provides:
- Compare kernel performance across configurations
- Test different batch sizes, rounds, unroll factors
- Correctness verification
- Statistical analysis

Usage:
    python scripts/comparison.py
    python scripts/comparison.py --sweep-unroll
    python scripts/comparison.py --sweep-batch
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
import time

from problem import Machine, Tree, Input, build_mem_image, reference_kernel2, N_CORES, VLEN
from perf_takehome import KernelBuilder


@dataclass
class TestResult:
    """Result of a single test run"""
    config: Dict
    cycles: int
    correct: bool
    build_time_ms: float
    run_time_ms: float
    error: Optional[str] = None


def run_kernel_test(forest_height: int = 10, batch_size: int = 256,
                   rounds: int = 16, seed: int = 123,
                   verify_correctness: bool = True) -> TestResult:
    """Run kernel and return detailed results"""
    config = {
        "forest_height": forest_height,
        "batch_size": batch_size,
        "rounds": rounds,
        "seed": seed,
    }

    try:
        # Setup
        random.seed(seed)
        forest = Tree.generate(forest_height)
        inp = Input.generate(forest, batch_size, rounds)
        mem = build_mem_image(forest, inp)

        # Build kernel
        build_start = time.perf_counter()
        kb = KernelBuilder(enable_debug_ops=verify_correctness)
        kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
        build_time = (time.perf_counter() - build_start) * 1000

        # Run kernel
        value_trace = {}
        run_start = time.perf_counter()
        machine = Machine(mem, kb.instrs, kb.debug_info(),
                         n_cores=N_CORES, value_trace=value_trace)
        machine.enable_pause = verify_correctness
        machine.enable_debug = verify_correctness
        machine.prints = False

        correct = True
        if verify_correctness:
            for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
                machine.run()
                inp_values_p = ref_mem[6]
                actual = machine.mem[inp_values_p:inp_values_p + len(inp.values)]
                expected = ref_mem[inp_values_p:inp_values_p + len(inp.values)]
                if actual != expected:
                    correct = False
                    break
        else:
            machine.run()

        run_time = (time.perf_counter() - run_start) * 1000

        return TestResult(
            config=config,
            cycles=machine.cycle,
            correct=correct,
            build_time_ms=build_time,
            run_time_ms=run_time,
        )

    except Exception as e:
        return TestResult(
            config=config,
            cycles=-1,
            correct=False,
            build_time_ms=0,
            run_time_ms=0,
            error=str(e),
        )


def compare_configs(configs: List[Dict], verify: bool = True) -> List[TestResult]:
    """Compare multiple configurations"""
    results = []

    print(f"{'Config':30s} | {'Cycles':>8} | {'Correct':>7} | {'Build':>8} | {'Run':>8}")
    print("-" * 75)

    for config in configs:
        result = run_kernel_test(
            forest_height=config.get("forest_height", 10),
            batch_size=config.get("batch_size", 256),
            rounds=config.get("rounds", 16),
            seed=config.get("seed", 123),
            verify_correctness=verify,
        )
        results.append(result)

        config_str = ", ".join(f"{k}={v}" for k, v in config.items() if k != "seed")
        if len(config_str) > 28:
            config_str = config_str[:25] + "..."

        status = "OK" if result.correct else "FAIL"
        if result.error:
            status = "ERR"
            print(f"{config_str:30s} | {'ERROR':>8} | {status:>7} | "
                  f"{result.build_time_ms:>7.1f}ms | {result.error[:20]}")
        else:
            print(f"{config_str:30s} | {result.cycles:>8,} | {status:>7} | "
                  f"{result.build_time_ms:>7.1f}ms | {result.run_time_ms:>7.1f}ms")

    return results


def sweep_parameter(param_name: str, values: List, base_config: Dict,
                   verify: bool = True) -> List[TestResult]:
    """Sweep a single parameter across values"""
    configs = []
    for val in values:
        config = base_config.copy()
        config[param_name] = val
        configs.append(config)

    print(f"\nSweeping {param_name}: {values}")
    print("=" * 75)
    return compare_configs(configs, verify)


def print_sweep_summary(results: List[TestResult], param_name: str):
    """Print summary of parameter sweep"""
    valid_results = [r for r in results if r.correct and r.cycles > 0]

    if not valid_results:
        print("\nNo valid results to summarize")
        return

    print(f"\n{'=' * 50}")
    print(f"SWEEP SUMMARY: {param_name}")
    print(f"{'=' * 50}")

    best = min(valid_results, key=lambda r: r.cycles)
    worst = max(valid_results, key=lambda r: r.cycles)

    print(f"\nBest:  {best.config[param_name]} -> {best.cycles:,} cycles")
    print(f"Worst: {worst.config[param_name]} -> {worst.cycles:,} cycles")
    print(f"Range: {worst.cycles - best.cycles:,} cycles ({100*(worst.cycles-best.cycles)/best.cycles:.1f}%)")

    # Show trend
    print(f"\n{param_name:>15} | {'Cycles':>10} | {'Delta':>10}")
    print("-" * 40)
    prev_cycles = None
    for r in results:
        if r.cycles > 0:
            delta = f"{r.cycles - prev_cycles:+,}" if prev_cycles else "-"
            print(f"{r.config[param_name]:>15} | {r.cycles:>10,} | {delta:>10}")
            prev_cycles = r.cycles


def run_comprehensive_comparison():
    """Run comprehensive comparison across standard configurations"""
    print("=" * 75)
    print("COMPREHENSIVE PERFORMANCE COMPARISON")
    print("=" * 75)

    # Test standard configuration
    print("\n[1] STANDARD CONFIGURATION")
    print("-" * 50)
    base_result = run_kernel_test(verify_correctness=True)
    print(f"  Cycles:  {base_result.cycles:,}")
    print(f"  Correct: {base_result.correct}")
    print(f"  Build:   {base_result.build_time_ms:.1f}ms")

    # Quick sanity tests
    print("\n[2] CORRECTNESS VERIFICATION")
    print("-" * 50)
    configs = [
        {"batch_size": 8, "rounds": 4},
        {"batch_size": 64, "rounds": 8},
        {"batch_size": 256, "rounds": 16},
    ]
    for config in configs:
        result = run_kernel_test(**config, verify_correctness=True)
        status = "PASS" if result.correct else "FAIL"
        print(f"  batch={config['batch_size']:>3}, rounds={config['rounds']:>2}: "
              f"{status} ({result.cycles:,} cycles)")

    return base_result


def main():
    parser = argparse.ArgumentParser(description="Compare kernel configurations")
    parser.add_argument("--sweep-batch", action="store_true",
                       help="Sweep batch sizes")
    parser.add_argument("--sweep-rounds", action="store_true",
                       help="Sweep round counts")
    parser.add_argument("--comprehensive", action="store_true",
                       help="Run comprehensive comparison")
    parser.add_argument("--no-verify", action="store_true",
                       help="Skip correctness verification (faster)")
    parser.add_argument("--batch-values", type=str, default="8,16,32,64,128,256",
                       help="Comma-separated batch sizes to test")
    parser.add_argument("--round-values", type=str, default="4,8,16,32",
                       help="Comma-separated round counts to test")
    args = parser.parse_args()

    verify = not args.no_verify

    if args.comprehensive:
        run_comprehensive_comparison()
    elif args.sweep_batch:
        batch_values = [int(x) for x in args.batch_values.split(",")]
        results = sweep_parameter("batch_size", batch_values,
                                 {"forest_height": 10, "rounds": 16}, verify)
        print_sweep_summary(results, "batch_size")
    elif args.sweep_rounds:
        round_values = [int(x) for x in args.round_values.split(",")]
        results = sweep_parameter("rounds", round_values,
                                 {"forest_height": 10, "batch_size": 256}, verify)
        print_sweep_summary(results, "rounds")
    else:
        # Default: single test
        result = run_kernel_test(verify_correctness=verify)
        print(f"\nResult: {result.cycles:,} cycles, correct={result.correct}")


if __name__ == "__main__":
    main()

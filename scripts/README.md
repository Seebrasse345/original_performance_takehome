# Diagnostic Toolkit for VLIW SIMD Kernel Optimization

This toolkit provides comprehensive analysis tools for understanding and optimizing the VLIW SIMD kernel performance.

## Quick Start

```bash
# Run full diagnostic (recommended first step)
python scripts/full_diagnostic.py

# Quick correctness test
python scripts/run_tests.py --quick

# Check bottlenecks
python scripts/bottleneck_analyzer.py
```

## Available Tools

### 1. full_diagnostic.py - Comprehensive Analysis
Runs all diagnostics and produces a complete report.

```bash
# Full analysis
python scripts/full_diagnostic.py

# Quick mode (skip detailed analysis)
python scripts/full_diagnostic.py --quick

# Save to file
python scripts/full_diagnostic.py -o report.txt
```

**Output includes:**
- Kernel statistics
- Correctness verification
- Bottleneck analysis
- Operation breakdown
- Cycle trace summary
- Scratch memory usage
- Scheduling analysis
- Optimization recommendations

---

### 2. bottleneck_analyzer.py - Engine Bottleneck Detection
Identifies which engine is limiting performance.

```bash
python scripts/bottleneck_analyzer.py
python scripts/bottleneck_analyzer.py --batch-size 512
python scripts/bottleneck_analyzer.py -q  # minimal output
```

**Key metrics:**
- Operations per engine
- Theoretical minimum cycles
- Slot utilization
- Bottleneck identification

---

### 3. op_breakdown.py - Operation Analysis
Detailed breakdown of all operations by type and engine.

```bash
# By operation type
python scripts/op_breakdown.py

# Estimated by depth level
python scripts/op_breakdown.py --by-depth
```

**Shows:**
- Operations by type within each engine
- Hash function cost analysis
- Per-depth operation estimates

---

### 4. cycle_trace.py - Cycle-by-Cycle Analysis
Shows what happens each cycle.

```bash
# Trace cycles 100-150
python scripts/cycle_trace.py --start 100 --count 50

# Summary only
python scripts/cycle_trace.py --summary

# Show individual ops
python scripts/cycle_trace.py --start 0 --count 20 --show-ops
```

**Analyzes:**
- Slot utilization per cycle
- Stall detection
- Engine saturation
- Utilization distribution

---

### 5. scratch_analyzer.py - Memory Usage Analysis
Analyzes scratch space usage and allocation.

```bash
python scripts/scratch_analyzer.py

# Estimate needs for configuration
python scripts/scratch_analyzer.py --estimate --unroll 32
```

**Reports:**
- Named allocations
- Temporary watermark
- Vector vs scalar usage
- Constant cache

---

### 6. scheduling_analyzer.py - Scheduling Quality
Analyzes instruction scheduling and parallelism.

```bash
python scripts/scheduling_analyzer.py

# Show critical path operations
python scripts/scheduling_analyzer.py --critical-path
```

**Metrics:**
- Dependency analysis
- Critical path length
- ILP (instruction level parallelism)
- Height distribution

---

### 7. comparison.py - A/B Testing
Compare configurations and sweep parameters.

```bash
# Single test
python scripts/comparison.py

# Sweep batch sizes
python scripts/comparison.py --sweep-batch

# Sweep rounds
python scripts/comparison.py --sweep-rounds

# Comprehensive comparison
python scripts/comparison.py --comprehensive
```

---

### 8. run_tests.py - Correctness Testing
Fast correctness verification.

```bash
# Quick single test
python scripts/run_tests.py --quick

# Test multiple seeds
python scripts/run_tests.py --seeds 123,456,789

# Test configurations
python scripts/run_tests.py --configs
```

---

## Architecture Reference

### Engine Slot Limits
| Engine | Slots/Cycle | Description |
|--------|-------------|-------------|
| alu    | 12          | Scalar arithmetic |
| valu   | 6           | Vector arithmetic |
| load   | 2           | Memory loads |
| store  | 2           | Memory stores |
| flow   | 1           | Control flow, vselect |

### Key Constants
- `VLEN = 8` - Vector length
- `SCRATCH_SIZE = 1536` - Total scratch space
- `N_CORES = 1` - Number of cores

### Understanding Bottlenecks

**Load-bound (most common):**
- 2 slots/cycle limits memory access
- Each depth 3+ round needs 8 loads per vector batch
- Preloading can help but uses flow ops

**Flow-bound (critical limitation):**
- Only 1 flow op per cycle!
- Each vselect counts as 1 flow op
- Trade-off: preloading saves loads but adds vselects

**VALU-bound:**
- Hash function uses ~12 VALU ops per element
- 6 slots/cycle available
- Use multiply_add for linear stages

## Typical Optimization Workflow

1. **Start with full diagnostic:**
   ```bash
   python scripts/full_diagnostic.py -o baseline.txt
   ```

2. **Identify bottleneck:**
   ```bash
   python scripts/bottleneck_analyzer.py
   ```

3. **Understand operation distribution:**
   ```bash
   python scripts/op_breakdown.py --by-depth
   ```

4. **Check scratch budget before changes:**
   ```bash
   python scripts/scratch_analyzer.py --estimate --unroll 32
   ```

5. **Make changes to perf_takehome.py**

6. **Verify correctness:**
   ```bash
   python scripts/run_tests.py --quick
   ```

7. **Compare with baseline:**
   ```bash
   python scripts/comparison.py
   ```

8. **Run full diagnostic again:**
   ```bash
   python scripts/full_diagnostic.py -o optimized.txt
   diff baseline.txt optimized.txt
   ```

## Output Interpretation

### Efficiency Percentage
`Efficiency = Theoretical_Min / Actual_Cycles * 100%`

- **90%+**: Excellent - close to hardware limit
- **80-90%**: Good - some scheduling overhead
- **70-80%**: Room for improvement
- **<70%**: Significant optimization potential

### Slot Utilization
Per-engine utilization shows how well each engine is used:
- **100%**: Engine fully utilized (may be bottleneck)
- **50-99%**: Good utilization
- **<50%**: Engine underutilized (opportunity for work migration)

### Critical Path
The longest dependency chain through the computation:
- Short critical path = more parallelism potential
- Long critical path = sequential bottleneck

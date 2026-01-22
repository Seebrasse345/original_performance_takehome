"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

import bisect
from dataclasses import dataclass
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


@dataclass
class Op:
    engine: Engine
    slot: tuple
    reads: tuple[int, ...]
    writes: tuple[int, ...]


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.const_vec_map = {}
        self.ops = []

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def _vec_addrs(self, base, length=VLEN):
        return tuple(range(base, base + length))

    def _emit_op(self, engine, slot, reads=(), writes=()):
        self.ops.append(
            Op(engine=engine, slot=slot, reads=tuple(reads), writes=tuple(writes))
        )

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def alloc_vec(self, name=None, length=VLEN):
        return self.alloc_scratch(name, length)

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self._emit_op("load", ("const", addr, val), writes=(addr,))
            self.const_map[val] = addr
        return self.const_map[val]

    def scratch_const_vec(self, val, name=None):
        if val not in self.const_vec_map:
            vec_addr = self.alloc_vec(name)
            scalar_addr = self.scratch_const(val)
            self._emit_op(
                "valu",
                ("vbroadcast", vec_addr, scalar_addr),
                reads=(scalar_addr,),
                writes=self._vec_addrs(vec_addr),
            )
            self.const_vec_map[val] = vec_addr
        return self.const_vec_map[val]

    def emit_alu(self, op, dest, a1, a2):
        self._emit_op("alu", (op, dest, a1, a2), reads=(a1, a2), writes=(dest,))

    def emit_valu(self, op, dest, a1, a2):
        reads = self._vec_addrs(a1) + self._vec_addrs(a2)
        writes = self._vec_addrs(dest)
        self._emit_op("valu", (op, dest, a1, a2), reads=reads, writes=writes)

    def emit_valu_madd(self, dest, a, b, c):
        reads = self._vec_addrs(a) + self._vec_addrs(b) + self._vec_addrs(c)
        writes = self._vec_addrs(dest)
        self._emit_op(
            "valu", ("multiply_add", dest, a, b, c), reads=reads, writes=writes
        )

    def emit_load(self, dest, addr):
        self._emit_op("load", ("load", dest, addr), reads=(addr,), writes=(dest,))

    def emit_vload(self, dest, addr):
        self._emit_op(
            "load", ("vload", dest, addr), reads=(addr,), writes=self._vec_addrs(dest)
        )

    def emit_load_offset(self, dest, addr, offset):
        self._emit_op(
            "load",
            ("load_offset", dest, addr, offset),
            reads=(addr + offset,),
            writes=(dest + offset,),
        )

    def emit_store(self, addr, src):
        self._emit_op("store", ("store", addr, src), reads=(addr, src))

    def emit_vstore(self, addr, src):
        reads = (addr,) + self._vec_addrs(src)
        self._emit_op("store", ("vstore", addr, src), reads=reads)

    def build_hash_vec(self, val_addr, tmp_addr):
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            c1 = self.const_vec_map[val1]
            c3 = self.const_vec_map[val3]
            self.emit_valu(op1, tmp_addr, val_addr, c1)
            self.emit_valu(op3, val_addr, val_addr, c3)
            self.emit_valu(op2, val_addr, tmp_addr, val_addr)

    def build_hash_scalar(self, val_addr, tmp_addr):
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            c1 = self.const_map[val1]
            c3 = self.const_map[val3]
            self.emit_alu(op1, tmp_addr, val_addr, c1)
            self.emit_alu(op3, val_addr, val_addr, c3)
            self.emit_alu(op2, val_addr, tmp_addr, val_addr)

    def build(self, ops: list["Op"]):
        if not ops:
            return []

        n_ops = len(ops)
        preds = [dict() for _ in range(n_ops)]
        succs = [[] for _ in range(n_ops)]
        last_read = {}
        last_write = {}

        for i, op in enumerate(ops):
            dep_map = preds[i]
            for addr in set(op.reads):
                pred = last_write.get(addr)
                if pred is not None:
                    dep_map[pred] = max(dep_map.get(pred, 0), 1)
            for addr in set(op.writes):
                pred = last_write.get(addr)
                if pred is not None:
                    dep_map[pred] = max(dep_map.get(pred, 0), 1)
                pred = last_read.get(addr)
                if pred is not None:
                    dep_map[pred] = max(dep_map.get(pred, 0), 0)
            for addr in op.reads:
                last_read[addr] = i
            for addr in op.writes:
                last_write[addr] = i

        indegree = [0] * n_ops
        ready_after = [0] * n_ops
        for i, dep_map in enumerate(preds):
            indegree[i] = len(dep_map)
            for pred, latency in dep_map.items():
                succs[pred].append((i, latency))

        ready = []
        for i in range(n_ops):
            if indegree[i] == 0:
                ready.append(i)
        ready.sort()

        instrs = []
        scheduled = 0
        cycle = 0
        engine_order = ["load", "store", "valu", "alu", "flow"]

        while scheduled < n_ops:
            bundle_ops = {engine: [] for engine in engine_order}
            slots_left = {engine: SLOT_LIMITS[engine] for engine in engine_order}
            scheduled_this_cycle = []

            made_progress = True
            while made_progress:
                made_progress = False
                for engine in engine_order:
                    if slots_left[engine] <= 0:
                        continue
                    chosen = None
                    for idx in ready:
                        op = ops[idx]
                        if op.engine != engine:
                            continue
                        if ready_after[idx] > cycle:
                            continue
                        chosen = idx
                        break
                    if chosen is None:
                        continue
                    bundle_ops[engine].append(chosen)
                    slots_left[engine] -= 1
                    scheduled_this_cycle.append(chosen)
                    ready.remove(chosen)
                    scheduled += 1

                    for succ, latency in succs[chosen]:
                        indegree[succ] -= 1
                        ready_after[succ] = max(ready_after[succ], cycle + latency)
                        if indegree[succ] == 0:
                            bisect.insort(ready, succ)
                    made_progress = True

            has_ops = any(bundle_ops[engine] for engine in engine_order)
            if not has_ops:
                instrs.append({"alu": []})
                cycle += 1
                continue

            instr = {}
            for engine in engine_order:
                if bundle_ops[engine]:
                    instr[engine] = [ops[i].slot for i in bundle_ops[engine]]
            instrs.append(instr)
            cycle += 1

        return instrs

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Vectorized implementation that stages the full batch in scratch,
        runs all rounds in-place, and writes back once at the end.
        """
        forest_values_p = 7
        inp_indices_p = forest_values_p + n_nodes
        inp_values_p = inp_indices_p + batch_size

        self.ops = []

        tmp_addr = self.alloc_scratch("tmp_addr")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")

        idx_base = self.alloc_scratch("idx", batch_size)
        val_base = self.alloc_scratch("val", batch_size)

        buffers = 3
        addr_bufs = [self.alloc_vec(f"addr_buf{bi}") for bi in range(buffers)]
        node_bufs = [self.alloc_vec(f"node_buf{bi}") for bi in range(buffers)]
        tmp_bufs = [self.alloc_vec(f"tmp_buf{bi}") for bi in range(buffers)]

        one_const = self.scratch_const(1, "one")
        two_const = self.scratch_const(2, "two")
        forest_const = self.scratch_const(forest_values_p, "forest_values_p")
        inp_idx_const = self.scratch_const(inp_indices_p, "inp_indices_p")
        inp_val_const = self.scratch_const(inp_values_p, "inp_values_p")
        n_nodes_const = self.scratch_const(n_nodes, "n_nodes")

        v_one = self.scratch_const_vec(1, "vone")
        v_two = self.scratch_const_vec(2, "vtwo")
        v_forest = self.scratch_const_vec(forest_values_p, "vforest_values_p")
        v_n_nodes = self.scratch_const_vec(n_nodes, "vn_nodes")

        for op1, val1, op2, op3, val3 in HASH_STAGES:
            self.scratch_const(val1)
            self.scratch_const(val3)
            self.scratch_const_vec(val1)
            self.scratch_const_vec(val3)

        vec_blocks = batch_size // VLEN
        tail_start = vec_blocks * VLEN

        block_offsets = []
        for b in range(vec_blocks):
            block_offsets.append(self.scratch_const(b * VLEN, f"off_{b * VLEN}"))

        tail_offsets = []
        for i in range(tail_start, batch_size):
            tail_offsets.append(self.scratch_const(i, f"off_{i}"))

        for b, offset_const in enumerate(block_offsets):
            offset = b * VLEN
            self.emit_alu("+", tmp_addr, inp_idx_const, offset_const)
            self.emit_vload(idx_base + offset, tmp_addr)
            self.emit_alu("+", tmp_addr, inp_val_const, offset_const)
            self.emit_vload(val_base + offset, tmp_addr)

        for i, offset_const in enumerate(tail_offsets):
            idx = tail_start + i
            self.emit_alu("+", tmp_addr, inp_idx_const, offset_const)
            self.emit_load(idx_base + idx, tmp_addr)
            self.emit_alu("+", tmp_addr, inp_val_const, offset_const)
            self.emit_load(val_base + idx, tmp_addr)

        self.instrs.extend(self.build(self.ops))
        self.instrs.append({"flow": [("pause",)]})

        self.ops = []

        for _round in range(rounds):
            for b in range(vec_blocks):
                offset = b * VLEN
                buf = b % buffers
                idx_addr = idx_base + offset
                val_addr = val_base + offset
                addr_buf = addr_bufs[buf]
                node_buf = node_bufs[buf]
                tmp_buf = tmp_bufs[buf]

                self.emit_valu("+", addr_buf, idx_addr, v_forest)
                for off in range(VLEN):
                    self.emit_load_offset(node_buf, addr_buf, off)
                self.emit_valu("^", val_addr, val_addr, node_buf)
                self.build_hash_vec(val_addr, tmp_buf)
                self.emit_valu("&", node_buf, val_addr, v_one)
                self.emit_valu("+", addr_buf, node_buf, v_one)
                self.emit_valu_madd(idx_addr, idx_addr, v_two, addr_buf)
                self.emit_valu("<", node_buf, idx_addr, v_n_nodes)
                self.emit_valu("*", idx_addr, idx_addr, node_buf)

            for i in range(tail_start, batch_size):
                idx_addr = idx_base + i
                val_addr = val_base + i
                self.emit_alu("+", tmp_addr, idx_addr, forest_const)
                self.emit_load(tmp_node_val, tmp_addr)
                self.emit_alu("^", val_addr, val_addr, tmp_node_val)
                self.build_hash_scalar(val_addr, tmp1)
                self.emit_alu("&", tmp1, val_addr, one_const)
                self.emit_alu("+", tmp1, tmp1, one_const)
                self.emit_alu("*", idx_addr, idx_addr, two_const)
                self.emit_alu("+", idx_addr, idx_addr, tmp1)
                self.emit_alu("<", tmp2, idx_addr, n_nodes_const)
                self.emit_alu("*", idx_addr, idx_addr, tmp2)

        for b, offset_const in enumerate(block_offsets):
            offset = b * VLEN
            self.emit_alu("+", tmp_addr, inp_idx_const, offset_const)
            self.emit_vstore(tmp_addr, idx_base + offset)
            self.emit_alu("+", tmp_addr, inp_val_const, offset_const)
            self.emit_vstore(tmp_addr, val_base + offset)

        for i, offset_const in enumerate(tail_offsets):
            idx = tail_start + i
            self.emit_alu("+", tmp_addr, inp_idx_const, offset_const)
            self.emit_store(tmp_addr, idx_base + idx)
            self.emit_alu("+", tmp_addr, inp_val_const, offset_const)
            self.emit_store(tmp_addr, val_base + idx)

        self.instrs.extend(self.build(self.ops))
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()

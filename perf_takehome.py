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

from collections import defaultdict
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


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.const_vec_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        if not vliw:
            instrs = []
            for engine, slot in slots:
                instrs.append({engine: [slot]})
            return instrs
        return self.build_vliw(slots)

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_vec(self, name=None):
        return self.alloc_scratch(name, VLEN)

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def scratch_const_vec(self, val, name=None):
        if val not in self.const_vec_map:
            scalar_addr = self.scratch_const(val)
            vec_addr = self.alloc_vec(name)
            self.add("valu", ("vbroadcast", vec_addr, scalar_addr))
            self.const_vec_map[val] = vec_addr
        return self.const_vec_map[val]

    def _slot_reads_writes(self, engine, slot):
        reads = set()
        writes = set()
        match engine:
            case "alu":
                _, dest, a1, a2 = slot
                reads.update([a1, a2])
                writes.add(dest)
            case "valu":
                match slot:
                    case ("vbroadcast", dest, src):
                        reads.add(src)
                        writes.update(range(dest, dest + VLEN))
                    case ("multiply_add", dest, a, b, c):
                        reads.update(range(a, a + VLEN))
                        reads.update(range(b, b + VLEN))
                        reads.update(range(c, c + VLEN))
                        writes.update(range(dest, dest + VLEN))
                    case (op, dest, a1, a2):
                        reads.update(range(a1, a1 + VLEN))
                        reads.update(range(a2, a2 + VLEN))
                        writes.update(range(dest, dest + VLEN))
                    case _:
                        pass
            case "load":
                match slot:
                    case ("load", dest, addr):
                        reads.add(addr)
                        writes.add(dest)
                    case ("load_offset", dest, addr, offset):
                        reads.add(addr + offset)
                        writes.add(dest + offset)
                    case ("vload", dest, addr):
                        reads.add(addr)
                        writes.update(range(dest, dest + VLEN))
                    case ("const", dest, _):
                        writes.add(dest)
                    case _:
                        pass
            case "store":
                match slot:
                    case ("store", addr, src):
                        reads.add(addr)
                        reads.add(src)
                    case ("vstore", addr, src):
                        reads.add(addr)
                        reads.update(range(src, src + VLEN))
                    case _:
                        pass
            case "flow":
                match slot:
                    case ("select", dest, cond, a, b):
                        reads.update([cond, a, b])
                        writes.add(dest)
                    case ("add_imm", dest, a, _):
                        reads.add(a)
                        writes.add(dest)
                    case ("vselect", dest, cond, a, b):
                        reads.update(range(cond, cond + VLEN))
                        reads.update(range(a, a + VLEN))
                        reads.update(range(b, b + VLEN))
                        writes.update(range(dest, dest + VLEN))
                    case ("coreid", dest):
                        writes.add(dest)
                    case ("trace_write", val):
                        reads.add(val)
                    case ("cond_jump", cond, _):
                        reads.add(cond)
                    case ("cond_jump_rel", cond, _):
                        reads.add(cond)
                    case ("jump_indirect", addr):
                        reads.add(addr)
                    case _:
                        pass
            case _:
                pass
        return reads, writes

    def build_vliw(self, slots):
        instrs = []
        current = {}
        counts = defaultdict(int)
        writes_in_cycle = set()

        def flush():
            nonlocal current, counts, writes_in_cycle
            if current:
                instrs.append(current)
            current = {}
            counts = defaultdict(int)
            writes_in_cycle = set()

        for engine, slot in slots:
            if engine is None or engine == "barrier":
                flush()
                continue
            reads, writes = self._slot_reads_writes(engine, slot)
            if (
                counts[engine] >= SLOT_LIMITS[engine]
                or reads & writes_in_cycle
                or writes & writes_in_cycle
            ):
                flush()
            current.setdefault(engine, []).append(slot)
            counts[engine] += 1
            writes_in_cycle |= writes

        flush()
        return instrs

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_hash_vec(self, val_hash_addr, tmp1, tmp2):
        stages = []
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            c1 = self.scratch_const_vec(val1)
            c3 = self.scratch_const_vec(val3)
            stages.append(
                [
                    ("valu", (op1, tmp1, val_hash_addr, c1)),
                    ("valu", (op3, tmp2, val_hash_addr, c3)),
                    ("valu", (op2, val_hash_addr, tmp1, tmp2)),
                ]
            )
        return stages

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Vectorized SIMD kernel with software-pipelined gather and hashing.
        """
        tmp_scalar = self.alloc_scratch("tmp_scalar")
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
            "dummy_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp_scalar, i))
            self.add("load", ("load", self.scratch[v], tmp_scalar))

        lane_offsets = self.alloc_vec("lane_offsets")
        for i in range(VLEN):
            self.add("load", ("const", lane_offsets + i, i))

        v_one = self.scratch_const_vec(1, "v_one")
        v_two = self.scratch_const_vec(2, "v_two")

        v_n_nodes = self.alloc_vec("v_n_nodes")
        self.add("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))
        v_batch_size = self.alloc_vec("v_batch_size")
        self.add("valu", ("vbroadcast", v_batch_size, self.scratch["batch_size"]))
        v_forest_values_p = self.alloc_vec("v_forest_values_p")
        self.add("valu", ("vbroadcast", v_forest_values_p, self.scratch["forest_values_p"]))
        v_dummy_base = self.alloc_vec("v_dummy_base")
        self.add("valu", ("vbroadcast", v_dummy_base, self.scratch["dummy_p"]))
        v_dummy_offsets = self.alloc_vec("v_dummy_offsets")
        self.add("valu", ("+", v_dummy_offsets, v_dummy_base, lane_offsets))

        for _, val1, _, _, val3 in HASH_STAGES:
            self.scratch_const_vec(val1, f"v_hash_{val1:08x}")
            self.scratch_const_vec(val3, f"v_hash_{val3:08x}")

        num_chunks = (batch_size + VLEN - 1) // VLEN
        for base in range(0, num_chunks * VLEN, VLEN):
            self.scratch_const(base)

        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting vectorized loop"))

        body = []

        streams = []
        for s in range(2):
            streams.append(
                {
                    "idx": self.alloc_vec(f"v_idx_{s}"),
                    "val": self.alloc_vec(f"v_val_{s}"),
                    "node": self.alloc_vec(f"v_node_{s}"),
                    "tmp1": self.alloc_vec(f"v_tmp1_{s}"),
                    "tmp2": self.alloc_vec(f"v_tmp2_{s}"),
                    "addr": self.alloc_vec(f"v_addr_{s}"),
                }
            )

        def interleave_ops(base_ops, extra_ops):
            if not extra_ops:
                return list(base_ops)
            extra = list(extra_ops)
            out = []
            if base_ops:
                out.append(base_ops[0])
                base_ops = base_ops[1:]
            for op in base_ops:
                out.append(op)
                if extra:
                    out.append(extra.pop(0))
            if extra:
                out.extend(extra)
            return out

        def idx_update_ops(stream):
            return [
                ("valu", ("&", stream["tmp1"], stream["val"], v_one)),
                ("valu", ("multiply_add", stream["idx"], stream["idx"], v_two, v_one)),
                ("valu", ("+", stream["idx"], stream["idx"], stream["tmp1"])),
                ("valu", ("<", stream["tmp2"], stream["idx"], v_n_nodes)),
                ("valu", ("*", stream["idx"], stream["idx"], stream["tmp2"])),
            ]

        def build_hash_ops(hash_stream, gather_stream, extra_ops=None, gather_delay_stages=0):
            stages = self.build_hash_vec(
                hash_stream["val"], hash_stream["tmp1"], hash_stream["tmp2"]
            )
            if gather_stream is None:
                delay = len(stages)
            else:
                delay = min(gather_delay_stages, len(stages))
            ops_pre = [("valu", ("^", hash_stream["val"], hash_stream["val"], hash_stream["node"]))]
            for si in range(delay):
                ops_pre.extend(stages[si])
            ops = interleave_ops(ops_pre, extra_ops)

            ops_mid = []
            load_offset = 0
            if gather_stream is not None:
                ops_mid.append(
                    (
                        "valu",
                        ("+", gather_stream["addr"], gather_stream["idx"], v_forest_values_p),
                    )
                )
            for si in range(delay, len(stages)):
                stage_ops = stages[si]
                ops_mid.extend(stage_ops[:2])
                if gather_stream is not None and load_offset < VLEN:
                    ops_mid.append(
                        ("load", ("load_offset", gather_stream["node"], gather_stream["addr"], load_offset))
                    )
                    ops_mid.append(
                        (
                            "load",
                            ("load_offset", gather_stream["node"], gather_stream["addr"], load_offset + 1),
                        )
                    )
                    load_offset += 2
                ops_mid.append(stage_ops[2])
            ops.extend(ops_mid)
            return ops, idx_update_ops(hash_stream)

        def emit_gather_only(stream):
            body.append(("valu", ("+", stream["addr"], stream["idx"], v_forest_values_p)))
            for off in range(VLEN):
                body.append(("load", ("load_offset", stream["node"], stream["addr"], off)))

        def emit_load_full(stream, base):
            base_const = self.scratch_const(base)
            body.append(("alu", ("+", tmp_scalar, self.scratch["inp_indices_p"], base_const)))
            body.append(("load", ("vload", stream["idx"], tmp_scalar)))
            body.append(("alu", ("+", tmp_scalar, self.scratch["inp_values_p"], base_const)))
            body.append(("load", ("vload", stream["val"], tmp_scalar)))

        def emit_store_full(stream, base):
            base_const = self.scratch_const(base)
            body.append(("alu", ("+", tmp_scalar, self.scratch["inp_indices_p"], base_const)))
            body.append(("store", ("vstore", tmp_scalar, stream["idx"])))
            body.append(("alu", ("+", tmp_scalar, self.scratch["inp_values_p"], base_const)))
            body.append(("store", ("vstore", tmp_scalar, stream["val"])))

        def emit_load_tail(stream, base):
            base_const = self.scratch_const(base)
            body.append(("valu", ("vbroadcast", stream["tmp1"], base_const)))
            body.append(("valu", ("+", stream["tmp1"], stream["tmp1"], lane_offsets)))
            body.append(("valu", ("<", stream["tmp2"], stream["tmp1"], v_batch_size)))
            body.append(("valu", ("-", stream["tmp1"], v_one, stream["tmp2"])))

            body.append(("alu", ("+", tmp_scalar, self.scratch["inp_indices_p"], base_const)))
            body.append(("valu", ("vbroadcast", stream["addr"], tmp_scalar)))
            body.append(("valu", ("+", stream["addr"], stream["addr"], lane_offsets)))
            body.append(("valu", ("*", stream["addr"], stream["addr"], stream["tmp2"])))
            body.append(("valu", ("*", stream["node"], v_dummy_offsets, stream["tmp1"])))
            body.append(("valu", ("+", stream["addr"], stream["addr"], stream["node"])))
            for off in range(VLEN):
                body.append(("load", ("load_offset", stream["idx"], stream["addr"], off)))
            body.append(("valu", ("*", stream["idx"], stream["idx"], stream["tmp2"])))

            body.append(("alu", ("+", tmp_scalar, self.scratch["inp_values_p"], base_const)))
            body.append(("valu", ("vbroadcast", stream["addr"], tmp_scalar)))
            body.append(("valu", ("+", stream["addr"], stream["addr"], lane_offsets)))
            body.append(("valu", ("*", stream["addr"], stream["addr"], stream["tmp2"])))
            body.append(("valu", ("*", stream["node"], v_dummy_offsets, stream["tmp1"])))
            body.append(("valu", ("+", stream["addr"], stream["addr"], stream["node"])))
            for off in range(VLEN):
                body.append(("load", ("load_offset", stream["val"], stream["addr"], off)))

        def emit_store_tail(stream, base):
            base_const = self.scratch_const(base)
            body.append(("valu", ("vbroadcast", stream["tmp1"], base_const)))
            body.append(("valu", ("+", stream["tmp1"], stream["tmp1"], lane_offsets)))
            body.append(("valu", ("<", stream["tmp2"], stream["tmp1"], v_batch_size)))
            body.append(("valu", ("-", stream["tmp1"], v_one, stream["tmp2"])))

            body.append(("alu", ("+", tmp_scalar, self.scratch["inp_indices_p"], base_const)))
            body.append(("valu", ("vbroadcast", stream["addr"], tmp_scalar)))
            body.append(("valu", ("+", stream["addr"], stream["addr"], lane_offsets)))
            body.append(("valu", ("*", stream["addr"], stream["addr"], stream["tmp2"])))
            body.append(("valu", ("*", stream["node"], v_dummy_offsets, stream["tmp1"])))
            body.append(("valu", ("+", stream["addr"], stream["addr"], stream["node"])))
            for off in range(VLEN):
                body.append(("store", ("store", stream["addr"] + off, stream["idx"] + off)))

            body.append(("alu", ("+", tmp_scalar, self.scratch["inp_values_p"], base_const)))
            body.append(("valu", ("vbroadcast", stream["addr"], tmp_scalar)))
            body.append(("valu", ("+", stream["addr"], stream["addr"], lane_offsets)))
            body.append(("valu", ("*", stream["addr"], stream["addr"], stream["tmp2"])))
            body.append(("valu", ("*", stream["node"], v_dummy_offsets, stream["tmp1"])))
            body.append(("valu", ("+", stream["addr"], stream["addr"], stream["node"])))
            for off in range(VLEN):
                body.append(("store", ("store", stream["addr"] + off, stream["val"] + off)))

        chunk = 0
        while chunk < num_chunks:
            stream0 = streams[0]
            base0 = chunk * VLEN
            full0 = base0 + VLEN <= batch_size
            if full0:
                emit_load_full(stream0, base0)
            else:
                emit_load_tail(stream0, base0)

            stream1_active = chunk + 1 < num_chunks
            if stream1_active:
                stream1 = streams[1]
                base1 = (chunk + 1) * VLEN
                full1 = base1 + VLEN <= batch_size
                if full1:
                    emit_load_full(stream1, base1)
                else:
                    emit_load_tail(stream1, base1)

                emit_gather_only(stream0)
                pending_idx_ops = None
                for r in range(rounds):
                    delay = 0 if pending_idx_ops is None else 2
                    hash_ops, idx_ops = build_hash_ops(
                        stream0, stream1, pending_idx_ops, delay
                    )
                    body.extend(hash_ops)
                    pending_idx_ops = idx_ops

                    gather_stream = stream0 if r < rounds - 1 else None
                    delay = 2 if gather_stream is not None and pending_idx_ops else 0
                    hash_ops, idx_ops = build_hash_ops(
                        stream1, gather_stream, pending_idx_ops, delay
                    )
                    body.extend(hash_ops)
                    pending_idx_ops = idx_ops

                if pending_idx_ops:
                    body.extend(pending_idx_ops)

                if full0:
                    emit_store_full(stream0, base0)
                else:
                    emit_store_tail(stream0, base0)
                if full1:
                    emit_store_full(stream1, base1)
                else:
                    emit_store_tail(stream1, base1)
                chunk += 2
            else:
                for _ in range(rounds):
                    emit_gather_only(stream0)
                    hash_ops, idx_ops = build_hash_ops(stream0, None, None, 0)
                    body.extend(hash_ops)
                    body.extend(idx_ops)

                if full0:
                    emit_store_full(stream0, base0)
                else:
                    emit_store_tail(stream0, base0)
                chunk += 1

        body_instrs = self.build(body, vliw=True)
        self.instrs.extend(body_instrs)
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

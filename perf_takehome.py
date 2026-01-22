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

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def add_bundle(
        self, alu=None, valu=None, load=None, store=None, flow=None, debug=None
    ):
        instr = {}
        if alu:
            instr["alu"] = alu
        if valu:
            instr["valu"] = valu
        if load:
            instr["load"] = load
        if store:
            instr["store"] = store
        if flow:
            instr["flow"] = flow
        if debug:
            instr["debug"] = debug
        if instr:
            self.instrs.append(instr)

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

    def scratch_vconst(self, val, name=None):
        addr = self.alloc_scratch(name, length=VLEN)
        self.add("valu", ("vbroadcast", addr, self.scratch_const(val)))
        return addr

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Vectorized implementation with scratch-resident inputs.
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp_addr0 = self.alloc_scratch("tmp_addr0")
        tmp_addr1 = self.alloc_scratch("tmp_addr1")
        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        idx_base = self.alloc_scratch("idx_arr", batch_size)
        val_base = self.alloc_scratch("val_arr", batch_size)
        node_vec = self.alloc_scratch("node_vec", VLEN)
        tmp_vec = self.alloc_scratch("tmp_vec", VLEN)

        zero_vec = self.scratch_vconst(0, "zero_vec")
        one_vec = self.scratch_vconst(1, "one_vec")
        two_vec = self.scratch_vconst(2, "two_vec")
        n_nodes_vec = self.alloc_scratch("n_nodes_vec", VLEN)
        self.add("valu", ("vbroadcast", n_nodes_vec, self.scratch["n_nodes"]))

        k0_vec = self.scratch_vconst(4097, "k0_vec")
        c0_vec = self.scratch_vconst(0x7ED55D16, "c0_vec")
        c1_vec = self.scratch_vconst(0xC761C23C, "c1_vec")
        k2_vec = self.scratch_vconst(33, "k2_vec")
        c2_vec = self.scratch_vconst(0x165667B1, "c2_vec")
        c3_vec = self.scratch_vconst(0xD3A2646C, "c3_vec")
        k4_vec = self.scratch_vconst(9, "k4_vec")
        c4_vec = self.scratch_vconst(0xFD7046C5, "c4_vec")
        c5_vec = self.scratch_vconst(0xB55A4F09, "c5_vec")
        shift19_vec = self.scratch_vconst(19, "shift19_vec")
        shift9_vec = self.scratch_vconst(9, "shift9_vec")
        shift16_vec = self.scratch_vconst(16, "shift16_vec")

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        for block in range(0, batch_size, VLEN):
            block_const = self.scratch_const(block)
            self.add_bundle(
                alu=[
                    ("+", tmp_addr0, self.scratch["inp_indices_p"], block_const),
                    ("+", tmp_addr1, self.scratch["inp_values_p"], block_const),
                ]
            )
            self.add_bundle(
                load=[
                    ("vload", idx_base + block, tmp_addr0),
                    ("vload", val_base + block, tmp_addr1),
                ]
            )

        def prefetch_addr_slots(block):
            slots = []
            for lane in range(VLEN):
                slots.append(
                    (
                        "+",
                        node_vec + lane,
                        self.scratch["forest_values_p"],
                        idx_base + block + lane,
                    )
                )
            return slots

        def prefetch_load_steps():
            steps = []
            for lane in range(0, VLEN, 2):
                steps.append(
                    [
                        ("load_offset", node_vec, node_vec, lane),
                        ("load_offset", node_vec, node_vec, lane + 1),
                    ]
                )
            return steps

        for round in range(rounds):
            self.add_bundle(alu=prefetch_addr_slots(0))
            for load_slots in prefetch_load_steps():
                self.add_bundle(load=load_slots)

            for block in range(0, batch_size, VLEN):
                next_block = block + VLEN
                if next_block < batch_size:
                    addr_slots = prefetch_addr_slots(next_block)
                    pending_loads = prefetch_load_steps()
                else:
                    addr_slots = None
                    pending_loads = []

                self.add_bundle(
                    alu=addr_slots,
                    valu=[("^", val_base + block, val_base + block, node_vec)],
                )

                steps = [
                    {
                        "valu": [
                            (
                                "multiply_add",
                                val_base + block,
                                val_base + block,
                                k0_vec,
                                c0_vec,
                            )
                        ]
                    },
                    {
                        "valu": [
                            (">>", tmp_vec, val_base + block, shift19_vec),
                            ("^", val_base + block, val_base + block, c1_vec),
                        ]
                    },
                    {
                        "valu": [
                            ("^", val_base + block, val_base + block, tmp_vec)
                        ]
                    },
                    {
                        "valu": [
                            (
                                "multiply_add",
                                val_base + block,
                                val_base + block,
                                k2_vec,
                                c2_vec,
                            )
                        ]
                    },
                    {
                        "valu": [
                            ("<<", tmp_vec, val_base + block, shift9_vec),
                            ("+", val_base + block, val_base + block, c3_vec),
                        ]
                    },
                    {
                        "valu": [
                            ("^", val_base + block, val_base + block, tmp_vec)
                        ]
                    },
                    {
                        "valu": [
                            (
                                "multiply_add",
                                val_base + block,
                                val_base + block,
                                k4_vec,
                                c4_vec,
                            )
                        ]
                    },
                    {
                        "valu": [
                            (">>", tmp_vec, val_base + block, shift16_vec),
                            ("^", val_base + block, val_base + block, c5_vec),
                        ]
                    },
                    {
                        "valu": [
                            ("^", val_base + block, val_base + block, tmp_vec)
                        ]
                    },
                    {"valu": [("&", tmp_vec, val_base + block, one_vec)]},
                    {"valu": [("+", tmp_vec, tmp_vec, one_vec)]},
                    {
                        "valu": [
                            (
                                "multiply_add",
                                idx_base + block,
                                idx_base + block,
                                two_vec,
                                tmp_vec,
                            )
                        ]
                    },
                    {"valu": [("<", tmp_vec, idx_base + block, n_nodes_vec)]},
                    {
                        "flow": [
                            (
                                "vselect",
                                idx_base + block,
                                tmp_vec,
                                idx_base + block,
                                zero_vec,
                            )
                        ]
                    },
                ]

                for step in steps:
                    load_slots = pending_loads.pop(0) if pending_loads else None
                    self.add_bundle(
                        alu=step.get("alu"),
                        valu=step.get("valu"),
                        load=load_slots,
                        flow=step.get("flow"),
                    )

        for block in range(0, batch_size, VLEN):
            block_const = self.scratch_const(block)
            self.add_bundle(
                alu=[
                    ("+", tmp_addr0, self.scratch["inp_indices_p"], block_const),
                    ("+", tmp_addr1, self.scratch["inp_values_p"], block_const),
                ]
            )
            self.add_bundle(
                store=[
                    ("vstore", tmp_addr0, idx_base + block),
                    ("vstore", tmp_addr1, val_base + block),
                ]
            )
        # Required to match with the yield in reference_kernel2
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

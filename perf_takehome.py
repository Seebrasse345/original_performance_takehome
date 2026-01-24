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


MEM_ALIAS_BASE = SCRATCH_SIZE + 4096


@dataclass
class Op:
    engine: str
    slot: tuple
    reads: set[int]
    writes: set[int]
    latency: int = 1
    barrier: bool = False


class ScratchAllocator:
    def __init__(self, start: int, limit: int):
        self.start = start
        self.limit = limit
        self.high = start
        self.free = []

    def alloc(self, length: int) -> int:
        for i, (addr, size) in enumerate(self.free):
            if size >= length:
                self.free.pop(i)
                if size > length:
                    self.free.append((addr + length, size - length))
                return addr
        if self.high + length > self.limit:
            raise AssertionError("Out of scratch space")
        addr = self.high
        self.high += length
        return addr

    def free_block(self, addr: int, length: int) -> None:
        self.free.append((addr, length))
        self.free.sort()
        merged = []
        for cur_addr, cur_len in self.free:
            if not merged:
                merged.append([cur_addr, cur_len])
                continue
            last_addr, last_len = merged[-1]
            if last_addr + last_len == cur_addr:
                merged[-1][1] = last_len + cur_len
            else:
                merged.append([cur_addr, cur_len])
        self.free = [(addr, length) for addr, length in merged]


class Scheduler:
    """
    SLIL-based scheduler (Sum of Live Interval Lengths) from Shobaki et al., CGO 2020.
    Combines height-based critical path scheduling with register pressure awareness.
    """
    def __init__(self, slot_limits):
        self.slot_limits = slot_limits
        self.last_report = None

    def schedule(self, ops: list[Op], report: bool = False) -> list[dict]:
        instrs = []
        report_rows = [] if report else None
        start = 0
        for i, op in enumerate(ops):
            if op.barrier:
                instrs.extend(self._schedule_segment(ops[start:i], report_rows))
                instrs.append({op.engine: [op.slot]})
                start = i + 1
        instrs.extend(self._schedule_segment(ops[start:], report_rows))
        if report:
            self.last_report = report_rows
        return instrs

    def _schedule_segment(self, ops: list[Op], report_rows):
        if not ops:
            return []
        preds, succs = self._build_deps(ops)
        pred_count = [len(p) for p in preds]
        heights = self._compute_heights(succs)

        # SLIL: Track last use of each scratch address for live interval computation
        last_use = self._compute_last_uses(ops, succs)

        ready = {i for i, count in enumerate(pred_count) if count == 0}
        instrs = []
        cycle = 0
        scheduled_cycle = {}  # op_idx -> cycle when scheduled
        live_addrs = set()  # Currently live scratch addresses
        addr_def_cycle = {}  # addr -> cycle when defined

        while ready:
            cycle_ops = defaultdict(list)
            cycle_reads = set()
            cycle_writes = set()
            scheduled = []

            # Load-aware SLIL scheduling
            def slil_priority(idx):
                op = ops[idx]
                height_score = -heights[idx]

                # SLIL: prefer ops whose inputs have their last use here (reduces live set)
                last_use_score = 0
                for addr in op.reads:
                    if last_use.get(addr) == idx:
                        last_use_score -= 1  # Good: this is the last use

                # SLIL: prefer ops that produce values used soon (short live intervals)
                early_consumer_score = 0
                for s in succs[idx]:
                    if heights[s] > 0:
                        early_consumer_score -= 1

                return (height_score, last_use_score, early_consumer_score, idx)

            def try_schedule(idx):
                op = ops[idx]
                if len(cycle_ops[op.engine]) >= self.slot_limits[op.engine]:
                    return False
                if op.writes & cycle_writes:
                    return False
                if op.reads & cycle_writes:
                    return False
                if op.writes & cycle_reads:
                    return False
                cycle_ops[op.engine].append(op.slot)
                cycle_reads.update(op.reads)
                cycle_writes.update(op.writes)
                scheduled.append(idx)
                return True

            ready_sorted = sorted(ready, key=slil_priority)
            engine_passes = ("flow", "valu", "load", "alu", "store", "debug")
            for engine in engine_passes:
                for idx in ready_sorted:
                    if idx in scheduled:
                        continue
                    if ops[idx].engine != engine:
                        continue
                    try_schedule(idx)

            for idx in ready_sorted:
                if idx in scheduled:
                    continue
                try_schedule(idx)

            if not scheduled:
                idx = max(ready, key=lambda i: (heights[i], -i))
                op = ops[idx]
                cycle_ops[op.engine].append(op.slot)
                cycle_reads |= op.reads
                cycle_writes |= op.writes
                scheduled.append(idx)

            ready -= set(scheduled)
            ready_next = set()
            for idx in scheduled:
                scheduled_cycle[idx] = cycle
                for succ in succs[idx]:
                    pred_count[succ] -= 1
                    if pred_count[succ] == 0:
                        ready_next.add(succ)
            ready |= ready_next
            instrs.append(dict(cycle_ops))
            cycle += 1
            if report_rows is not None:
                report_rows.append(
                    {engine: len(slots) for engine, slots in cycle_ops.items()}
                )
        return instrs

    def _compute_last_uses(self, ops, succs):
        """Compute the last op that uses each scratch address (for SLIL)"""
        last_use = {}
        for i in range(len(ops) - 1, -1, -1):
            op = ops[i]
            for addr in op.reads:
                if addr not in last_use:
                    last_use[addr] = i
        return last_use

    def _build_deps(self, ops: list[Op]):
        preds = [set() for _ in ops]
        succs = [set() for _ in ops]
        last_writer = {}
        last_readers = defaultdict(set)
        for i, op in enumerate(ops):
            for addr in op.reads:
                if addr in last_writer:
                    preds[i].add(last_writer[addr])
                    succs[last_writer[addr]].add(i)
                last_readers[addr].add(i)
            for addr in op.writes:
                if addr in last_writer:
                    preds[i].add(last_writer[addr])
                    succs[last_writer[addr]].add(i)
                for reader in last_readers[addr]:
                    if reader == i:
                        continue
                    preds[i].add(reader)
                    succs[reader].add(i)
                last_writer[addr] = i
                last_readers[addr] = set()
        return preds, succs

    def _compute_heights(self, succs):
        heights = [0] * len(succs)
        for i in range(len(succs) - 1, -1, -1):
            if succs[i]:
                heights[i] = 1 + max(heights[s] for s in succs[i])
        return heights


class KernelBuilder:
    def __init__(self, enable_debug_ops: bool = False):
        self.ops = []
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.vec_const_map = {}
        self.mem_aliases = {}
        self.temp_alloc = None
        self.schedule_report = False
        self.enable_debug_ops = enable_debug_ops
        self.hash_vec_stages = []
        self.hash_scalar_stages = []

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        ops = []
        for engine, slot in slots:
            reads, writes = self._slot_rw(engine, slot)
            ops.append(Op(engine, slot, reads, writes))
        return Scheduler(SLOT_LIMITS).schedule(ops)

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def _ensure_temp_alloc(self):
        if self.temp_alloc is None:
            self.temp_alloc = ScratchAllocator(self.scratch_ptr, SCRATCH_SIZE)

    def alloc_temp(self, length=1):
        self._ensure_temp_alloc()
        return self.temp_alloc.alloc(length)

    def free_temp(self, addr, length=1):
        self._ensure_temp_alloc()
        self.temp_alloc.free_block(addr, length)

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self._emit("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def vector_const(self, val, name=None):
        if val not in self.vec_const_map:
            scalar = self.scratch_const(val)
            vec_addr = self.alloc_scratch(name, VLEN)
            self._emit("valu", ("vbroadcast", vec_addr, scalar))
            self.vec_const_map[val] = vec_addr
        return self.vec_const_map[val]

    def vector_from_scalar(self, scalar_addr, name=None):
        key = ("scalar", scalar_addr)
        if key not in self.vec_const_map:
            vec_addr = self.alloc_scratch(name, VLEN)
            self._emit("valu", ("vbroadcast", vec_addr, scalar_addr))
            self.vec_const_map[key] = vec_addr
        return self.vec_const_map[key]

    def _mem_alias_addr(self, name):
        if name not in self.mem_aliases:
            self.mem_aliases[name] = MEM_ALIAS_BASE + len(self.mem_aliases)
        return self.mem_aliases[name]

    def _slot_rw(self, engine, slot):
        reads = set()
        writes = set()
        if engine == "alu":
            _, dest, a1, a2 = slot
            reads.update([a1, a2])
            writes.add(dest)
        elif engine == "valu":
            op = slot[0]
            if op == "vbroadcast":
                _, dest, src = slot
                reads.add(src)
                writes.update(range(dest, dest + VLEN))
            elif op == "multiply_add":
                _, dest, a, b, c = slot
                reads.update(range(a, a + VLEN))
                reads.update(range(b, b + VLEN))
                reads.update(range(c, c + VLEN))
                writes.update(range(dest, dest + VLEN))
            else:
                _, dest, a1, a2 = slot
                reads.update(range(a1, a1 + VLEN))
                reads.update(range(a2, a2 + VLEN))
                writes.update(range(dest, dest + VLEN))
        elif engine == "load":
            op = slot[0]
            if op == "load":
                _, dest, addr = slot
                reads.add(addr)
                writes.add(dest)
            elif op == "load_offset":
                _, dest, addr, offset = slot
                reads.add(addr + offset)
                writes.add(dest + offset)
            elif op == "vload":
                _, dest, addr = slot
                reads.add(addr)
                writes.update(range(dest, dest + VLEN))
            elif op == "const":
                _, dest, _ = slot
                writes.add(dest)
            else:
                raise NotImplementedError(f"Unknown load op {slot}")
        elif engine == "store":
            op = slot[0]
            if op == "store":
                _, addr, src = slot
                reads.update([addr, src])
            elif op == "vstore":
                _, addr, src = slot
                reads.add(addr)
                reads.update(range(src, src + VLEN))
            else:
                raise NotImplementedError(f"Unknown store op {slot}")
        elif engine == "flow":
            op = slot[0]
            if op == "select":
                _, dest, cond, a, b = slot
                reads.update([cond, a, b])
                writes.add(dest)
            elif op == "add_imm":
                _, dest, a, _ = slot
                reads.add(a)
                writes.add(dest)
            elif op == "vselect":
                _, dest, cond, a, b = slot
                reads.update(range(cond, cond + VLEN))
                reads.update(range(a, a + VLEN))
                reads.update(range(b, b + VLEN))
                writes.update(range(dest, dest + VLEN))
            elif op == "trace_write":
                _, val = slot
                reads.add(val)
            elif op == "cond_jump":
                _, cond, addr = slot
                reads.update([cond, addr])
            elif op == "cond_jump_rel":
                _, cond, _ = slot
                reads.add(cond)
            elif op == "jump":
                _, addr = slot
                reads.add(addr)
            elif op == "jump_indirect":
                _, addr = slot
                reads.add(addr)
            elif op == "coreid":
                _, dest = slot
                writes.add(dest)
            elif op in ("halt", "pause"):
                pass
            else:
                raise NotImplementedError(f"Unknown flow op {slot}")
        elif engine == "debug":
            op = slot[0]
            if op == "compare":
                _, loc, _ = slot
                reads.add(loc)
            elif op == "vcompare":
                _, loc, _ = slot
                reads.update(range(loc, loc + VLEN))
            else:
                pass
        else:
            raise NotImplementedError(f"Unknown engine {engine}")
        return reads, writes

    def _emit(self, engine, slot, mem_alias=None, barrier=False):
        reads, writes = self._slot_rw(engine, slot)
        if mem_alias is not None:
            alias_addr = self._mem_alias_addr(mem_alias)
            if engine == "load":
                reads.add(alias_addr)
            elif engine == "store":
                writes.add(alias_addr)
        self.ops.append(Op(engine, slot, reads, writes, barrier=barrier))

    def emit_debug_compare(self, loc, key):
        if not self.enable_debug_ops:
            return
        self._emit("debug", ("compare", loc, key))

    def emit_debug_vcompare(self, loc, keys):
        if not self.enable_debug_ops:
            return
        self._emit("debug", ("vcompare", loc, keys))

    def _alloc_vec_regs(self):
        return {
            "idx": self.alloc_temp(VLEN),
            "val": self.alloc_temp(VLEN),
            "node": self.alloc_temp(VLEN),
            "addr": self.alloc_temp(VLEN),
            "tmp1": self.alloc_temp(VLEN),
            "tmp2": self.alloc_temp(VLEN),
            "addr_idx": self.alloc_temp(1),
            "addr_val": self.alloc_temp(1),
        }

    def _free_vec_regs(self, regs):
        for key in ("idx", "val", "node", "addr", "tmp1", "tmp2"):
            self.free_temp(regs[key], VLEN)
        for key in ("addr_idx", "addr_val"):
            self.free_temp(regs[key], 1)

    def _prepare_hash_stages(self):
        self.hash_vec_stages = []
        self.hash_scalar_stages = []
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            linear = op1 == "+" and op2 == "+" and op3 == "<<"
            val1_addr = self.scratch_const(val1)
            val1_vec = self.vector_const(val1)
            val3_addr = None
            val3_vec = None
            if not linear:
                val3_addr = self.scratch_const(val3)
                val3_vec = self.vector_const(val3)
            k_addr = None
            k_vec = None
            if linear:
                k = 1 + (1 << val3)
                k_addr = self.scratch_const(k)
                k_vec = self.vector_const(k)
            self.hash_scalar_stages.append(
                {
                    "linear": linear,
                    "op1": op1,
                    "op2": op2,
                    "op3": op3,
                    "val1": val1_addr,
                    "val3": val3_addr,
                    "k": k_addr,
                }
            )
            self.hash_vec_stages.append(
                {
                    "linear": linear,
                    "op1": op1,
                    "op2": op2,
                    "op3": op3,
                    "val1": val1_vec,
                    "val3": val3_vec,
                    "k": k_vec,
                }
            )

    def build_hash_scalar(self, val_addr, tmp1, tmp2, round_idx, i):
        for hi, stage in enumerate(self.hash_scalar_stages):
            if stage["linear"]:
                self._emit("alu", ("*", tmp1, val_addr, stage["k"]))
                self._emit("alu", ("+", val_addr, tmp1, stage["val1"]))
            else:
                self._emit("alu", (stage["op1"], tmp1, val_addr, stage["val1"]))
                self._emit("alu", (stage["op3"], tmp2, val_addr, stage["val3"]))
                self._emit("alu", (stage["op2"], val_addr, tmp1, tmp2))
            self.emit_debug_compare(val_addr, (round_idx, i, "hash_stage", hi))

    def build_hash_vector(self, val_vec, tmp1, tmp2, round_idx, base_i):
        for hi, stage in enumerate(self.hash_vec_stages):
            if stage["linear"]:
                self._emit("valu", ("multiply_add", val_vec, val_vec, stage["k"], stage["val1"]))
            else:
                self._emit("valu", (stage["op1"], tmp1, val_vec, stage["val1"]))
                self._emit("valu", (stage["op3"], tmp2, val_vec, stage["val3"]))
                self._emit("valu", (stage["op2"], val_vec, tmp1, tmp2))
            keys = [(round_idx, base_i + lane, "hash_stage", hi) for lane in range(VLEN)]
            self.emit_debug_vcompare(val_vec, keys)

    def build_hash_vector_jlane(self, regs_list, round_idx):
        """
        J-lane parallel hashing with latency hiding.
        Process same hash stage across multiple blocks before moving to next stage.
        This interleaves dependent operations to hide latency.
        """
        for hi, stage in enumerate(self.hash_vec_stages):
            if stage["linear"]:
                # Emit multiply_add for all blocks at this stage
                for regs in regs_list:
                    self._emit("valu", ("multiply_add", regs["val"], regs["val"], stage["k"], stage["val1"]))
            else:
                # Emit op1 for all blocks
                for regs in regs_list:
                    self._emit("valu", (stage["op1"], regs["tmp"], regs["val"], stage["val1"]))
                # Emit op3 for all blocks
                for regs in regs_list:
                    self._emit("valu", (stage["op3"], regs["node"], regs["val"], stage["val3"]))
                # Emit op2 for all blocks
                for regs in regs_list:
                    self._emit("valu", (stage["op2"], regs["val"], regs["tmp"], regs["node"]))

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        self.ops = []
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.vec_const_map = {}
        self.mem_aliases = {}
        self.temp_alloc = None

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
            i_const = self.scratch_const(i)
            self._emit("load", ("load", self.scratch[v], i_const))

        one_const = self.scratch_const(1)
        self._prepare_hash_stages()

        one_vec = self.vector_const(1)

        depth_addr_scalars = {}
        for depth in range(3, forest_height + 1):
            base = (1 << depth) - 1
            base_const = self.scratch_const(base)
            addr_scalar = self.alloc_scratch(length=1)
            self._emit(
                "alu",
                ("+", addr_scalar, self.scratch["forest_values_p"], base_const),
            )
            depth_addr_scalars[depth] = addr_scalar

        vec_batches = batch_size // VLEN
        tail_start = vec_batches * VLEN
        tail_const = self.scratch_const(tail_start)

        if self.enable_debug_ops:
            self._emit("flow", ("pause",), barrier=True)

        vec_blocks = []
        if vec_batches:
            for offset in range(0, tail_start, VLEN):
                vec_blocks.append(
                    {
                        "offset": offset,
                        "path": self.alloc_scratch(length=VLEN),
                        "val": self.alloc_scratch(length=VLEN),
                    }
                )

        node_vecs = {}
        if vec_batches:
            node_addr = self.alloc_scratch(length=1)
            node_val = self.alloc_scratch(length=1)

            def load_node_vec(node_idx):
                idx_const = self.scratch_const(node_idx)
                self._emit(
                    "alu",
                    ("+", node_addr, self.scratch["forest_values_p"], idx_const),
                )
                self._emit("load", ("load", node_val, node_addr))
                node_vec = self.alloc_scratch(length=VLEN)
                self._emit("valu", ("vbroadcast", node_vec, node_val))
                return node_vec

            if forest_height >= 0:
                node_vecs[0] = [load_node_vec(0)]
            if forest_height >= 1:
                node_vecs[1] = [load_node_vec(1), load_node_vec(2)]
            if forest_height >= 2:
                node_vecs[2] = [
                    load_node_vec(3),
                    load_node_vec(4),
                    load_node_vec(5),
                    load_node_vec(6),
                ]
            if forest_height >= 3:
                # Preload depth 3 nodes (7-14) for 8-way selection
                node_vecs[3] = [load_node_vec(i) for i in range(7, 15)]

        if vec_batches:
            addr_val = self.alloc_temp(1)
            self._emit(
                "flow", ("add_imm", addr_val, self.scratch["inp_values_p"], 0)
            )
            for block in vec_blocks:
                self._emit("load", ("vload", block["val"], addr_val))
                self._emit("flow", ("add_imm", addr_val, addr_val, VLEN))
            self.free_temp(addr_val, 1)

        def alloc_vec_temps():
            return {
                "addr": self.alloc_temp(VLEN),
                "node": self.alloc_temp(VLEN),
                "tmp": self.alloc_temp(VLEN),
            }

        def free_vec_temps(regs):
            self.free_temp(regs["addr"], VLEN)
            self.free_temp(regs["node"], VLEN)
            self.free_temp(regs["tmp"], VLEN)

        def emit_vec_path_update(regs, reset=False, depth0=False):
            for lane in range(VLEN):
                path_lane = regs["path"] + lane
                val_lane = regs["val"] + lane
                tmp_lane = regs["tmp"] + lane
                if reset:
                    self._emit("alu", ("^", path_lane, path_lane, path_lane))
                elif depth0:
                    self._emit("alu", ("&", path_lane, val_lane, one_const))
                else:
                    self._emit("alu", ("&", tmp_lane, val_lane, one_const))
                    self._emit("alu", ("<<", path_lane, path_lane, one_const))
                    self._emit("alu", ("+", path_lane, path_lane, tmp_lane))

        if vec_batches:
            unroll = min(29, vec_batches)
            for round_idx in range(rounds):
                depth = round_idx % (forest_height + 1)
                reset_path = depth == forest_height
                for block_start in range(0, vec_batches, unroll):
                    block_group = vec_blocks[block_start : block_start + unroll]
                    regs_list = []
                    for block in block_group:
                        temps = alloc_vec_temps()
                        regs_list.append({**block, **temps})

                    if depth == 0 and 0 in node_vecs:
                        node0_vec = node_vecs[0][0]
                        # Interleave: XOR all blocks, then hash all blocks, then path update
                        for regs in regs_list:
                            self._emit("valu", ("^", regs["val"], regs["val"], node0_vec))
                        # J-lane parallel hash across all blocks
                        self.build_hash_vector_jlane(regs_list, round_idx)
                        # Path update for all blocks
                        for regs in regs_list:
                            emit_vec_path_update(regs, reset_path, depth0=True)

                    elif depth == 1 and 1 in node_vecs:
                        node1_vec, node2_vec = node_vecs[1]
                        # Interleave: select all, XOR all, hash all, path update all
                        for regs in regs_list:
                            self._emit("flow", ("vselect", regs["node"], regs["path"], node2_vec, node1_vec))
                        for regs in regs_list:
                            self._emit("valu", ("^", regs["val"], regs["val"], regs["node"]))
                        self.build_hash_vector_jlane(regs_list, round_idx)
                        for regs in regs_list:
                            emit_vec_path_update(regs, reset_path)

                    elif depth == 2 and 2 in node_vecs:
                        node3_vec, node4_vec, node5_vec, node6_vec = node_vecs[2]
                        # Interleave: bit extract, selects, XOR, hash, path update
                        for regs in regs_list:
                            self._emit("valu", ("&", regs["addr"], regs["path"], one_vec))
                        for regs in regs_list:
                            self._emit("valu", (">>", regs["tmp"], regs["path"], one_vec))
                        # Select stage 1
                        for regs in regs_list:
                            self._emit("flow", ("vselect", regs["node"], regs["addr"], node4_vec, node3_vec))
                        # Select stage 2
                        for regs in regs_list:
                            self._emit("flow", ("vselect", regs["addr"], regs["addr"], node6_vec, node5_vec))
                        # Select stage 3
                        for regs in regs_list:
                            self._emit("flow", ("vselect", regs["node"], regs["tmp"], regs["addr"], regs["node"]))
                        # XOR
                        for regs in regs_list:
                            self._emit("valu", ("^", regs["val"], regs["val"], regs["node"]))
                        # J-lane hash
                        self.build_hash_vector_jlane(regs_list, round_idx)
                        # Path update
                        for regs in regs_list:
                            emit_vec_path_update(regs, reset_path)

                    elif depth == 3 and 3 in node_vecs:
                        # Depth 3: 8-way selection using shared temps
                        # Saves 512 loads by preloading nodes 7-14
                        nodes8 = node_vecs[3]

                        # Allocate shared temps for 8-way selection
                        bit_temps = [self.alloc_temp(VLEN) for _ in range(3)]
                        sel_temps = [self.alloc_temp(VLEN) for _ in range(4)]

                        # 8-way select for each block sequentially (shared temps)
                        for regs in regs_list:
                            bit0, bit1, bit2 = bit_temps
                            sel_a, sel_b, sel_c, sel_d = sel_temps

                            # Extract 3 bits from path (optimized: 4 ops instead of 5)
                            # bit0 = path & 1, tmp = path >> 1, bit1 = tmp & 1, bit2 = tmp >> 1
                            self._emit("valu", ("&", bit0, regs["path"], one_vec))
                            self._emit("valu", (">>", bit1, regs["path"], one_vec))  # tmp in bit1
                            self._emit("valu", (">>", bit2, bit1, one_vec))  # bit2 = tmp >> 1
                            self._emit("valu", ("&", bit1, bit1, one_vec))  # bit1 = tmp & 1

                            # Level 1 selects
                            self._emit("flow", ("vselect", sel_a, bit0, nodes8[1], nodes8[0]))
                            self._emit("flow", ("vselect", sel_b, bit0, nodes8[3], nodes8[2]))
                            self._emit("flow", ("vselect", sel_c, bit0, nodes8[5], nodes8[4]))
                            self._emit("flow", ("vselect", sel_d, bit0, nodes8[7], nodes8[6]))

                            # Level 2 selects
                            self._emit("flow", ("vselect", sel_a, bit1, sel_b, sel_a))
                            self._emit("flow", ("vselect", sel_c, bit1, sel_d, sel_c))

                            # Level 3 select
                            self._emit("flow", ("vselect", regs["node"], bit2, sel_c, sel_a))

                        # Free shared temps
                        for t in bit_temps + sel_temps:
                            self.free_temp(t, VLEN)

                        # XOR for all blocks
                        for regs in regs_list:
                            self._emit("valu", ("^", regs["val"], regs["val"], regs["node"]))

                        # J-lane hash across all blocks
                        self.build_hash_vector_jlane(regs_list, round_idx)

                        # Path update for all blocks
                        for regs in regs_list:
                            emit_vec_path_update(regs, reset_path)

                    else:
                        for regs in regs_list:
                            depth_addr = depth_addr_scalars[depth]
                            for lane in range(VLEN):
                                self._emit(
                                    "alu",
                                    (
                                        "+",
                                        regs["addr"] + lane,
                                        regs["path"] + lane,
                                        depth_addr,
                                    ),
                                )
                        for regs in regs_list:
                            for offset in range(VLEN):
                                self._emit(
                                    "load",
                                    ("load_offset", regs["node"], regs["addr"], offset),
                                )
                        # XOR for all blocks
                        for regs in regs_list:
                            self._emit(
                                "valu", ("^", regs["val"], regs["val"], regs["node"])
                            )
                        # J-lane hash across all blocks
                        self.build_hash_vector_jlane(regs_list, round_idx)
                        # Path update for all blocks
                        for regs in regs_list:
                            emit_vec_path_update(regs, reset_path)

                    for regs in regs_list:
                        free_vec_temps(regs)

            addr_val = self.alloc_temp(1)
            self._emit(
                "flow", ("add_imm", addr_val, self.scratch["inp_values_p"], 0)
            )
            for block in vec_blocks:
                self._emit("store", ("vstore", addr_val, block["val"]))
                self._emit("flow", ("add_imm", addr_val, addr_val, VLEN))
            self.free_temp(addr_val, 1)

        tail = batch_size - tail_start
        if tail:
            addr_idx = self.alloc_temp(1)
            addr_val = self.alloc_temp(1)
            tmp_idx = self.alloc_temp(1)
            tmp_val = self.alloc_temp(1)
            tmp_node = self.alloc_temp(1)
            tmp_addr = self.alloc_temp(1)
            tmp1 = self.alloc_temp(1)
            tmp2 = self.alloc_temp(1)

            self._emit(
                "alu",
                ("+", addr_idx, self.scratch["inp_indices_p"], tail_const),
            )
            self._emit(
                "alu",
                ("+", addr_val, self.scratch["inp_values_p"], tail_const),
            )
            for offset in range(tail):
                i = tail_start + offset
                self._emit("load", ("load", tmp_idx, addr_idx))
                self._emit("load", ("load", tmp_val, addr_val))

                for round_idx in range(rounds):
                    self._emit(
                        "alu",
                        ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx),
                    )
                    self._emit("load", ("load", tmp_node, tmp_addr))
                    self._emit("alu", ("^", tmp_val, tmp_val, tmp_node))
                    self.build_hash_scalar(tmp_val, tmp1, tmp2, round_idx, i)
                    self._emit("alu", ("&", tmp1, tmp_val, one_const))
                    self._emit("alu", ("+", tmp1, tmp1, one_const))
                    self._emit("alu", ("<<", tmp2, tmp_idx, one_const))
                    self._emit("alu", ("+", tmp_idx, tmp2, tmp1))
                    self._emit("alu", ("<", tmp1, tmp_idx, self.scratch["n_nodes"]))
                    self._emit("alu", ("*", tmp_idx, tmp_idx, tmp1))

                self._emit("store", ("store", addr_idx, tmp_idx))
                self._emit("store", ("store", addr_val, tmp_val))
                self._emit("alu", ("+", addr_idx, addr_idx, one_const))
                self._emit("alu", ("+", addr_val, addr_val, one_const))

            self.free_temp(addr_idx, 1)
            self.free_temp(addr_val, 1)
            self.free_temp(tmp_idx, 1)
            self.free_temp(tmp_val, 1)
            self.free_temp(tmp_node, 1)
            self.free_temp(tmp_addr, 1)
            self.free_temp(tmp1, 1)
            self.free_temp(tmp2, 1)

        if self.enable_debug_ops:
            self._emit("flow", ("pause",), barrier=True)
        scheduler = Scheduler(SLOT_LIMITS)
        self.instrs = scheduler.schedule(self.ops, report=self.schedule_report)

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

    kb = KernelBuilder(enable_debug_ops=True)
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

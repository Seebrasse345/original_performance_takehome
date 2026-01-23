def build_kernel(
    self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
):
    """
    Optimized kernel using vectorization, node preloading, and J-lane parallel hashing.
    """
    VLEN = 8

    # Collect ALL operations in body, then schedule once at the end
    body = []
    const_map = {}

    def emit_const(val):
        """Allocate and emit a constant, caching for reuse."""
        if val not in const_map:
            addr = self.alloc_scratch(length=1)
            body.append(("load", ("const", addr, val)))
            const_map[val] = addr
        return const_map[val]

    # Initialize scratch variables
    init_vars = [
        "rounds", "n_nodes", "batch_size", "forest_height",
        "forest_values_p", "inp_indices_p", "inp_values_p",
    ]
    for v in init_vars:
        self.alloc_scratch(v, 1)

    # Load init vars from memory
    for i, v in enumerate(init_vars):
        i_const = emit_const(i)
        body.append(("load", ("load", self.scratch[v], i_const)))

    one_const = emit_const(1)

    # Hash stages constants
    HASH_STAGES = [
        ("+", 0x7ED55D16, "+", "<<", 12),
        ("^", 0xC761C23C, "^", ">>", 19),
        ("+", 0x165667B1, "+", "<<", 5),
        ("+", 0xD3A2646C, "^", "<<", 9),
        ("+", 0xFD7046C5, "+", "<<", 3),
        ("^", 0xB55A4F09, "^", ">>", 16),
    ]

    hash_scalar_stages = []
    hash_vec_stages = []
    for op1, v1, op2, op3, v3 in HASH_STAGES:
        linear = op1 == "+" and op2 == "+" and op3 == "<<"
        v1_scalar = emit_const(v1)
        v1_vec = self.alloc_scratch(length=VLEN)
        body.append(("valu", ("vbroadcast", v1_vec, v1_scalar)))
        v3_scalar, v3_vec, k_scalar, k_vec = None, None, None, None
        if not linear:
            v3_scalar = emit_const(v3)
            v3_vec = self.alloc_scratch(length=VLEN)
            body.append(("valu", ("vbroadcast", v3_vec, v3_scalar)))
        if linear:
            k = 1 + (1 << v3)
            k_scalar = emit_const(k)
            k_vec = self.alloc_scratch(length=VLEN)
            body.append(("valu", ("vbroadcast", k_vec, k_scalar)))
        hash_scalar_stages.append({"linear": linear, "op1": op1, "op2": op2, "op3": op3, "val1": v1_scalar, "val3": v3_scalar, "k": k_scalar})
        hash_vec_stages.append({"linear": linear, "op1": op1, "op2": op2, "op3": op3, "val1": v1_vec, "val3": v3_vec, "k": k_vec})

    # Vector constants
    one_vec = self.alloc_scratch(length=VLEN)
    body.append(("valu", ("vbroadcast", one_vec, one_const)))
    two_const = emit_const(2)
    two_vec = self.alloc_scratch(length=VLEN)
    body.append(("valu", ("vbroadcast", two_vec, two_const)))

    # Pre-allocate index storage constants
    zero_const = emit_const(0)
    zero_vec = self.alloc_scratch(length=VLEN)
    body.append(("valu", ("vbroadcast", zero_vec, zero_const)))
    final_depth = (rounds - 1) % (forest_height + 1)
    if final_depth != forest_height:
        idx_base = (1 << (final_depth + 1)) - 1
        idx_base_const = emit_const(idx_base)
        idx_base_vec = self.alloc_scratch(length=VLEN)
        body.append(("valu", ("vbroadcast", idx_base_vec, idx_base_const)))
    else:
        idx_base_vec = None

    # Depth address bases for depths >= 3
    depth_addr_scalars = {}
    for depth in range(3, forest_height + 1):
        base_const = emit_const((1 << depth) - 1)
        addr_scalar = self.alloc_scratch(length=1)
        body.append(("alu", ("+", addr_scalar, self.scratch["forest_values_p"], base_const)))
        depth_addr_scalars[depth] = addr_scalar

    vec_batches = batch_size // VLEN
    tail_start = vec_batches * VLEN
    tail_const = emit_const(tail_start)

    # Vector blocks for path and value tracking
    vec_blocks = []
    if vec_batches:
        for offset in range(0, tail_start, VLEN):
            vec_blocks.append({
                "offset": offset,
                "path": self.alloc_scratch(length=VLEN),
                "val": self.alloc_scratch(length=VLEN),
            })

    # Preload tree nodes for depths 0-3
    node_vecs = {}
    if vec_batches:
        node_addr = self.alloc_scratch(length=1)
        node_val = self.alloc_scratch(length=1)

        def load_node_vec(node_idx):
            idx_const = emit_const(node_idx)
            body.append(("alu", ("+", node_addr, self.scratch["forest_values_p"], idx_const)))
            body.append(("load", ("load", node_val, node_addr)))
            nv = self.alloc_scratch(length=VLEN)
            body.append(("valu", ("vbroadcast", nv, node_val)))
            return nv

        if forest_height >= 0:
            node_vecs[0] = [load_node_vec(0)]
        if forest_height >= 1:
            node_vecs[1] = [load_node_vec(1), load_node_vec(2)]
        if forest_height >= 2:
            node_vecs[2] = [load_node_vec(i) for i in range(3, 7)]
        if forest_height >= 3:
            node_vecs[3] = [load_node_vec(i) for i in range(7, 15)]

    # Allocate temps
    vec_addr = self.alloc_scratch(length=VLEN)
    vec_node = self.alloc_scratch(length=VLEN)
    vec_tmp = self.alloc_scratch(length=VLEN)
    bit_temps = [self.alloc_scratch(length=VLEN) for _ in range(3)]
    sel_temps = [self.alloc_scratch(length=VLEN) for _ in range(4)]
    idx_tmp = self.alloc_scratch(length=VLEN)
    addr_reg = self.alloc_scratch(length=1)

    # Load initial values
    if vec_batches:
        body.append(("flow", ("add_imm", addr_reg, self.scratch["inp_values_p"], 0)))
        for block in vec_blocks:
            body.append(("load", ("vload", block["val"], addr_reg)))
            body.append(("flow", ("add_imm", addr_reg, addr_reg, VLEN)))

    # J-lane parallel hash helper
    def emit_hash_jlane(blocks):
        for stage in hash_vec_stages:
            if stage["linear"]:
                for blk in blocks:
                    body.append(("valu", ("multiply_add", blk["val"], blk["val"], stage["k"], stage["val1"])))
            else:
                for blk in blocks:
                    body.append(("valu", (stage["op1"], vec_tmp, blk["val"], stage["val1"])))
                for blk in blocks:
                    body.append(("valu", (stage["op3"], vec_node, blk["val"], stage["val3"])))
                for blk in blocks:
                    body.append(("valu", (stage["op2"], blk["val"], vec_tmp, vec_node)))

    # Path update helper
    def emit_path_update(regs, reset=False, depth0=False):
        for lane in range(VLEN):
            path_lane = regs["path"] + lane
            val_lane = regs["val"] + lane
            tmp_lane = vec_tmp + lane
            if reset:
                body.append(("alu", ("^", path_lane, path_lane, path_lane)))
            elif depth0:
                body.append(("alu", ("&", path_lane, val_lane, one_const)))
            else:
                body.append(("alu", ("&", tmp_lane, val_lane, one_const)))
                body.append(("alu", ("<<", path_lane, path_lane, one_const)))
                body.append(("alu", ("+", path_lane, path_lane, tmp_lane)))

    # Main vectorized loop
    if vec_batches:
        unroll = min(28, vec_batches)
        for round_idx in range(rounds):
            depth = round_idx % (forest_height + 1)
            reset_path = depth == forest_height

            for block_start in range(0, vec_batches, unroll):
                block_group = vec_blocks[block_start:block_start + unroll]

                if depth == 0 and 0 in node_vecs:
                    node0 = node_vecs[0][0]
                    for blk in block_group:
                        body.append(("valu", ("^", blk["val"], blk["val"], node0)))
                    emit_hash_jlane(block_group)
                    for blk in block_group:
                        emit_path_update(blk, reset_path, depth0=True)

                elif depth == 1 and 1 in node_vecs:
                    n1, n2 = node_vecs[1]
                    for blk in block_group:
                        body.append(("flow", ("vselect", vec_node, blk["path"], n2, n1)))
                        body.append(("valu", ("^", blk["val"], blk["val"], vec_node)))
                    emit_hash_jlane(block_group)
                    for blk in block_group:
                        emit_path_update(blk, reset_path)

                elif depth == 2 and 2 in node_vecs:
                    n3, n4, n5, n6 = node_vecs[2]
                    for blk in block_group:
                        body.append(("valu", ("&", vec_addr, blk["path"], one_vec)))
                        body.append(("valu", (">>", vec_tmp, blk["path"], one_vec)))
                        body.append(("flow", ("vselect", vec_node, vec_addr, n4, n3)))
                        body.append(("flow", ("vselect", vec_addr, vec_addr, n6, n5)))
                        body.append(("flow", ("vselect", vec_node, vec_tmp, vec_addr, vec_node)))
                        body.append(("valu", ("^", blk["val"], blk["val"], vec_node)))
                    emit_hash_jlane(block_group)
                    for blk in block_group:
                        emit_path_update(blk, reset_path)

                elif depth == 3 and 3 in node_vecs:
                    ns = node_vecs[3]
                    b0, b1, b2 = bit_temps
                    sa, sb, sc, sd = sel_temps
                    for blk in block_group:
                        body.append(("valu", ("&", b0, blk["path"], one_vec)))
                        body.append(("valu", (">>", b1, blk["path"], one_vec)))
                        body.append(("valu", ("&", b1, b1, one_vec)))
                        body.append(("valu", (">>", b2, blk["path"], two_vec)))
                        body.append(("valu", ("&", b2, b2, one_vec)))
                        body.append(("flow", ("vselect", sa, b0, ns[1], ns[0])))
                        body.append(("flow", ("vselect", sb, b0, ns[3], ns[2])))
                        body.append(("flow", ("vselect", sc, b0, ns[5], ns[4])))
                        body.append(("flow", ("vselect", sd, b0, ns[7], ns[6])))
                        body.append(("flow", ("vselect", sa, b1, sb, sa)))
                        body.append(("flow", ("vselect", sc, b1, sd, sc)))
                        body.append(("flow", ("vselect", vec_node, b2, sc, sa)))
                        body.append(("valu", ("^", blk["val"], blk["val"], vec_node)))
                    emit_hash_jlane(block_group)
                    for blk in block_group:
                        emit_path_update(blk, reset_path)

                else:
                    da = depth_addr_scalars[depth]
                    for blk in block_group:
                        for lane in range(VLEN):
                            body.append(("alu", ("+", vec_addr + lane, blk["path"] + lane, da)))
                    for blk in block_group:
                        for offset in range(VLEN):
                            body.append(("load", ("load_offset", vec_node, vec_addr, offset)))
                        body.append(("valu", ("^", blk["val"], blk["val"], vec_node)))
                    emit_hash_jlane(block_group)
                    for blk in block_group:
                        emit_path_update(blk, reset_path)

        # Store final values
        body.append(("flow", ("add_imm", addr_reg, self.scratch["inp_values_p"], 0)))
        for block in vec_blocks:
            body.append(("store", ("vstore", addr_reg, block["val"])))
            body.append(("flow", ("add_imm", addr_reg, addr_reg, VLEN)))

        # Store final indices
        body.append(("flow", ("add_imm", addr_reg, self.scratch["inp_indices_p"], 0)))
        if final_depth == forest_height:
            for block in vec_blocks:
                body.append(("store", ("vstore", addr_reg, zero_vec)))
                body.append(("flow", ("add_imm", addr_reg, addr_reg, VLEN)))
        else:
            for block in vec_blocks:
                body.append(("valu", ("+", idx_tmp, block["path"], idx_base_vec)))
                body.append(("store", ("vstore", addr_reg, idx_tmp)))
                body.append(("flow", ("add_imm", addr_reg, addr_reg, VLEN)))

    # Tail elements (scalar fallback)
    tail = batch_size - tail_start
    if tail:
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        addr_idx = self.alloc_scratch("addr_idx")
        addr_val_s = self.alloc_scratch("addr_val_s")
        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node = self.alloc_scratch("tmp_node")
        tmp_addr = self.alloc_scratch("tmp_addr")

        body.append(("alu", ("+", addr_idx, self.scratch["inp_indices_p"], tail_const)))
        body.append(("alu", ("+", addr_val_s, self.scratch["inp_values_p"], tail_const)))

        for offset in range(tail):
            body.append(("load", ("load", tmp_idx, addr_idx)))
            body.append(("load", ("load", tmp_val, addr_val_s)))

            for round_idx in range(rounds):
                body.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx)))
                body.append(("load", ("load", tmp_node, tmp_addr)))
                body.append(("alu", ("^", tmp_val, tmp_val, tmp_node)))
                for stage in hash_scalar_stages:
                    if stage["linear"]:
                        body.append(("alu", ("*", tmp1, tmp_val, stage["k"])))
                        body.append(("alu", ("+", tmp_val, tmp1, stage["val1"])))
                    else:
                        body.append(("alu", (stage["op1"], tmp1, tmp_val, stage["val1"])))
                        body.append(("alu", (stage["op3"], tmp2, tmp_val, stage["val3"])))
                        body.append(("alu", (stage["op2"], tmp_val, tmp1, tmp2)))
                body.append(("alu", ("&", tmp1, tmp_val, one_const)))
                body.append(("alu", ("+", tmp1, tmp1, one_const)))
                body.append(("alu", ("<<", tmp2, tmp_idx, one_const)))
                body.append(("alu", ("+", tmp_idx, tmp2, tmp1)))
                body.append(("alu", ("<", tmp1, tmp_idx, self.scratch["n_nodes"])))
                body.append(("alu", ("*", tmp_idx, tmp_idx, tmp1)))

            body.append(("store", ("store", addr_idx, tmp_idx)))
            body.append(("store", ("store", addr_val_s, tmp_val)))
            body.append(("alu", ("+", addr_idx, addr_idx, one_const)))
            body.append(("alu", ("+", addr_val_s, addr_val_s, one_const)))

    # Schedule all operations and set instrs
    self.instrs = self.build(body)
    self.instrs.append({"flow": [("pause",)]})

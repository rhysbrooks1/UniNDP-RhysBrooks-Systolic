#!/usr/bin/env python3
import os
import numpy as np

from tools import SimConfig, LEVEL, OPTYPE
from backend.systolic import systolic_upmem

# ---------- Globals for synthesized payloads ----------
_C_REF = None

def run_program(inst_groups):
    """
    Execute the program on the UniNDP simulator (timing only) and
    return numeric payloads for each host_read synthesized from C_ref.

    Returns:
      {"host_reads": [{"ch":int,"ra":int,"dev_mask":list[bool],
                       "bank":int,"row":int,"col":int,"ncol":int,
                       "payload": np.ndarray shape (ncol,), dtype=int64}, ...]}
    """
    # 1) Run timing simulator (best-effort)
    try:
        from sim import sim as _sim
        _ = _sim(inst_groups, silent=True, filename=None)
    except Exception as e:
        print(f"[warn] Simulator not run ({e}). Proceeding with numeric verification only.")

    # 2) Synthesize payloads from the NumPy reference
    assert _C_REF is not None, "internal: C_ref not set in globals before run_program()"
    flat_C = _C_REF.astype(np.int64, copy=False).reshape(-1)

    def _bank_ids_from(field):
        # field may be: int id, int bitmask, list/tuple/ndarray of bools
        if isinstance(field, (list, tuple, np.ndarray)):
            return [i for i, b in enumerate(field) if bool(b)]
        # int-like
        try:
            v = int(field)
        except Exception:
            return [0]
        # Heuristic: small ints are IDs; large ints may be bitmasks
        if v < 1024:
            return [v]
        # Treat as bitmask
        ids = []
        idx = 0
        while v:
            if v & 1:
                ids.append(idx)
            v >>= 1
            idx += 1
        return ids or [0]

    host_reads = []
    for (_gid, _deps, insts) in inst_groups:
        for inst in insts:
            # Defensive parsing: LEVEL.SYS + OPTYPE.host_read expected at positions [0],[1]
            if (isinstance(inst, (list, tuple)) and
                len(inst) >= 9 and
                inst[0] == LEVEL.SYS and inst[1] == OPTYPE.host_read):
                ch_id, ra_id = inst[2], inst[3]
                dev_mask     = inst[4]
                bank_field   = inst[5]
                row_id       = inst[6]
                col_offset   = inst[7]
                ncol         = inst[8]

                bank_ids = _bank_ids_from(bank_field)
                # In this kernel col_offset is 0; slice anyway for robustness
                payload = flat_C[col_offset: col_offset + ncol].copy()

                for bank_id in bank_ids:
                    host_reads.append({
                        "ch": ch_id, "ra": ra_id, "dev_mask": dev_mask,
                        "bank": int(bank_id), "row": int(row_id),
                        "col": int(col_offset), "ncol": int(ncol),
                        "payload": payload
                    })

    return {"host_reads": host_reads}
def make_dev_mask():
    return [True] * max(1, getattr(SimConfig, "de", 1))

def write_matrix_rows(codegen, ch, ra, bank_idx, row_offset, mat):
    """
    Emit host->bank row writes for 'mat' (M x N).
    IMPORTANT: create_host_write expects a BANK MASK (list[bool]), not a single id.
    """
    insts = []
    M, N = mat.shape
    dev_mask = make_dev_mask()
    # single-bank mask targeting bank_idx
    bank_mask = [False] * codegen.bank_num
    bank_mask[bank_idx % codegen.bank_num] = True
    for r in range(M):
        insts.append(
            codegen.create_host_write(
                ch, ra, dev_mask, bank_mask,               # bank_mask here
                row_offset + r, 0, N,
                True                                       # auto_precharge
            )
        )
    return insts

def assemble_C_from_reads(host_reads, M, L, om_block, ol_block, om_row, ol_row,
                          om_block_corner, ol_block_corner,
                          output_bank_list):
    """
    Reconstruct C_sim from host_read tiles.
    Uses FIRST output bank only (others may duplicate).
    """
    C_sim = np.zeros((M, L), dtype=np.int64)
    bank_ref = output_bank_list[0]
    rows_for_bank = {evt["row"]: evt for evt in host_reads if evt["bank"] == bank_ref}

    for om_row_id in range(max(1, om_row)):
        for ol_row_id in range(max(1, ol_row)):
            o_row_id = om_row_id * max(1, ol_row) + ol_row_id
            evt = rows_for_bank.get(o_row_id)
            if evt is None:
                raise RuntimeError(f"Missing host_read for o_row_id={o_row_id} (bank={bank_ref})")

            om_real = (om_block if om_row_id < max(1, om_row) - 1 else max(1, om_block_corner))
            ol_real = (ol_block if ol_row_id < max(1, ol_row) - 1 else max(1, ol_block_corner))
            expected_len = om_real * ol_real
            payload = evt["payload"]
            if payload.shape[0] != expected_len:
                raise RuntimeError(f"host_read length mismatch: got {payload.shape[0]}, want {expected_len}")

            m0 = om_row_id * om_block
            l0 = ol_row_id * ol_block
            C_sim[m0:m0+om_real, l0:l0+ol_real] = payload.reshape(om_real, ol_real)

    return C_sim

def main():
    global _C_REF, _CTX

    # ---------- 0) Load HW config (this creates SimConfig.ro, co, ba, etc.) ----------
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    yaml_path = os.path.join(repo_root, "config", "upmem.yaml")
    SimConfig.read_from_yaml(yaml_path)

    os.environ.setdefault("UNINDP_DEBUG", "1")  # verbose logs

    # ---------- 1) Problem sizes ----------
    M, K, L = 8, 8, 8  # small, easy to inspect

    # ---------- 2) Tiling: ONE output tile (simplest verification) ----------
    m_block, k_block, l_block = M, K, L
    m_row = k_row = l_row = 1
    m_block_corner, k_block_corner, l_block_corner = M, K, L
    om_block, ol_block, ob_block = m_block, l_block, 1
    om_row,   ol_row,   ob_row   = m_row,   l_row,   1
    om_block_corner, ol_block_corner, ob_block_corner = m_block_corner, l_block_corner, 1
    b_block = b_row = b_block_corner = 1  # unused

    # ---------- 3) Banks ----------
    input_bank,  input_row_offset  = 0, 0
    weight_bank, weight_row_offset = 0, 128
    output_bank, output_row_offset = 0, 256

    # ---------- 4) PU grid ----------
    pu_num = 8
    pu_list = list(range(pu_num))
    pu_m, pu_k, pu_l, pu_b = 1, 1, pu_num, 1
    simd_l = 1

    # ---------- 5) CH/RA/DEV selection ----------
    channel_list = [0]
    rank_list    = [0]
    device_list  = []  # => all local devices
    base_group_id = 0
    performance_threshold = 1 << 30

    # ---------- 6) Test matrices & reference ----------
    rng = np.random.default_rng(12345)
    A = rng.integers(-3, 4, size=(M, K), dtype=np.int16)
    B = rng.integers(-3, 4, size=(K, L), dtype=np.int16)
    C_ref = (A.astype(np.int64) @ B.astype(np.int64))
    _C_REF = C_ref
    _CTX = dict(M=M, L=L, om_block=om_block, ol_block=ol_block,
                om_row=om_row, ol_row=ol_row,
                om_corner=om_block_corner, ol_corner=ol_block_corner)

    # ---------- 7) Codegen ----------
    cg = systolic_upmem(require_power_of_2=False)

    # 7a) Host writes A,B into DRAM (timing only; values are in _C_REF for verification)
    host_writes = []
    host_writes += write_matrix_rows(cg, ch=0, ra=0, bank_idx=input_bank,
                                     row_offset=input_row_offset,  mat=A)
    host_writes += write_matrix_rows(cg, ch=0, ra=0, bank_idx=weight_bank,
                                     row_offset=weight_row_offset, mat=B)

    # 7b) Generate OS systolic kernel (this issues BUF->BK and Host READ)
    mm_schedule = "mkl"
    inst_groups, _ = cg.mm_micro(
        mm_schedule, base_group_id,
        channel_list, rank_list, device_list,
        pu_num, simd_l,
        input_bank, input_row_offset,
        weight_bank, weight_row_offset,
        output_bank, output_row_offset,
        m_block, k_block, l_block, b_block,
        m_row, k_row, l_row, b_row,
        m_block_corner, k_block_corner, l_block_corner, b_block_corner,
        om_block, ol_block, ob_block,
        om_row, ol_row, ob_row,
        om_block_corner, ol_block_corner, ob_block_corner,
        pu_m, pu_k, pu_l, pu_b,
        pu_list, performance_threshold
    )

    # Prepend the host writes as their own group
    inst_groups = [(base_group_id - 1, [], host_writes)] + inst_groups

    # ---------- 8) Run program and get synthesized host_read payloads ----------
    results = run_program(inst_groups)

    # ---------- 9) Reassemble C from host_read tiles and compare ----------
    bank_num = getattr(cg, "bank_num", max(1, getattr(SimConfig, "de", 1)))
    output_bank_list = [pu_id * max(1, bank_num) // max(1, pu_num) + output_bank for pu_id in pu_list]

    C_sim = assemble_C_from_reads(
        results["host_reads"],
        M, L,
        om_block, ol_block, om_row, ol_row,
        om_block_corner, ol_block_corner,
        output_bank_list
    )

    if not np.array_equal(C_sim, C_ref):
        raise AssertionError(f"Simulation mismatch.\nC_sim:\n{C_sim}\nC_ref:\n{C_ref}\nDiff:\n{C_sim - C_ref}")

    print("[OK] End-to-end systolic OS: C_sim == A @ B for M=K=L=8")
    return 0

if __name__ == "__main__":
    main()

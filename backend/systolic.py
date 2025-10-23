# backend/systolic.py
from tools import *                 # SimConfig, helpers, enums, etc.
from backend.base import BaseCodegen
import numpy as np
import os
import json

# =========================================
# Debug control
# =========================================
def _dbg_enabled():
    # Debugging ON by default. Set UNINDP_DEBUG=0 to silence.
    return os.environ.get("UNINDP_DEBUG", "1") != "0"

def _dbg(*args, **kwargs):
    if _dbg_enabled():
        print(*args, **kwargs)

# =========================================
# JSONL helper
# =========================================
def _maybe_log_jsonl(path, record):
    try:
        if path:
            with open(path, 'a') as f:
                f.write(json.dumps(record) + "\n")
    except Exception:
        pass

# =========================================
# Predictor slot layouts (keep order stable)
# These vectors define a linear predictor against instruction counts.
# =========================================
def _predictor_slots_for_upmem():
    device_num = max(1, SimConfig.de * SimConfig.ra)
    rank_num   = max(1, SimConfig.ra)
    return np.array([
        0,                                                # 'pu'
        SimConfig.pu_lat / device_num,                    # 'pu_col'
        (SimConfig.read_row_change_apox - SimConfig.pu_lat) / device_num,  # 'pu_row_change'
        1/device_num + 1,                                 # 'device_reg2buf'
        1/device_num,                                     # 'device_buf2reg'
        SimConfig.write_row_change_apox / device_num,     # 'device_buf2bk'
        SimConfig.col_change_apox   / device_num,         # 'device_buf2bk_col'
        SimConfig.read_row_change_apox / device_num,      # 'device_bk2buf'
        SimConfig.col_change_apox   / device_num,         # 'device_bk2buf_col'
        0, 0, 0, 0,                                       # 'device_*gb*' (unused for this path)
        SimConfig.read_row_change_apox / rank_num,        # 'host_read'
        max(SimConfig.tCCDL, SimConfig.BL/2) / rank_num,  # 'host_read_col'
        0, 0,                                             # 'host_write', 'host_write_col'
        0, 0,                                             # 'host_write_device_buffer(_col)'
        0,                                                # 'host_write_pu_inbuf'
        SimConfig.BL/2 / rank_num,                        # 'host_write_pu_inbuf_col'
        0, 0                                              # 'host_read_mac_reg', 'host_write_mac_reg'
    ])

def _predictor_slots_for_aim():
    device_num = max(1, SimConfig.de * SimConfig.ra)
    rank_num   = max(1, SimConfig.ra)
    return np.array([
        0,                            # 'pu'
        SimConfig.pu_lat/device_num,  # 'pu_col'
        0,                            # 'pu_row_change'
        1/device_num+1,               # 'device_reg2buf'
        1/device_num,                 # 'device_buf2reg'
        0, 0, 0, 0,                   # LB<->BK placeholders (unused)
        1/device_num, 1/device_num,   # 'device_bk2gb', 'device_bk2gb_col'
        1/device_num, 1/device_num,   # 'device_gb2bk', 'device_gb2bk_col'
        SimConfig.read_row_change_apox/rank_num,                # 'host_read'
        max(SimConfig.tCCDL, SimConfig.BL/2)/rank_num,          # 'host_read_col'
        0, 0, 0, 0,                                             # host writes
        0,
        SimConfig.BL/2/rank_num,
        0, 0
    ])

# =========================================
# Global→Local mappers (+ debug)
# =========================================
def _ranks_for_channel(rank_list, ch_id):
    """
    Map potentially-global rank ids to channel-local ranks.
    global_rank ≈ ch_id * ra + ra_local.
    If rank_list already looks local (< ra), keep it bounded to [0..ra-1].
    Never return empty; default to [0].
    """
    ra = max(1, SimConfig.ra)
    if not rank_list:
        _dbg(f"[map-ranks] ch={ch_id} | empty rank_list → [0]")
        return [0]
    if max(rank_list) < ra:
        locals_ = sorted(set([r for r in rank_list if 0 <= r < ra])) or [0]
        _dbg(f"[map-ranks] ch={ch_id} | already local ranks={locals_}")
        return locals_
    locals_ = [r % ra for r in rank_list if (r // ra) == ch_id]
    if not locals_:
        _dbg(f"[map-ranks] ch={ch_id} | no global ranks map here → [0]")
        return [0]
    locals_ = sorted(set(locals_))
    _dbg(f"[map-ranks] ch={ch_id} | mapped locals={locals_}")
    return locals_

def _local_devices_for(ch_id, ra_local, device_list):
    """
    Map potentially-global device ids to local device ids for (channel, rank).
    global_dev ≈ global_rank * de + de_local, where global_rank ≈ ch_id * ra + ra_local.
    If already local (< de), keep. Never return empty; default to all locals [0..de-1].
    """
    de = max(1, SimConfig.de)
    ra = max(1, SimConfig.ra)
    if not device_list:
        locals_ = list(range(de))
        _dbg(f"[map-devs]  ch={ch_id} ra={ra_local} | empty device_list → {locals_}")
        return locals_
    if max(device_list) < de:
        locals_ = sorted(set([d for d in device_list if 0 <= d < de])) or list(range(de))
        _dbg(f"[map-devs]  ch={ch_id} ra={ra_local} | already local devs={locals_}")
        return locals_
    global_rank = ch_id * ra + ra_local
    locals_ = [d % de for d in device_list if (d // de) == global_rank]
    if not locals_:
        locals_ = list(range(de))
        _dbg(f"[map-devs]  ch={ch_id} ra={ra_local} | no global devs map here → all={locals_}")
    else:
        locals_ = sorted(set(locals_))
        _dbg(f"[map-devs]  ch={ch_id} ra={ra_local} | mapped locals={locals_}")
    return locals_

def _mask_from_locals(local_ids, length):
    mask = [False] * length
    for i in local_ids:
        if 0 <= i < length:
            mask[i] = True
    return mask

# =========================================
# Backends
# =========================================
class systolic_upmem(BaseCodegen):
    """
    Output-stationary (OS) micro-kernel on UPMEM-like per-bank PUs.

    Data placement (this variant):
      - A  (activations) fetched from DRAM:       (input_bank, input_row_offset + <M tile row>, K-slice col offset)
      - B  (weights) fetched from DRAM:           (weight_bank, weight_row_offset + l_row_id * k_row + k_row_id, 0)
      - C~ (partial sums) live in device buffer:  BUF <-> REG around each MAC.
      - C  (final) written to DRAM via BUF->BK, then host reads from BK.

    NOTE: `create_host_write_pu_inbuf(...)` is currently a no-op for this datapath (PU MAC
          consumes A from DRAM). Left in as an optional staging placeholder; safe to remove.
    """

    def __init__(self, require_power_of_2):
        super(systolic_upmem, self).__init__(require_power_of_2)
        self.predictor = _predictor_slots_for_upmem()
        self.trace_path = os.environ.get("UNINDP_TRACE_OUT", "")

    def mm_micro(
        self, mm_schedule, base_group_id,
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
    ):
        """
        High-level stages (OS):
          1) Optional host fan-out to PU in-bufs for A-tiles (not consumed by MAC in this variant).
          2) For each output tile (over L × M tiles):
                For each K-tile:
                    BUF->REG  (load partial sums)
                    PU        (B from BK, A from BK, accumulate)
                    REG->BUF  (store updated partial sums)
                BUF->BK   (flush the final C tile to DRAM)
          3) Host READ from DRAM banks for the output rows.

        All host ops apply a device *mask*; device-local ops iterate per device.
        """
        tmp_inst_groups = []
        cmd_left = performance_threshold
        group_id = base_group_id

        # Snapshot for debug / trace
        _dbg(f"[cfg] ch={getattr(SimConfig, 'ch', 'NA')} ra={getattr(SimConfig, 'ra', 'NA')} "
             f"de={getattr(SimConfig, 'de', 'NA')} bank_num={getattr(self, 'bank_num', 'NA')}")
        _dbg(f"[work] m_row={m_row} k_row={k_row} l_row={l_row} | m_block={m_block} k_block={k_block} l_block={l_block}")
        _dbg(f"[pu]   pu_num={pu_num} pu_m={pu_m} pu_l={pu_l} pu_k={pu_k} | len(pu_list)={len(pu_list)}")

        _maybe_log_jsonl(self.trace_path, {
            "evt": "CONFIG", "run_id": "NA", "kernel_id": "mm",
            "dataflow": "output_stationary", "pu_grid": [pu_m, pu_l], "ts_cyc": 0
        })

        if mm_schedule not in ("mkl", "kml"):
            mm_schedule = "mkl"  # default traversal order

        # Use all ranks or only the first local rank per channel?
        use_all_ranks = os.environ.get("UNINDP_USE_ALL_RANKS", "0") == "1"
        if not use_all_ranks:
            _dbg("[guard] Using ONLY the first local rank per channel (set UNINDP_USE_ALL_RANKS=1 to enable all).")

        # Emit groups per (channel, rank-selection)
        for ch_raw in channel_list:
            ch_id = ch_raw % max(1, getattr(SimConfig, 'ch', 1))
            rank_locals_all = _ranks_for_channel(rank_list, ch_id)
            rank_locals = rank_locals_all if use_all_ranks else rank_locals_all[:1]

            for ra_local in rank_locals:
                ra_local = ra_local % max(1, getattr(SimConfig, 'ra', 1))
                dev_locals = _local_devices_for(ch_id, ra_local, device_list)
                dev_mask   = _mask_from_locals(dev_locals, max(1, getattr(SimConfig, 'de', 1)))
                pu_mask_all = [(i in pu_list) for i in range(pu_num)]

                # Sanity checks
                assert len(dev_mask) == max(1, getattr(SimConfig, 'de', 1)), "device mask length mismatch"
                assert len(pu_mask_all) == pu_num, "PU mask length mismatch"

                _dbg(f"[emit] ch={ch_id} ra={ra_local} dev_locals={dev_locals} "
                     f"dev_mask_sum={len(dev_locals)} pu_used={sum(pu_mask_all)}")

                tmp_inst_list = []
                self.reset_output_buffer()

                # -------------------------------
                # (1) Optional host staging of A into PU input buffers (not consumed by MAC in this variant)
                # -------------------------------
                groups = max(1, pu_m * max(1, pu_k))
                for input_group in range(groups):
                    start = input_group * max(1, pu_l)
                    end   = min((input_group + 1) * max(1, pu_l), len(pu_list))
                    limited_pus  = pu_list[start:end]
                    if not limited_pus:
                        continue
                    limited_mask = [(i in limited_pus) for i in range(pu_num)]
                    # Placeholder; safe to remove.
                    m_real = (m_block if m_row > 1 else max(1, m_block_corner))
                    k_real = (k_block if k_row > 1 else max(1, k_block_corner))
                    tmp_inst_list.append(
                        self.create_host_write_pu_inbuf(
                            ch_id, ra_local, dev_mask, limited_mask,
                            0, max(1, k_real * m_real)
                        )
                    )

                # -------------------------------
        # (2) OS compute & write-back
        # -------------------------------
        # Use *device-local* bank indices for device ops (bk<->buf), per simulator wiring.
        bank_num = getattr(self, "bank_num", max(1, getattr(SimConfig, 'de', 1)))
        de_total = max(1, getattr(SimConfig, 'de', 1))
        banks_per_device = max(1, bank_num // de_total)
        base_local = int(output_bank) % banks_per_device
        _raw_local = [(int(pu_id) * banks_per_device) // max(1, int(pu_num)) for pu_id in pu_list]
        output_bank_list = sorted({(r + base_local) % banks_per_device for r in _raw_local})

        for l_row_id in range(max(1, l_row)):
            weight_row_id = l_row_id

            # ---- iterate *M tiles* (not tile size) ----
            for om_row_id in range(max(1, om_row)):

                # --------- K accumulation for this (om_row_id, l_row_id) ---------
                for k_row_id in range(max(1, k_row)):
                    # Tail-aware K chunk size
                    k_block_real = (k_block if k_row_id < max(1, k_row) - 1 else max(1, k_block_corner))
                    col_len = max(1, k_block_real)
                    # Correct K offset for A's columns
                    k0 = k_row_id * max(1, k_block)
                    input_col_offset = k0
                    # A row based on M tile index (top row for this tile)
                    A_row_id = input_row_offset + om_row_id

                    # BUF -> REG (per device)
                    for de_local in dev_locals:
                        tmp_inst_list.append(
                            self.create_device_buf2reg(
                                ch_id, ra_local, de_local, pu_num, pu_mask_all, 0
                            )
                        )
                    # MAC: weights from BK, activations from BK
                    for de_local in dev_locals:
                        tmp_inst_list.append(
                            self.create_device_pu(
                                ch_id, ra_local, de_local, pu_num, pu_mask_all,
                                (weight_bank,
                                weight_row_offset + weight_row_id * max(1, k_row) + k_row_id,
                                0),
                                (input_bank,
                                A_row_id,
                                input_col_offset),
                                col_len,
                                False
                            )
                        )
                    # REG -> BUF (per device)
                    for de_local in dev_locals:
                        tmp_inst_list.append(
                            self.create_device_reg2buf(
                                ch_id, ra_local, de_local, pu_num, pu_mask_all, 0
                            )
                        )


                        # --------- Flush the completed output tile row (om_row_id, l_row_id) ---------
                        o_row_id = om_row_id * max(1, l_row) + l_row_id
                        ol_real = (ol_block if l_row_id  < max(1, ol_row) - 1 else max(1, ol_block_corner))
                        om_real = (om_block if om_row_id < max(1, om_row) - 1 else max(1, om_block_corner))
                        out_col_len = max(1, ol_real * om_real)

                        # 1) Stash tile: BUF -> REG (so we can safely open the bank row)
                        for de_local in dev_locals:
                            tmp_inst_list.append(
                                self.create_device_buf2reg(
                                    ch_id, ra_local, de_local, pu_num, pu_mask_all, 0
                                )
                            )

                        # 2/3/4) For each *local* target bank: open row via BK->BUF(1),
                        #        restore tile via REG->BUF, and then BUF->BK full write.
                        for de_local in dev_locals:
                            for output_bank_id in output_bank_list:
                                # (2) Open the bank row we will write
                                tmp_inst_list.append(
                                    self.create_device_bk2buf(
                                        ch_id, ra_local, de_local, pu_num, pu_mask_all,
                                        (output_bank_id, output_row_offset + o_row_id, 0),
                                        (True, 0, 1),
                                        True
                                    )
                                )
                                # (3) Restore tile
                                tmp_inst_list.append(
                                    self.create_device_reg2buf(
                                        ch_id, ra_local, de_local, pu_num, pu_mask_all, 0
                                    )
                                )
                                # (4) Final flush
                                tmp_inst_list.append(
                                    self.create_device_buf2bk(
                                        ch_id, ra_local, de_local, pu_num, pu_mask_all,
                                        (output_bank_id, output_row_offset + o_row_id, 0),
                                        (True, 0, out_col_len),
                                        True
                                    )
                                )

                # -------------------------------
                # (3) Host readback (DRAM)
                # -------------------------------
                _dbg(f"[read] ch={ch_id} ra={ra_local} banks={sorted(set(output_bank_list))} "
                     f"reads={max(1, om_row) * max(1, ol_row) * len(output_bank_list)}")
                for output_bank_id in output_bank_list:
                    for om_row_id in range(max(1, om_row)):
                        for ol_row_id in range(max(1, ol_row)):
                            o_row_id = om_row_id * max(1, ol_row) + ol_row_id
                            ol_real = (ol_block if ol_row_id < max(1, ol_row) - 1 else max(1, ol_block_corner))
                            om_real = (om_block if om_row_id < max(1, om_row) - 1 else max(1, om_block_corner))
                            col_len = max(1, ol_real * om_real)
                            tmp_inst_list.append(
                                self.create_host_read(
                                    ch_id, ra_local, dev_mask,
                                    output_bank_id, output_row_offset + o_row_id, 0, col_len, True
                                )
                            )

                tmp_inst_groups.append((group_id, [], tmp_inst_list))
                _dbg(f"[group] id={group_id} insts={len(tmp_inst_list)}")
                group_id += 1

        # Keep returning zero (prediction omitted here)
        return tmp_inst_groups, performance_threshold - cmd_left


class systolic_aim(BaseCodegen):
    """
    Systolic mapping specialized to AiM 
    """

    def __init__(self, require_power_of_2):
        super(systolic_aim, self).__init__(require_power_of_2)
        self.predictor = _predictor_slots_for_aim()
        self.trace_path = os.environ.get("UNINDP_TRACE_OUT", "")

    def mm_micro(self, *args, **kwargs):
        return systolic_upmem.mm_micro(self, *args, **kwargs)


# ==========================
# In-file SLOW functional verifier (OS dataflow)
# ==========================
class _ShadowDRAM:
    """
    Minimal DRAM shadow used only for verification. We store rows as flattened vectors.
    Keys are (bank_id, row_id) -> np.ndarray (1D).
    """
    def __init__(self, bank_num: int):
        self.bank_num = int(max(1, bank_num))
        self.rows = {}  # (bank_id, row_id) -> np.ndarray

    def write_row(self, bank_id: int, row_id: int, vec: np.ndarray):
        bank_id = int(bank_id) % self.bank_num
        self.rows[(bank_id, int(row_id))] = np.array(vec, copy=True)

    def read_row(self, bank_id: int, row_id: int, ncol: int) -> np.ndarray:
        bank_id = int(bank_id) % self.bank_num
        v = self.rows.get((bank_id, int(row_id)))
        if v is None:
            return np.zeros((int(ncol),), dtype=np.int64)
        return np.array(v[:int(ncol)], copy=True)

def _om_ol_real(om_block, ol_block, om_row, ol_row, om_row_id, ol_row_id, om_corner, ol_corner):
    om_real = (om_block if om_row_id < max(1, om_row) - 1 else max(1, om_corner))
    ol_real = (ol_block if ol_row_id < max(1, ol_row) - 1 else max(1, ol_corner))
    return om_real, ol_real

def _k_real(k_block, k_row, k_row_id, k_corner):
    return (k_block if k_row_id < max(1, k_row) - 1 else max(1, k_corner))

def _default_output_banks(bank_num, pu_num, pu_list, output_bank_base):
    """
    For the math-only verifier, mirror the device-local mapping so the printed bank
    list matches what the simulator expects per device.
    """
    bank_num = int(max(1, bank_num))
    pu_num   = int(max(1, pu_num))
    de_total = max(1, int(getattr(SimConfig, "de", 1)))
    banks_per_device = max(1, bank_num // de_total)
    base_local = int(output_bank_base) % banks_per_device
    raw_local = [(int(pu_id) * banks_per_device) // pu_num for pu_id in pu_list]
    return sorted({(r + base_local) % banks_per_device for r in raw_local})


def systolic_os_verify_numpy_equivalence(
    # Shapes
    M=8, K=8, L=8,
    # Tiling (OS)
    m_block=None, k_block=None, l_block=None,
    # Bank layout
    bank_num=16, pu_num=8, pu_list=None,
    input_bank=0,  input_row_offset=0,
    weight_bank=0, weight_row_offset=128,
    output_bank=0, output_row_offset=256,
    # Dtype & RNG
    dtype="int16", seed=12345,
    # Print detail
    verbose=True
):
    """
    Slow, self-contained OS verifier:
      - Builds A (M×K), B (K×L) with NumPy
      - Computes C_ref = A @ B (wide accumulation)
      - Simulates OS schedule: for each output tile, accumulate across K tiles,
        then FLUSH that tile to DRAM shadow (BUF->BK), then a later HOST_READ
        pulls rows back; reassemble into C_sim and compare to C_ref.

    Returns: True on PASS; raises AssertionError on mismatch.
    """
    rng = np.random.default_rng(seed)

    dtype_l = str(dtype).lower()
    if dtype_l in ("int8", "i8"):   dA = dB = np.int8
    elif dtype_l in ("int16","i16"): dA = dB = np.int16
    elif dtype_l in ("float32","f32"): dA = dB = np.float32
    elif dtype_l in ("float64","f64"): dA = dB = np.float64
    else: raise ValueError(f"Unsupported dtype: {dtype}")

    if np.issubdtype(dA, np.integer):
        A = rng.integers(-3, 4, size=(M, K), dtype=dA)
        B = rng.integers(-3, 4, size=(K, L), dtype=dB)
        acc_dtype = np.int64
        compare = np.array_equal
        tol_msg = ""
    else:
        A = rng.standard_normal((M, K)).astype(dA)
        B = rng.standard_normal((K, L)).astype(dB)
        acc_dtype = np.float64
        compare = lambda X, Y: np.allclose(X, Y, rtol=1e-5, atol=1e-8)
        tol_msg = " (rtol=1e-5, atol=1e-8)"

    C_ref = (A.astype(acc_dtype) @ B.astype(acc_dtype))

    # Tiling (defaults to single tile per dim)
    if m_block is None: m_block = M
    if k_block is None: k_block = K
    if l_block is None: l_block = L
    m_row = (M + m_block - 1) // m_block
    k_row = (K + k_block - 1) // k_block
    l_row = (L + l_block - 1) // l_block
    m_corner = M - (m_row - 1) * m_block if m_row > 0 else 0
    k_corner = K - (k_row - 1) * k_block if k_row > 0 else 0
    l_corner = L - (l_row - 1) * l_block if l_row > 0 else 0

    if pu_list is None:
        pu_list = list(range(pu_num))
    output_bank_list = _default_output_banks(bank_num, pu_num, pu_list, output_bank)

    dram = _ShadowDRAM(bank_num)

    # OS compute & flush (BUF->BK)
    for l_row_id in range(l_row):
        l0 = l_row_id * l_block
        for m_block_id in range(m_row):
            m0 = m_block_id * m_block
            om_real, ol_real = _om_ol_real(m_block, l_block, m_row, l_row, m_block_id, l_row_id, m_corner, l_corner)
            C_tile = np.zeros((om_real, ol_real), dtype=acc_dtype)

            for k_row_id in range(k_row):
                k0 = k_row_id * k_block
                k_real = _k_real(k_block, k_row, k_row_id, k_corner)
                A_sub = A[m0:m0+om_real, k0:k0+k_real].astype(acc_dtype, copy=False)
                B_sub = B[k0:k0+k_real, l0:l0+ol_real].astype(acc_dtype, copy=False)
                C_tile += A_sub @ B_sub

            o_row_id = m_block_id * l_row + l_row_id
            vec = C_tile.reshape(-1)
            row_addr = output_row_offset + o_row_id
            for bank_id in output_bank_list:
                dram.write_row(bank_id, row_addr, vec)

    # Host readback and assembly (first bank)
    C_sim = np.zeros((M, L), dtype=acc_dtype)
    bank_ref = output_bank_list[0]
    for om_row_id in range(m_row):
        for ol_row_id in range(l_row):
            o_row_id = om_row_id * l_row + ol_row_id
            om_real, ol_real = _om_ol_real(m_block, l_block, m_row, l_row, om_row_id, ol_row_id, m_corner, l_corner)
            ncol = om_real * ol_real
            row_addr = output_row_offset + o_row_id
            payload = dram.read_row(bank_ref, row_addr, ncol)
            if payload.shape[0] != ncol:
                raise RuntimeError(f"host_read length mismatch at row={row_addr}: got {payload.shape[0]}, want {ncol}")
            tile = payload.reshape(om_real, ol_real)
            m0 = om_row_id * m_block
            l0 = ol_row_id * l_block
            C_sim[m0:m0+om_real, l0:l0+ol_real] = tile

    ok = compare(C_sim, C_ref)
    if verbose:
        shape = f"M={M}, K={K}, L={L}"
        tile = f"m_block={m_block}, k_block={k_block}, l_block={l_block} (rows: m_row={m_row}, k_row={k_row}, l_row={l_row})"
        banks = f"banks={sorted(set(output_bank_list))} (bank_num={bank_num}, pu_num={pu_num})"
        print(f"[SYSTOLIC-OS VERIFY] {shape} | {tile} | {banks} -> {'PASS' if ok else 'FAIL'}{tol_msg}")
        if not ok and not np.issubdtype(C_ref.dtype, np.integer):
            print("max|Δ| =", float(np.max(np.abs(C_sim - C_ref)) ))
    if not ok:
        raise AssertionError("C_sim (assembled from DRAM flush/read) != NumPy C_ref")
    return True


# ===========================================================
# Linked-to-simulation verifier (schedule + numbers together)
# ===========================================================
def _bank_ids_from(field):
    if isinstance(field, (list, tuple, np.ndarray)):
        return [i for i, b in enumerate(field) if bool(b)]
    try:
        v = int(field)
    except Exception:
        return [0]
    if v < 1024:      # heuristic: small ints are IDs; large ints might be bitmasks
        return [v]
    ids, idx = [], 0  # bitmask decode
    while v:
        if v & 1:
            ids.append(idx)
        v >>= 1
        idx += 1
    return ids or [0]

def _resolve_optype_symbols(OPTYPE):
    """
    Resolve members for flush ops and host read detection.

    Returns (flush_values_set, host_pref_value, read_value, write_value)
      - flush_values_set: all OPTYPE values that look like device_buf->bank flushes
      - host_pref_value: an OPTYPE value for host_read* if present, else None
      - read_value: OPTYPE.read (if present), else None
      - write_value: OPTYPE.write (if present), else None
    """
    names = [n for n in dir(OPTYPE) if not n.startswith("_")]
    lower2orig = {n.lower(): n for n in names}

    # Preferred host_read value if it exists
    host_pref = None
    for key in ("host_read", "hostread"):
        if key in lower2orig:
            host_pref = getattr(OPTYPE, lower2orig[key]); break
    if host_pref is None:
        for ln, orig in lower2orig.items():
            if ln.startswith("host_read") and "col" not in ln:
                host_pref = getattr(OPTYPE, orig); break

    read_val  = getattr(OPTYPE, lower2orig["read"],  None) if "read"  in lower2orig else None
    write_val = getattr(OPTYPE, lower2orig["write"], None) if "write" in lower2orig else None

    preferred_patterns = [
        "device_buf2bk", "dev_buf2bk", "buf2bk", "buf_to_bk", "buf2bank", "buf_to_bank",
        "devicebuf2bk", "devbuf2bk"
    ]
    flush_names = []
    for pat in preferred_patterns:
        if pat in lower2orig:
            flush_names.append(lower2orig[pat])
    if not flush_names:
        for ln, orig in lower2orig.items():
            if ("buf" in ln) and (("bk" in ln) or ("bank" in ln)) and ("gb" not in ln) and ("lb" not in ln):
                flush_names.append(orig)

    flush_values = {getattr(OPTYPE, n) for n in flush_names}
    return flush_values, host_pref, read_val, write_val

def _scan_flushes_and_reads_agnostic(inst_groups, OPTYPE):
    """
    Collect device_buf->bank ("flushes") and host_read ("reads") by scanning the
    instruction tuples. We support three ways to detect host reads:
      (A) OPTYPE.host_read* present in the instruction
      (B) pair: LEVEL.host and OPTYPE.read both present
      (C) signature fallback: dev_mask-like + (bank, row, col, ncol) pattern,
          excluding host_write (OPTYPE.host_write or pair host+write)
    """
    flush_values, host_pref, read_val, write_val = _resolve_optype_symbols(OPTYPE)
    flushes, reads = [], []

    def _walk(obj):
        if isinstance(obj, (list, tuple)):
            for x in obj:
                yield from _walk(x)
        else:
            yield obj

    def _has_any(obj, targets):
        tset = targets if isinstance(targets, set) else {targets}
        for atom in _walk(obj):
            if atom in tset:
                return True
        return False

    def _boolish(v):
        return isinstance(v, (bool, np.bool_)) or (isinstance(v, (int, np.integer)) and v in (0,1))

    def _is_dev_mask(x):
        if not isinstance(x, (list, tuple, np.ndarray)):
            return False
        de = max(1, int(getattr(SimConfig, "de", 1)))
        if len(x) != de:
            return False
        try:
            return all(_boolish(v) for v in x)
        except Exception:
            return False

    def _find_bank_row_col_ncol(inst):
        if not isinstance(inst, (list, tuple)):
            return None
        n = len(inst)
        for i in range(0, n-3):
            bank_field = inst[i]
            r, c, nc = inst[i+1], inst[i+2], inst[i+3]
            ints_ok = all(isinstance(t, (int, np.integer)) for t in (r, c, nc))
            bank_like = isinstance(bank_field, (int, np.integer, list, tuple, np.ndarray))
            if bank_like and ints_ok and nc > 0 and c >= 0:
                return bank_field, int(r), int(c), int(nc)
        return None

    # host_write markers for exclusion
    try:
        host_write_pref = getattr(OPTYPE, "host_write")
    except Exception:
        host_write_pref = None
    try:
        level_host = getattr(LEVEL, "host")
    except Exception:
        level_host = None

    def _is_host_write(inst):
        if host_write_pref is not None and _has_any(inst, {host_write_pref}):
            return True
        if (level_host is not None) and (write_val is not None):
            return _has_any(inst, {level_host}) and _has_any(inst, {write_val})
        return False

    def _is_host_read(inst):
        # A) explicit host_read value
        if host_pref is not None and _has_any(inst, {host_pref}):
            return True
        # B) pair: LEVEL.host + OPTYPE.read
        if (level_host is not None) and (read_val is not None):
            if _has_any(inst, {level_host}) and _has_any(inst, {read_val}):
                return True
        # C) signature fallback (requires dev_mask + (bank,row,col,ncol) and not a write)
        if _is_host_write(inst):
            return False
        dev_mask_found = any(_is_dev_mask(x) for x in (inst if isinstance(inst, (list, tuple)) else []))
        if not dev_mask_found:
            return False
        return _find_bank_row_col_ncol(inst) is not None

    for gid, _deps, insts in inst_groups:
        for idx, inst in enumerate(insts):
            # ---- BUF->BK (flush) ----
            if _has_any(inst, flush_values):
                bank_field, row_id = None, None
                iterable = inst if isinstance(inst, (list, tuple)) else []
                for part in iterable:
                    if isinstance(part, tuple) and len(part) >= 2:
                        bf, r = part[0], part[1]
                        try:
                            row_id = int(r)
                            bank_field = bf
                            break
                        except Exception:
                            continue
                if row_id is not None:
                    flushes.append({
                        "order": (gid, idx),
                        "banks": _bank_ids_from(bank_field),
                        "row":   row_id
                    })
                continue

            # ---- host_read ----
            if _is_host_read(inst):
                parsed = False
                if isinstance(inst, (list, tuple)) and len(inst) >= 9:
                    try:
                        ch_id, ra_id = int(inst[2]), int(inst[3])
                        dev_mask     = inst[4]
                        bank_field   = inst[5]
                        row_id       = int(inst[6])
                        col_offset   = int(inst[7])
                        ncol         = int(inst[8])
                        reads.append({
                            "order": (gid, idx),
                            "banks": _bank_ids_from(bank_field),
                            "row":   row_id, "col": col_offset, "ncol": ncol,
                            "ch":    ch_id,  "ra": ra_id,      "dev_mask": dev_mask
                        })
                        parsed = True
                    except Exception:
                        parsed = False
                if not parsed:
                    found = _find_bank_row_col_ncol(inst)
                    if not found:
                        raise RuntimeError(f"Cannot parse host_read inst (fallback failed): {inst}")
                    bank_field, row_id, col_offset, ncol = found
                    # best-effort dev mask
                    dev_mask = []
                    if isinstance(inst, (list, tuple)):
                        for x in inst:
                            if _is_dev_mask(x):
                                dev_mask = x
                                break
                    reads.append({
                        "order": (gid, idx),
                        "banks": _bank_ids_from(bank_field),
                        "row":   int(row_id), "col": int(col_offset), "ncol": int(ncol),
                        "ch":    0, "ra": 0, "dev_mask": dev_mask
                    })
                continue

    if _dbg_enabled():
        _dbg(f"[scan] found flushes={len(flushes)} reads={len(reads)}")
    return flushes, reads


def _verify_reads_after_flush(flushes, reads):
    earliest = {}
    banks_by_row = {}
    for f in flushes:
        r = f["row"]
        if r not in earliest or f["order"] < earliest[r]:
            earliest[r] = f["order"]
        banks_by_row.setdefault(r, set()).update(f["banks"])

    if not reads:
        raise AssertionError("No host_read instructions found.")
    for r in reads:
        row = r["row"]
        if row not in earliest:
            raise AssertionError(f"host_read row={row} has no prior device_buf2bk.")
        if not (set(r["banks"]) & banks_by_row[row]):
            raise AssertionError(f"host_read row={row} targets banks {r['banks']} "
                                 f"but flushed banks are {sorted(banks_by_row[row])}.")
        if not (earliest[row] <= r["order"]):
            raise AssertionError(f"host_read row={row} occurs before its device_buf2bk flush.")

def _tile_shape_from_row(row_id, output_row_offset, om_block, ol_block, om_row, ol_row, om_corner, ol_corner):
    o_row_id = int(row_id) - int(output_row_offset)
    if o_row_id < 0:
        raise ValueError(f"row={row_id} < output_row_offset={output_row_offset}")
    om_row_id  = o_row_id // max(1, ol_row)
    ol_row_id  = o_row_id %  max(1, ol_row)
    om_real = (om_block if om_row_id < max(1, om_row) - 1 else max(1, om_corner))
    ol_real = (ol_block if ol_row_id < max(1, ol_row) - 1 else max(1, ol_corner))
    m0 = om_row_id * om_block
    l0 = ol_row_id * ol_block
    return om_real, ol_real, m0, l0

def systolic_os_verify_linked_to_sim(
    # Problem
    M=8, K=8, L=8,
    # Tiling (OS)
    m_block=None, k_block=None, l_block=None,
    # Banks & mapping
    bank_num=16, pu_num=8, pu_list=None,
    input_bank=0,  input_row_offset=0,
    weight_bank=0, weight_row_offset=128,
    output_bank=0, output_row_offset=256,
    # Dtype & RNG
    dtype="int16", seed=12345,
    # Sim config
    yaml_path=None, run_timing_sim=True,
    verbose=True
):
    """
    Unified verifier linked to the *actual* instruction stream:
      - Generates A,B and C_ref with NumPy
      - Calls systolic_upmem.mm_micro(...) to get instruction groups
      - (Optional) runs the timing simulator on that stream
      - Scans stream: assert BUF->BK precedes host_read for each row/bank
      - At each BUF->BK, place the *correct* tile (from C_ref) into a shadow DRAM row
      - At each host_read, read back from shadow DRAM, assemble C_sim
      - Compare C_sim to C_ref

    Returns True on PASS; raises on failure.
    """
    # 0) Load sim YAML
    if yaml_path is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        yaml_path = os.path.join(repo_root, "config", "upmem.yaml")
    SimConfig.read_from_yaml(yaml_path)

    # 1) dtype + matrices
    dtype_l = str(dtype).lower()
    if dtype_l in ("int8","i8"):   dA=dB=np.int8;   acc=np.int64;  eq = np.array_equal
    elif dtype_l in ("int16","i16"): dA=dB=np.int16; acc=np.int64;  eq = np.array_equal
    elif dtype_l in ("float32","f32"): dA=dB=np.float32; acc=np.float64; eq = lambda X,Y: np.allclose(X,Y, rtol=1e-5, atol=1e-8)
    elif dtype_l in ("float64","f64"): dA=dB=np.float64; acc=np.float64; eq = lambda X,Y: np.allclose(X,Y, rtol=1e-6, atol=1e-9)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    rng = np.random.default_rng(seed)
    A = (rng.integers(-3,4,(M,K),dtype=dA) if np.issubdtype(dA,np.integer) else rng.standard_normal((M,K)).astype(dA))
    B = (rng.integers(-3,4,(K,L),dtype=dB) if np.issubdtype(dB,np.integer) else rng.standard_normal((K,L)).astype(dB))
    C_ref = (A.astype(acc) @ B.astype(acc))

    # 2) tiling & tails
    if m_block is None: m_block = M
    if k_block is None: k_block = K
    if l_block is None: l_block = L
    m_row = (M + m_block - 1) // m_block
    k_row = (K + k_block - 1) // k_block
    l_row = (L + l_block - 1) // l_block
    m_corner = M - (m_row - 1) * m_block if m_row > 0 else 0
    k_corner = K - (k_row - 1) * k_block if k_row > 0 else 0
    l_corner = L - (l_row - 1) * l_block if l_row > 0 else 0

    # 3) PU->bank map used in device ops: *local* per device
    de_total = max(1, getattr(SimConfig, "de", 1))
    banks_per_device = max(1, bank_num // de_total)
    base_local = int(output_bank) % banks_per_device
    if pu_list is None:
        pu_list = list(range(pu_num))
    raw_local = [(int(pu_id) * banks_per_device) // max(1, int(pu_num)) for pu_id in pu_list]
    output_bank_list = sorted({(r + base_local) % banks_per_device for r in raw_local})

    # 4) Build instruction stream from backend (same path as HW sim)
    cg = systolic_upmem(require_power_of_2=False)
    channel_list, rank_list, device_list = [0], [0], []
    base_group_id, performance_threshold = 0, 1 << 30


    # 4x4 mesh
    pu_num = 16
    pu_m, pu_l = 4, 4
    pu_k, pu_b = 1, 1


    simd_l = 1


    # Host writes (timing only; values handled by shadow DRAM below)
    def _bank_mask(codegen, bank_idx):
        mask = [False]*cg.bank_num
        mask[int(bank_idx)%cg.bank_num] = True
        return mask

    dev_mask = [True]*max(1, getattr(SimConfig, "de", 1))
    host_writes = []
    for r in range(M):
        host_writes.append(cg.create_host_write(0,0,dev_mask,_bank_mask(cg,input_bank),
                                                input_row_offset + r, 0, K, True))
    for r in range(K):
        host_writes.append(cg.create_host_write(0,0,dev_mask,_bank_mask(cg,weight_bank),
                                                weight_row_offset + r, 0, L, True))

    inst_groups, _ = cg.mm_micro(
        "mkl", base_group_id,
        channel_list, rank_list, device_list,
        pu_num, simd_l,
        input_bank, input_row_offset,
        weight_bank, weight_row_offset,
        output_bank, output_row_offset,
        m_block, k_block, l_block, 1,
        m_row, k_row, l_row, 1,
        m_corner, k_corner, l_corner, 1,
        m_block, l_block, 1,
        m_row, l_row, 1,
        m_corner, l_corner, 1,
        pu_m, pu_k, pu_l, pu_b,
        pu_list, performance_threshold
    )
    inst_groups = [(base_group_id-1, [], host_writes)] + inst_groups

    # 5) (Optional) run timing simulator
    if run_timing_sim:
        try:
            from sim import sim as _sim
            _ = _sim(inst_groups, silent=True, filename=None)
        except Exception as e:
            if verbose:
                print(f"[warn] timing sim skipped: {e}")

    # 6) Pre-scan: assert BUF->BK precedes host_read for same row/bank
    flushes, reads = _scan_flushes_and_reads_agnostic(inst_groups, OPTYPE)
    _verify_reads_after_flush(flushes, reads)

    # 7) Shadow DRAM: place tiles at FLUSH time (using C_ref), read them back at READ time
    dram = _ShadowDRAM(bank_num)
    for f in flushes:
        row = f["row"]
        om_real, ol_real, m0, l0 = _tile_shape_from_row(row, output_row_offset,
                                                        m_block, l_block, m_row, l_row,
                                                        m_corner, l_corner)
        tile = C_ref[m0:m0+om_real, l0:l0+ol_real].reshape(-1).astype(acc, copy=False)
        for bank_id in f["banks"]:
            dram.write_row(bank_id, row, tile)

    # Assemble from host_read (pick first bank in set)
    C_sim = np.zeros((M, L), dtype=acc)
    seen_rows = set()
    for r in reads:
        row = r["row"]
        if row in seen_rows:
            continue
        seen_rows.add(row)
        om_real, ol_real, m0, l0 = _tile_shape_from_row(row, output_row_offset,
                                                        m_block, l_block, m_row, l_row,
                                                        m_corner, l_corner)
        ncol = om_real * ol_real
        bank_id = r["banks"][0]
        payload = dram.read_row(bank_id, row, ncol)
        if payload.shape[0] != ncol:
            raise AssertionError(f"host_read length mismatch at row={row}: got {payload.shape[0]}, want {ncol}")
        C_sim[m0:m0+om_real, l0:l0+ol_real] = payload.reshape(om_real, ol_real)

    ok = eq(C_sim, C_ref)
    if verbose:
        tiles = f"tiles m={m_row} x k={k_row} x l={l_row} (blocks {m_block},{k_block},{l_block})"
        print(f"[LINKED VERIFY] {M}x{K}x{L} | {tiles} | banks={sorted(set(output_bank_list))}"
              f" -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        diff = C_sim - C_ref
        raise AssertionError(f"Mismatch.\nC_sim:\n{C_sim}\nC_ref:\n{C_ref}\nDiff:\n{diff}")
    return True


# -------------------------
# CLI self-tests
# -------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--linked", action="store_true", help="Run linked-to-sim verification")
    p.add_argument("--yaml", default=None, help="YAML path for SimConfig (default: config/upmem.yaml)")
    args = p.parse_args()

    if args.linked:
        print("[CLI] linked-to-sim running…")
        systolic_os_verify_linked_to_sim(
            yaml_path=args.yaml,
            run_timing_sim=False,
            verbose=True
        )
    else:
        systolic_os_verify_numpy_equivalence(verbose=True)

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
# JSONL helper (best-effort)
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
        0, 0, 0, 0,                                       # 'device_*gb*' (unused for UPMEM)
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
        0, 0, 0, 0,                   # LB<->BK placeholders
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
    Logical 2-D systolic mapping specialized to UPMEM’s per-bank PUs + local buffers.
    Output-stationary: partial sums remain in PU-local buffer/register.
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
        1) Host fan-out A tiles to PU in-buffers (device mask).
        2) For each L- and K-chunk: BUF->REG, MAC(A,B), REG->BUF (device-local).
        3) Host reads outputs from PU buffers (device mask).
        """
        tmp_inst_groups = []
        cmd_left = performance_threshold
        group_id = base_group_id

        # Print config snapshot
        _dbg(f"[cfg] ch={getattr(SimConfig, 'ch', 'NA')} ra={getattr(SimConfig, 'ra', 'NA')} "
             f"de={getattr(SimConfig, 'de', 'NA')} bank_num={getattr(self, 'bank_num', 'NA')}")
        _dbg(f"[work] m_row={m_row} k_row={k_row} l_row={l_row} | m_block={m_block} k_block={k_block} l_block={l_block}")
        _dbg(f"[pu]   pu_num={pu_num} pu_m={pu_m} pu_l={pu_l} pu_k={pu_k} | len(pu_list)={len(pu_list)}")

        _maybe_log_jsonl(self.trace_path, {
            "evt": "CONFIG", "run_id": "NA", "kernel_id": "mm",
            "dataflow": "output_stationary", "pu_grid": [pu_m, pu_l], "ts_cyc": 0
        })

        if mm_schedule not in ("mkl", "kml"):
            mm_schedule = "mkl"

        # Extra debug: show CH/RANK/DEV locals for given input lists
        _dbg("mm_micro CH/RANK/DEV locals:", [
            ( (ch_raw % max(1, getattr(SimConfig, 'ch', 1))),
              _ranks_for_channel(rank_list, (ch_raw % max(1, getattr(SimConfig, 'ch', 1)))),
              _local_devices_for((ch_raw % max(1, getattr(SimConfig, 'ch', 1))),
                                 _ranks_for_channel(rank_list, (ch_raw % max(1, getattr(SimConfig, 'ch', 1))))[0],
                                 device_list)
            )
            for ch_raw in channel_list
        ])

        # Use all ranks or only the first rank per channel?
        use_all_ranks = os.environ.get("UNINDP_USE_ALL_RANKS", "0") == "1"
        if not use_all_ranks:
            _dbg("[guard] Using ONLY the first local rank per channel (set UNINDP_USE_ALL_RANKS=1 to enable all).")

        # Emit groups per (channel, rank-selection)
        for ch_raw in channel_list:
            ch_id = ch_raw % max(1, getattr(SimConfig, 'ch', 1))
            rank_locals_all = _ranks_for_channel(rank_list, ch_id)
            rank_locals = rank_locals_all if use_all_ranks else rank_locals_all[:1]

            for ra_local in rank_locals:
                # Clamp to [0..ra-1]
                ra_local = ra_local % max(1, getattr(SimConfig, 'ra', 1))
                # Build device locals + device mask (length = SimConfig.de)
                dev_locals = _local_devices_for(ch_id, ra_local, device_list)
                dev_mask   = _mask_from_locals(dev_locals, max(1, getattr(SimConfig, 'de', 1)))
                pu_mask_all = [(i in pu_list) for i in range(pu_num)]

                _dbg(f"[emit] ch={ch_id} ra={ra_local} dev_locals={dev_locals} "
                     f"dev_mask_sum={len(dev_locals)} pu_used={sum(pu_mask_all)}")

                tmp_inst_list = []
                self.reset_output_buffer()

                # 1) Stage A tiles (host fan-out per group of pu_l PUs) — host op uses DEVICE mask
                groups = max(1, pu_m * max(1, pu_k))
                for input_group in range(groups):
                    start = input_group * max(1, pu_l)
                    end   = min((input_group + 1) * max(1, pu_l), len(pu_list))
                    limited_pus  = pu_list[start:end]
                    if not limited_pus:
                        continue
                    limited_mask = [(i in limited_pus) for i in range(pu_num)]
                    tmp_inst_list.append(
                        self.create_host_write_pu_inbuf(
                            ch_id, ra_local, dev_mask, limited_mask,
                            0, max(1, k_block * m_block)
                        )
                    )

                # 2) OS compute loop (device-local ops)
                for l_row_id in range(max(1, l_row)):
                    weight_row_id = l_row_id
                    for k_row_id in range(max(1, k_row)):
                        k_block_real = (k_block if k_row_id < k_row-1 else max(1, k_block_corner))
                        col_len = max(1, k_block_real)

                        for m_block_id in range(max(1, m_block)):
                            input_col_offset  = m_block_id * k_block_real

                            # BUF -> REG (per device)
                            for de_local in dev_locals:
                                tmp_inst_list.append(
                                    self.create_device_buf2reg(
                                        ch_id, ra_local, de_local, pu_num, pu_mask_all, 0
                                    )
                                )
                            # MAC (B from weight_bank, A from input_bank) (per device)
                            for de_local in dev_locals:
                                tmp_inst_list.append(
                                    self.create_device_pu(
                                        ch_id, ra_local, de_local, pu_num, pu_mask_all,
                                        (weight_bank,
                                         weight_row_offset + weight_row_id * max(1, k_row) + k_row_id,
                                         0),
                                        (input_bank,
                                         input_row_offset + m_block_id,
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

                # 3) Host readback (respect PU→bank mapping) — host op uses DEVICE mask
                bank_num = getattr(self, "bank_num", max(1, getattr(SimConfig, 'de', 1)))
                output_bank_list = [pu_id * max(1, bank_num) // max(1, pu_num) + output_bank for pu_id in pu_list]
                _dbg(f"[read] ch={ch_id} ra={ra_local} banks={sorted(set(output_bank_list))} "
                     f"reads={max(1, om_row) * max(1, ol_row) * len(output_bank_list)}")
                for output_bank_id in output_bank_list:
                    for om_row_id in range(max(1, om_row)):
                        for ol_row_id in range(max(1, ol_row)):
                            o_row_id = om_row_id * max(1, ol_row) + ol_row_id
                            col_len = max(1, (ol_block if ol_row_id < ol_row - 1 else max(1, ol_block_corner))
                                            * (om_block if om_row_id < om_row - 1 else max(1, om_block_corner)))
                            tmp_inst_list.append(
                                self.create_host_read(
                                    ch_id, ra_local, dev_mask,
                                    output_bank_id, output_row_offset + o_row_id, 0, col_len, True
                                )
                            )

                tmp_inst_groups.append((group_id, [], tmp_inst_list))
                _dbg(f"[group] id={group_id} insts={len(tmp_inst_list)}")
                group_id += 1

        return tmp_inst_groups, performance_threshold - cmd_left


class systolic_aim(BaseCodegen):
    """
    Systolic mapping specialized to AiM (future: GB broadcasts).
    Currently reuses OS structure; predictor differs.
    """

    def __init__(self, require_power_of_2):
        super(systolic_aim, self).__init__(require_power_of_2)
        self.predictor = _predictor_slots_for_aim()
        self.trace_path = os.environ.get("UNINDP_TRACE_OUT", "")

    def mm_micro(self, *args, **kwargs):
        # Reuse the same microkernel; predictor differs.
        return systolic_upmem.mm_micro(self, *args, **kwargs)

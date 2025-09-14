from tools import *
from sim.hw_system import *
from sim.inst_queue import *
import tqdm
import pickle as pkl
from math import inf
import numpy as np

def tqdm_replacement(iterable_object,*args,**kwargs):
    return iterable_object

def _required_channels_from_commands(commands):
    """
    Scan the generated command stream and find the maximum channel id used.
    Simulator will provision SimConfig.ch = max_ch + 1 when not sim_verify.
    """
    max_ch = 0
    try:
        for grp in commands:
            # group tuple typically: (group_id, <meta>, [inst, inst, ...])
            if not isinstance(grp, (list, tuple)) or len(grp) < 3:
                continue
            inst_list = grp[2]
            for inst in inst_list:
                if isinstance(inst, (list, tuple)) and len(inst) > 2:
                    ch_val = inst[2]
                    if isinstance(ch_val, int):
                        if ch_val > max_ch:
                            max_ch = ch_val
    except Exception:
        # best-effort only
        pass
    return max_ch + 1

def sim(commands, silent=False, filename=None, sim_verify=False):
    """
    Run the UniNDP simulator over a list of instruction groups.
    IMPORTANT: We no longer force SimConfig.ch=1 unconditionally. Instead, we
    size the channel count to cover whatever the command stream uses.
    """
    # If commands are serialized, load them first so we can size channels properly.
    if filename is not None:
        with open(filename, 'rb') as f:
            commands = pkl.load(f)

    # Determine how many channels are actually referenced by the command stream.
    # Keep the old "fast path" (ch=1) only if all commands target channel 0.
    real_ch = SimConfig.ch
    if not sim_verify:
        needed_ch = _required_channels_from_commands(commands)
        # If everything is on ch=0, we still enjoy the fast path.
        SimConfig.ch = max(1, needed_ch)

    # Build simulator state
    np_bankstate = np.zeros(
        (SimConfig.ch, SimConfig.ra, SimConfig.de, SimConfig.bg * SimConfig.ba, 4),
        dtype=np.int64
    )
    de_state_num = 1 + 1 + max(SimConfig.de_pu)  # bus, buffer, pu
    ra_state_num = SimConfig.de + 1 + SimConfig.ra_pu  # bus, buffer, pu
    ch_state_num = 0  # bus, buffer, pu
    sys_state_num = SimConfig.ch
    resource_state = np.zeros(
        sys_state_num + SimConfig.ch * (ch_state_num + SimConfig.ra * (ra_state_num + (SimConfig.de * de_state_num))),
        dtype=np.int64
    )
    HW = HW_system(np_bankstate, resource_state)

    # Create inst queue
    queue = inst_queue()

    total_cmd = 0
    for cmd in commands:
        queue.add_group(cmd[0], cmd[1], cmd[2])
        total_cmd += len(cmd[2])

    global_tick = 0
    issue_cmd = None
    issue_group = None

    # if silent:
    #     tqdm_copy = tqdm.tqdm
    #     tqdm.tqdm = tqdm_replacement

    iter_bar = range(total_cmd + 1) if silent else tqdm.tqdm(range(total_cmd + 1))

    for _ in iter_bar:
        # 1) advance time for previously chosen command
        if issue_cmd is not None:
            if add_tick > 0:
                global_tick += add_tick
                # Fast state updates (vector subtraction + clamp)
                np_bankstate -= int(add_tick)
                np_bankstate.clip(0, out=np_bankstate)
                resource_state -= int(add_tick)
                resource_state.clip(0, out=resource_state)

            # 2) commit previously chosen command
            queue.issue_inst(issue_group)
            queue.clear_empty_group(issue_group)
            HW.issue_inst(issue_cmd, issue_group)
            issue_cmd = None
            issue_group = None

        # 3) choose next command to issue
        add_tick = inf
        issuable_cmd = queue.get_inst()
        for tmp in issuable_cmd:
            group_id, inst_tmp = tmp
            tmp_issue_lat = HW.check_inst(inst_tmp, group_id)
            if tmp_issue_lat < add_tick:
                add_tick = tmp_issue_lat
                issue_cmd = inst_tmp
                issue_group = group_id

    # Final drain
    global_tick += np.max(resource_state)

    # restore original channel count
    SimConfig.ch = real_ch
    return global_tick

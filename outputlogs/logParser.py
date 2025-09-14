import ast

instruction_string = """
(<LEVEL.SYS: 4>, <OPTYPE.reg2buf: 2>, 0, 0, [True, True, True, True, True, True, True, True], [True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], 0, 0, 4, True)
(<LEVEL.RA: 2>, <OPTYPE.pu: 1>, 0, 0, (4, [True, True, True, True]), (1, 0, 0, 0), (0, 0, 0, 0), 4, False)
(<LEVEL.SYS: 4>, <OPTYPE.host_read_rank_pu_reg: 8>, 0, 0, [True, True, True, True])
(<LEVEL.RA: 2>, <OPTYPE.pu: 1>, 0, 0, (4, [True, True, True, True]), (1, 0, 0, 4), (0, 0, 0, 0), 4, False)
(<LEVEL.SYS: 4>, <OPTYPE.host_read_rank_pu_reg: 8>, 0, 0, [True, True, True, True])
(<LEVEL.RA: 2>, <OPTYPE.pu: 1>, 0, 0, (4, [True, True, True, True]), (1, 0, 0, 8), (0, 0, 0, 0), 4, False)
(<LEVEL.SYS: 4>, <OPTYPE.host_read_rank_pu_reg: 8>, 0, 0, [True, True, True, True])
(<LEVEL.RA: 2>, <OPTYPE.pu: 1>, 0, 0, (4, [True, True, True, True]), (1, 0, 0, 12), (0, 0, 0, 0), 4, False)
(<LEVEL.SYS: 4>, <OPTYPE.host_read_rank_pu_reg: 8>, 0, 0, [True, True, True, True])
(<LEVEL.RA: 2>, <OPTYPE.pu: 1>, 0, 0, (4, [True, True, True, True]), (1, 0, 0, 16), (0, 0, 0, 0), 4, False)
(<LEVEL.SYS: 4>, <OPTYPE.host_read_rank_pu_reg: 8>, 0, 0, [True, True, True, True])
(<LEVEL.RA: 2>, <OPTYPE.pu: 1>, 0, 0, (4, [True, True, True, True]), (1, 0, 0, 20), (0, 0, 0, 0), 4, False)
(<LEVEL.SYS: 4>, <OPTYPE.host_read_rank_pu_reg: 8>, 0, 0, [True, True, True, True])
(<LEVEL.RA: 2>, <OPTYPE.pu: 1>, 0, 0, (4, [True, True, True, True]), (1, 0, 0, 24), (0, 0, 0, 0), 4, False)
(<LEVEL.SYS: 4>, <OPTYPE.host_read_rank_pu_reg: 8>, 0, 0, [True, True, True, True])
(<LEVEL.RA: 2>, <OPTYPE.pu: 1>, 0, 0, (4, [True, True, True, True]), (1, 0, 0, 28), (0, 0, 0, 0), 4, False)
(<LEVEL.SYS: 4>, <OPTYPE.host_read_rank_pu_reg: 8>, 0, 0, [True, True, True, True])
(<LEVEL.RA: 2>, <OPTYPE.pu: 1>, 0, 0, (4, [True, True, True, True]), (1, 0, 0, 32), (0, 0, 0, 0), 4, False)
(<LEVEL.SYS: 4>, <OPTYPE.host_read_rank_pu_reg: 8>, 0, 0, [True, True, True, True])
(<LEVEL.RA: 2>, <OPTYPE.pu: 1>, 0, 0, (4, [True, True, True, True]), (1, 0, 0, 36), (0, 0, 0, 0), 4, False)
(<LEVEL.SYS: 4>, <OPTYPE.host_read_rank_pu_reg: 8>, 0, 0, [True, True, True, True])
(<LEVEL.RA: 2>, <OPTYPE.pu: 1>, 0, 0, (4, [True, True, True, True]), (1, 0, 0, 40), (0, 0, 0, 0), 4, False)
(<LEVEL.SYS: 4>, <OPTYPE.host_read_rank_pu_reg: 8>, 0, 0, [True, True, True, True])
(<LEVEL.RA: 2>, <OPTYPE.pu: 1>, 0, 0, (4, [True, True, True, True]), (1, 0, 0, 44), (0, 0, 0, 0), 4, False)
(<LEVEL.SYS: 4>, <OPTYPE.host_read_rank_pu_reg: 8>, 0, 0, [True, True, True, True])
(<LEVEL.RA: 2>, <OPTYPE.pu: 1>, 0, 0, (4, [True, True, True, True]), (1, 0, 0, 48), (0, 0, 0, 0), 4, False)
(<LEVEL.SYS: 4>, <OPTYPE.host_read_rank_pu_reg: 8>, 0, 0, [True, True, True, True])
(<LEVEL.RA: 2>, <OPTYPE.pu: 1>, 0, 0, (4, [True, True, True, True]), (1, 0, 0, 52), (0, 0, 0, 0), 4, False)
(<LEVEL.SYS: 4>, <OPTYPE.host_read_rank_pu_reg: 8>, 0, 0, [True, True, True, True])
(<LEVEL.RA: 2>, <OPTYPE.pu: 1>, 0, 0, (4, [True, True, True, True]), (1, 0, 0, 56), (0, 0, 0, 0), 4, False)
(<LEVEL.SYS: 4>, <OPTYPE.host_read_rank_pu_reg: 8>, 0, 0, [True, True, True, True])
(<LEVEL.RA: 2>, <OPTYPE.pu: 1>, 0, 0, (4, [True, True, True, True]), (1, 0, 0, 60), (0, 0, 0, 0), 4, False)
(<LEVEL.SYS: 4>, <OPTYPE.host_read_rank_pu_reg: 8>, 0, 0, [True, True, True, True])
(<LEVEL.RA: 2>, <OPTYPE.pu: 1>, 0, 0, (4, [True, True, True, True]), (1, 0, 0, 64), (0, 0, 0, 0), 4, False)
(<LEVEL.SYS: 4>, <OPTYPE.host_read_rank_pu_reg: 8>, 0, 0, [True, True, True, True])
(<LEVEL.RA: 2>, <OPTYPE.pu: 1>, 0, 0, (4, [True, True, True, True]), (1, 0, 0, 68), (0, 0, 0, 0), 4, False)
(<LEVEL.SYS: 4>, <OPTYPE.host_read_rank_pu_reg: 8>, 0, 0, [True, True, True, True])
(<LEVEL.RA: 2>, <OPTYPE.pu: 1>, 0, 0, (4, [True, True, True, True]), (1, 0, 0, 72), (0, 0, 0, 0), 4, False)
(<LEVEL.SYS: 4>, <OPTYPE.host_read_rank_pu_reg: 8>, 0, 0, [True, True, True, True])
(<LEVEL.RA: 2>, <OPTYPE.pu: 1>, 0, 0, (4, [True, True, True, True]), (1, 0, 0, 76), (0, 0, 0, 0), 4, False)
(<LEVEL.SYS: 4>, <OPTYPE.host_read_rank_pu_reg: 8>, 0, 0, [True, True, True, True])
(<LEVEL.RA: 2>, <OPTYPE.pu: 1>, 0, 0, (4, [True, True, True, True]), (1, 0, 0, 80), (0, 0, 0, 0), 4, False)
(<LEVEL.SYS: 4>, <OPTYPE.host_read_rank_pu_reg: 8>, 0, 0, [True, True, True, True])
(<LEVEL.RA: 2>, <OPTYPE.pu: 1>, 0, 0, (4, [True, True, True, True]), (1, 0, 0, 84), (0, 0, 0, 0), 4, False)
(<LEVEL.SYS: 4>, <OPTYPE.host_read_rank_pu_reg: 8>, 0, 0, [True, True, True, True])
(<LEVEL.RA: 2>, <OPTYPE.pu: 1>, 0, 0, (4, [True, True, True, True]), (1, 0, 0, 88), (0, 0, 0, 0), 4, False)
(<LEVEL.SYS: 4>, <OPTYPE.host_read_rank_pu_reg: 8>, 0, 0, [True, True, True, True])
(<LEVEL.RA: 2>, <OPTYPE.pu: 1>, 0, 0, (4, [True, True, True, True]), (1, 0, 0, 92), (0, 0, 0, 0), 4, False)
(<LEVEL.SYS: 4>, <OPTYPE.host_read_rank_pu_reg: 8>, 0, 0, [True, True, True, True])
(<LEVEL.RA: 2>, <OPTYPE.pu: 1>, 0, 0, (4, [True, True, True, True]), (1, 0, 0, 96), (0, 0, 0, 0), 4, False)
(<LEVEL.SYS: 4>, <OPTYPE.host_read_rank_pu_reg: 8>, 0, 0, [True, True, True, True])
(<LEVEL.RA: 2>, <OPTYPE.pu: 1>, 0, 0, (4, [True, True, True, True]), (1, 0, 0, 100), (0, 0, 0, 0), 4, False)
(<LEVEL.SYS: 4>, <OPTYPE.host_read_rank_pu_reg: 8>, 0, 0, [True, True, True, True])
(<LEVEL.RA: 2>, <OPTYPE.pu: 1>, 0, 0, (4, [True, True, True, True]), (1, 0, 0, 104), (0, 0, 0, 0), 4, False)
(<LEVEL.SYS: 4>, <OPTYPE.host_read_rank_pu_reg: 8>, 0, 0, [True, True, True, True])
(<LEVEL.RA: 2>, <OPTYPE.pu: 1>, 0, 0, (4, [True, True, True, True]), (1, 0, 0, 108), (0, 0, 0, 0), 4, False)
(<LEVEL.SYS: 4>, <OPTYPE.host_read_rank_pu_reg: 8>, 0, 0, [True, True, True, True])
(<LEVEL.RA: 2>, <OPTYPE.pu: 1>, 0, 0, (4, [True, True, True, True]), (1, 0, 0, 112), (0, 0, 0, 0), 4, False)
(<LEVEL.SYS: 4>, <OPTYPE.host_read_rank_pu_reg: 8>, 0, 0, [True, True, True, True])
(<LEVEL.RA: 2>, <OPTYPE.pu: 1>, 0, 0, (4, [True, True, True, True]), (1, 0, 0, 116), (0, 0, 0, 0), 4, False)
(<LEVEL.SYS: 4>, <OPTYPE.host_read_rank_pu_reg: 8>, 0, 0, [True, True, True, True])
(<LEVEL.RA: 2>, <OPTYPE.pu: 1>, 0, 0, (4, [True, True, True, True]), (1, 0, 0, 120), (0, 0, 0, 0), 4, False)
(<LEVEL.SYS: 4>, <OPTYPE.host_read_rank_pu_reg: 8>, 0, 0, [True, True, True, True])
(<LEVEL.RA: 2>, <OPTYPE.pu: 1>, 0, 0, (4, [True, True, True, True]), (1, 0, 0, 124), (0, 0, 0, 0), 4, False)
(<LEVEL.SYS: 4>, <OPTYPE.host_read_rank_pu_reg: 8>, 0, 0, [True, True, True, True])
(<LEVEL.RA: 2>, <OPTYPE.pu: 1>, 0, 0, (4, [True, True, True, True]), (1, 0, 0, 128), (0, 0, 0, 0), 4, False)
(<LEVEL.SYS: 4>, <OPTYPE.host_read_rank_pu_reg: 8>, 0, 0, [True, True, True, True])
(<LEVEL.RA: 2>, <OPTYPE.pu: 1>, 0, 0, (4, [True, True, True, True]), (1, 0, 0, 132), (0, 0, 0, 0), 4, False)
(<LEVEL.SYS: 4>, <OPTYPE.host_read_rank_pu_reg: 8>, 0, 0, [True, True, True, True])
(<LEVEL.RA: 2>, <OPTYPE.pu: 1>, 0, 0, (4, [True, True, True, True]), (1, 0, 0, 136), (0, 0, 0, 0), 4, False)
(<LEVEL.SYS: 4>, <OPTYPE.host_read_rank_pu_reg: 8>, 0, 0, [True, True, True, True])
(<LEVEL.RA: 2>, <OPTYPE.pu: 1>, 0, 0, (4, [True, True, True, True]), (1, 0, 0, 140), (0, 0, 0, 0), 4, False)
(<LEVEL.SYS: 4>, <OPTYPE.host_read_rank_pu_reg: 8>, 0, 0, [True, True, True, True])
(<LEVEL.RA: 2>, <OPTYPE.pu: 1>, 0, 0, (4, [True, True, True, True]), (1, 0, 0, 144), (0, 0, 0, 0), 4, False)
(<LEVEL.SYS: 4>, <OPTYPE.host_read_rank_pu_reg: 8>, 0, 0, [True, True, True, True])
(<LEVEL.RA: 2>, <OPTYPE.pu: 1>, 0, 0, (4, [True, True, True, True]), (1, 0, 0, 148), (0, 0, 0, 0), 4, False)
(<LEVEL.SYS: 4>, <OPTYPE.host_read_rank_pu_reg: 8>, 0, 0, [True, True, True, True])
(<LEVEL.RA: 2>, <OPTYPE.pu: 1>, 0, 0, (4, [True, True, True, True]), (1, 0, 0, 152), (0, 0, 0, 0), 4, False)
(<LEVEL.SYS: 4>, <OPTYPE.host_read_rank_pu_reg: 8>, 0, 0, [True, True, True, True])
(<LEVEL.RA: 2>, <OPTYPE.pu: 1>, 0, 0, (4, [True, True, True, True]), (1, 0, 0, 156), (0, 0, 0, 0), 4, False)
(<LEVEL.SYS: 4>, <OPTYPE.host_read_rank_pu_reg: 8>, 0, 0, [True, True, True, True])
(<LEVEL.RA: 2>, <OPTYPE.pu: 1>, 0, 0, (4, [True, True, True, True]), (1, 0, 0, 160), (0, 0, 0, 0), 4, False)
(<LEVEL.SYS: 4>, <OPTYPE.host_read_rank_pu_reg: 8>, 0, 0, [True, True, True, True])
(<LEVEL.RA: 2>, <OPTYPE.pu: 1>, 0, 0, (4, [True, True, True, True]), (1, 0, 0, 164), (0, 0, 0, 0), 4, False)
"""

# Clean and split input into lines
lines = [line.strip() for line in instruction_string.strip().split('\n') if line.strip()]

def describe_instruction(instr):
    try:
        # Normalize enum strings for parsing
        instr = instr.replace("<LEVEL.", "'LEVEL.").replace("<OPTYPE.", "'OPTYPE.")
        instr = instr.replace(">", "'")
        parsed = ast.literal_eval(instr)

        level = parsed[0].split('.')[1]
        optype = parsed[1].split('.')[1]

        desc = f"Operation: {optype.upper()} at level {level.upper()}"

        if optype == 'pu':
            pu_mask = parsed[6][1]
            input_addr = parsed[7]
            weight_addr = parsed[8]
            desc += f" | PU Mask: {pu_mask} | Input Addr: {input_addr} | Weight Addr: {weight_addr}"

        elif optype in ['bk2gb', 'buf2reg']:
            desc += f" | Mask: {parsed[4]}"

        return desc

    except Exception as e:
        return f"Failed to parse instruction: {e}"

# Run the formatter
for line in lines:
    print(describe_instruction(line))

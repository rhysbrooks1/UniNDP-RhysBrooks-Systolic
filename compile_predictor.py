import time
from frontend import *
from midend import *

# Import all standard backends
from backend import *
# Ensure systolic classes are present even if backend/__init__.py was not updated
try:
    systolic_upmem  # type: ignore[name-defined]
    systolic_aim    # type: ignore[name-defined]
except NameError:
    # Fallback explicit import (safe even if already imported)
    from backend.systolic import systolic_upmem, systolic_aim  # type: ignore

from sim import sim
from tools import *
import argparse
from math import inf
import tqdm
import csv
import os

# NOTE: if you want to disable tqdm in the framework, you can use the following
# def tqdm_replacement(iterable_object,*args,**kwargs):
#     return iterable_object
# tqdm_copy = tqdm.tqdm
# tqdm.tqdm = tqdm_replacement

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--architecture', '-A', type=str, default='aim')
    argparser.add_argument('--name', '-N', type=str, default='gemm')
    argparser.add_argument('--outdir', '-O', type=str, default='test')
    argparser.add_argument('--workload', '-W', type=str, default='mm', help="""workload type:
                           mm [M, K (Reduce_dimension), N, B],
                           elewise [C, H, W, B],
                           softmax [C(Softmax Dimension), H, W, B],
                           layernorm [C(norm), H(norm), W(norm), B],
                           batchnorm [C, H(norm), W(norm), B(norm)]""")
    argparser.add_argument('--workloadsize', '-S', nargs='+', type=int, default=[5000,5000,1,1])
    argparser.add_argument('--po2', '-P', action='store_true')
    argparser.add_argument('--nosim', '-NS', action='store_true')
    argparser.add_argument('--refpred', '-RP', action='store_true')
    argparser.add_argument('--cmdthre', '-T', type=float, default=3.0)
    argparser.add_argument('--topk', '-K', type=int, default=30)
    argparser.add_argument('--quicksearch', '-Q', action='store_true')
    argparser.add_argument('--allow_under_ultize', '-UU', action='store_true')
    args = argparser.parse_args()

    # set up log file
    output_dir_name = os.path.join('test_pred', args.outdir)
    os.makedirs(f"{output_dir_name}/csv", exist_ok=True)
    os.makedirs(f"{output_dir_name}/log", exist_ok=True)
    outcsv_name = f"./{output_dir_name}/csv/_{args.name}.csv"
    log_name = f"./{output_dir_name}/log/_{args.name}.log"
    log_file = open(log_name, 'w')

    # set up workload size
    #   mm workload size: [M, K, N, B]
    #   elewise / softmax / layernorm workload size: [C, H, W, B]
    assert len(args.workloadsize) == 4, f"Invalid workload size: {args.workloadsize}"
    if args.workload == 'mm':  # [M, K (Reduce_dimension), N, B]
        mm_size = (args.workloadsize[0] * args.workloadsize[3],
                   args.workloadsize[1],
                   args.workloadsize[2],
                   1)
    elif args.workload == 'elewise':  # [C, H, W, B]
        mm_size = (1,
                   args.workloadsize[0] * args.workloadsize[1] * args.workloadsize[2] * args.workloadsize[3],
                   1,
                   1)  # only k
    elif args.workload == 'softmax':  # [C(Softmax Dimension), H, W, B]
        mm_size = (1,
                   args.workloadsize[0],
                   args.workloadsize[1] * args.workloadsize[2] * args.workloadsize[3],
                   1)  # k,n
    elif args.workload == 'layernorm':  # [C(norm), H(norm), W(norm), B]
        mm_size = (1,
                   args.workloadsize[0] * args.workloadsize[1] * args.workloadsize[2],
                   args.workloadsize[3],
                   1)
    elif args.workload == 'batchnorm':  # [C, H(norm), W(norm), B(norm)]
        mm_size = (1,
                   args.workloadsize[3] * args.workloadsize[1] * args.workloadsize[2],
                   args.workloadsize[0],
                   1)
    else:
        raise ValueError(f"Unknown workload: {args.workload}")

    # ----------------------------------------------------------------------
    # 1) ARCHITECTURE SELECTION  (register systolic-* here)
    # ----------------------------------------------------------------------
    if args.architecture == 'aim':
        SimConfig.read_from_yaml('./config/gddr6-aim.yaml')
        SimConfig.de_pu = [16]
        Codegen = aim16
    elif args.architecture == 'aim8':
        SimConfig.read_from_yaml('./config/gddr6-aim.yaml')
        SimConfig.de_pu = [8]
        Codegen = aim8
    elif args.architecture == 'hbm-pim':
        SimConfig.read_from_yaml('./config/hbm-pim.yaml')
        Codegen = hbmpim
    elif args.architecture == 'upmem':
        SimConfig.read_from_yaml('./config/upmem.yaml')
        Codegen = upmem
    elif args.architecture == 'systolic-upmem':
        SimConfig.read_from_yaml('./config/upmem.yaml')
        Codegen = systolic_upmem
    elif args.architecture == 'systolic-aim':
        SimConfig.read_from_yaml('./config/gddr6-aim.yaml')
        SimConfig.de_pu = [16]  # same default as aim for MM
        Codegen = systolic_aim
    elif args.architecture == 'dimmining':
        SimConfig.read_from_yaml('./config/dimmining.yaml')
        Codegen = dimmining
    else:
        raise ValueError(f"Unknown hw architecture: {args.architecture}")

    # set NDP level
    if args.architecture == 'dimmining':
        SimConfig.pu_level = LEVEL.RA
    else:
        SimConfig.pu_level = LEVEL.DE

    print("sim config: ", SimConfig.__dict__, file=log_file)

    # ----------------------------------------------------------------------
    # 2) DESIGN SPACE
    # ----------------------------------------------------------------------
    start_tick = time.time()
    design_space = []
    partition_tool = Partition(require_power_of_2=args.po2)
    partition_space = partition_tool.get_partition_space_mm(mm_size)
    filtered_partition_space = partition_tool.choose_from_partition_space_mm(partition_space)
    if not args.allow_under_ultize:
        partition_space = filtered_partition_space
    print(len(partition_space), file=log_file)

    for index in tqdm.tqdm(range(len(partition_space))):
        compute_level, pu_num, partition = partition_space[index]
        simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row = partition_tool.mem_partition_mm(mm_size, partition)
        for input_choice in reversed(mkl_Input_to_row):
            if args.architecture in ['aim', 'aim8', 'systolic-aim']:
                if ml_Out_to_row:
                    output_choice = ml_Out_to_row[0]
                    design_space.append((compute_level, pu_num, partition, simd_k, input_choice, simd_l, output_choice))
                continue
            for output_choice in reversed(ml_Out_to_row):
                design_space.append((compute_level, pu_num, partition, simd_k, input_choice, simd_l, output_choice))

    print(f"constructed design space, size = {len(design_space)}")
    get_info = Codegen(require_power_of_2=args.po2).inst_info

    os.makedirs(os.path.dirname(outcsv_name), exist_ok=True)
    csvfile = open(outcsv_name, 'w', newline='')
    writer = csv.writer(csvfile)

    # ----------------------------------------------------------------------
    # 3) BASELINE SELECTION  (treat systolic-* like base arch)
    # ----------------------------------------------------------------------
    print("processing baseline")
    baseline = None

    if args.architecture in ['aim', 'aim8', 'systolic-aim']:
        for compute_level, pu_num, partition, simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row in design_space:
            if partition[3][0] * partition[3][1] == 1 and (
                mkl_Input_to_row[0][0] * mkl_Input_to_row[0][2] * mkl_Input_to_row[0][3] == 1
                or mkl_Input_to_row[0][1] * simd_k == mm_size[1]
            ):
                baseline = compute_level, pu_num, partition, simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row
                print(f"baseline strategy: {baseline}", file=log_file)
                break

    elif args.architecture == 'hbm-pim':
        for compute_level, pu_num, partition, simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row in design_space:
            if partition[3][0] * partition[3][1] == 1 and mkl_Input_to_row[0][1] == 8:
                baseline = compute_level, pu_num, partition, simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row
                print(f"baseline strategy: {baseline}", file=log_file)
                break

    elif args.architecture in ['upmem', 'systolic-upmem']:
        for compute_level, pu_num, partition, simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row in design_space:
            if partition[3][0] * partition[3][1] == 1 and (
                mkl_Input_to_row[0][0] * mkl_Input_to_row[0][2] * mkl_Input_to_row[0][3] == 1
                or mkl_Input_to_row[0][1] * simd_k == mm_size[1]
            ):
                baseline = compute_level, pu_num, partition, simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row
                print(f"baseline strategy: {baseline}", file=log_file)
                break
        if baseline is None:
            for compute_level, pu_num, partition, simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row in design_space:
                if (
                    mkl_Input_to_row[0][0] * mkl_Input_to_row[0][2] * mkl_Input_to_row[0][3] == 1
                    or mkl_Input_to_row[0][1] * simd_k == mm_size[1]
                ):
                    baseline = compute_level, pu_num, partition, simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row
                    print(f"baseline strategy: {baseline}", file=log_file)
                    break

    elif args.architecture == 'dimmining':
        for compute_level, pu_num, partition, simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row in design_space:
            if partition[2][0] * partition[2][1] == 1 and (
                mkl_Input_to_row[0][0] * mkl_Input_to_row[0][2] * mkl_Input_to_row[0][3] == 1
                or mkl_Input_to_row[0][1] * simd_k == mm_size[1]
            ):
                baseline = compute_level, pu_num, partition, simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row
                print(f"baseline strategy: {baseline}", file=log_file)
                break

    if baseline is None:  # corner case
        baseline = design_space[0]
        print(f"baseline strategy: {baseline}", file=log_file)

    compute_level, pu_num, partition, simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row = baseline

    # ----------------------------------------------------------------------
    # 4) CODEGEN + SIM FOR BASELINE
    # ----------------------------------------------------------------------
    mapping_tool = Mapping(require_power_of_2=args.po2)
    hw_id_list = mapping_tool.assign_hw(partition)

    input_bank, input_row_offset, \
    weight_bank, weight_row_offset, \
    output_bank, output_row_offset = mapping_tool.assign_dram(
        pu_num, mkl_Input_to_row, ml_Out_to_row, partition
    )

    codegen_tool = Codegen(require_power_of_2=args.po2)
    codegen_tool.set_gen()
    gen_code, inst_count, predict_result = codegen_tool.codegen(
        args.workload, compute_level, pu_num, partition,
        simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row,
        hw_id_list, (input_bank, input_row_offset,
                     weight_bank, weight_row_offset,
                     output_bank, output_row_offset),
        cmd_threshold=0
    )
    baseline_inst_num, baseline_pu_dram_num, baseline_host_dram_num, baseline_row_change_num = codegen_tool.get_matrix()
    baseline_sim_result = sim(gen_code, silent=True, filename=None)
    csvfile.flush()
    print(f"baseline result: {baseline_sim_result}", file=log_file)

    # ----------------------------------------------------------------------
    # 5) SEARCH
    # ----------------------------------------------------------------------
    if args.quicksearch:
        print(f"quick search, size = {len(design_space)}, topk = {args.topk}, cmdthre = {args.cmdthre}")
        predict_result_list = []
        min_codelen = 0
        # round 1: top-k by predictor
        for index in tqdm.tqdm(range(len(design_space))):
            thre = min_codelen * args.cmdthre
            design_point = design_space[index]
            compute_level, pu_num, partition, simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row = design_point
            mapping_tool = Mapping(require_power_of_2=args.po2)
            hw_id_list = mapping_tool.assign_hw(partition)
            input_bank, input_row_offset, weight_bank, weight_row_offset, output_bank, output_row_offset = mapping_tool.assign_dram(
                pu_num, mkl_Input_to_row, ml_Out_to_row, partition
            )
            codegen_tool = Codegen(require_power_of_2=args.po2)
            gen_code, inst_count, predict_result = codegen_tool.codegen(
                args.workload, compute_level, pu_num, partition,
                simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row,
                hw_id_list, (input_bank, input_row_offset,
                             weight_bank, weight_row_offset,
                             output_bank, output_row_offset),
                cmd_threshold=thre
            )
            if gen_code is not None:
                predict_result_list.append((predict_result, design_point))
                if len(predict_result_list) > args.topk:
                    predict_result_list = sorted(predict_result_list, key=lambda x: x[0])[:args.topk]

        design_space = [x[1] for x in predict_result_list]
        for pr in predict_result_list:
            print(f"predict_result: {pr}", file=log_file)

        if args.nosim:
            best_predict_result = predict_result_list[0][0]
            for i in range(len(predict_result_list)):
                if predict_result_list[i][0] > best_predict_result:
                    break
            design_space = design_space[:i]

        # round 2: simulate the remaining
        print("2nd round: get the best")
        best_result = inf
        best_design = []
        for index in tqdm.tqdm(range(len(design_space))):
            design_point = design_space[index]
            compute_level, pu_num, partition, simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row = design_point
            mapping_tool = Mapping(require_power_of_2=args.po2)
            hw_id_list = mapping_tool.assign_hw(partition)
            input_bank, input_row_offset, weight_bank, weight_row_offset, output_bank, output_row_offset = mapping_tool.assign_dram(
                pu_num, mkl_Input_to_row, ml_Out_to_row, partition
            )
            codegen_tool = Codegen(require_power_of_2=args.po2)
            codegen_tool.set_gen()
            gen_code, inst_count, predict_result = codegen_tool.codegen(
                args.workload, compute_level, pu_num, partition,
                simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row,
                hw_id_list, (input_bank, input_row_offset,
                             weight_bank, weight_row_offset,
                             output_bank, output_row_offset),
                cmd_threshold=0
            )
            if gen_code is not None:
                sim_result = sim(gen_code, silent=True, filename=None)
                inst_num, pu_dram_num, host_dram_num, row_change_num = codegen_tool.get_matrix()
                if sim_result < best_result or (args.nosim and sim_result > best_result):
                    best_result = sim_result
                    best_design = [inst_num, pu_dram_num, host_dram_num, row_change_num]

        writer.writerow(["name", "workload", "best_result", "baseline_result", "speedup"])
        writer.writerow([args.name, args.workload, best_result, baseline_sim_result, round(baseline_sim_result/best_result, 2)])
        csvfile.flush()

    else:  # brute force
        print(f"search using brute force, size = {len(design_space)}, cmdthre = {args.cmdthre}")
        writer.writerow(["compilation_strategy", "predict_result", "sim_result"])
        min_codelen = {}
        best_result = inf
        best_design = []
        for index in tqdm.tqdm(range(len(design_space))):
            design_point = design_space[index]
            compute_level, pu_num, partition, simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row = design_point
            mapping_tool = Mapping(require_power_of_2=args.po2)
            hw_id_list = mapping_tool.assign_hw(partition)
            input_bank, input_row_offset, weight_bank, weight_row_offset, output_bank, output_row_offset = mapping_tool.assign_dram(
                pu_num, mkl_Input_to_row, ml_Out_to_row, partition
            )
            if pu_num not in min_codelen.keys():
                thre = 0
            else:
                thre = args.cmdthre * min_codelen[pu_num]
            codegen_tool = Codegen(require_power_of_2=args.po2)
            _, _, predict_result = codegen_tool.codegen(
                args.workload, compute_level, pu_num, partition,
                simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row,
                hw_id_list, (input_bank, input_row_offset,
                             weight_bank, weight_row_offset,
                             output_bank, output_row_offset),
                cmd_threshold=thre
            )
            codegen_tool = Codegen(require_power_of_2=args.po2)
            codegen_tool.set_gen()
            gen_code, inst_count, _ = codegen_tool.codegen(
                args.workload, compute_level, pu_num, partition,
                simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row,
                hw_id_list, (input_bank, input_row_offset,
                             weight_bank, weight_row_offset,
                             output_bank, output_row_offset),
                cmd_threshold=thre
            )
            if gen_code is not None:
                if pu_num not in min_codelen.keys() or predict_result < min_codelen[pu_num]:
                    min_codelen[pu_num] = predict_result
                inst_num, pu_dram_num, host_dram_num, row_change_num = codegen_tool.get_matrix()
                sim_result = 0 if args.refpred else sim(gen_code, silent=True, filename=None)
                if sim_result < best_result:
                    best_result = sim_result
                    best_design = [inst_num, pu_dram_num, host_dram_num, row_change_num]

                writer.writerow([str(
                    [compute_level, pu_num, partition] + [mul(partition[i]) for i in range(4 if SimConfig.pu_level == LEVEL.DE else 3)]
                    + [simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row]
                ), predict_result, sim_result])
                csvfile.flush()

    end_tick = time.time()
    print(f"best_result: {best_result}", file=log_file)
    print(f"best_design: {best_design}", file=log_file)
    print(f"compile_time: {end_tick-start_tick}", file=log_file)
    if not args.refpred:
        print(f"speedup: {baseline_sim_result / best_result}", file=log_file)
    csvfile.close()

if __name__ == '__main__':
    main()

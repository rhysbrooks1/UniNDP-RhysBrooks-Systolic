

# UniNDP (Fork) — Quick Start & Artifact Guide

> This repository is a **fork** of the HPCA’25 UniNDP artifact.
> It preserves the original CLI and experiment flow, and adds:
>
> * A robust **systolic OS functional verifier** (math-only and linked-to-sim).
> * Extra debug logging for DRAM row discipline and instruction scheduling.
> * Minor fixes for enum-agnostic stream scanning and Python 3–first UX.

If you want the upstream artifact guide, see the original repo. This document covers what you need for **this repo**.


## 0) Setup

```bash
# clone repo
git clone https://github.com/rhysbrooks1/UniNDP-RhysBrooks-Systolic
cd UniNDP-RhysBrooks-Systolic

# (recommended) create a venv
python3 -m venv .venv
source .venv/bin/activate

# install requirements
pip install -r requirements.txt
```

> OS assumptions: Linux (Ubuntu/Debian tested).
> Key deps: Python 3.10+ (3.12 OK), PyYAML, numpy, tqdm, csv.
> If you hit a “No module named tqdm” error later, run: `pip install tqdm`.

---

## 1) Documentation (Fork Notes)

You can use the compiler and simulator like upstream. This fork adds **systolic verification utilities** you can run directly (see §3.3).

### 1.1 Compiler Usage (same CLI as upstream)

Compile a **single operator**:

```bash
python3 compile.py \
  -A {architecture_name} \
  -W {workload_type} \
  -S {input_size} {reduce_size} {output_size} {batch_size} \
  -O {output_dir}
```

Common flags:

* `-Q` quiet compile logs
* `-K N` choose top-N design points to simulate
* `-WS <workspace>` set workspace root

CLI help:

```bash
python3 compile.py -h
```

### 1.2 Batch Compile via Script (unchanged)

#### Step 0: prepare workloads

See `workload/README.md` for CSV format.

#### Step 1: run the batch script

```bash
# from repo root
./process_workload.sh {workload_file_name} {architecture_name} {workspace_name}
# example:
./process_workload.sh mm.csv aim myspace
```

This issues background `nohup ... &` commands and writes output logs to `nohup.out`.

#### Step 2: monitor / terminate batch

```bash
watch 'ps -aux | grep compile'
# kill all running compiles
kill -9 $(ps -aux | grep compile | grep -v grep | awk '{print $2}')
```

#### Step 3: export results

* Per-arch results go under:
  `./{workspace_name}/{workload_name}_{architecture_name}/csv` and `/log`

Create per-op CSV summaries:

```bash
cp script/combine_op.sh op/
cd op
bash combine_op.sh
# => op/mm_{arch}.csv (etc.)
```

---

## 2) Artifact Evaluation 

The commands below mirror the paper’s AE, and work on this fork without change. (Only difference: you must use `python3`.)

### 2.1 MM & MVM (Sec. VII-B, Table V & VI)

**Step 1: compile (~2.5h)**

```bash
# under repo root
bash ./process_workload.sh mm.csv {arch} {topK} op
# example (AiM, top 30):
bash ./process_workload.sh mm.csv aim 30 op
```

* `{arch}`: upmem | aim | aim8 | hbm-pim | dimmining
* `{topK}`: optional (default 30)
* outputs stored under `op/`

**Step 2: export**

```bash
cp script/combine_op.sh op/
cd op
bash combine_op.sh
# => ./op/mm_{arch}.csv per-arch
```

### 2.2 End-to-End (Sec. VII-B, Fig. 7 & 8)

**Step 1: compile**

```bash
bash ./process_workload.sh {workload_filename} {arch} {topK} e2e
# example
bash ./process_workload.sh resnet_18_224.csv aim 30 e2e
```

* workloads: `resnet_18_224.csv`, `vgg11_224.csv`, `llama2_7B_decode_tk32.csv`, `llama2_7B_prefill.csv`, `llama2_13B_decode_tk32.csv`, `llama2_13B_prefill.csv`
* `{arch}`: upmem | aim | dimmining

**Step 2: export (Fig.7)**

```bash
cp script/combine_e2e.sh e2e/
cd e2e
bash combine_e2e.sh
# => e2e/combined_results.csv
```

**Step 3: key metrics (Fig.8)**

In CSV:

* total instructions: `cmd`
* DRAM Access cmds: `pu_dram_access + host_dram_access`
* row changes: `row_change`

### 2.3 Simulator Verification (Sec. VII-C)

> ⚠ Samsung PIMSimulator requires sizes multiple of 4096.

**UniNDP Sim:**

```bash
python3 sim_verify.py -S {input_size} {output_size}
# logs under verify_result/log/[input_size,output_size].log
```

**Samsung PIMSimulator (external):**
Follow their README to build (`scons`), then edit:

`PIMSimulator/src/tests/PIMBenchTestCases.cpp`:

```cpp
TEST_F(PIMBenchFixture, gemv)
{
    setPIMBenchTestCase(KernelType::GEMV, {output_size}, {input_size});
    executePIMKernel();
}
```

Run:

```bash
./sim --gtest_filter=PIMBenchFixture.gemv > log.txt
```

### 2.4 Predictor Verification (Fig.10)

```bash
python3 compile_predictor.py -A upmem    -S 1 4096 4096 1 -O upmem_pred
python3 compile_predictor.py -A aim      -S 1 4096 4096 1 -O aim_16_pred
python3 compile_predictor.py -A aim8     -S 1 4096 4096 1 -O aim_8_pred
python3 compile_predictor.py -A dimmining -S 1 4096 4096 1 -O dimmining_pred
```

Speed-up toggle:

```bash
python3 compile_predictor.py -A dimmining -S 1 4096 4096 1 -O dimmining_pred -RP
```

Predictor-only:

```bash
python3 -OO compile_predictor.py -A aim -S 4096 4096 4096 1 -O with_sim -Q -K 30
python3 -OO compile_predictor.py -A aim -S 4096 4096 4096 1 -O no_sim  -Q -K 30 -NS
```

### 2.5 Pruning (Fig.9)

```bash
python3 compile_detail.py -A aim -S 1 1000 1000 1 -O pruning_test -UU
# results: pruning_and_breakdown/pruning_test/csv/_gemm.csv
```

Compare pruning on/off:

```bash
python3 -OO compile_detail.py -A aim -S 4096 4096 4096 1 -O with_prune -Q -K 30
python3 -OO compile_detail.py -A aim -S 4096 4096 4096 1 -O no_prune  -Q -K 30 -UU
```

### 2.6 Breakdown (Fig.11)

```bash
python3 -OO compile_detail.py -A upmem   -S 4096 6656 832 1 -O upmem_breakdown   -Q -K 50
python3 -OO compile_detail.py -A aim     -S 4096 6656 832 1 -O aim_breakdown     -Q -K 50
python3 -OO compile_detail.py -A aim8    -S 4096 6656 832 1 -O aim8_breakdown    -Q -K 50
```

### 2.7 Insight 2 (Sec. VII-G-2)

HBM-PIM input buffer size sensitivity:

```bash
# MM (4096)
python3 compile.py -A hbm-pim -W mm -S 4096 4096 4096 1 -Q -K 5 -IB 256 -O mm_inbuf_256  -WS buffer_insight
python3 compile.py -A hbm-pim -W mm -S 4096 4096 4096 1 -Q -K 5 -IB 512 -O mm_inbuf_512  -WS buffer_insight

# MVM (4096)
python3 compile.py -A hbm-pim -W mm -S 1 4096 4096 1 -Q -K 5 -IB 256 -O mvm_inbuf_256 -WS buffer_insight
python3 compile.py -A hbm-pim -W mm -S 1 4096 4096 1 -Q -K 5 -IB 512 -O mvm_inbuf_512 -WS buffer_insight
```

---

## 3) Fork-Specific Utilities (what’s new)

### 3.1 YAML & Debugging

* Default UPMEM config: `config/upmem.yaml`
* You can override via `--yaml config/<file>.yaml` or env:

  * `UNINDP_DEBUG=1` (verbose)
  * `UNINDP_TRACE_OUT=./runs/<name>/trace.jsonl` (JSONL trace)
  * `UNINDP_USE_ALL_RANKS=1` (enable multi-rank per channel in backend)

### 3.2 Systolic OS: **Math-only** Verifier (fast sanity)

```bash
export PYTHONPATH=.
python3 backend/systolic.py
# Expect “[SYSTOLIC-OS VERIFY] … -> PASS”
```

Custom sizes/tiles:

```bash
python3 - <<'PY'
from backend.systolic import systolic_os_verify_numpy_equivalence as verify
verify(M=8,K=8,L=8, m_block=4,k_block=4,l_block=4, dtype="int16", seed=123, verbose=True)
PY
```

### 3.3 Systolic OS: **Linked-to-Sim** Verifier (checks stream order & DRAM rows)

```bash
export PYTHONPATH=.
python3 backend/systolic.py --linked --yaml config/upmem.yaml
# Expect “[LINKED VERIFY] … -> PASS”
```

This verifier:

* Builds the real instruction stream (`mm_micro`) for your settings.
* Scans for **BUF→BK** flushes and **host_read** ops (enum-agnostic).
* Mirrors scheduler rows into a **shadow DRAM** at flush time.
* Assembles the result at host_read and compares with NumPy reference.


### 3.4 Dumping Instruction Streams

Use `compile.py` with `UNINDP_DEBUG=1` to print “group/inst” summaries during compile/sim, or add a targeted print in `backend/base.py` where `create_*` ops are produced.

---

## 4) Tips / Common Issues

* **Missing `tqdm`**: `pip install tqdm`
* **YAML path**: if you see defaults only, pass `--yaml config/upmem.yaml`
* **“openrow != target_row”**: your write path violates DRAM row policy. This fork opens the row via a tiny `bk2buf(1)` before `buf2bk` flushes in the OS kernel. If you changed bank mapping, ensure **device-local** bank IDs are used for device ops.
* **Bank mapping**: This fork maps PUs to **device-local** banks (per-device index space) for all `bk<->buf` ops; host reads still work with the per-device bank enumeration returned by the backend.

---

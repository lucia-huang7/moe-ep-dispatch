# GPU validation

## Hardware and transport

The host contains eight H100 SXM GPUs with NV18 connectivity across all pairs. Preflight confirms all 56 directed CUDA peer-access capabilities, 18 active NVLink links per GPU, eight active 400 Gb/s NDR ports, `EnableStreamMemOPs=1`, and the `gdrdrv` and `nvidia_peermem` modules. Bidirectional 4 MiB peer copies between two GPUs pass exact checks.

A 1,000-transfer `ib_write_bw` run with 64 KiB host-memory messages between two local HCAs measured 356.81 Gb/s average. This is a same-node host-memory RDMA check. The native GPU tests separately verify IBGDA initialization, GPU NIC handlers, GPU-resident NIC buffers and HCA counter activity.

The eight-rank packed-path validation disables native NVLink and NCCL InfiniBand. All eight HCA port counters increase during the run. It passes 768 rank-local cases with hook completion and a nondefault CUDA stream. The maximum native/reference absolute difference is 0.0078125, and the maximum difference from the CPU FP64 oracle is 0.04537865368331051 for that run.

## Numerical alignment

Native dispatch order is not assumed to match the PyTorch reference. Three BF16 channels encode rank, token-index high bits and token-index low bits. Rows are matched within each expert using this stable identity, then compared exactly across the complete hidden dimension. The three-channel encoding avoids losing token identities above 256 in BF16.

Each expert has its own two BF16 weight matrices, bias and ReLU. Routing weights are nonuniform FP32 values. The final oracle independently evaluates those quantized weights in CPU FP64 and accumulates routes directly, without reusing the communication inverse permutation.

<img src="../assets/tables/gpu-validation-1.png" alt="Numerical comparison criteria" width="720">

All tolerances are declared before executing each run. The final validation runner additionally records maximum error normalized by reference absolute maximum and relative L2 error. A one-percent normalized maximum threshold is applied to the native/reference comparison; per-element checks remain unchanged.

An earlier FP32 GPU baseline/optimized bitwise check differed by 1.19e-7 because atomic accumulation order can change. GPU baseline/optimized comparisons now use fixed dtype-specific tolerances. Exact transport checks, CPU bitwise comparisons and the FP64 oracle remain separate. Ordered GPU accumulation has its own bitwise controls and uses additional route-slot storage.

## Test families

<img src="../assets/tables/gpu-validation-2.png" alt="Validation workloads" width="720">

Case counts include individual ranks. These are separate runs with different coverage and should not be added into a single test-suite count. Intermediate validation runs remain in the experiment archive.

Scenarios include balanced routes, concentration on two experts, a dropped top-k route, an empty source rank, globally empty batches and all routes dropped. Duplicate expert routes are covered by the PyTorch reference and rejected by the native adapter.

The randomized CPU checks vary token count, hidden width, expert output width, local expert count, dtype and top-k routes. The linear expert oracle is evaluated independently. Each dispatch is followed by mutation of the caller's routing weights to NaN; outputs must still match the original finite weights preserved in the context. Both five-call and three-call outputs agree bitwise in these CPU runs.

## Large-input confirmation

Each rank receives 512 + 3r tokens, except the declared empty/drop scenarios. All 384 cases pass the existing elementwise tolerances and the additional one-percent native/reference normalized maximum criterion. Event completion, a nondefault stream and forced RDMA are enabled.

<img src="../assets/tables/gpu-validation-3.png" alt="Large-input numerical validation" width="940">

Normalized maximum is `max(abs(actual-reference)) / max(abs(reference))`; agreement on an all-zero reference is zero. The FP64 column is an observed metric; its elementwise tolerance is unchanged. [Hidden 2048](../results/packed-precision-large-2048.json), [hidden 7168](../results/packed-precision-large-7168.json).

The capacity regression verifies an inclusive limit of 128. Balanced, skewed and dropped routing pass at 127 and 128 tokens per rank; globally empty inputs pass, and 129 tokens are rejected before dispatch without consuming buffer state. [Capacity controls](../results/capacity-boundary.json).

## Adapter and layout controls

Adapter contract checks run on a nondefault stream and force RDMA. They reject an overlapping dispatch, destruction before combine, a foreign context, an unreserved output tensor, a consumed context and repeated combine. The registered getter must return the same reserved object for one dispatch.

Invalid receive rows and the registered output are filled with NaNs before valid expert rows are computed. Combined outputs remain finite and match the independent FP64 oracle. This checks that padding is excluded from both expert computation and combine, including all-empty and all-dropped inputs.

The C++ binding regression rejects foreign strided storage, incorrect strides, wrong dtype and ordinary noncontiguous input. A subsequent valid operation must pass, proving those failures did not consume the next alternating buffer. A final compatibility case populates registered storage and passes the contiguous placeholder used by the pinned upstream test.

## Diagnostic controls

The localization runner reports the first failing execution stage, not simply the largest final error.

<img src="../assets/tables/gpu-validation-4.png" alt="Controlled fault localization" width="720">

Every control reports its expected stage. These are injected diagnostic faults, not observations of an upstream production race.

## Upstream pressure

The pressure wrapper calls the pinned upstream `test_main` without changing its BF16/FP8, rounding, hook, zero-copy or preallocated-output checks. It repeats each seed twice after its reference run and requires identical per-rank hashes. Every completed seed is atomically checkpointed with all rank hashes. A deadline is checked between seeds, with an external timeout as a second bound.

The short two-rank calibration completed 15 seeds with one matching repetition per seed. The eight-rank calibration completed 11 seeds with one matching repetition per seed. The completed long run used two matching repetitions per seed, completing 806 seeds and 2,418 upstream runs over 8,531.5 seconds. Every repetition matched its reference hash, and the process exited successfully. The completed record is [available here](../results/upstream-pressure-world8-long.json). Its identity-like expert operations complement, rather than replace, the independent two-layer MLP oracle.

## Build controls

The Linux environment pins PyTorch 2.7.1+cu126, NCCL 2.26.2, NVSHMEM 3.3.9 and all 33 Python distributions. The fixed image provides driver 570.211.01 and CUDA compiler 12.8.93.

Python development headers match Python 3.12.3-1ubuntu0.11 and were installed from Ubuntu snapshot `20260228T000000Z`. The snapshot also supplied libexpat 2.6.1-2ubuntu0.4 and zlib development headers 1:1.3.dfsg-3.1ubuntu2.1. No blanket operating-system upgrade is part of setup.

The build includes the virtual environment's bin directory in PATH for Ninja and lets DeepEP discover the versioned NVSHMEM host library. The final manifest hashes the C++ binding source, local patch, compiled extension, source diff and Python lock. Earlier strict-prototype builds remain distinct from the final compatibility-preserving binding.

## Run selected checks

```bash
source scripts/gpu_environment.sh
export CUDA_VISIBLE_DEVICES=0,1
.venv/bin/python -m scripts.validate_distributed --backend nccl --world-size 2 --output results/nccl-fresh.json
.venv/bin/torchrun --standalone --nproc_per_node=2 -m scripts.validate_deepep \
  --packed-experts --packing zero_copy --known-output-size \
  --completion event --stream non_default --output results/native-fresh.json
.venv/bin/torchrun --standalone --nproc_per_node=2 -m scripts.validate_adapter_contract \
  --completion event --rdma-only --output results/contract-fresh.json
```

Completed rank-local cases are atomically checkpointed under `results/checkpoints/` before the final cross-rank aggregate is written. This progress directory is ignored by Git. Use a new output path for every launch. The native adapter permits one outstanding dispatch; all ranks must execute matching collective sequences.

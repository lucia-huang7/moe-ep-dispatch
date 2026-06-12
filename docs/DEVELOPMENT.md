# Development

Use Python 3.10 or newer with a PyTorch installation providing torch.distributed and Gloo for the CPU reference tests. The initial validation uses PyTorch 2.9.1 on macOS with separate spawned processes and one CPU thread per worker.

```bash
python -m pytest -q
python -m scripts.validate_distributed --world-size 2 --output results/new-world2.json
python -m scripts.validate_distributed --world-size 3 --output results/new-world3.json
```

Output paths must be fresh. JSON records include source hashes, per-case absolute errors, preset tolerances and actual collective input/output payload sizes. Payload accounting includes self-directed entries and is not physical interconnect traffic.

## Contract

The forward-only dispatcher accepts contiguous floating token matrices, int64 expert IDs and FP32 routing weights on the same device. Experts are evenly partitioned by contiguous global ID ranges across the process group. A route ID of -1 is dropped; repeated expert IDs are supported as distinct weighted routes. Top-k must be positive. Tokens may be empty on some or all ranks. Expert computation preserves the dispatched row order and may change the feature width.

All ranks must agree on hidden width, expert count, dtype, group and collective call order. Rank-local validation failures cannot safely recover a collective already entered by other ranks. Run hardware and numerical preflight before performance measurements. A rank-local exception is not a distributed recovery mechanism.

Dispatch returns its own context, including split tuples, stable permutations, valid route slots and a copy of the weights. A later dispatch does not overwrite an earlier call's routing state. Context tensors must be treated as read-only by callers. Combine inverts expert grouping, reverses the exchange and accumulates weighted results in FP32 before casting to the expert output dtype.

## Numerical checks

The direct FP64 oracle loops over each token and route and evaluates the selected expert without using dispatcher permutations. Each expert has a different matrix and bias; weights are nonuniform. A separate exact check verifies incoming token rows and local expert IDs against source-rank/token/route identities. Wrong expert weights form a negative control that must fail the final oracle.

The initial matrix fixes absolute and relative tolerances at 2e-6 for FP32, 8e-3 for FP16 and 6e-2 for BF16 before comparing results. These thresholds cover the small tested expert matrices; larger GPU shapes need separately defined numerical contracts. Source hashes identify each recorded validation version.

## Native GPU path

The native adapter has a narrower contract than the general reference. It supports forward-only BF16 tokens, distinct top-k expert routes, the pinned hidden dimensions, and one outstanding context. Event and hook completion modes are explicitly waited before data consumption. See [the integration contract](DEEPEP.md).

The environment script defaults to the eight verified compute HCAs and GPUs zero and one. Set `NVSHMEM_HCA_LIST` and `CUDA_VISIBLE_DEVICES` for a different verified topology before sourcing it. The script preserves those overrides.

```bash
source scripts/gpu_environment.sh
.venv/bin/torchrun --standalone --nproc_per_node=2 -m scripts.benchmark_pipeline \
  --methods baseline_grouped optimized_grouped native_grouped native_packed \
  --rounds 3 --output results/pipeline-fresh.json
```

Three rounds over four methods exercise all 24 permutations and yield 72 samples per method for each workload. Use three fresh process launches for the process-level aggregation shown in the report. Profiling runs separately from these timings.

```bash
.venv/bin/torchrun --standalone --nproc_per_node=2 -m scripts.profile_pipeline \
  --packing zero_copy --known-output-size --grouped-experts --packed-experts \
  --output results/profile-fresh.json --trace-prefix results/traces/fresh
```

## Randomized routing checks

```bash
python -m scripts.fuzz_reference --world-size 3 --seeds 200 --output results/fuzz-fresh.json
```

These CPU/Gloo cases vary shapes and routes, use an independent FP64 linear expert oracle, verify exact dispatch rows, and mutate caller-owned weights after dispatch to check context snapshots. They are separate from the native two-layer MLP checks.

## Report graphics

```bash
python -m scripts.summarize_results
python -m scripts.plot_tables
python -m scripts.plot_method
```

The table renderer refreshes pipeline values from the aggregated measurements and applies the shared table layouts. Method diagrams and tables use Times New Roman, with outlined SVG exports alongside PNG images.

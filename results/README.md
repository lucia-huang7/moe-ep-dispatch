# Result index

## Final pipeline grid

`pipeline-packed-world{2,4,8}-rep{1,2,3}.json` contains nine fresh process launches. Each launch records 16 workloads, four methods, 72 samples/method/workload, every rank sample and validation errors. `pipeline-summary.json` aggregates these files and records their SHA256 hashes.

The measured methods share grouped expert computation. Native compact and packed methods both use registered zero-copy combine storage; the packed variant also reads the native receive layout directly and writes the expert GEMM into registered output. Default native transport permits NVLink.

`pipeline-packed-rdma.json` is a separate forced-RDMA configuration with NCCL IB disabled. It has four workloads and one process launch.

## Ablations and attribution

`reference-ablation.json`, `reference-grouping.json`, `packing-ablation.json`, `native-grouping.json`, `completion-ablation.json`, `packed-completion-ablation.json` and `packed-ablation.json` isolate metadata, grouping, padding allocation, completion and registered-buffer choices. Timings from these separate launches are not chained into cumulative speedups.

`profile-before.json`, `profile-after.json` and `profile-packed.json` contain CPU/CUDA event aggregates from separate profiler runs. Their durations are not uninstrumented benchmark latencies.

## Correctness

The stage-aligned Gloo records, NCCL record and native records describe their tested shapes, tolerances and rank-local cases. `deepep-packed-world8.json` covers 768 rank-local cases with forced RDMA. `adapter-contract-{sync,event,hook}.json` checks lifecycle and poisoned padding on eight ranks.

`zero-copy-contract-final.json` validates the final binding convention; `precision-localization-final.json` contains controlled injected faults. Both identify the final build.

## Environment

`hardware-summary.json` contains public topology facts. Native build manifests identify the pinned source, patch and extension binary. Full host logs, profiler traces and intermediate experiment archives are stored outside the project.

## Final confirmation

`upstream-pressure-world8-long.json` records the completed 806-seed pressure run. `packed-precision-large-{2048,7168}.json` adds 192 large-input cases per width with normalized error metrics. `precision-localization-final.json` and `zero-copy-contract-final.json` repeat the diagnostic and binding controls with final build metadata.

`capacity-boundary.json` checks the inclusive 128-token capacity and rejected overflow. `pipeline-guard-confirmation.json` is a separate fresh-process performance confirmation after that Python guard correction.

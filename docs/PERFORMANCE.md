# Pipeline performance

## Final comparison

The final grid contains 48 workloads across 2/4/8 GPUs, hidden widths 2048/7168, four input token counts and balanced/skewed routing. Each point is repeated in three fresh process groups. Four methods run in every permutation for three rounds, producing 72 samples per method per workload per launch.

The five-call to three-call reference change lowers full-operation median latency by 7.7%–13.2% across the grid. The packed native path lowers compact native latency by 13.1%–23.7%, using the median of the three process medians. All 144 matched process comparisons favor the packed path; the paired reductions span 13.1%–23.9%. These are descriptive observations from three launches, not a confidence interval.

Both native variants use the same communication implementation and registered zero-copy combine storage. The difference is the expert data path. Compact mode materializes received rows and expert labels, builds a compact expert output, then copies valid rows into registered storage. Packed mode reads valid prefixes in place and writes the second expert GEMM into registered storage.

The final measurements include identical grouped expert MLPs for all methods. Earlier masked-versus-grouped experiments are separate ablations, so that compute-path improvement is not attributed to changing the communication backend.

## Profiler evidence

Counts below are rank-zero aggregates over five complete iterations. Profiling is separate from latency measurements.

<img src="../assets/tables/performance-1.png" alt="Profiler counts over five iterations" width="940">

The grouped compact path replaces boolean expert indexing and redundant output-size work. Its copy count can increase even while overall latency improves, so raw operation counts alone do not explain performance. The packed path then removes receive concatenation and much of the copy work without reducing the remaining explicit synchronization count.

The initial reference profiler run records 85 stream synchronizations over five iterations; reusing counts, known output sizes and grouped experts reduces that count to 35. These profiler counts are not standalone communication-kernel timings.

## Completion choices

The packed completion ablation measures synchronous, event and hook modes in one counterbalanced process launch. Hook mode has the lowest median in 13 of 16 workloads, synchronous mode in three, and event mode in none. Their small host-pipeline differences do not establish useful communication/compute overlap. The final multi-process grid uses synchronous completion consistently.

## Forced RDMA

The separate eight-rank transport run disables native NVLink and NCCL InfiniBand. Native traffic uses its RDMA path while the same-node NCCL reference can use its local GPU transport. These are distinct transport configurations, not an isolated network-kernel comparison.

All rows below use hidden 7168, four experts per rank and top-k two. Values are milliseconds from one process launch, with 72 counterbalanced samples per method.

<img src="../assets/tables/performance-2.png" alt="Forced-RDMA pipeline latency (ms)" width="940">

Packing improves the native adapter in each row, but the larger skewed RDMA case remains slower than the NCCL reference. Concentrating all routes on two global experts also concentrates expert computation on one rank. The measured boundary is therefore the complete adapter-plus-expert workload under that routing and transport configuration.

## Scope

The grid uses a native capacity of 1024 tokens per rank, top-k two and expert intermediate width 128. It exercises the communication and integration path rather than a complete deployed MoE language model. Increasing rank count also increases total input tokens and total experts. Shapes, routing, capacity and completion mode must accompany any reported throughput or speedup.

[Complete measurements](MEASUREMENTS.md), [aggregation and timing](METHODOLOGY.md), [source records](../results/README.md).

## Final guard confirmation

After aligning the Python capacity check with the inclusive C++ bound, a fresh two-rank launch repeated hidden-7168 workloads at 16 and 512 tokens on rank zero, balanced and skewed routing. Both native methods use the same valid-input computation as the original grid. Sixty counterbalanced samples per method confirm reductions of 13.0%–23.9% for the packed path. This separate launch is not pooled into the three-process summary.

<img src="../assets/tables/performance-3.png" alt="Final capacity-guard confirmation" width="940">

[Confirmation samples](../results/pipeline-guard-confirmation.json).

# Measurement methodology

## Complete operation

The latency boundary starts before dispatch and ends after weighted combine and CUDA synchronization. It includes Python routing, validation, metadata transfers, native calls, expert MLP work, packing, allocations and completion waits. It excludes the control barrier and synchronization immediately before the timed region. These are host wall-clock pipeline measurements, not standalone communication-kernel latency.

Each rank records each invocation. For a matching invocation index, the reported sample is the maximum duration across ranks. Input generation, reference generation and correctness checks run before warm-up and timing.

The comparison uses the same input tensors, distinct BF16 expert matrices, ReLU, biases, FP32 routing weights and final BF16 output. Performance inputs are random BF16 values; precision runs additionally encode stable identities in three channels. Each expert MLP has intermediate width 128. Top-k is two and there are four experts per rank. Rank r receives N + 3r tokens. Balanced routing selects distinct random experts; skewed routing sends all tokens to global experts zero and one.

The five-call reference retains redundant rank-count exchanges and their consistency checks. The three-call path derives incoming rank counts from expert counts and reuses dispatch splits during combine. Its improvement includes the removal of the redundant exchanges and associated checks, not only network transfer time.

The native capacity is fixed at 1024 tokens per rank throughout the final token grid. Its padded expert layout therefore reserves substantially more rows than are valid for the smallest inputs. The packing ablation includes that fixed-capacity cost; the final compact-versus-packed comparison uses registered zero-copy combine storage for both paths. A deployment with smaller capacity may change the allocation-related benefit.

## Experimental order

Every workload warms up each method five times. The schedule contains every permutation of the selected methods in each counterbalanced round. Three methods and ten rounds yield 60 samples per method; four methods and three rounds yield 72. Four-method ablations with ten rounds yield 240 samples per method.

Earlier four-method JSON files contain a fixed text label saying six permutations. Their actual samples follow all 24 permutations, as implemented by `itertools.permutations(methods)` and reflected in their sample counts. The final packed-input experiment records the dynamically derived permutation count.

The final grid runs in three fresh process groups for each of two, four and eight GPUs. Summary figures take the median of the three process-level medians. Error bars span those three medians; they are not confidence intervals. Within-process percentiles and individual rank samples remain available in the JSON records. Ablation timings from a different launch are not chained into a cumulative speedup.

## Transport and scaling

Default latency comparisons permit native NVLink. The RDMA-only comparison disables the native NVLink path and sets `NCCL_IB_DISABLE=1`; it is a separate transport configuration. HCA counter changes and runtime configuration establish transport use, but they do not separate every control-plane byte from payload traffic.

The scaling plot holds tokens and experts per rank approximately fixed, including the documented rank-dependent token offset. Increasing rank count therefore increases both global tokens and total experts. It is workload scaling on one eight-GPU host, not a fixed-total-work strong-scaling experiment or a multi-node network benchmark.

Global input throughput is total original input tokens divided by full-operation latency. Routed token copies and top-k expansion are not counted as additional input tokens. This is an MoE-layer synthetic workload, not autoregressive model decoding throughput.

## Attribution

CPU/CUDA profiler runs are separate from the uninstrumented timing runs. Labeled regions identify dispatch, expert and combine work. `cudaStreamSynchronize` counts describe explicit synchronization calls seen by the profiler over five iterations; they are not a total count of every possible GPU dependency. Profiler CPU totals and CUDA kernel totals are reported separately and are not added to infer pipeline latency.

The packed native path avoids receive compaction and output repacking while retaining host count retrieval and input validation. Event/hook completion modes wait before consuming results; these tests do not by themselves establish useful compute/communication overlap.

## Precision

Transport alignment uses exact token and expert identity. Native order may differ within an expert, so rows are matched before numeric comparisons. Expert computations use distinct weights and biases, and final outputs are also compared with an independently evaluated CPU FP64 oracle.

FP32 atomic accumulation order can differ on CUDA. Native/reference and GPU baseline/optimized comparisons use fixed dtype-specific tolerances. Ordered accumulation is a separate implementation and memory-cost tradeoff; its bitwise controls are reported independently.

Synthetic fault injection tests whether stage diagnostics identify deliberately corrupted dispatch values, expert matrices, nonfinite outputs, combine order or routing weights. Upstream pressure tests check their own BF16/FP8 invariants and repeated-seed hashes using identity-like experts. These two test families complement the independent MLP oracle and are not interchangeable case counts.

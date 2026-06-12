# Native DeepEP integration

## Pinned API

The adapter targets [DeepEP v1.2.1](https://github.com/deepseek-ai/DeepEP/tree/9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee), commit `9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee`. It uses the original `Buffer`, `low_latency_dispatch`, `low_latency_combine` and `get_next_low_latency_combine_buffer` interfaces. The setup script checks out this revision before applying the local binding patch.

<img src="../assets/tables/deepep-1.png" alt="Validated runtime environment" width="720">

All installed Python distributions are pinned and hashed in the Linux lock.

## Registered-view API extension

The pinned binding requires `x.is_contiguous()` before invoking low-latency combine. The registered combine-buffer getter returns a view with padding between experts and rows. Passing that returned view directly therefore fails at the binding, before a communication kernel is launched.

[The patch](../patches/deepep-v1.2.1-zero-copy-layout.patch) accepts a noncontiguous input only when zero-copy mode is enabled, its data pointer belongs to the next registered send buffer, and all expected strides match. These checks precede the alternating-buffer-index update. Ordinary inputs still require contiguous storage.

The legacy upstream convention remains supported. A caller may populate the registered buffer and pass a separate contiguous placeholder to zero-copy combine. The kernel consumes the registered storage, so the placeholder's contents are not the payload. Rejecting every foreign pointer would break this established calling convention.

This extends direct-view input support rather than changing the existing placeholder protocol. The regression test checks foreign strided storage, wrong strides, wrong dtype, ordinary noncontiguous input, a valid operation after all rejected calls, and the legacy contiguous placeholder. Rejections must not advance the buffer index. The patch changes the C++ binding, not the communication kernels.

## Packed expert path

`dispatch_packed()` exposes the native expert-major receive view and valid row counts without compacting it with `torch.cat`. Each expert MLP reads only its valid prefix. Its second matrix multiplication writes directly into a reserved registered combine view, followed by in-place bias addition. `combine_registered()` accepts the exact view reserved for that dispatch.

The comparison retains the same BF16 matrices, ReLU, biases and routing weights. It removes compact receive materialization, expert labels, the compact expert-output tensor and the subsequent output-to-registered-buffer copy. It does not fuse the expert GEMMs into a communication kernel.

Host row counts still require a device-to-host synchronization. Forward input validation also synchronizes. These costs remain in end-to-end timings.

## Completion and lifecycle

Synchronous mode uses the binding's default completion. Event mode waits with `event.current_stream_wait()`. Hook mode invokes the receive hook before consuming received data. Neither mode is described as compute/communication overlap merely because it exposes an asynchronous API.

Each adapter permits one outstanding dispatch. A combine consumes its context, releases its reserved view and permits the next dispatch. A foreign context, duplicate combine, overlapping dispatch, or premature destruction raises an error before entering a collective. All ranks must follow the same operation order; a rank-local validation exception is not a distributed recovery protocol.

The native adapter is forward-only, accepts contiguous BF16 token inputs, contiguous int64 routes and contiguous FP32 routing weights, and rejects duplicate valid expert IDs within one token. The PyTorch reference additionally tests duplicate routes. Experts are contiguous and evenly sharded across ranks. Supported hidden widths follow the pinned binding. Token capacity is inclusive, matching the C++ check `num_tokens <= num_max_dispatch_tokens_per_rank`; larger inputs are rejected before dispatch. The upstream Python docstring describes a strict bound, but its C++ implementation and supplied tests allow equality.

## Transport checks

The eight-GPU host exposes NV18 connectivity across GPU pairs, all 56 directed CUDA peer-access capabilities and eight active 400 Gb/s NDR InfiniBand ports. Runtime checks verify peer copies, verbs devices, GPUDirect support and IBGDA initialization.

Default native measurements permit NVLink. Forced-RDMA validation sets `allow_nvlink_for_low_latency_mode=False`, disables NCCL InfiniBand for reference/control traffic, and records HCA port-counter changes. All measurements use a single host. The host-memory `ib_write_bw` check between two HCAs establishes a usable same-node RDMA path; it is separate from native GPU timing and does not establish inter-node bandwidth.

## Source and scope

The all-to-all implementation follows the five-exchange versus three-exchange reference exercise. This repository adds distinct two-layer experts, nonuniform weights, independent FP64 checks, identity-aligned stage comparisons, explicit contexts, completion controls and performance ablations. The native backend is the pinned upstream DeepEP implementation with the documented binding patch. Passing these tests does not reproduce or establish the root cause of a separate production race incident.

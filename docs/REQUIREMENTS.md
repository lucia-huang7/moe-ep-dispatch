# Project requirements

## Objective

Implement a transparent PyTorch expert-parallel MoE path, reduce redundant all-to-all metadata exchanges, and use it as an independent numerical reference for a version-pinned DeepEP backend.

## Required deliverables

<img src="../assets/tables/requirements-1.png" alt="Implementation and validation scope" width="720">

## Hardware requirements

For two-rank GPU reference tests, use two CUDA GPUs on one host with a working NCCL transport. For the intended native DeepEP comparison, prefer two full H100 SXM GPUs with NVLink; a model-name badge must be followed by runtime topology and CUDA peer-access validation.

The pinned revision permits NVLink in low-latency mode but still allocates an NVSHMEM-backed low-latency buffer. Its documented installation prerequisites additionally include supported RDMA devices and a usable NVSHMEM transport, including the IBGDA prerequisites documented by the selected revision. NVLink is not evidence of RDMA availability. Confirm device exposure, verbs support, GPUDirect and provider-supported host configuration before paying for a host for this path. Do not change host driver security settings automatically.

NVLink and RDMA serve different paths: intranode normal dispatch/combine can use NVLink; low-latency/remote paths have additional transport requirements. A dual-GPU machine is the proposed experiment size, not a guarantee that every upstream benchmark configuration supports world size two unchanged.

## Software requirements for the course-era API

The inspectable historical reference is DeepEP v1.2.1, commit `9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee`. Its documented requirements include:

- Linux CUDA development environment with a compatible C++ compiler and toolkit.
- Python 3.8 or later and CUDA-enabled PyTorch 2.1 or later.
- CUDA 12.3 or later for Hopper SM90.
- NVSHMEM 3.3.9 or later for enabled RDMA/low-latency functionality.
- Working torch.distributed/NCCL with one rank per distinct GPU.
- Buffer-era APIs matched to the exact course calls; review shape constraints and feature flags at the pinned revision.

The validated stack is Python 3.12.3, PyTorch 2.7.1+cu126, CUDA compiler 12.8.93, NCCL 2.26.2, and NVSHMEM 3.3.9.

Current DeepEP main uses different APIs and dependencies. Installing it does not provide a drop-in replacement for the course-era Buffer examples.

## Sources

- [Historical DeepEP requirements and APIs](https://github.com/deepseek-ai/DeepEP/tree/9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee)
- [Historical NVSHMEM and IBGDA prerequisites](https://github.com/deepseek-ai/DeepEP/blob/9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee/third-party/README.md)
- [Current DeepEP](https://github.com/deepseek-ai/DeepEP)

Course-specific source links and private hardware quotes are retained outside the public project directory. Course authors' production incidents and performance numbers are not project results.

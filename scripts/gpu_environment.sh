#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export NVSHMEM_DIR
NVSHMEM_DIR=$(.venv/bin/python -c 'import nvidia.nvshmem; print(nvidia.nvshmem.__path__[0])')
export LD_LIBRARY_PATH="$NVSHMEM_DIR/lib:${LD_LIBRARY_PATH:-}"
export NVSHMEM_REMOTE_TRANSPORT=ibrc
export NVSHMEM_IB_ENABLE_IBGDA=1
export NVSHMEM_IBGDA_NIC_HANDLER=gpu
export NVSHMEM_HCA_LIST="${NVSHMEM_HCA_LIST:-mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1}"
export NVSHMEM_DEBUG=INFO
export NVSHMEM_DEBUG_SUBSYS=INIT,TRANSPORT
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

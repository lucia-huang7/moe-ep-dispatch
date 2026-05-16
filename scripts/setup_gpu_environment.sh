#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ ! -f /usr/include/python3.12/Python.h ]]; then
    echo "Install matching Python development headers from the fixed Ubuntu snapshot; see docs/GPU_VALIDATION.md." >&2
    exit 1
fi
python3 -m venv .venv
.venv/bin/python -m pip install --no-deps uv==0.6.17
.venv/bin/uv pip sync environment/requirements-linux.lock --python .venv/bin/python --require-hashes
mkdir -p vendor results
if [[ ! -d vendor/DeepEP/.git ]]; then
    git clone https://github.com/deepseek-ai/DeepEP.git vendor/DeepEP
fi
git -C vendor/DeepEP checkout --detach 9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee
patch_file="$PWD/patches/deepep-v1.2.1-zero-copy-layout.patch"
if git -C vendor/DeepEP apply --reverse --check "$patch_file" >/dev/null 2>&1; then
    :
else
    git -C vendor/DeepEP apply --check "$patch_file"
    git -C vendor/DeepEP apply "$patch_file"
fi
export NVSHMEM_DIR
NVSHMEM_DIR=$(.venv/bin/python -c 'import nvidia.nvshmem; print(nvidia.nvshmem.__path__[0])')
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$PWD/.venv/bin:$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=9.0
export MAX_JOBS=8
export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:${LD_LIBRARY_PATH:-}"
# Let the pinned build detect the wheel library soname.
env -u NVSHMEM_DIR .venv/bin/python -m pip install --no-clean --no-build-isolation --no-deps ./vendor/DeepEP
.venv/bin/python -c 'import torch,deep_ep; print(torch.__version__,torch.version.cuda,deep_ep.__file__)'
.venv/bin/python -m pip freeze > results/gpu-package-versions.txt

.venv/bin/python -m scripts.record_build_manifest --output results/deepep-build-manifest.json

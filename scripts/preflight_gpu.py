"""Record GPU topology and RDMA prerequisites without changing host settings."""
import argparse
import ctypes
import json
import platform
import shutil
import subprocess
from pathlib import Path


def run(command):
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=45)
        return {'command': command, 'returncode': result.returncode,
                'stdout': result.stdout, 'stderr': result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'command': command, 'error': str(exc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a fresh output path')
    commands = [
        ['nvidia-smi', '--query-gpu=index,name,uuid,driver_version,memory.total', '--format=csv'],
        ['nvidia-smi', 'topo', '-m'],
        ['nvidia-smi', 'nvlink', '--status'],
        [shutil.which('nvcc') or '/usr/local/cuda/bin/nvcc', '--version'], ['ibv_devinfo', '-l'], ['ibstat'],
        ['bash', '-lc', 'cat /proc/driver/nvidia/params | grep -E "EnableStreamMemOPs|RegistryDwords"'],
        ['bash', '-lc', 'lsmod | grep -E "gdrdrv|nvidia_peermem|mlx5"'],
        ['bash', '-lc', 'for d in /sys/class/infiniband/*; do echo "$(basename "$d") $(cat "$d/ports/1/rate") $(cat "$d/ports/1/link_layer")"; done'],
        ['systemctl', 'list-timers', 'moe-budget-stop.timer', '--no-pager'],
    ]
    payload = {'platform': platform.platform(), 'checks': [run(c) for c in commands]}
    driver = ctypes.CDLL('libcuda.so.1')
    count = ctypes.c_int()
    assert driver.cuInit(0) == 0
    assert driver.cuDeviceGetCount(ctypes.byref(count)) == 0
    matrix = []
    for source in range(count.value):
        row = []
        for target in range(count.value):
            available = ctypes.c_int()
            status = driver.cuDeviceCanAccessPeer(ctypes.byref(available), source, target)
            row.append({'status': status, 'can_access': bool(available.value)})
        matrix.append(row)
    payload.update(gpu_count=count.value, peer_access=matrix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({'gpu_count': count.value, 'peer_0_to_1': matrix[0][1], 'output': str(args.output)}))


if __name__ == '__main__':
    main()

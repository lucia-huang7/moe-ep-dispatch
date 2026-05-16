"""Validate direct CUDA peer copies in both directions without host staging."""
import argparse
import ctypes
import json
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a fresh output path')
    runtime = ctypes.CDLL('libcudart.so.12')
    runtime.cudaMemcpyPeer.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    runtime.cudaMemcpyPeer.restype = ctypes.c_int
    cases = []
    count = 1 << 20
    expected = torch.arange(count, dtype=torch.int32)
    for source, target in [(0, 1), (1, 0)]:
        assert torch.cuda.can_device_access_peer(source, target)
        x = expected.to(f'cuda:{source}')
        y = torch.empty(count, dtype=torch.int32, device=f'cuda:{target}')
        torch.cuda.set_device(source)
        peer_status = runtime.cudaDeviceEnablePeerAccess(target, 0)
        if peer_status not in (0, 704):
            raise RuntimeError(f'Peer enable failed with status {peer_status}')
        runtime.cudaGetLastError()
        torch.cuda.synchronize(source)
        status = runtime.cudaMemcpyPeer(y.data_ptr(), target, x.data_ptr(), source, x.numel() * x.element_size())
        if status:
            raise RuntimeError(f'cudaMemcpyPeer failed with status {status}')
        torch.cuda.synchronize(target)
        torch.testing.assert_close(y.cpu(), expected, atol=0, rtol=0)
        cases.append({'source': source, 'target': target, 'bytes': x.numel() * x.element_size(), 'exact': True})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'torch': torch.__version__, 'cases': cases}, indent=2) + '\n')
    print('PASS: two bidirectional 4 MiB CUDA peer copies')


if __name__ == '__main__':
    main()

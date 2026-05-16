"""Fingerprint the installed native extension and its pinned source patch."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import torch
import deep_ep_cpp


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('results/deepep-build-manifest.json'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / 'vendor/DeepEP'
    binary = Path(deep_ep_cpp.__file__)
    commit = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    difference = subprocess.check_output(['git','-C',str(source),'diff','--','csrc/deep_ep.cpp'])
    payload = {'recorded_at':datetime.now(timezone.utc).isoformat(), 'upstream_commit':commit,
               'source_diff_sha256':hashlib.sha256(difference).hexdigest(),
               'binding_source_sha256':digest(source/'csrc/deep_ep.cpp'),
               'extension_sha256':digest(binary), 'extension_filename':binary.name,
               'patch_sha256':digest(root/'patches/deepep-v1.2.1-zero-copy-layout.patch'),
               'torch':torch.__version__, 'cuda_runtime':torch.version.cuda,
               'nvshmem':importlib.metadata.version('nvidia-nvshmem-cu12'),
               'python':sys.version.split()[0], 'requirements_sha256':digest(root/'environment/requirements-linux.lock')}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2)+'\n')
    archive = args.output.parent/'builds'/f'{payload["extension_sha256"][:20]}.json'
    archive.parent.mkdir(parents=True,exist_ok=True)
    if not archive.exists():
        archive.write_text(json.dumps(payload,indent=2)+'\n')
    print(f'Recorded native build {payload["extension_sha256"][:20]}',flush=True)


if __name__ == '__main__':
    main()

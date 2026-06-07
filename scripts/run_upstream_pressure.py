"""Bound the pinned upstream precision/hash pressure test with atomic checkpoints."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(local_rank, world, args):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/'vendor/DeepEP/tests'))
    from utils import init_dist
    from test_low_latency import test_main
    from deep_ep import Buffer
    torch.set_num_threads(1)
    rank, ranks, group = init_dist(local_rank, world)
    experts = ranks * args.experts_per_rank
    size = Buffer.get_low_latency_rdma_size_hint(args.tokens, args.hidden, ranks, experts)
    buffer = Buffer(group, num_rdma_bytes=size, low_latency_mode=True,
                    num_qps_per_rank=args.experts_per_rank, allow_nvlink_for_low_latency_mode=not args.rdma_only,
                    explicitly_destroy=True)
    completed = []
    started = time.time()
    try:
        seed = 0
        while True:
            stop = torch.tensor([time.time() >= args.deadline_epoch if rank == 0 else False], dtype=torch.int32, device='cuda')
            dist.broadcast(stop, src=0)
            if bool(stop.item()):
                break
            reference_hash = test_main(args.tokens, args.hidden, experts, args.top_k, rank, ranks, group, buffer, seed=seed)
            for _ in range(args.repeat):
                actual_hash = test_main(args.tokens, args.hidden, experts, args.top_k, rank, ranks, group, buffer, seed=seed)
                if actual_hash != reference_hash:
                    raise AssertionError(f'Upstream hash changed at seed {seed}: {actual_hash} != {reference_hash}')
            hashes = [None] * ranks
            dist.all_gather_object(hashes, int(reference_hash), group=group)
            if rank == 0:
                completed.append({'seed':seed, 'rank_hashes':hashes, 'matching_repetitions':args.repeat,
                                  'completed_utc':datetime.now(timezone.utc).isoformat()})
                args.output.parent.mkdir(parents=True,exist_ok=True)
                temporary = args.output.with_suffix('.partial.json')
                temporary.write_text(json.dumps({'scope':'Pinned upstream identity-expert precision/hash stress, including BF16/FP8 and synchronous/hook variants; not a model-level race reproduction',
                                                 'world_size':ranks,'hidden':args.hidden,'tokens_per_rank':args.tokens,
                                                 'num_experts':experts,'top_k':args.top_k,'finished':False,
                                                 'native_build':json.loads((root/'results/deepep-build-manifest.json').read_text()),
                                                 'allow_nvlink':not args.rdma_only,'elapsed_seconds':time.time()-started,
                                                 'completed_seeds':completed,'deadline_epoch':args.deadline_epoch},indent=2)+'\n')
                temporary.replace(args.output)
                print(f'CHECKPOINT completed_seed={seed} elapsed={time.time()-started:.1f}s',flush=True)
            seed += 1
        if rank == 0:
            if args.output.exists():
                payload = json.loads(args.output.read_text())
                payload.update(finished=True, elapsed_seconds=time.time()-started,
                               finished_utc=datetime.now(timezone.utc).isoformat())
                temporary = args.output.with_suffix('.partial.json')
                temporary.write_text(json.dumps(payload,indent=2)+'\n')
                temporary.replace(args.output)
            print(f'COMPLETE: bounded upstream stress completed {seed} seeds',flush=True)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)  # Do not enter collective teardown after a rank-local failure.
    finally:
        buffer.destroy()
        dist.barrier()
        dist.destroy_process_group()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--world-size',type=int,default=8)
    parser.add_argument('--hidden',type=int,default=7168)
    parser.add_argument('--tokens',type=int,default=128)
    parser.add_argument('--experts-per-rank',type=int,default=4)
    parser.add_argument('--top-k',type=int,default=2)
    parser.add_argument('--repeat',type=int,default=2)
    parser.add_argument('--deadline-epoch',type=float,required=True)
    parser.add_argument('--rdma-only',action='store_true')
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a fresh output path')
    if time.time() >= args.deadline_epoch:
        raise ValueError('Deadline has already passed')
    # The upstream helper interprets WORLD_SIZE/RANK as node counts, not torchrun ranks.
    os.environ.pop('WORLD_SIZE',None)
    os.environ.pop('RANK',None)
    mp.spawn(worker,args=(args.world_size,args),nprocs=args.world_size,join=True)


if __name__ == '__main__':
    main()

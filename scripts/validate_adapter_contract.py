"""Check context ownership, lifecycle and poisoned padding on the native fast path."""
import argparse
from dataclasses import replace
from datetime import timedelta
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from moe_lab.deepep_backend import DeepEPLowLatencyDispatcher
from moe_lab.expert_mlp import evaluate_experts_into
from scripts.validate_deepep import inputs, expert_weights, oracle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--completion', choices=['sync', 'event', 'hook'], default='event')
    parser.add_argument('--rdma-only', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a fresh output path')
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    torch.set_num_threads(1)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group('nccl', timeout=timedelta(seconds=120))
    native = DeepEPLowLatencyDispatcher(world*4, 2048, completion=args.completion,
                                        packing='zero_copy', allow_nvlink=not args.rdma_only)
    records = []
    def rejected(name, function, error_type):
        try:
            function()
        except error_type:
            records.append({'rank': rank, 'case': name, 'passed': True})
        else:
            raise AssertionError(f'{name} did not reject invalid use')
    try:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for scenario in ('balanced', 'skewed', 'empty_rank', 'all_empty', 'all_dropped'):
                cx, ci, cw = inputs(rank, world, 2048, scenario)
                cpu_matrices = expert_weights(world, 2048)
                matrices = [w.cuda() for w in cpu_matrices]
                x, ids, weights = [t.cuda() for t in (cx, ci, cw)]
                expected = oracle(cx, ci, cw, cpu_matrices).cuda()
                batch = native.dispatch_packed(x, ids, weights)
                rejected(f'{scenario}/overlapping_dispatch', lambda: native.dispatch_packed(x, ids, weights), RuntimeError)
                rejected(f'{scenario}/premature_close', native.close, RuntimeError)
                foreign = replace(batch.context, owner=object())
                rejected(f'{scenario}/foreign_context', lambda: native.registered_output(foreign), ValueError)
                registered = native.registered_output(batch.context)
                assert native.registered_output(batch.context) is registered
                rejected(f'{scenario}/unreserved_output', lambda: native.combine_registered(registered.clone(), batch.context), ValueError)
                # Valid row counts must prevent either expert computation or combine
                # from observing invalid padding, even when it contains NaNs.
                for expert, count in enumerate(batch.context.counts):
                    batch.packed_tokens[expert, count:].fill_(float('nan'))
                registered.fill_(float('nan'))
                evaluate_experts_into(batch, matrices, rank, registered)
                actual = native.combine_registered(registered, batch.context)
                assert torch.isfinite(actual).all()
                torch.testing.assert_close(actual.double(), expected, atol=.06, rtol=.06)
                records.append({'rank':rank, 'case':f'{scenario}/poisoned_padding', 'passed':True})
                rejected(f'{scenario}/consumed_context', lambda: native.registered_output(batch.context), ValueError)
                rejected(f'{scenario}/repeated_combine', lambda: native.combine_registered(registered, batch.context), ValueError)
        stream.synchronize()
        gathered = [None] * world
        dist.all_gather_object(gathered, records)
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            payload = {'world_size':world, 'completion':args.completion, 'stream':'non_default',
                       'allow_nvlink':not args.rdma_only, 'native_build':json.loads(Path('results/deepep-build-manifest.json').read_text()),
                       'cases':[r for rs in gathered for r in rs]}
            args.output.write_text(json.dumps(payload,indent=2)+'\n')
            print(f'PASS: {len(records)*world} adapter lifecycle and padding controls', flush=True)
    finally:
        if native._pending is None:
            native.close()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()

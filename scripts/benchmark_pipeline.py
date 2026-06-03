"""Counterbalanced end-to-end MoE timings with per-rank wall-clock samples."""
import argparse
from datetime import timedelta
import itertools
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
import traceback
import torch
import torch.distributed as dist
from moe_lab.dispatch import AllToAllDispatcher
from moe_lab.deepep_backend import DeepEPLowLatencyDispatcher
from scripts.validate_deepep import expert_weights, experts
from moe_lab.expert_mlp import evaluate_experts_into


NATIVE_MODES = {'native_deepep': ('sync', 'zeros', False),
                'native_hint': ('sync', 'zeros', True),
                'native_empty': ('sync', 'empty', True),
                'native_zero_copy': ('sync', 'zero_copy', True),
                'native_event': ('event', 'zero_copy', True),
                'native_hook': ('hook', 'zero_copy', True)}
NATIVE_MODES.update({'native_grouped': ('sync','zero_copy',True),
                     'native_grouped_event': ('event','zero_copy',True),
                     'native_grouped_hook': ('hook','zero_copy',True)})
NATIVE_MODES.update({'native_direct': ('sync','zero_copy',True),
                     'native_direct_event': ('event','zero_copy',True),
                     'native_direct_hook': ('hook','zero_copy',True)})
NATIVE_MODES.update({'native_packed': ('sync','zero_copy',True),
                     'native_packed_event': ('event','zero_copy',True),
                     'native_packed_hook': ('hook','zero_copy',True)})
REGISTERED = {'native_direct', 'native_direct_event', 'native_direct_hook'}
PACKED = {'native_packed', 'native_packed_event', 'native_packed_hook'}
REGISTERED.update(PACKED)
GROUPED = {'baseline_grouped', 'optimized_grouped', 'native_grouped',
           'native_grouped_event', 'native_grouped_hook'}


def make_inputs(rank, world, tokens, hidden, scenario):
    generator = torch.Generator().manual_seed(107 + rank)
    count = tokens + rank * 3
    x = torch.randn(count, hidden, generator=generator).bfloat16().cuda()
    ids = torch.stack([torch.randperm(world * 4, generator=generator)[:2] for _ in range(count)]).cuda()
    if scenario == 'skewed':
        ids[:] = torch.tensor([0, 1], device=x.device)
    weights = torch.rand(count, 2, generator=generator).cuda()
    weights /= weights.sum(1, keepdim=True)
    return x, ids, weights


@torch.no_grad()
def pipeline(dispatcher, x, ids, weights, matrices, rank, *, contiguous_experts=False, registered_experts=False, packed_experts=False):
    batch = dispatcher.dispatch_packed(x, ids, weights) if packed_experts else dispatcher.dispatch(x, ids, weights)
    if registered_experts:
        registered = dispatcher.registered_output(batch.context)
        evaluate_experts_into(batch, matrices, rank, registered)
        return dispatcher.combine_registered(registered, batch.context)
    output = experts(batch, matrices, rank, contiguous=contiguous_experts)
    return dispatcher.combine(output, batch.context)


def summarize(values):
    ordered = sorted(values)
    return {'median_ms': statistics.median(values), 'min_ms': min(values),
            'p25_ms': ordered[len(ordered)//4], 'p75_ms': ordered[3*len(ordered)//4],
            'p95_ms': ordered[min(len(ordered)-1, math.ceil(.95*len(ordered))-1)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokens', nargs='+', type=int, default=[16, 64, 256, 512])
    parser.add_argument('--hidden', nargs='+', type=int, default=[2048, 7168])
    parser.add_argument('--scenarios', nargs='+', choices=['balanced', 'skewed'], default=['balanced', 'skewed'])
    parser.add_argument('--rdma-only', action='store_true')
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--methods', nargs='+', default=['baseline_five', 'optimized_three', 'native_deepep'],
                        choices=['baseline_five', 'optimized_three', 'optimized_hint', 'optimized_ordered',
                                 'baseline_grouped', 'optimized_grouped', *NATIVE_MODES])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a fresh output path')
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    torch.set_num_threads(1)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group('nccl', timeout=timedelta(seconds=120))
    methods = args.methods
    if len(set(methods)) != len(methods) or not 2 <= len(methods) <= 4:
        raise ValueError('Choose two to four distinct methods')
    records, native = [], None
    try:
        capacity = 2 ** math.ceil(math.log2(max(args.tokens) + (world-1)*3 + 1))
        for hidden in args.hidden:
            matrices = [w.cuda() for w in expert_weights(world, hidden)]
            native = DeepEPLowLatencyDispatcher(world*4, hidden, max_tokens=capacity, allow_nvlink=not args.rdma_only)
            implementations = {'baseline_five': AllToAllDispatcher(world*4, optimized=False),
                               'optimized_three': AllToAllDispatcher(world*4),
                               'optimized_hint': AllToAllDispatcher(world*4, known_output_size=True),
                               'optimized_ordered': AllToAllDispatcher(world*4, known_output_size=True, combine_mode='ordered'),
                               'baseline_grouped': AllToAllDispatcher(world*4, optimized=False, known_output_size=True, host_expert_counts=True),
                               'optimized_grouped': AllToAllDispatcher(world*4, known_output_size=True, host_expert_counts=True),
                               **{key: native for key in NATIVE_MODES}}
            for tokens, scenario in itertools.product(args.tokens, args.scenarios):
                print(f'rank={rank} hidden={hidden} tokens={tokens} scenario={scenario}', flush=True)
                x, ids, weights = make_inputs(rank, world, tokens, hidden, scenario)
                def invoke(method):
                    if method in NATIVE_MODES:
                        native.completion, native.packing, native.known_output_size = NATIVE_MODES[method]
                    return pipeline(implementations[method], x, ids, weights, matrices, rank,
                                    contiguous_experts=method in GROUPED, registered_experts=method in REGISTERED,
                                    packed_experts=method in PACKED)
                reference = pipeline(implementations['baseline_five'], x, ids, weights, matrices, rank)
                validation = {}
                for method in methods:
                    output = invoke(method)
                    torch.testing.assert_close(output, reference, atol=.03, rtol=.03)
                    validation[method] = float((output.float()-reference.float()).abs().max())
                    for _ in range(args.warmup):
                        invoke(method)
                samples = {m: [] for m in methods}
                schedule = list(itertools.permutations(methods)) * args.rounds
                for order in schedule:
                    for method in order:
                        dist.barrier()
                        torch.cuda.synchronize()
                        begin = time.perf_counter_ns()
                        invoke(method)
                        torch.cuda.synchronize()
                        samples[method].append((time.perf_counter_ns()-begin)/1e6)
                gathered = [None] * world
                dist.all_gather_object(gathered, {'rank': rank, 'samples': samples, 'validation': validation})
                if rank == 0:
                    total_tokens = sum(tokens+r*3 for r in range(world))
                    worst = {m: [max(g['samples'][m][i] for g in gathered)
                                 for i in range(len(samples[m]))] for m in methods}
                    record = {'hidden': hidden, 'tokens_rank_zero': tokens, 'scenario': scenario,
                              'world_size': world, 'top_k': 2, 'experts': world*4,
                              'expert_intermediate': 128, 'total_input_tokens': total_tokens,
                              'samples_per_method': len(schedule), 'rank_results': gathered,
                              'worst_rank_samples_ms': worst,
                              'summary': {m: dict(summarize(v), tokens_per_second=total_tokens/(statistics.median(v)/1000))
                                          for m,v in worst.items()}}
                    records.append(record)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    temp = args.output.with_suffix('.partial.json')
                    root = Path(__file__).resolve().parents[1]
                    source_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for folder in ['moe_lab', 'scripts'] for p in sorted((root/folder).glob('*.py'))}
                    temp.write_text(json.dumps({'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
                                                'source_hashes': source_hashes,
                                                'native_modes': NATIVE_MODES, 'methods': methods, 'allow_nvlink': not args.rdma_only,
                                                'native_build':json.loads((root/'results/deepep-build-manifest.json').read_text()) if (root/'results/deepep-build-manifest.json').exists() else None,
                                                'measurement': 'Synchronized wall-clock full dispatch/expert/combine; worst rank per sample; barriers excluded',
                                                'ordering': f'All {math.factorial(len(methods))} permutations, repeated counterbalanced rounds',
                                                'records': records}, indent=2)+'\n')
                    temp.replace(args.output)
                for implementation in implementations.values():
                    if hasattr(implementation, 'collectives'):
                        implementation.collectives.clear()
            native.close()
            native = None
        if rank == 0:
            print(f'COMPLETE: {len(records)} end-to-end benchmark workloads', flush=True)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if native is not None and native._pending is None:
            native.close()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
